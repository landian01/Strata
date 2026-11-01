"""Opt-in, Windows-only private stop channel. No listener, service, or public shutdown route.

The gateway starts server.py in the foreground with --control-file ABS and --shutdown-timeout 300.
Its cmdStop is: python -B <overlay>/serve/control.py stop --file ABS --pid PID --timeout 360
The same control path serializes different model profiles. Its parent must already be owner-only.
Files are deliberately retained: a stop client needs the final result, and a later start must verify stale identity.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # direct script use with the embedded Python, from any cwd
from serve import winjob
from strata_control import private_files

WINDOWS = os.name == "nt"
MAX_RECORD = 16384
STATES = {"loading", "ready", "stopping", "stopped", "aborted", "forced"}

if WINDOWS:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _adv = ctypes.WinDLL("advapi32", use_last_error=True)
    _k32.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _k32.LocalFree.argtypes = (ctypes.c_void_p,)
    _k32.GetVolumePathNameW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    _k32.GetVolumeInformationW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR,
                                         wintypes.DWORD)

    class _FileInfo(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
                    ("access", wintypes.FILETIME), ("write", wintypes.FILETIME), ("volume", wintypes.DWORD),
                    ("size_high", wintypes.DWORD), ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                    ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]

    class _Overlapped(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_size_t), ("internal_high", ctypes.c_size_t),
                    ("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD), ("event", wintypes.HANDLE)]

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("descriptor", ctypes.c_void_p), ("inherit", wintypes.BOOL)]

    _k32.GetFileInformationByHandle.argtypes = (wintypes.HANDLE, ctypes.POINTER(_FileInfo))
    _k32.LockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                               wintypes.DWORD, ctypes.POINTER(_Overlapped))
    _k32.UnlockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.POINTER(_Overlapped))
    _adv.GetSecurityInfo.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_void_p))
    _adv.GetSecurityInfo.restype = wintypes.DWORD
    _adv.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = (
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p)
    _adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)


def _error(what):
    return ctypes.WinError(ctypes.get_last_error(), what)


def _private_sddl(sddl: str, sid: str) -> bool:
    # Require the real owner and a protected DACL, exactly one unconditional full-control allow for that owner.
    owner = re.search(r"O:([^:()]+?)(?=[GDS]:|$)", sddl)
    dacl = re.search(r"D:([^()]*)", sddl)
    entries = re.findall(r"\(([^()]*)\)", sddl)
    if not owner or not dacl or "P" not in dacl[1] or len(entries) != 1:
        return False
    fields = entries[0].split(";")
    return (private_files._same_sid(owner[1], sid) and len(fields) == 6 and fields[0] == "A"
            and fields[1] in ("", "OICI", "CIOI") and fields[2] in ("FA", "0x1f01ff")
            and not fields[3] and not fields[4] and private_files._same_sid(fields[5], sid))


def _check_handle(handle, *, directory: bool, private: bool = True):
    info = _FileInfo()
    if not _k32.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise _error("cannot inspect control path handle")
    if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
        raise ValueError("control paths must be ordinary files/directories, never reparse points")
    if not directory and info.links != 1:
        raise ValueError("control files must not have hard links")
    if not private:
        return
    descriptor = ctypes.c_void_p()
    code = _adv.GetSecurityInfo(handle, 1, 1 | 4, None, None, None, None, ctypes.byref(descriptor))
    if code:
        raise ctypes.WinError(code, "cannot inspect control path security")
    try:
        text = wintypes.LPWSTR()
        if not _adv.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor, 1, 1 | 4, ctypes.byref(text), None):
            raise _error("cannot decode control path security")
        try:
            if not _private_sddl(text.value, private_files.current_user_sid()):
                raise ValueError("control files and their parent must have an owner-only protected ACL")
        finally:
            _k32.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        _k32.LocalFree(descriptor)


def _path(value: str) -> Path:
    if not WINDOWS:
        raise ValueError("--control-file currently requires Windows")
    path = Path(value)
    if (not path.is_absolute() or not re.match(r"^[a-zA-Z]:[\\/]", value)
            or any(c in value[2:] for c in (":", "\x00"))
            or any(p.endswith((".", " ")) or p in (".", "..") for p in re.split(r"[\\/]", value[3:]))):
        raise ValueError("control file must be a normal absolute local path (no UNC, device, ADS or dot paths)")
    return path


class _Area:
    """Pin every directory, denying rename/delete while open, and check the immediate parent's private ACL."""

    def __init__(self, value: str):
        self.path = _path(value)
        self.handles = []
        try:
            directories = list(reversed(self.path.parent.parents)) + [self.path.parent]
            for folder in directories:
                handle = _k32.CreateFileW(str(folder), 0x20000 | 0x80, 3, None, 3, 0x02200000, None)
                if handle == wintypes.HANDLE(-1).value:
                    raise _error("cannot pin control parent")
                self.handles.append(handle)
                _check_handle(handle, directory=True, private=folder == self.path.parent)
            root, flags = ctypes.create_unicode_buffer(32768), wintypes.DWORD()
            if not _k32.GetVolumePathNameW(str(self.path.parent), root, len(root)):
                raise _error("cannot establish control volume")
            if not _k32.GetVolumeInformationW(root.value, None, 0, None, None, ctypes.byref(flags), None, 0):
                raise _error("cannot establish control volume ACL support")
            if not flags.value & 8:
                raise ValueError("control volume must support persistent ACLs")
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in reversed(self.handles):
            _k32.CloseHandle(handle)
        self.handles.clear()


class _File:
    """A checked, non-inheritable file HANDLE. Byte locks beyond EOF do not block clients reading the JSON."""

    def __init__(self, path: Path, *, create: bool):
        self.path, self.new, self.file = path, False, None
        descriptor = ctypes.c_void_p()
        sid = private_files.current_user_sid()
        if not _adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                f"O:{sid}D:P(A;;FA;;;{sid})", 1, ctypes.byref(descriptor), None):
            raise _error("cannot construct private control ACL")
        try:
            security = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
            handle = _k32.CreateFileW(str(path), 0xc0000000, 3, ctypes.byref(security),
                                      1 if create else 3, 0x02200000, None)
            if handle == wintypes.HANDLE(-1).value and create and ctypes.get_last_error() in (80, 183):
                handle = _k32.CreateFileW(str(path), 0xc0000000, 3, None, 3, 0x02200000, None)
            elif handle != wintypes.HANDLE(-1).value and create:
                self.new = True
            if handle == wintypes.HANDLE(-1).value:
                raise _error("cannot open private control file")
        finally:
            _k32.LocalFree(descriptor)
        try:
            _check_handle(handle, directory=False)
            fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
        except BaseException:
            _k32.CloseHandle(handle)
            raise
        self.file = os.fdopen(fd, "r+b", buffering=0)
        self.handle = handle
        self.overlapped = _Overlapped(offset=0x7ffffffe)

    def lock(self):
        if not _k32.LockFileEx(self.handle, 3, 0, 1, 0, ctypes.byref(self.overlapped)):
            raise _error("control file is in use (native byte lock)")

    def unlock(self):
        if not _k32.UnlockFileEx(self.handle, 0, 1, 0, ctypes.byref(self.overlapped)):
            raise _error("cannot release control byte lock")

    def read(self) -> dict:
        _check_handle(self.handle, directory=False)
        self.file.seek(0)
        raw = self.file.read(MAX_RECORD + 1)
        if len(raw) > MAX_RECORD:
            raise ValueError("control record is too large")
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError("invalid control record")
        return record

    def write(self, record: dict):
        _check_handle(self.handle, directory=False)
        raw = json.dumps(record, separators=(",", ":")).encode("ascii")
        if len(raw) > MAX_RECORD:
            raise ValueError("control record is too large")
        self.file.seek(0)
        if self.file.write(raw) != len(raw):
            raise OSError("incomplete control record write")
        self.file.truncate()
        os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def _binding(record: dict) -> dict:
    if (type(record.get("version")) is not int or record["version"] != 1
            or type(record.get("pid")) is not int or not 0 < record["pid"] <= 0xffffffff
            or not isinstance(record.get("token"), str) or not re.fullmatch(r"[1-9][0-9]{0,19}", record["token"])
            or not isinstance(record.get("nonce"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["nonce"])):
        raise ValueError("invalid control identity")
    return {k: record[k] for k in ("version", "pid", "token", "nonce")}


def _record(record: dict) -> dict:
    _binding(record)
    if (record.get("kind") != "strata-native-control" or not isinstance(record.get("state"), str)
            or record["state"] not in STATES
            or type(record.get("ready")) is not bool or type(record.get("exit_code")) not in (int, type(None))):
        raise ValueError("invalid control status")
    return record


def _request(record: dict, binding: dict) -> str:
    if (_binding(record) != binding or record.get("kind") != "strata-native-stop"
            or record.get("action") not in ("none", "stop")):
        raise ValueError("stop request does not match this instance")
    return record["action"]


def _old_is_gone(record):
    try:
        process = winjob.ProcessIdentity(record["pid"])
    except OSError as exc:
        if getattr(exc, "winerror", None) == 87:  # ERROR_INVALID_PARAMETER: no such process. Access denied is NOT absence.
            return
        raise
    try:
        if process.token == record["token"] and process.alive():
            raise ValueError("control file still belongs to a live instance")
    finally:
        process.close()


class Controller:
    """Helper owned by the native foreground server, not another backend/owner process."""

    def __init__(self, value: str, request_stop, *, grace: float):
        if not math.isfinite(grace) or grace <= 0:
            raise ValueError("control grace must be positive and finite")
        self.area = self.status = self.requests = self.parent = None
        self.guard = threading.RLock()
        self.done = threading.Event()
        self.ready = self.latched = self.completed = False
        self.deadline = None
        self.callback, self.grace = request_stop, grace
        try:
            self.area = _Area(value)
            self.status = _File(self.area.path, create=True)
            self.status.lock()                      # held until process exit, even after finish()
            old = None if self.status.new else _record(self.status.read())
            if old is not None:
                _old_is_gone(old)
            self.requests = _File(Path(str(self.area.path) + ".stop"), create=True)
            self.requests.lock()
            try:
                if not self.requests.new:
                    if old is None:
                        raise ValueError("unbound existing stop file")
                    _request(self.requests.read(), _binding(old))
                self.parent = winjob.direct_parent()  # stable HANDLE, never just poll the parent's PID
                winjob.require_self_job()             # mandatory before polling or spawning any engine
                current = winjob.ProcessIdentity(os.getpid())
                try:
                    self.binding = {"version": 1, "pid": current.pid, "token": current.token,
                                    "nonce": os.urandom(32).hex()}
                finally:
                    current.close()
                self.requests.write({**self.binding, "kind": "strata-native-stop", "action": "none"})
                self._publish("loading", None)
            finally:
                self.requests.unlock()
        except BaseException:
            self.release()
            raise
        atexit.register(self.abort)
        self.thread = threading.Thread(target=self._watch, name="strata-native-control", daemon=True)
        self.thread.start()

    def _publish(self, state, exit_code, **details):
        self.status.write({**self.binding, "kind": "strata-native-control", "state": state,
                           "ready": self.ready, "exit_code": exit_code, **details})

    def mark_ready(self):
        with self.guard:
            self.ready = True
            if not self.latched:
                self._publish("ready", None)

    def stop_once(self, reason: str):
        with self.guard:
            if self.latched or self.completed:
                return
            self.latched = True
            self.deadline = time.monotonic() + self.grace
            try:
                self._publish("stopping", None, reason=reason)
            except (OSError, ValueError):
                pass                               # the self-job deadline must survive a damaged status file
        self.callback(reason)                       # exactly one interrupt, including repeated stop requests

    def _watch(self):
        while not self.done.wait(0.2):
            try:
                if not self.parent.alive():
                    self.stop_once("parent-lost")
                if not self.latched:
                    try:
                        self.requests.lock()
                    except OSError as exc:
                        if getattr(exc, "winerror", None) != 33:  # ERROR_LOCK_VIOLATION: writer is publishing
                            raise
                    else:
                        try:
                            if _request(self.requests.read(), self.binding) == "stop":
                                self.stop_once("control-stop")
                        finally:
                            self.requests.unlock()
            except (OSError, ValueError):
                self.stop_once("control-error")
            if self.deadline is not None and time.monotonic() >= self.deadline:
                try:
                    with self.guard:
                        self._publish("forced", 9, reason="control-deadline")
                finally:
                    winjob.terminate_self_job(9)
                return

    def finish(self, exit_code: int, *, forced: bool = False):
        with self.guard:
            self._publish("forced" if forced else "stopped", exit_code,
                          graceful=exit_code == 0 and not forced)
            self.completed = True
            self.done.set()

    def abort(self):
        # The existing server registers lifeline cleanup before this helper. At interpreter exit this records an
        # interrupted/failed startup, NOT a drain/QUIT success. The self job is still held until the OS closes it.
        with self.guard:
            if self.completed:
                return
            self.done.set()
            try:
                self._publish("aborted", 10, reason="ready-abort" if self.ready else "nonready-abort", graceful=False)
            except (OSError, ValueError):
                pass
            self.completed = True

    def release(self):
        """Constructor failure / tests only. Normal server lifetime MUST retain the control lock until exit."""
        for resource in (self.requests, self.status, self.parent, self.area):
            if resource is not None:
                resource.close()


def close_failed(engine) -> bool:
    """StrataEngine and replica members retain forced-close / nonzero-exit evidence after proc is cleared."""
    members = getattr(engine, "engines", None) or []
    return (bool(getattr(engine, "close_forced", False))
            or getattr(engine, "close_exit_code", 0) not in (0, None)
            or any(close_failed(child) for child in members))


def _stop_result(pid, record, code, outcome, *, error=None):
    nonce = record.get("nonce", "") if record else ""
    instance = hashlib.sha256(nonce.encode("ascii")).hexdigest()[:24] if nonce else None
    print(json.dumps({"pid": pid, "instance": instance, "ready": record.get("ready") if record else None,
                      "phase": record.get("state") if record else None, "graceful": code == 0,
                      "forced": bool(record and record.get("state") == "forced"), "outcome": outcome,
                      "exit_code": code, **({"error": str(error)} if error else {})}), flush=True)
    return code


def _read_status(file):
    # A status update is an in-place write under the lifetime mutex. A client never takes that mutex, so retry
    # only a transient partial JSON write; ACL/identity failures still fail closed immediately.
    for attempt in range(20):
        try:
            return _record(file.read())
        except json.JSONDecodeError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def stop(value: str, pid: int, timeout: float) -> int:
    area = status = requests = process = record = None
    try:
        if type(pid) is not int or pid <= 0 or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("stop needs a positive PID and finite timeout in (0, 3600]")
        area = _Area(value)
        status = _File(area.path, create=False)
        record = _read_status(status)
        binding = _binding(record)
        if binding["pid"] != pid:
            raise ValueError("requested PID does not own the control file")
        process = winjob.ProcessIdentity(pid)
        if process.token != binding["token"] or not process.alive():
            raise ValueError("process creation token does not match a live control instance")
        requests = _File(Path(str(area.path) + ".stop"), create=False)
        deadline = time.monotonic() + timeout
        # Serialize concurrent cmdStop publishers. The server's lifetime mutex is on the OTHER file.
        while True:
            try:
                requests.lock()
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) != 33 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        try:
            _request(requests.read(), binding)
            if _binding(_read_status(status)) != binding or not process.alive():
                raise ValueError("control instance changed before stop publication")
            requests.write({**binding, "kind": "strata-native-stop", "action": "stop"})
        finally:
            requests.unlock()
        while process.alive():
            if time.monotonic() >= deadline:
                return _stop_result(pid, _read_status(status), 7, "timeout-unconfirmed")
            time.sleep(0.1)
        # Only a held HANDLE's exit, a matching final record AND actual exit code zero establish graceful success.
        final = _read_status(status)
        if (_binding(final) != binding or final["state"] != "stopped" or final["exit_code"] != 0
                or not final["ready"] or final.get("graceful") is not True or process.exit_code() != 0):
            return _stop_result(pid, final, 9, "exit-unconfirmed")
        return _stop_result(pid, final, 0, "graceful")
    except (OSError, ValueError) as exc:
        return _stop_result(pid, record, 2, "refused", error=exc)
    finally:
        for resource in (requests, status, process, area):
            if resource is not None:
                resource.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    command = commands.add_parser("stop", help="request native drain/close and verify the process exit")
    command.add_argument("--file", required=True)
    command.add_argument("--pid", required=True, type=int)
    command.add_argument("--timeout", type=float, default=360)
    args = ap.parse_args(argv)
    try:
        return stop(args.file, args.pid, args.timeout)
    except (OSError, ValueError) as exc:
        return _stop_result(args.pid, None, 2, "refused", error=exc)


if __name__ == "__main__":
    sys.exit(main())
