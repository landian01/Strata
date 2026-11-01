"""serve/winjob.py - on Windows, tie the processes Strata starts to the server's own lifetime.

Closing the console window, Task Manager or a crash end the server without running its cleanup, and the engine,
the vision encoder and the MCP servers kept running on their own. `contain(proc)` puts a child in a job object that
is set to kill everything in it when its last handle closes - and the only handle is this process's, which the OS
closes however the server ends. Processes the child starts later join the same job. The server itself stays out of
the job, so a browser that `--open` starts is not tied to it. Each child (and the server) is also opted out of Windows
power throttling, see `no_throttle` (#691). Elsewhere, and if the job cannot be made, a no-op.
"""
from __future__ import annotations

import os
import threading

_job = None
_self_job = None                 # opt-in control mode: includes us, so never close it in a running interpreter
_lock = threading.Lock()

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _k32.SetHandleInformation.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD)
    _k32.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
    _k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
    _k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))

    def _make_job():
        job = _k32.CreateJobObjectW(None, None)     # no security attributes: the handle is not inherited
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
                                            ctypes.sizeof(info)):
            _k32.CloseHandle(job)
            return None
        return job

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    _k32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE

    def no_throttle(handle) -> bool:
        """Opt a process out of Windows power throttling (EcoQoS).  Without it Windows 11 treats a process whose console
        window is minimized as background work and, on a hybrid CPU, moves its threads to the E-cores: generation
        slowed by about 20% while the server window was minimized (#691, i9-13980HX: 36.6 -> 43.2 tok/s)."""
        for mask in (0x1 | 0x4, 0x1):               # EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION, both off (Windows 11);
            info = _PowerThrottling(1, mask, 0)     # Windows 10 knows the first only: ERROR_INVALID_PARAMETER for both
            if _k32.SetProcessInformation(handle, 4, ctypes.byref(info), ctypes.sizeof(info)):   # ProcessPowerThrottling
                return True
        return False

    no_throttle(_k32.GetCurrentProcess())           # the server itself (tokenizing, streaming) too
else:
    def no_throttle(handle) -> bool:
        return False


def require_self_job() -> None:
    """Fail closed before spawning anything. Our non-inheritable handle owns us and all future descendants.

    No BREAKAWAY flags are allowed. Nested-job refusal (including a restrictive gateway job) is a startup error,
    never a warning. The handle intentionally lives until process exit, when the OS closes it.
    """
    global _self_job
    if os.name != "nt":
        raise OSError("--control-file currently requires Windows")
    with _lock:
        if _self_job:
            return
        job = _make_job()
        if not job:
            raise ctypes.WinError(ctypes.get_last_error(), "cannot create control-mode self job")
        if not _k32.SetHandleInformation(job, 1, 0):
            _k32.CloseHandle(job)
            raise ctypes.WinError(ctypes.get_last_error(), "cannot make self-job handle non-inheritable")
        if not _k32.AssignProcessToJobObject(job, _k32.GetCurrentProcess()):
            error = ctypes.get_last_error()
            _k32.CloseHandle(job)
            raise ctypes.WinError(error, "cannot assign server to control-mode self job")
        _self_job = job
        inside = wintypes.BOOL()
        if not _k32.IsProcessInJob(_k32.GetCurrentProcess(), job, ctypes.byref(inside)) or not inside.value:
            raise OSError("cannot verify control-mode self job")


def terminate_self_job(exit_code: int = 9) -> None:
    """Deadline fallback: end only this server's strict job, including this process. Never search by process name."""
    if os.name != "nt" or not _self_job:
        raise OSError("no verified control-mode self job")
    if not _k32.TerminateJobObject(_self_job, exit_code):
        raise ctypes.WinError(ctypes.get_last_error(), "cannot terminate control-mode self job")


class ProcessIdentity:
    """A held process HANDLE plus kernel creation time. PID reuse cannot change what is waited on."""

    def __init__(self, pid: int):
        if os.name != "nt" or type(pid) is not int or not 0 < pid <= 0xffffffff:
            raise ValueError("a positive Windows PID is required")
        self.pid = pid
        self.handle = _k32.OpenProcess(0x100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error(), "cannot open process identity")
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not _k32.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
                raise ctypes.WinError(ctypes.get_last_error(), "cannot read process creation time")
            self.token = str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        except BaseException:
            self.close()
            raise

    def alive(self) -> bool:
        result = _k32.WaitForSingleObject(self.handle, 0)
        if result not in (0, 258):
            raise OSError("cannot wait for process identity")
        return result == 258

    def exit_code(self) -> int:
        if self.alive():
            raise OSError("process has not exited")
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error(), "cannot read process exit code")
        return code.value

    def close(self) -> None:
        if self.handle:
            _k32.CloseHandle(self.handle)
            self.handle = None


def direct_parent() -> ProcessIdentity:
    """Open our direct parent before loading. Refuse an already-lost or reused parent PID."""
    if os.name != "nt":
        raise OSError("parent HANDLE monitoring requires Windows")

    class BasicInfo(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_void_p), ("peb", ctypes.c_void_p),
                    ("reserved2", ctypes.c_void_p * 2), ("pid", ctypes.c_size_t), ("parent", ctypes.c_size_t)]

    query = ctypes.WinDLL("ntdll").NtQueryInformationProcess
    query.argtypes = (wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p)
    query.restype = ctypes.c_long
    info = BasicInfo()
    if query(_k32.GetCurrentProcess(), 0, ctypes.byref(info), ctypes.sizeof(info), None) != 0:
        raise OSError("cannot establish direct parent identity")
    parent = ProcessIdentity(int(info.parent))
    try:
        current = ProcessIdentity(os.getpid())
        try:
            if int(parent.token) >= int(current.token) or not parent.alive():
                raise OSError("direct parent exited or its PID was reused before monitoring")
        finally:
            current.close()
    except BaseException:
        parent.close()
        raise
    return parent


def terminate() -> bool:
    """End every process in the job now (the deadline kill of an owner that must not leave a tree behind). False when
    there is no job (nothing was contained, or not Windows)."""
    if os.name != "nt" or not _job:
        return False
    return bool(_k32.TerminateJobObject(_job, 1))


def contain(proc) -> bool:
    """Put a started subprocess.Popen in the kill-on-close job. True if it is in; False (harmlessly) otherwise."""
    global _job
    if os.name != "nt" or proc is None:
        return False
    if _self_job:
        # Descendants inherit the strict job at CreateProcess, before any child code can run. Do not silently fall
        # back to the old best-effort child job in control mode.
        inside = wintypes.BOOL()
        if not _k32.IsProcessInJob(int(proc._handle), _self_job, ctypes.byref(inside)) or not inside.value:
            raise OSError("control-mode child did not inherit the server's self job")
        no_throttle(int(proc._handle))
        return True
    with _lock:
        if _job is None:
            _job = _make_job() or 0
        if not _job:
            return False
    try:
        no_throttle(int(proc._handle))
        return bool(_k32.AssignProcessToJobObject(_job, int(proc._handle)))
    except (AttributeError, OSError, ValueError):
        return False
