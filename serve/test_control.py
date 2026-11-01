"""CPU-only unittest for native private control. No model, GPU, engine binary or pytest.

Run even with an embedded Python from an unrelated cwd:
    python -B D:/Strata/source/serve/test_control.py
Subprocess fixtures use files / inherited stdio, never captured Windows pipes.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import control, winjob
from strata_control import private_files

PYTHON = getattr(sys, "_base_executable", None) or sys.executable
BINDING = {"version": 1, "pid": 123, "token": "456", "nonce": "a" * 64}


def fixture(mode, directory):
    directory = Path(directory)
    control_path = str(directory / "backend.json")
    if mode == "launcher":
        child = subprocess.Popen([PYTHON, "-B", str(ROOT / "serve/server.py"), "--engine", "mock", "--port", "0",
                                  "--control-file", control_path, "--read-only-config", "--shutdown-timeout", "1"],
                                 stdin=subprocess.DEVNULL)
        (directory / "server.pid").write_text(str(child.pid))
        time.sleep(60)
        return
    if mode == "loading":
        from serve import server
        def interrupt(reason):
            import _thread
            _thread.interrupt_main()
        control.Controller(control_path, interrupt, grace=5)
        server.CONTROL_READY_POLL_S = 0.2
        original_queue = server.queue.Queue
        class ReadyQueue(original_queue):
            def get(self, *args, **kwargs):
                (directory / "ready-wait.flag").write_text("waiting")
                return super().get(*args, **kwargs)
        def silent_engine(what, args, **kwargs):
            child = server.lifeline.popen([PYTHON, "-B", "-c", "import time; time.sleep(60)"], "engine", **kwargs)
            (directory / "child.pid").write_text(str(child.pid))
            return child
        # Exercise the actual startup READY queue, not a sleep-loop stand-in.
        with mock.patch.object(server, "popen", side_effect=silent_engine), mock.patch.object(server.queue, "Queue", ReadyQueue):
            server.StrataEngine(PYTHON, [])
        raise AssertionError("silent engine unexpectedly became ready")
    if mode in ("tree", "deadline", "jobtree"):
        guard = None
        if mode == "jobtree":
            winjob.require_self_job()
        else:
            guard = control.Controller(control_path, lambda reason: None, grace=0.6)
        grandchild_path = str(directory / "grandchild.pid")
        code = ("import subprocess,sys,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); "
                f"Path({grandchild_path!r}).write_text(str(p.pid)); time.sleep(60)")
        child = subprocess.Popen([PYTHON, "-B", "-c", code], stdin=subprocess.DEVNULL)
        (directory / "child.pid").write_text(str(child.pid))
        if guard is not None:
            guard.mark_ready()
        else:
            (directory / "job.json").write_text(json.dumps({"contained": winjob.contain(child),
                                                           "inheritable": os.get_handle_inheritable(winjob._self_job)}))
        time.sleep(60)
        return
    raise ValueError(mode)


class Protocol(unittest.TestCase):
    def test_binding_rejects_wrong_types_and_malformed_tokens(self):
        self.assertEqual(control._binding(BINDING), BINDING)
        for key, value in (("pid", True), ("pid", 0), ("token", "0"), ("token", "x"),
                           ("nonce", "a" * 63), ("version", 2)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                control._binding({**BINDING, key: value})

    def test_request_is_instance_bound(self):
        request = {**BINDING, "kind": "strata-native-stop", "action": "stop"}
        self.assertEqual(control._request(request, BINDING), "stop")
        for change in ({"pid": 124}, {"token": "457"}, {"nonce": "b" * 64}, {"action": "kill"}):
            with self.assertRaises(ValueError):
                control._request({**request, **change}, BINDING)

    def test_private_acl_is_strict(self):
        sid = "S-1-5-21-1-2-3-1001"
        self.assertTrue(control._private_sddl(f"O:{sid}D:P(A;;FA;;;{sid})", sid))
        self.assertTrue(control._private_sddl(f"O:{sid}D:PAI(A;OICI;FA;;;{sid})", sid))
        for sddl in (f"O:{sid}D:(A;;FA;;;{sid})", f"O:{sid}D:P(A;;FA;;;WD)",
                     f"O:{sid}D:P(A;;FA;;;{sid})(A;;FA;;;SY)", f"O:{sid}D:P(A;IO;FA;;;{sid})",
                     f"O:BAD:D:P(A;;FA;;;{sid})", f"O:{sid}D:P(A;;FR;;;{sid})"):
            self.assertFalse(control._private_sddl(sddl, sid), sddl)

    def test_close_evidence_including_replicas(self):
        from types import SimpleNamespace as NS
        self.assertFalse(control.close_failed(NS()))
        self.assertTrue(control.close_failed(NS(close_forced=True)))
        self.assertTrue(control.close_failed(NS(close_exit_code=7)))
        self.assertTrue(control.close_failed(NS(engines=[NS(), NS(close_forced=True)])))

    def test_stop_json_does_not_disclose_nonce(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(control._stop_result(123, {**BINDING, "ready": True, "state": "stopped"}, 0,
                                                 "graceful"), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["graceful"])
        self.assertEqual(len(result["instance"]), 24)
        self.assertNotIn(BINDING["nonce"], output.getvalue())


class ControllerLogic(unittest.TestCase):
    def make_controller(self):
        status = mock.Mock(new=True)
        requests = mock.Mock(new=True)
        parent = mock.Mock()
        parent.alive.return_value = True
        identity = mock.Mock(pid=123, token="456")
        callback = mock.Mock()
        with mock.patch.object(control, "_Area") as area, mock.patch.object(control, "_File", side_effect=[status, requests]), \
             mock.patch.object(winjob, "direct_parent", return_value=parent), \
             mock.patch.object(winjob, "ProcessIdentity", return_value=identity), \
             mock.patch.object(winjob, "require_self_job"), mock.patch.object(threading.Thread, "start"):
            area.return_value.path = Path("D:/private/backend.json")
            controller = control.Controller("D:/private/backend.json", callback, grace=1)
        self.addCleanup(controller.release)
        self.addCleanup(controller.abort)
        return controller, status, requests, parent, callback

    def test_latched_stop_and_nonready_abort_are_not_graceful(self):
        controller, status, requests, parent, callback = self.make_controller()
        self.assertEqual(status.write.call_args.args[0]["state"], "loading")
        controller.stop_once("control-stop")
        controller.stop_once("parent-lost")
        controller.mark_ready()
        callback.assert_called_once_with("control-stop")
        self.assertEqual(status.write.call_args.args[0]["state"], "stopping")
        controller.ready = False
        controller.abort()
        final = status.write.call_args.args[0]
        self.assertEqual(final["state"], "aborted")
        self.assertEqual(final["reason"], "nonready-abort")
        self.assertFalse(final["graceful"])

    def test_parent_loss_deadline_calls_only_self_job(self):
        controller, status, requests, parent, callback = self.make_controller()
        parent.alive.return_value = False
        controller.done = mock.Mock()
        controller.done.wait.return_value = False
        with mock.patch.object(control.time, "monotonic", side_effect=[100, 102]), \
             mock.patch.object(winjob, "terminate_self_job") as terminate:
            controller._watch()
        callback.assert_called_once_with("parent-lost")
        terminate.assert_called_once_with(9)
        self.assertEqual(status.write.call_args.args[0]["state"], "forced")

    def test_stop_requires_final_record_and_actual_zero_exit(self):
        for state, actual, expected in (("stopped", 0, 0), ("stopped", 7, 9), ("forced", 0, 9), ("aborted", 0, 9)):
            with self.subTest(state=state, actual=actual):
                ready = {**BINDING, "kind": "strata-native-control", "state": "ready", "ready": True,
                         "exit_code": None}
                final = {**ready, "state": state, "exit_code": 0, "graceful": True}
                status = mock.Mock()
                status.read.side_effect = [ready, ready, final]
                requests = mock.Mock()
                requests.read.return_value = {**BINDING, "kind": "strata-native-stop", "action": "none"}
                process = mock.Mock(token=BINDING["token"])
                process.alive.side_effect = [True, True, False]
                process.exit_code.return_value = actual
                output = io.StringIO()
                with mock.patch.object(control, "_Area") as area, \
                     mock.patch.object(control, "_File", side_effect=[status, requests]), \
                     mock.patch.object(winjob, "ProcessIdentity", return_value=process), contextlib.redirect_stdout(output):
                    area.return_value.path = Path("D:/private/backend.json")
                    self.assertEqual(control.stop("D:/private/backend.json", 123, 1), expected)
                self.assertEqual(json.loads(output.getvalue())["graceful"], expected == 0)


@unittest.skipUnless(os.name == "nt", "Windows kernel handles")
class KernelJobs(unittest.TestCase):
    def test_direct_parent_handle_and_creation_identity(self):
        parent = winjob.direct_parent()
        current = winjob.ProcessIdentity(os.getpid())
        try:
            self.assertTrue(parent.alive())
            self.assertTrue(current.alive())
            self.assertLess(int(parent.token), int(current.token))
            self.assertEqual(parent.pid, os.getppid())
        finally:
            current.close()
            parent.close()

    def test_job_failure_and_missing_inherited_job_are_fatal(self):
        with mock.patch.object(winjob, "_self_job", None), mock.patch.object(winjob, "_make_job", return_value=None):
            with self.assertRaises(OSError):
                winjob.require_self_job()
        from types import SimpleNamespace
        with mock.patch.object(winjob, "_self_job", 1), mock.patch.object(winjob._k32, "IsProcessInJob", return_value=0):
            with self.assertRaisesRegex(OSError, "did not inherit"):
                winjob.contain(SimpleNamespace(_handle=2))

    def test_actual_self_job_kills_child_and_grandchild_on_crash(self):
        directory = Path(tempfile.mkdtemp(prefix=".control-kernel-", dir=ROOT))
        process = None
        identities = []
        try:
            with open(directory / "fixture.log", "wb") as log:
                process = subprocess.Popen([PYTHON, "-B", str(Path(__file__)), "--fixture", "jobtree", str(directory)],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if (directory / "grandchild.pid").exists() and (directory / "job.json").exists():
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
                self.assertTrue((directory / "job.json").exists(), (directory / "fixture.log").read_text())
                self.assertEqual(json.loads((directory / "job.json").read_text()),
                                 {"contained": True, "inheritable": False})
                identities = [winjob.ProcessIdentity(int((directory / name).read_text()))
                              for name in ("child.pid", "grandchild.pid")]
                self.assertTrue(all(identity.alive() for identity in identities))
                process.kill()
                process.wait(5)
                deadline = time.monotonic() + 5
                while any(identity.alive() for identity in identities) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(all(not identity.alive() for identity in identities))
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(5)
            for identity in identities:
                identity.close()
            target = directory.absolute()
            self.assertEqual(target.parent, ROOT.absolute())
            self.assertTrue(target.name.startswith(".control-kernel-"))
            self.assertEqual(target.resolve(), target)
            shutil.rmtree(target)


@unittest.skipUnless(os.name == "nt", "Windows security and kernel handles")
class WindowsControl(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix=".control-tests-", dir=ROOT))
        private_files.restrict_to_current_user(self.directory, directory=True)
        self.path = self.directory / "backend.json"
        self.processes = []
        self.files = []
        self.controllers = []
        self.identities = []

    def tearDown(self):
        for controller in self.controllers:
            controller.abort()
            controller.thread.join(2)
            controller.release()
        for process in self.processes:
            if process.poll() is None:
                process.kill()                    # exact Popen object created by this test, no PID/name search
            process.wait(10)
        for identity in self.identities:
            identity.close()
        for file in self.files:
            file.close()
        # Before this recursive delete, verify the absolute target is exactly our unique scratch directory.
        target = self.directory.absolute()
        self.assertEqual(target.parent, ROOT.absolute())
        self.assertTrue(target.name.startswith(".control-tests-"))
        self.assertEqual(target.resolve(), target)
        shutil.rmtree(target)

    def spawn(self, args):
        log = open(self.directory / f"process-{len(self.processes)}.log", "wb")
        self.files.append(log)
        process = subprocess.Popen([PYTHON, "-B", *args], stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=self.directory)
        self.processes.append(process)
        return process

    def wait_for(self, fn, seconds=12):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                value = fn()
                if value:
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        logs = "\n".join(p.read_text(errors="replace") for p in self.directory.glob("*.log"))
        self.fail("fixture wait timed out\n" + logs)

    def status(self):
        return json.loads(self.path.read_bytes())

    def make_controller(self):
        # Security/file locking/parent HANDLE are real; avoid assigning the unittest runner itself to a kill-on-exit job.
        callback = mock.Mock()
        with mock.patch.object(winjob, "require_self_job"):
            controller = control.Controller(str(self.path), callback, grace=10)
        self.controllers.append(controller)
        return controller, callback

    def run_stop(self, pid, timeout=8):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = control.stop(str(self.path), pid, timeout)
        return code, json.loads(output.getvalue())

    def test_absolute_normal_paths_only(self):
        for path in ("relative.json", r"\\server\share\control.json", r"\\?\D:\control.json",
                     r"D:\control.json:stream", r"D:\somewhere\..\control.json", "D:\\folder.\\control.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                control._path(path)

    def test_private_file_and_parent_and_acl_volume_required(self):
        area = control._Area(str(self.path))
        area.close()
        file = control._File(self.path, create=True)
        try:
            self.assertTrue(file.new)
            self.assertFalse(os.get_handle_inheritable(file.handle))
        finally:
            file.close()
        # No security repair of an unsafe existing file or parent is permitted.
        with mock.patch.object(control, "_private_sddl", return_value=False):
            with self.assertRaises(ValueError):
                control._File(self.path, create=False)
            with self.assertRaises(ValueError):
                control._Area(str(self.path))
        with mock.patch.object(control._k32, "GetVolumeInformationW", return_value=1):
            with self.assertRaisesRegex(ValueError, "persistent ACL"):
                control._Area(str(self.path))
        with mock.patch.object(control._k32, "GetVolumeInformationW", return_value=0):
            with self.assertRaises(OSError):
                control._Area(str(self.path))

    def test_hard_link_and_symlink_rejected(self):
        original = self.directory / "original.json"
        original.write_text("{}")
        private_files.restrict_to_current_user(original)
        os.link(original, self.path)
        with self.assertRaisesRegex(ValueError, "hard links"):
            control._File(self.path, create=False)
        symlink = self.directory / "symlink.json"
        try:
            os.symlink(original, symlink)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("symlink privilege unavailable; hardlink rejection did pass")
            raise
        with self.assertRaisesRegex(ValueError, "reparse"):
            control._File(symlink, create=False)

    def test_shared_path_mutex_and_stop_latch(self):
        controller, callback = self.make_controller()
        with mock.patch.object(winjob, "require_self_job"):
            with self.assertRaises(OSError):
                control.Controller(str(self.path), lambda reason: None, grace=10)
        request = control._File(Path(str(self.path) + ".stop"), create=False)
        try:
            request.lock()
            request.write({**controller.binding, "kind": "strata-native-stop", "action": "stop"})
            request.unlock()
            self.wait_for(lambda: callback.called)
            controller.stop_once("again")
            controller.stop_once("parent-lost")
            self.assertEqual(callback.call_count, 1)
            self.assertEqual(self.status()["state"], "stopping")
            controller.mark_ready()              # never reset a latched stop when loading finishes
            self.assertEqual(self.status()["state"], "stopping")
        finally:
            request.close()

    def test_fail_closed_before_self_job_and_strict_stale_validation(self):
        self.path.write_text("{}")
        private_files.restrict_to_current_user(self.path)
        with mock.patch.object(winjob, "require_self_job") as job:
            with self.assertRaises(ValueError):
                control.Controller(str(self.path), lambda reason: None, grace=10)
            job.assert_not_called()

    def test_self_job_failure_is_fatal_before_ready(self):
        with mock.patch.object(winjob, "require_self_job", side_effect=OSError("job refused")):
            with self.assertRaisesRegex(OSError, "job refused"):
                control.Controller(str(self.path), lambda reason: None, grace=10)
        self.assertEqual(self.path.read_bytes(), b"")

    def test_wrong_pid_and_token_never_write_stop(self):
        controller, _ = self.make_controller()
        self.assertEqual(self.run_stop(os.getpid() + 100000)[0], 2)
        held = control._File(self.path, create=False)
        try:
            held.write({**self.status(), "token": "1"})
            self.assertEqual(self.run_stop(os.getpid())[0], 2)
        finally:
            held.close()
        self.assertEqual(json.loads(Path(str(self.path) + ".stop").read_bytes())["action"], "none")

    def test_mock_server_direct_cli_graceful_and_read_only(self):
        snapshot = self.directory / "private-snapshot.json"
        snapshot.write_text('{}')
        sidecar = snapshot.with_suffix("").with_name(snapshot.stem + ".shared-settings.json")
        sidecar.write_text('{"temperature":0.123}')
        before = snapshot.read_bytes(), sidecar.read_bytes()
        process = self.spawn([str(ROOT / "serve/server.py"), "--engine", "mock", "--config", str(snapshot),
                              "--control-file", str(self.path), "--read-only-config", "--shutdown-timeout", "1",
                              "--port", "0"])
        self.wait_for(lambda: self.path.exists() and self.status().get("state") == "ready")
        # Discover only the mock server's actual bound URL from its own log.
        log_path = self.directory / "process-0.log"
        import re
        base = self.wait_for(lambda: (re.search(r"ready: (http://[^ /]+)", log_path.read_text()) or [None, None])[1])
        for path in ("/config", "/settings"):
            req = urllib.request.Request(base + path, data=b"{}", headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(error.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/config", timeout=5)
        self.assertNotIn(str(snapshot), error.exception.read().decode())
        with urllib.request.urlopen(base + "/settings", timeout=5) as response:
            self.assertEqual(json.load(response), {"shared": False, "defaults": {}})
        # Test the actual one-shot direct-script CLI, not only its imported function.
        client = self.spawn([str(ROOT / "serve/control.py"), "stop", "--file", str(self.path), "--pid",
                             str(process.pid), "--timeout", "8"])
        self.assertEqual(client.wait(12), 0)
        self.assertEqual(process.wait(5), 0)
        result = json.loads((self.directory / "process-1.log").read_text())
        self.assertTrue(result["graceful"])
        self.assertEqual(result["pid"], process.pid)
        self.assertTrue(result["ready"])
        self.assertEqual(before, (snapshot.read_bytes(), sidecar.read_bytes()))
        # A fully verified dead instance may be safely reused; no stale request stops the next model.
        second = self.spawn([str(ROOT / "serve/server.py"), "--engine", "mock", "--control-file", str(self.path),
                             "--read-only-config", "--shutdown-timeout", "1", "--port", "0"])
        self.wait_for(lambda: self.status().get("state") == "ready" and self.status()["pid"] == second.pid)
        self.assertNotEqual(result["instance"], control.hashlib.sha256(self.status()["nonce"].encode()).hexdigest()[:24])
        self.assertEqual(self.run_stop(second.pid)[0], 0)
        self.assertEqual(second.wait(5), 0)

    def test_startup_stop_is_explicit_nonready_abort(self):
        process = self.spawn([str(Path(__file__)), "--fixture", "loading", str(self.directory)])
        self.wait_for(lambda: (self.directory / "ready-wait.flag").exists())
        child = winjob.ProcessIdentity(int((self.directory / "child.pid").read_text()))
        self.identities.append(child)
        started = time.monotonic()
        code, result = self.run_stop(process.pid)
        self.assertEqual(code, 9)
        self.assertFalse(result["graceful"])
        self.assertFalse(result["ready"])
        self.assertEqual(self.status()["reason"], "nonready-abort")
        self.assertNotEqual(process.wait(5), 0)
        self.assertLess(time.monotonic() - started, 4)
        self.wait_for(lambda: not child.alive())

    def test_direct_parent_loss_enters_native_drain(self):
        launcher = self.spawn([str(Path(__file__)), "--fixture", "launcher", str(self.directory)])
        self.wait_for(lambda: (self.directory / "server.pid").exists())
        server_pid = int((self.directory / "server.pid").read_text())
        server = winjob.ProcessIdentity(server_pid)
        self.identities.append(server)
        self.wait_for(lambda: self.path.exists() and self.status().get("state") == "ready")
        launcher.kill()
        launcher.wait(5)
        self.wait_for(lambda: not server.alive())
        self.assertEqual(server.exit_code(), 0)
        self.assertEqual(self.status()["state"], "stopped")
        self.assertTrue(self.status()["graceful"])

    def tree_fixture(self, mode):
        process = self.spawn([str(Path(__file__)), "--fixture", mode, str(self.directory)])
        self.wait_for(lambda: (self.directory / "grandchild.pid").exists() and self.path.exists()
                      and self.status().get("state") == "ready")
        identities = [winjob.ProcessIdentity(int((self.directory / name).read_text()))
                      for name in ("child.pid", "grandchild.pid")]
        self.identities.extend(identities)
        return process, identities

    def test_self_job_contains_descendants_when_server_crashes(self):
        process, identities = self.tree_fixture("tree")
        process.kill()
        process.wait(5)
        self.wait_for(lambda: all(not identity.alive() for identity in identities))
        # Crash artifacts are reused only after checking original process identity is gone.
        controller, _ = self.make_controller()
        self.assertEqual(self.status()["state"], "loading")
        self.assertEqual(controller.binding["pid"], os.getpid())

    def test_deadline_terminates_only_self_job_and_reports_forced(self):
        process, identities = self.tree_fixture("deadline")
        code, result = self.run_stop(process.pid)
        self.assertEqual(code, 9)
        self.assertTrue(result["forced"])
        self.assertFalse(result["graceful"])
        self.assertEqual(process.wait(5), 9)
        self.wait_for(lambda: all(not identity.alive() for identity in identities))


class ReadOnlyHttp(unittest.TestCase):
    def test_read_only_refuses_writes_and_never_reads_private_config(self):
        from serve import server
        tok = server.ByteTokenizer()
        service = server.Service(server.MockEngine(tok, ["hello"]), tok,
                                 server.ChatTemplate(ROOT / "serve/chat_template.jinja"))
        service.read_only_config = True
        service.config_path = "D:/private/do-not-disclose-snapshot.json"
        httpd = server.serve(service, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with mock.patch.object(server.runconfig, "load", side_effect=AssertionError("must not read snapshot")):
                for path, data in (("/config", None), ("/config", b"{}"), ("/settings", b"{}")):
                    request = urllib.request.Request(base + path, data=data,
                                                     headers={"Content-Type": "application/json"})
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(request, timeout=5)
                    self.assertEqual(error.exception.code, 403)
                    self.assertNotIn("snapshot", error.exception.read().decode())
            self.assertEqual(service.shared, {})
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_default_settings_post_still_works(self):
        from serve import server
        tok = server.ByteTokenizer()
        service = server.Service(server.MockEngine(tok, ["hello"]), tok,
                                 server.ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = server.serve(service, port=0)
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/settings",
                                             data=b'{"defaults":{"temperature":0.25}}',
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.load(response)["defaults"]["temperature"], 0.25)
        finally:
            httpd.shutdown()
            httpd.server_close()


class EngineCloseEvidence(unittest.TestCase):
    def test_quit_terminate_kill_and_nonzero_are_distinguished(self):
        from serve.server import StrataEngine
        for outcome in ("quit", "terminate", "kill", "nonzero"):
            with self.subTest(outcome=outcome):
                engine = StrataEngine("no-engine-is-spawned", [], lazy=True)
                process = mock.Mock(stdin=io.StringIO(), stdout=io.StringIO())
                process.poll.side_effect = [None, 7 if outcome == "nonzero" else 0]
                if outcome == "terminate":
                    process.wait.side_effect = [subprocess.TimeoutExpired("fake", 20), 0]
                elif outcome == "kill":
                    process.wait.side_effect = [subprocess.TimeoutExpired("fake", 20),
                                                subprocess.TimeoutExpired("fake", 5), 0]
                engine.proc = process
                engine.close()
                self.assertEqual(control.close_failed(engine), outcome != "quit")
                self.assertIsNone(engine.proc)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fixture":
        fixture(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
