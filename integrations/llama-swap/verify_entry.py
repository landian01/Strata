"""Real-model acceptance of the one-shot Windows entry. Not a runtime component.

Checks Status during cold warm-up, explicit loading cancellation, restart after
that cancellation, a real answer, and the deployment's own Stop command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from contracts import assert_files, assert_metrics, expected_metrics, observe_engine, validation_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--expected-engine-version", help="assert live INFO engine=, not package metadata")
    args = parser.parse_args()
    manifest = json.loads((args.deployment / "manifest.json").read_text(encoding="utf-8-sig"))
    assert_files(manifest)
    expected_version = args.expected_engine_version or manifest.get("expected_engine_version")
    if args.expected_engine_version and manifest.get("expected_engine_version") not in (None, args.expected_engine_version):
        raise ValueError("expected engine version disagrees with the pinned manifest")
    sys.path.insert(0, manifest["runtime"])
    import psutil
    from serve import winjob
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(validation_root(manifest, args.validation_root)):
        raise ValueError("a NEW directory under the declared validation root is required")
    output.mkdir(parents=True)
    report = {"ok": False, "scope": "real IQ3 loading cancellation and one-shot entry smoke tests", "steps": []}
    start = args.deployment / "Start-Strata.cmd"
    stop = args.deployment / "Stop-Strata.cmd"
    control = Path(manifest["control_file"])
    opened, warmups, identities = [], [], []

    def hold(record):
        identity = winjob.ProcessIdentity(record["pid"])
        identities.append(identity)
        assert identity.token == record["token"] and identity.alive(), "native loading identity mismatch"
        return identity

    def final_status(initial, identity, *, ready):
        record = json.loads(control.read_bytes())
        assert all(record[key] == initial[key] for key in ("pid", "token", "nonce")), "native final identity mismatch"
        assert not identity.alive() and record["ready"] is ready, "native final liveness or readiness mismatch"
        if ready:
            assert record["state"] == "stopped" and record.get("graceful") is True and record["exit_code"] == identity.exit_code() == 0, "native ready shutdown was not verified graceful"
        else:
            assert record["state"] == "aborted" and record["reason"] == "nonready-abort" and record.get("graceful") is False, "native loading shutdown was not an explicit nonready abort"
            assert record["exit_code"] != 0 and identity.exit_code() != 0, "native loading abort had an unexpected zero exit"
        # The nonce is a private control credential, not acceptance-log content.
        return {**{key: value for key, value in record.items() if key != "nonce"},
                "instance": hashlib.sha256(record["nonce"].encode("ascii")).hexdigest()[:24]}

    def save():
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def note(name, **facts):
        report["steps"].append({"name": name, **facts})
        save()
        print(name + ": " + json.dumps(facts), flush=True)

    def command(name, arguments, *, background=False, stopping=False):
        log_path = output / (name + ".log")
        stream = log_path.open("wb")
        opened.append(stream)
        proc = subprocess.Popen([os.environ["ComSpec"], "/d", "/c", str(stop if stopping else start), *arguments],
                                cwd=manifest["root"], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        if background:
            warmups.append(proc)
            return proc
        code = proc.wait(1200)
        stream.flush()
        text = log_path.read_text(encoding="utf-8-sig", errors="replace")
        if code:
            raise AssertionError((name, code, text))
        return json.loads(text)

    def loading(previous=None):
        end = time.monotonic() + 45
        while time.monotonic() < end:
            try:
                record = json.loads(control.read_bytes())
                if record.get("pid") != previous and record.get("state") == "loading":
                    return record
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                pass  # creation / in-place JSON write only; permission and identity failures stay fatal
            time.sleep(0.1)
        raise TimeoutError("native service did not publish a new loading identity")

    try:
        canceled = command("warm-cancel", ["launch", "--preset", "IQ3"], background=True)
        before = loading()
        canceled_identity = hold(before)
        status = command("status-while-loading", ["launch", "--status"])
        assert status["ok"] and status["running"], status
        note("status-during-loading", running=status["running"])
        stopped = command("cancel-loading-via-own-stop", [], stopping=True)
        canceled.wait(60)
        assert stopped["ok"] and stopped["model_stop"] == "loading-aborted" and not stopped["graceful_model_stop"], stopped
        assert canceled.returncode != 0  # the canceled Warm must not pretend it completed
        aborted = final_status(before, canceled_identity, ready=False)
        note("loading-canceled-without-false-graceful", stop=stopped, warm_exit=canceled.returncode, final_native=aborted)
        assert command("status-after-cancel", ["launch", "--status"])["state"] == "stopped"
        warm = command("warm-restart", ["launch", "--preset", "IQ3"], background=True)
        current = loading(before["pid"])
        assert current["pid"] != before["pid"] and current["nonce"] != before["nonce"]
        current_identity = hold(current)
        status = command("status-during-second-warm", ["launch", "--status"])
        assert status["ok"] and status["running"], status
        assert warm.wait(1200) == 0
        note("warm-after-cancel-completes", native_pid=current["pid"])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        payload = {"model": "iq3", "messages": [{"role": "user", "content": "Answer only with the number: 17 + 25 ="}],
                   "temperature": 0, "reasoning_effort": "none", "max_tokens": 64}
        request = urllib.request.Request(f"http://127.0.0.1:{manifest['port']}/v1/chat/completions",
                                         data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=1200) as response:
            answer = json.load(response)
        assert answer["choices"][0]["message"]["content"].strip() == "42", answer
        note("native-entry-real-answer", answer="42", model=answer["model"])
        binding = manifest["bindings"][0]
        cfg = json.loads(Path(binding["config"]).read_text(encoding="utf-8-sig"))
        metrics_url = f"http://127.0.0.1:{manifest['port']}/upstream/{urllib.parse.quote(binding['model'], safe='')}/metrics"
        with opener.open(metrics_url, timeout=20) as response:
            metrics = json.load(response)
        expected = expected_metrics(binding, cfg, expected_version)
        assert_metrics(metrics["engine"], expected)
        inspection = observe_engine(psutil, psutil.Process(current["pid"]), cfg, binding)
        note("real-engine-contract", expected=expected, live_process_fields=inspection)
        if inspection["status"] == "NOT_VERIFIED":
            report["verification_limits"] = ["Live child process fields NOT_VERIFIED: " + inspection["reason"]]
        stopped = command("ready-model-own-stop", [], stopping=True)
        assert stopped["ok"] and stopped["model_stop"] == "graceful" and stopped["graceful_model_stop"], stopped
        assert command("final-status", ["launch", "--status"])["state"] == "stopped"
        final = final_status(current, current_identity, ready=True)
        assert_files(manifest)
        note("own-stop-confirms-real-graceful-exit", result=stopped, final_native=final, protected_unchanged=True)
        report["ok"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        try:
            report["cleanup"] = command("failure-cleanup", [], stopping=True)
        except BaseException as cleanup:
            report["cleanup_error"] = repr(cleanup)
        raise
    finally:
        for process in warmups:
            if process.poll() is None:
                process.wait(60)
        for stream in opened:
            stream.close()
        for identity in identities:
            identity.close()
        save()
    print(json.dumps({"ok": True, "result": str(output / "result.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
