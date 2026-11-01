"""Real-model acceptance of the one-shot Windows entry. Not a runtime component.

Checks Status during cold warm-up, explicit loading cancellation, restart after
that cancellation, a real answer, and the deployment's own Stop command.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.deployment / "manifest.json").read_text(encoding="utf-8-sig"))
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(manifest["root"]) / "production/validation"):
        raise ValueError("a NEW validation directory is required")
    output.mkdir()
    report = {"ok": False, "scope": "real IQ3 loading cancellation and one-shot entry smoke tests", "steps": []}
    start = args.deployment / "Start-Strata.cmd"
    stop = args.deployment / "Stop-Strata.cmd"
    control = Path(manifest["control_file"])
    opened, warmups = [], []

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
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise TimeoutError("native service did not publish a new loading identity")

    try:
        canceled = command("warm-cancel", ["launch", "--preset", "IQ3"], background=True)
        before = loading()
        status = command("status-while-loading", ["launch", "--status"])
        assert status["ok"] and status["running"], status
        note("status-during-loading", running=status["running"])
        stopped = command("cancel-loading-via-own-stop", [], stopping=True)
        canceled.wait(60)
        assert stopped["ok"] and stopped["model_stop"] == "loading-aborted" and not stopped["graceful_model_stop"], stopped
        assert canceled.returncode != 0  # the canceled Warm must not pretend it completed
        note("loading-canceled-without-false-graceful", stop=stopped, warm_exit=canceled.returncode)
        assert command("status-after-cancel", ["launch", "--status"])["state"] == "stopped"
        warm = command("warm-restart", ["launch", "--preset", "IQ3"], background=True)
        current = loading(before["pid"])
        assert current["pid"] != before["pid"]
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
        stopped = command("ready-model-own-stop", [], stopping=True)
        assert stopped["ok"] and stopped["model_stop"] == "graceful" and stopped["graceful_model_stop"], stopped
        assert command("final-status", ["launch", "--status"])["state"] == "stopped"
        note("own-stop-confirms-real-graceful-exit", result=stopped)
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
        save()
    print(json.dumps({"ok": True, "result": str(output / "result.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
