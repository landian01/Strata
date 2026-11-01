"""Exercise a staged direct gateway, with CPU mock servers or the two pinned real models.

This is a test runner, never a runtime owner/backend. Every run keeps its own logs
and result JSON. Existing launchers/configs and the installed Terminal are not edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def free_port():
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in (8081, 18181, 18182):
            return port


def wait_for(fn, timeout=30):
    end, last = time.monotonic() + timeout, None
    while time.monotonic() < end:
        try:
            value = fn()
            if value:
                return value
        except (OSError, ValueError, KeyError) as exc:
            last = repr(exc)
        time.sleep(0.1)
    raise TimeoutError(f"condition did not become true in {timeout}s: {last}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--mode", choices=("cpu", "live"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = load(args.deployment / "manifest.json")
    root = Path(manifest["root"])
    output = args.output.resolve()
    if not output.is_relative_to(root / "production/validation") or output.exists():
        raise ValueError("output must be a NEW directory under ROOT/production/validation")
    output.mkdir(parents=True)
    sys.path.insert(0, str(root / "terminal/0.1.51"))
    import psutil
    from strata_control import private_files
    control_dir = output / "control"
    private_files.private_mkdir(control_dir, exist_ok=False)
    private_files.restrict_to_current_user(control_dir, directory=True)
    control_file = control_dir / "backend.json"
    report = {"ok": False, "mode": args.mode, "scope": "Routing and lifecycle smoke tests, not model quality/performance benchmarking",
              "steps": [], "initial_protected": [{**item, "observed": sha(item["path"])} for item in manifest["protected"]]}
    all_expected = manifest["protected"] + manifest["assets"]
    if any(sha(item["path"]) != item["sha256"] for item in all_expected):
        raise ValueError("a pinned deployment input changed")
    config = load(manifest["gateway_config"])
    gateway_port = free_port() if args.mode == "cpu" else manifest["port"]
    backend_ports = [free_port(), free_port()] if args.mode == "cpu" else [b["port"] for b in manifest["bindings"]]
    ids = [b["model"] for b in manifest["bindings"]]
    for index, model in enumerate(ids):
        spec = config["models"][model]
        command = list(manifest["bindings"][index]["command"])
        command[command.index("--port") + 1] = str(backend_ports[index])
        command[command.index("--control-file") + 1] = str(control_file)
        if args.mode == "cpu":
            mock_config = output / f"mock-{index}.json"
            save(mock_config, {"model_name": model, "open_browser": False})
            command[command.index("--engine") + 1] = "mock"
            command[command.index("--config") + 1] = str(mock_config)
            command[command.index("--shutdown-timeout") + 1] = "20"
            command += ["--script", "</think>\n\n" + ("A bounded native streaming answer. " * 16), "--script-delay", "0.025"]
        spec["cmd"] = subprocess.list2cmdline(command)
        stop = [command[0], "-B", str(Path(manifest["runtime"]) / "serve/control.py"), "stop", "--file", str(control_file),
                "--pid", "${PID}", "--timeout", "360" if args.mode == "live" else "40"]
        spec["cmdStop"] = subprocess.list2cmdline(stop)
        spec["proxy"] = f"http://127.0.0.1:{backend_ports[index]}"
    if args.mode == "cpu":
        config["healthCheckTimeout"], config["unloadTimeout"] = 40, 40
    config_path = output / "gateway.json"
    save(config_path, config)
    base = f"http://127.0.0.1:{gateway_port}"
    owned_backends = []
    gateway = None
    stdout = (output / "gateway.stdout.log").open("wb")
    stderr = (output / "gateway.stderr.log").open("wb")

    def note(name, **facts):
        report["steps"].append({"name": name, "at": time.time(), **facts})
        save(output / "result.json", report)
        print(name + ": " + json.dumps(facts, ensure_ascii=False), flush=True)

    def request(path, payload=None, timeout=1200):
        data = None if payload is None else json.dumps(payload).encode()
        if payload == {} and path.startswith("/api/models/"):
            data = b""  # gateway control handlers take no body; avoid unread-body TCP resets on Windows
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with HTTP.open(req, timeout=timeout) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                body = json.loads(raw)
            except ValueError:
                body = {"raw": raw.decode("utf-8", "replace")}
            return exc.code, body

    def body(model, stream=False, long=False):
        return {"model": model, "messages": [{"role": "user", "content":
            "Write a detailed numbered list of 100 different practical uses of a computer. Do not stop before item 100."
            if long else "Answer only with the number: 17 + 25 ="}],
            "temperature": 0, "reasoning_effort": "none", "max_tokens": 384 if long else 64, "stream": stream}

    def identity(canonical):
        record = wait_for(lambda: load(control_file) if load(control_file).get("ready") else None, 15)
        process = psutil.Process(record["pid"])
        item = {"pid": process.pid, "created": process.create_time(), "cmdline": process.cmdline()}
        owned_backends.append(item)
        assert process.ppid() == gateway.pid, item
        assert str(Path(manifest["runtime"]) / "serve/server.py") in item["cmdline"], item
        assert not any("swap_backend.py" in part or "auto_launcher.py" in part for part in item["cmdline"]), item
        if args.mode == "live":
            binding = next(row for row in manifest["bindings"] if row["model"] == canonical)
            status, metrics = request("/upstream/" + urllib.parse.quote(canonical, safe="") + "/metrics")
            assert status == 200, metrics
            engine = metrics["engine"]
            cfg = load(binding["config"])
            flags = dict(zip(cfg["args"][::2], cfg["args"][1::2]))
            main_slots, helper_slots = binding["expected_slots"]
            expected = {"model": canonical, "expert_slots_primary": main_slots,
                        "expert_slots": main_slots + helper_slots, "pool_workers": 5, "mtp_max": 6,
                        "max_context": int(flags["--max-context"]), "kv": flags["--kv"],
                        "kv_resident": int(flags["--kv-resident"])}
            for key, value in expected.items():
                assert engine.get(key) == value, (key, engine.get(key), value)
            item["engine"] = expected
        return item

    def gone(item):
        try:
            return psutil.Process(item["pid"]).create_time() != item["created"]
        except psutil.NoSuchProcess:
            return True

    def smoke(alias, canonical):
        status, result = request("/v1/chat/completions", body(alias))
        if status != 200 or result.get("model") != canonical:
            raise AssertionError((status, result))
        choice = result["choices"][0]
        text = choice["message"].get("content", "").strip()
        if not text or (args.mode == "live" and (text != "42" or choice["finish_reason"] != "stop")):
            raise AssertionError(result)
        note("answer", alias=alias, model=canonical, answer=text, finish_reason=choice["finish_reason"])
        return identity(canonical)

    def streaming(alias, box, started, *, disconnect=False):
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body(alias, True, True)).encode(),
                                         headers={"Content-Type": "application/json"})
            chunks, done = 0, False
            with HTTP.open(req, timeout=1200) as response:
                for line in response:
                    if line.startswith(b"data: "):
                        if line[6:].strip() == b"[DONE]":
                            done = True
                            break
                        event = json.loads(line[6:])
                        if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                            chunks += 1
                            started.set()
                            if disconnect:
                                break
            box.update(chunks=chunks, done=done, ended=time.monotonic())
        except BaseException as exc:
            box["error"] = repr(exc)
            started.set()

    try:
        for port in [gateway_port, *backend_ports]:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", port))
        env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_") and k not in
               ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTHONPATH", "PYTHONHOME")}
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
        gateway = subprocess.Popen([manifest["gateway"], "-config", str(config_path), "-listen",
                                    f"127.0.0.1:{gateway_port}", "-watch-config=false"],
                                   cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        wait_for(lambda: request("/api/version", timeout=2)[0] == 200, 30)
        note("gateway-ready", pid=gateway.pid, ports=[gateway_port, *backend_ports])
        status, catalog = request("/v1/models")
        assert status == 200 and {row["id"] for row in catalog["data"]} == set(ids), catalog
        assert request("/v1/chat/completions", body("not-a-configured-model"))[0] == 404
        note("catalog-and-unknown-model", unknown_status=404)
        first = smoke("iq3", ids[0])
        again = smoke(ids[0], ids[0])
        assert first["pid"] == again["pid"] and first["created"] == again["created"]
        note("same-model-reuses-native-process", process=first)
        for endpoint in ("config", "settings"):
            assert request("/upstream/" + urllib.parse.quote(ids[0], safe="") + "/" + endpoint, {})[0] == 403
        note("native-readonly-config-and-settings", statuses=[403, 403])
        stream_box, signal = {}, threading.Event()
        stream_thread = threading.Thread(target=streaming, args=("iq3", stream_box, signal), daemon=True)
        stream_thread.start()
        if not signal.wait(90) or "error" in stream_box:
            raise AssertionError(stream_box)
        second_box = {}
        def second_request():
            try:
                second_box["identity"] = smoke("iq4", ids[1])
                second_box["ended"] = time.monotonic()
            except BaseException as exc:
                second_box["error"] = repr(exc)
        second = threading.Thread(target=second_request, daemon=True)
        second.start()
        time.sleep(0.4)
        assert stream_thread.is_alive() and second.is_alive() and not gone(first), (stream_box, second_box)
        note("cross-model-request-waits-for-stream", old_native_alive=True)
        stream_thread.join(120)
        second.join(1200)
        assert not stream_thread.is_alive() and not second.is_alive() and stream_box.get("done") and "error" not in second_box, (stream_box, second_box)
        assert gone(first), first
        native_iq4 = second_box["identity"]
        note("cross-model-stream-complete", stream=stream_box, new_native=native_iq4, old_native_exited=True)
        cancel_box, signal = {}, threading.Event()
        client = threading.Thread(target=streaming, args=("iq4", cancel_box, signal), kwargs={"disconnect": True}, daemon=True)
        client.start()
        assert signal.wait(90)
        client.join(30)
        assert "error" not in cancel_box and cancel_box.get("chunks"), cancel_box
        recovered = smoke("iq4", ids[1])
        assert recovered["pid"] == native_iq4["pid"]
        note("client-disconnect-recovered", stream=cancel_box, same_native=True)
        back = smoke("iq3", ids[0])
        assert gone(native_iq4)
        note("switch-back", process=back, old_native_exited=True)
        # Explicit Pause is different from normal swapping: test native drain while HTTP is still in flight.
        drain_box, signal = {}, threading.Event()
        drain_stream = threading.Thread(target=streaming, args=("iq3", drain_box, signal), daemon=True)
        drain_stream.start()
        assert signal.wait(90)
        status, paused = request("/api/models/pause", {}, timeout=450)
        drain_stream.join(120)
        assert status == 200 and drain_box.get("done") and "error" not in drain_box, (status, paused, drain_box)
        assert gone(back)
        note("explicit-pause-drains-stream", response=paused, stream=drain_box, native_exited=True)
        assert request("/api/models/resume", {})[0] == 200
        final = smoke("iq4", ids[1])
        # End only the gateway through its retained creation handle; native parent guard / Job must reap the model.
        gateway.kill()
        gateway.wait(20)
        wait_for(lambda: gone(final), 60)
        note("gateway-abrupt-exit-reaps-native", native_exited=True, gateway_exit=gateway.returncode)
        for port in [gateway_port, *backend_ports]:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", port))
        assert all(sha(item["path"]) == item["sha256"] for item in all_expected)
        report.update(ok=True, all_protected_unchanged=True, all_test_ports_released=True)
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if gateway is not None and gateway.poll() is None:
            try:
                request("/api/models/pause", {}, timeout=450)
            except BaseException as exc:
                report["cleanup_pause_error"] = repr(exc)
            gateway.kill()  # the retained Popen handle, never a PID-only lookup
            gateway.wait(20)
        stdout.close()
        stderr.close()
        report["backend_identities"] = owned_backends
        report["all_observed_backends_gone"] = all(gone(item) for item in owned_backends)
        report["gateway_log"] = str(output / "gateway.stdout.log")
        logs = (output / "gateway.stdout.log").read_text(encoding="utf-8", errors="replace")
        logs += (output / "gateway.stderr.log").read_text(encoding="utf-8", errors="replace")
        report["graceful_stop_reports"] = len(re.findall(r'"outcome":\s*"graceful"', logs))
        report["stop_failures_in_log"] = "stop signal failed" in logs
        report["forced_stop_reports"] = len(re.findall(r'"forced":\s*true', logs))
        if report["ok"] and (report["graceful_stop_reports"] < 3 or report["stop_failures_in_log"] or report["forced_stop_reports"]):
            report.update(ok=False, error="native graceful stop evidence missing or forced/failed stop was logged")
        save(output / "result.json", report)
    print(json.dumps({"ok": report["ok"], "result": str(output / "result.json")}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
