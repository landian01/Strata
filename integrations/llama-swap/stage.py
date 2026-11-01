"""Stage a pinned, direct llama-swap deployment. One-shot; never starts a model.

The engine, installed Control/Terminal and model files are read-only inputs.
Legacy calls use the installed native service plus the reviewed control patch.
Explicit upgrade calls snapshot complete reviewed source without that old patch.
Neither route adds a long-lived backend or owner process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checked(path: Path, expected: str) -> dict:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"pinned input changed: {path}: {actual}, expected {expected}")
    return {"path": str(path), "sha256": actual}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--patch", type=Path, default=HERE / "native-control.patch",
                        help="reviewed native-only overlay patch (defaults to the versioned patch beside this tool)")
    parser.add_argument("--terminal-root", type=Path, help="unpacked Windows Control bundle (Python and dependencies)")
    parser.add_argument("--engine-dir", type=Path, help="immutable custom engine folder containing strata.exe and DLLs")
    parser.add_argument("--native-source", type=Path, help="complete reviewed candidate source; NEVER apply the legacy patch")
    parser.add_argument("--baseline-manifest", type=Path, help="accepted direct-native manifest; all baseline inputs remain pinned")
    parser.add_argument("--expected-engine-version", help="required live INFO engine= version, e.g. 0.1.55")
    parser.add_argument("--engine-commit", help="engine source provenance (does not substitute for binary hashes)")
    parser.add_argument("--state-directory", type=Path, help="NEW private candidate state directory")
    parser.add_argument("--declare-mtp", action="store_true", help="explicitly freeze an existing MTP choice; never edits args")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--backend-ports", type=int, nargs=2, default=(18181, 18182))
    args = parser.parse_args(argv)
    explicit = (args.terminal_root, args.engine_dir, args.native_source, args.baseline_manifest)
    if any(explicit):
        if not all(explicit):
            parser.error("upgrade staging requires --terminal-root, --engine-dir, --native-source and --baseline-manifest together")
        if len({args.port, *args.backend_ports}) != 3 or not all(1024 <= p <= 65535 for p in (args.port, *args.backend_ports)):
            parser.error("three distinct unprivileged ports are required")
        from candidate_stage import stage_candidate
        return stage_candidate(args)
    if args.state_directory or args.declare_mtp or args.expected_engine_version or args.engine_commit:
        parser.error("upgrade-only options require the four explicit upgrade paths")
    root, output = args.root.resolve(), args.output.resolve()
    if output.parent != root / "production" or output.exists():
        raise ValueError("output must be a NEW directory directly inside ROOT/production")
    if len({args.port, *args.backend_ports}) != 3 or not all(1024 <= p <= 65535 for p in (args.port, *args.backend_ports)):
        raise ValueError("three distinct unprivileged ports are required")
    terminal = root / "terminal/0.1.51"
    python = terminal / "python/python.exe"
    release = read_json(root / "production/release.json")
    old = read_json(root / "production/auto-route-manifest.json")
    registry = read_json(root / "production/launch-registry.json")
    protected = [checked(Path(item["path"]), item["sha256"]) for item in release["engine_files"]]
    protected.append(checked(Path(old["exe"]), old["exe_sha256"]))
    protected.append({"path": str(python), "sha256": digest(python)})
    for entry in registry["presets"]:
        config = Path(entry["config"])
        protected.append(checked(config, entry["sha256"]))
        cfg = read_json(config)
        flags = dict(zip(cfg["args"][::2], cfg["args"][1::2]))
        protected.append(checked(Path(flags["--expert-profile"]), entry["expert_profile_sha256"]))
        if cfg["gpu"] != [0, 1] or cfg["gpu_roles"] != {"0": "main", "1": "helper"} or flags["--layer-split"] != "":
            raise ValueError("the accepted physical GPU order / helper configuration changed")
    if not args.patch.is_file():
        raise ValueError("reviewed native-only patch is required")

    output.mkdir()
    runtime = output / "native"
    runtime.mkdir()
    # No Python installation, engine executable, weights, packs or DLLs are copied.
    for name in ("serve", "strata_control", "strata_cli", "tools", "ref", "data"):
        source = terminal / name
        if source.is_dir():
            shutil.copytree(source, runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"))
    for name in ("setup.py", "strata.py", "VERSION", "COMMIT", "BUNDLE.json"):
        if (terminal / name).is_file():
            shutil.copy2(terminal / name, runtime / name)
    # Apply only the reviewed changes; preserve fixes already present in the installed package.
    subprocess.run(["git", "apply", "--check", str(args.patch.resolve())], cwd=runtime, check=True)
    subprocess.run(["git", "apply", str(args.patch.resolve())], cwd=runtime, check=True)
    shutil.copy2(SOURCE / "serve/control.py", runtime / "serve/control.py")
    shutil.copy2(args.patch, output / "native-control.patch")
    shutil.copy2(HERE / "Invoke-Strata.ps1", output / "Invoke-Strata.ps1")
    for name in ("Start-Strata.cmd", "Stop-Strata.cmd"):
        template = (HERE / (name + ".in")).read_text(encoding="utf-8")
        rendered = template.replace("@DEPLOYMENT@", str(output)).replace("@ROOT@", str(root))
        (output / name).write_text(rendered, encoding="utf-8", newline="\r\n")
    configs = output / "configs"
    configs.mkdir()
    state = root / "runtime" / output.name
    state.mkdir()
    # Control credentials live here, separate from deployable code and logs.
    sys.path.insert(0, str(terminal))
    from strata_control import private_files
    private_files.restrict_to_current_user(state, directory=True)
    control_dir = state / "control"
    private_files.private_mkdir(control_dir, parents=True, exist_ok=False)
    private_files.restrict_to_current_user(control_dir, directory=True)
    logs = output / "logs"
    logs.mkdir()
    control_file = control_dir / "backend.json"
    bindings = []
    models = {}
    for index, old_binding in enumerate(old["bindings"]):
        name = "iq3" if index == 0 else "iq4"
        config_path = configs / f"{name}.json"
        shutil.copy2(old_binding["config"], config_path)
        original = read_json(config_path)
        port = args.backend_ports[index]
        command = [str(python), "-B", str(runtime / "serve/server.py"), "--engine", "strata", "--config", str(config_path),
                   "--host", "127.0.0.1", "--port", str(port), "--control-file", str(control_file),
                   "--read-only-config", "--shutdown-timeout", "300"]
        stop = [str(python), "-B", str(runtime / "serve/control.py"), "stop", "--file", str(control_file),
                "--pid", "${PID}", "--timeout", "360"]
        binding = {**old_binding, "config": str(config_path), "port": port,
                   "original_config": old_binding["config"], "command": command,
                   "config_sha256": digest(config_path)}
        bindings.append(binding)
        models[old_binding["model"]] = {
            "name": old_binding["name"] + " (direct native service)",
            "cmd": subprocess.list2cmdline(command), "cmdStop": subprocess.list2cmdline(stop),
            "proxy": f"http://127.0.0.1:{port}", "checkEndpoint": "/health", "useModelName": old_binding["model"],
            "aliases": old_binding["aliases"], "ttl": 0, "concurrencyLimit": 8,
            "env": ["STRATA_NO_PRESET=1", "STRATA_NO_ACQUIRE=1", "PYTHONUTF8=1", "PYTHONIOENCODING=utf-8",
                    "PYTHONNOUSERSITE=1", f"STRATA_DATA_DIR={root}", f"STRATA_INSTALL_DIR={root / 'install'}",
                    f"STRATA_COMPONENTS_DIR={root / 'components'}", f"STRATA_PROFILE_DIR={root / 'profiles'}",
                    f"STRATA_RUNTIME_DIR={state}"],
            "timeouts": {"connect": 10, "keepAlive": 30, "responseHeader": 600,
                         "tlsHandshake": 10, "expectContinue": 1, "idleConn": 90},
        }
    gateway_config = {
        "healthCheckTimeout": 600, "unloadTimeout": 360, "globalTTL": 0,
        "logLevel": "info", "logTimeFormat": "rfc3339", "logToStdout": "both",
        "captureBuffer": 0, "sendLoadingState": False, "includeAliasesInList": False,
        "performance": {"disabled": True}, "helpagent": {"disable": True},
        "security": {"cors": {"allowedOrigins": [f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}",
                                                     "http://127.0.0.1:8570"]}},
        "models": models,
        "routing": {"router": {"use": "group", "settings": {"groups": {"strata-dual-gpu": {
            "swap": True, "exclusive": True, "members": list(models)}}}}},
    }
    gateway_path = output / "llama-swap.yaml"
    write_json(gateway_path, gateway_config)  # JSON is valid YAML; no optional YAML parser is needed.
    assets = [gateway_path, output / "Invoke-Strata.ps1", output / "native-control.patch",
              output / "Start-Strata.cmd", output / "Stop-Strata.cmd"]
    assets += list(configs.glob("*.json"))
    assets += [runtime / "serve" / name for name in ("server.py", "control.py", "winjob.py")]
    protected = list({item["path"]: item for item in protected}.values())
    manifest = {
        "schema_version": 1, "mode": "direct-native", "root": str(root), "directory": str(output),
        "runtime": str(runtime), "state_directory": str(state), "control_file": str(control_file),
        "gateway": old["exe"], "gateway_version": old["version"], "gateway_config": str(gateway_path),
        "port": args.port, "host": "127.0.0.1", "bindings": bindings,
        "protected": protected, "assets": [{"path": str(path), "sha256": digest(path)} for path in assets],
        "native_patch_sha256": digest(output / "native-control.patch"),
        "source_engine_commit": release["source_commit"], "source_native_revision": "uncommitted; see acceptance record",
        "models_and_engine_unchanged": True,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"ok": True, "staged_only": True, "directory": str(output),
                      "engine_and_weights_copied": False, "bindings": [b["model"] for b in bindings]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
