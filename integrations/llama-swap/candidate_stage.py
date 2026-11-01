"""Explicit upgrade staging. The legacy stage path deliberately remains separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from contracts import assert_config, assert_effective, assert_files, file_record, load, probe, sha, tree_files

HERE = Path(__file__).resolve().parent
PAYLOAD_DIRS = ("serve", "strata_control", "strata_cli", "tools", "ref", "data")
PAYLOAD_FILES = ("setup.py", "strata.py", "VERSION", "COMMIT", "BUNDLE.json", "requirements.txt", "requirements-monitor.txt")


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_candidate(args):
    root, output = args.root.resolve(), args.output.resolve()
    terminal, engine, source, baseline_path = (p.resolve() for p in
        (args.terminal_root, args.engine_dir, args.native_source, args.baseline_manifest))
    # A candidate can live beside the worktree, without editing production. Never stage into source itself.
    allowed = output.parent == root / "production" or (output.is_relative_to(source.parent) and not output.is_relative_to(source))
    input_dirs = (source, terminal, engine, baseline_path.parent)
    if output.exists() or not allowed or any(output.is_relative_to(path) for path in input_dirs):
        raise ValueError("output must be NEW, directly in ROOT/production or beside the candidate source, never inside an input")
    baseline = load(baseline_path)
    if baseline.get("mode") != "direct-native" or Path(baseline["root"]).resolve() != root or len(baseline["bindings"]) != 2:
        raise ValueError("baseline must be this root's accepted two-model direct-native manifest")
    assert_files(baseline)
    python = terminal / "python/python.exe"
    bundle_path = terminal / "BUNDLE.json"
    bundle = load(bundle_path)
    control_version = (terminal / "VERSION").read_text(encoding="utf-8").strip()
    if bundle.get("component") != "strata-control" or bundle.get("version") != control_version or not python.is_file():
        raise ValueError("terminal-root must hold a consistent unpacked Windows Control bundle and its Python")
    exe = engine / "strata.exe"
    if not exe.is_file():
        raise ValueError("engine-dir must contain strata.exe and its runtime libraries")
    build = load(engine / "BUILD.json") if (engine / "BUILD.json").is_file() else {}
    version = args.expected_engine_version or build.get("version")
    if not version or (build.get("version") and build["version"] != version):
        raise ValueError("declare --expected-engine-version (and agree with BUILD.json when present)")
    lib_dirs = [str(engine)]
    for name in build.get("lib_dirs", []):
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("BUILD.json lib_dirs must stay inside the engine package")
        directory = (engine / name).resolve()
        if not directory.is_relative_to(engine) or not directory.is_dir():
            raise ValueError(f"missing or external engine library directory: {name}")
        if str(directory) not in lib_dirs:
            lib_dirs.append(str(directory))
    for relative in ("serve/server.py", "serve/control.py", "serve/winjob.py", "strata_control/private_files.py", "tools/strata_tokenizer.py"):
        if not (source / relative).is_file():
            raise ValueError(f"native-source is missing required runtime code: {relative}")
    if "--control-file" not in (source / "serve/server.py").read_text(encoding="utf-8"):
        raise ValueError("native-source must already include the reviewed native control integration; no old patch is applied")
    protected = [*baseline["protected"], *baseline["assets"], file_record(baseline_path), file_record(python)]
    protected += [file_record(p) for p in (bundle_path, terminal / "VERSION", terminal / "COMMIT") if p.is_file()]
    protected += [file_record(p) for p in tree_files(engine)]
    output.mkdir(parents=True)
    runtime = output / "native"
    runtime.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", ".git")
    for name in PAYLOAD_DIRS:
        if (source / name).is_dir():
            shutil.copytree(source / name, runtime / name, ignore=ignore)
    for name in PAYLOAD_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, runtime / name)
    source_files = []
    for copied in tree_files(runtime):
        original = source / copied.relative_to(runtime)
        record = file_record(original)
        if record["sha256"] != sha(copied):
            raise ValueError(f"native-source changed during snapshot: {original}")
        source_files.append(record)
    # Source checkouts need not contain the archive's VERSION file. Preserve the
    # selected Control version metadata without pretending it came from Git.
    runtime_version = runtime / "VERSION"
    metadata_sources = []
    if runtime_version.is_file():
        if runtime_version.read_text(encoding="utf-8").strip() != control_version:
            raise ValueError("native-source VERSION and selected Control version disagree")
    else:
        shutil.copy2(terminal / "VERSION", runtime_version)
        metadata_sources.append({"runtime_path": str(runtime_version), "source": file_record(terminal / "VERSION")})
    for name in ("Invoke-Strata.ps1",):
        shutil.copy2(HERE / name, output / name)
    for name in ("Start-Strata.cmd", "Stop-Strata.cmd"):
        rendered = (HERE / (name + ".in")).read_text(encoding="utf-8").replace("@DEPLOYMENT@", str(output)).replace("@ROOT@", str(root))
        (output / name).write_text(rendered, encoding="utf-8", newline="\r\n")
    configs, logs = output / "configs", output / "logs"
    configs.mkdir()
    logs.mkdir()
    state = (args.state_directory or (root / "runtime" / output.name if output.parent == root / "production"
                                     else output.parent / "runtime" / output.name)).resolve()
    if state.exists() or any(state.is_relative_to(path) for path in input_dirs):
        raise ValueError("state-directory must be NEW and outside inputs; never reuse the baseline's control credentials")
    control_file = state / "control/backend.json"
    gateway_config = load(baseline["gateway_config"])
    cors = gateway_config.get("security", {}).get("cors", {})
    if "allowedOrigins" in cors:
        origins = {f"http://{host}:{baseline['port']}": f"http://{host}:{args.port}" for host in ("127.0.0.1", "localhost")}
        cors["allowedOrigins"] = [origins.get(value, value) for value in cors["allowedOrigins"]]
    bindings, comparisons = [], []
    for index, old in enumerate(baseline["bindings"]):
        original = load(old["config"])
        cfg = {**original, "exe": str(exe), "lib_dirs": list(lib_dirs), "cwd": str(runtime),
               "log": str(logs / f"model-{index}.engine.log")}
        declarations = {}
        if args.declare_mtp:
            if "drafting" not in original and "--mtp" in original["args"] and original.get("mtp") is not False:
                declarations["drafting"] = "mtp"
                cfg.update(declarations)
            elif original.get("drafting") != "mtp":
                raise ValueError("--declare-mtp needs the baseline's enabled MTP and no contradictory drafting declaration")
        assert_config(original, cfg, declarations)
        config_path = configs / f"{'iq3' if index == 0 else 'iq4'}.json"
        save(config_path, cfg)
        before = probe(Path(old["command"][0]), baseline["runtime"], old["config"], logs / f"baseline-{index}.effective.json")
        after = probe(python, runtime, config_path, logs / f"candidate-{index}.effective.json")
        assert_effective(before, after)
        command = [str(python), "-B", str(runtime / "serve/server.py"), "--engine", "strata", "--config", str(config_path),
                   "--host", "127.0.0.1", "--port", str(args.backend_ports[index]), "--control-file", str(control_file),
                   "--read-only-config", "--shutdown-timeout", "300"]
        stop = [str(python), "-B", str(runtime / "serve/control.py"), "stop", "--file", str(control_file), "--pid", "${PID}", "--timeout", "360"]
        bindings.append({**old, "config": str(config_path), "original_config": old.get("original_config", old["config"]),
                         "baseline_config": old["config"], "config_sha256": sha(config_path), "port": args.backend_ports[index],
                         "command": command, "config_declarations": declarations, "effective_args": after["effective_args"],
                         "expected_tuning_env": after["tuning_env"], "effective_gpu_order": after["gpu_order"]})
        comparisons.append({"model": old["model"], "unchanged_args_env_gpu": True, "scope": after["scope"],
                            "relocated": {key: {"before": original.get(key), "after": cfg[key]} for key in ("exe", "lib_dirs", "cwd", "log")},
                            "declarations": declarations, "before": str(logs / f"baseline-{index}.effective.json"),
                            "after": str(logs / f"candidate-{index}.effective.json")})
        spec = gateway_config["models"][old["model"]]
        spec.update(cmd=subprocess.list2cmdline(command), cmdStop=subprocess.list2cmdline(stop), proxy=f"http://127.0.0.1:{args.backend_ports[index]}")
        # Preserve every gateway environment value except the per-deployment runtime directory.
        spec["env"] = [value for value in spec["env"] if not value.startswith("STRATA_RUNTIME_DIR=")] + [f"STRATA_RUNTIME_DIR={state}"]
    # ACL setup is a one-shot child of the selected Control Python, never an owner daemon.
    state.parent.mkdir(parents=True, exist_ok=True)
    with (logs / "private-files.log").open("wb") as stream:
        subprocess.run([str(python), "-B", "-c", "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                        "from strata_control import private_files as p; s=Path(sys.argv[2]); "
                        "p.private_mkdir(s,parents=True,exist_ok=False); p.restrict_to_current_user(s,directory=True); "
                        "p.private_mkdir(s/'control',exist_ok=False); p.restrict_to_current_user(s/'control',directory=True)",
                        str(runtime), str(state)], stdout=stream, stderr=stream, check=True)
    gateway_path = output / "llama-swap.yaml"
    save(gateway_path, gateway_config)
    save(output / "config-comparison.json", comparisons)
    native_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assets = [p for p in tree_files(output) if not p.is_relative_to(logs)]
    assets += sorted(logs.glob("*.effective.json"))  # immutable CPU evidence, not the append-only serving logs
    native_tree = [{"path": path.relative_to(runtime).as_posix(), "sha256": sha(path)} for path in tree_files(runtime)]
    native_tree_sha = hashlib.sha256(json.dumps(native_tree, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = {"schema_version": 2, "mode": "direct-native", "root": str(root), "directory": str(output),
                "runtime": str(runtime), "state_directory": str(state), "control_file": str(control_file),
                "validation_root": str((root / "production/validation") if output.parent == root / "production" else output.parent / "validation"),
                "terminal_root": str(terminal), "control_version": control_version, "engine_directory": str(engine),
                "expected_engine_version": version, "engine_version_evidence": "declared; live INFO engine= must verify",
                "source_engine_commit": args.engine_commit or build.get("source_commit") or build.get("commit"),
                "source_native_revision": native_revision, "native_source": str(source), "native_source_files": source_files,
                "native_tree_sha256": native_tree_sha, "native_file_count": len(native_tree), "runtime_metadata_sources": metadata_sources,
                "native_source_revision_scope": "HEAD is provenance only; exact copied working-tree files are SHA-256 pinned",
                "baseline_manifest": file_record(baseline_path), "gateway": baseline["gateway"], "gateway_version": baseline["gateway_version"],
                "gateway_config": str(gateway_path), "port": args.port, "host": "127.0.0.1", "bindings": bindings,
                "protected": list({item["path"]: item for item in protected}.values()), "assets": [file_record(p) for p in assets],
                "models_unchanged": True, "engine_unchanged": False, "models_and_engine_unchanged": False,
                "native_patch_applied": False, "config_comparison": str(output / "config-comparison.json"),
                "verification_limits": ["CPU shaping is not live child process/environment inspection", "No model started by staging",
                                        "No model quality or performance acceptance", "Original model paths/args are retained; weight bytes are not recopied or rehashed"]}
    # Inputs are checked again after copying. A partial output is deliberately retained on failure.
    assert_files(manifest)
    for record in source_files:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(f"native-source changed during staging: {record['path']}")
    save(output / "manifest.json", manifest)
    print(json.dumps({"ok": True, "staged_only": True, "directory": str(output), "control_version": control_version,
                      "expected_engine_version": version, "engine_and_weights_copied": False}))
    return 0
