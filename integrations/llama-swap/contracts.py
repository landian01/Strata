"""CPU-only deployment contracts. This module never owns a serving process."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

RELOCATED = frozenset(("exe", "lib_dirs", "cwd", "log"))
DECLARATIONS = {"drafting": "mtp"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def tree_files(root):
    return sorted((p for p in Path(root).rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".log")), key=str)


def assert_files(manifest):
    for item in manifest["protected"] + manifest["assets"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"pinned deployment input changed: {item['path']}")


def assert_config(original, candidate, declarations=None):
    """No tuning changes; only four path fields and explicitly requested declarations."""
    declarations = declarations or {}
    for key, value in declarations.items():
        if key not in DECLARATIONS or value != DECLARATIONS[key] or key in original:
            raise ValueError(f"unsupported or replacing declaration: {key}")
    expected = {k: v for k, v in original.items() if k not in RELOCATED}
    expected.update(declarations)
    actual = {k: v for k, v in candidate.items() if k not in RELOCATED}
    if actual != expected:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)
                         or (k in expected) != (k in actual))
        raise ValueError(f"non-path model configuration changed: {changed}")


def flags(args):
    """Read named values without assuming every option has a value or losing empty strings."""
    return {key: args[i + 1] for i, key in enumerate(args[:-1]) if key.startswith("--")
            and not args[i + 1].startswith("--")}


def expected_metrics(binding, cfg, version=None):
    values = flags(cfg["args"])
    main, helper = binding["expected_slots"]
    expected = {"model": binding["model"], "expert_slots_primary": main, "expert_slots": main + helper,
                "pool_workers": int(values["--pool-workers"]), "mtp_max": int(values["--spec"]),
                "max_context": int(values["--max-context"]), "kv": values["--kv"],
                "kv_resident": int(values["--kv-resident"])}
    if version:
        expected["engine"] = version  # engine's INFO engine=, NOT the BUILD.json fallback
        expected["version"] = version
    return expected


def assert_metrics(engine, expected):
    for key, value in expected.items():
        if engine.get(key) != value:
            raise AssertionError((key, engine.get(key), value))


def observe_engine(psutil, native, cfg, binding):
    """Best effort read-only inspection; Windows access denial is NOT verified evidence."""
    result = {"status": "NOT_VERIFIED", "scope": "live child process inspection", "children": []}
    try:
        children = native.children(recursive=True)
    except psutil.AccessDenied as exc:
        result["reason"] = repr(exc)
        return result
    found = False
    expected_exe = os.path.normcase(os.path.abspath(cfg["exe"]))
    for child in children:
        row = {"pid": child.pid, "fields": {}}
        result["children"].append(row)
        for field in ("exe", "cmdline"):
            try:
                row["fields"][field] = getattr(child, field)()
            except psutil.AccessDenied as exc:
                row["fields"][field] = {"status": "NOT_VERIFIED", "reason": repr(exc)}
        observed = row["fields"].get("exe")
        if not isinstance(observed, str):
            continue
        if os.path.normcase(os.path.abspath(observed)) != expected_exe:
            if Path(observed).name.lower() == "strata.exe":
                raise AssertionError(("wrong live engine executable", observed, cfg["exe"]))
            continue
        found = True
        try:
            env = child.environ()
        except psutil.AccessDenied as exc:
            env = {"status": "NOT_VERIFIED", "reason": repr(exc)}
        row["fields"]["environ"] = env
        commands = row["fields"]["cmdline"]
        if isinstance(commands, list):
            expected = [cfg["exe"], "--serve", *binding.get("effective_args", cfg["args"])]
            if commands != expected:
                raise AssertionError(("live engine command", commands, expected))
        if isinstance(env, dict) and env.get("status") != "NOT_VERIFIED":
            expected_env = {k: str(v) for k, v in cfg.get("env", {}).items()}
            for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
                if key in binding.get("expected_tuning_env", {}):
                    expected_env[key] = binding["expected_tuning_env"][key]
            for key, value in expected_env.items():
                if env.get(key) != value:
                    raise AssertionError(("live engine environment", key, env.get(key), value))
            # Do not retain unrelated inherited secrets in the acceptance report.
            row["fields"]["environ"] = {k: env.get(k) for k in expected_env}
        row["status"] = "VERIFIED" if isinstance(commands, list) and isinstance(env, dict) and env.get("status") != "NOT_VERIFIED" else "NOT_VERIFIED"
    if found and all(row.get("status") == "VERIFIED" for row in result["children"]):
        result["status"] = "VERIFIED"
    if result["status"] == "NOT_VERIFIED":
        result["reason"] = "some engine child fields are unavailable or no engine executable could be identified; INFO/CPU contracts are separate evidence"
    return result


def validation_root(manifest, override=None):
    return Path(override or manifest.get("validation_root") or Path(manifest["root"]) / "production/validation").resolve()


def probe(python, runtime, config, output):
    """Isolated imports: baseline and candidate may have different module versions."""
    output = Path(output)
    with output.with_suffix(".stdout.log").open("wb") as stdout, output.with_suffix(".stderr.log").open("wb") as stderr:
        subprocess.run([str(python), "-B", str(Path(__file__).resolve()), "probe", str(runtime), str(config), str(output)],
                       stdout=stdout, stderr=stderr, check=True, timeout=120)
    return load(output)


def startup_args(server, cfg):
    """Cover main's extra suffix without guessing tokenizer- or binary-dependent flags."""
    if cfg.get("effort_position", "start") != "start":
        raise ValueError("this upgrade contract supports only default/start effort_position; end requires a separate tokenizer/binary proof")
    suffix = server.effort_end_args(cfg, cfg["exe"], None) or []
    if suffix:
        raise ValueError("default/start effort_position unexpectedly adds engine arguments")
    return server.engine_args(cfg) + suffix


def assert_effective(before, after):
    for key in ("effective_args", "tuning_env", "path_suffix_sha256", "gpu_order"):
        if before[key] != after[key]:
            raise ValueError(f"CPU effective configuration changed: {key}: {before[key]!r} -> {after[key]!r}")
    if before["effective_args"] != before["configured_args"] or after["effective_args"] != after["configured_args"]:
        raise ValueError("the runtime implicitly rewrites the accepted engine args")


def probe_main(runtime, config, output):
    """Run shaping functions only. No engine, hardware probe, socket or subprocess may start."""
    sys.path.insert(0, str(Path(runtime).resolve()))
    cfg = load(config)
    # Match the one-shot entry's inherited-tuning cleanup. Do not read/tune a GPU.
    for key in list(os.environ):
        if key.startswith("STRATA_") or key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTHONPATH", "PYTHONHOME"):
            del os.environ[key]
    os.environ.update(STRATA_NO_PRESET="1", STRATA_NO_ACQUIRE="1")
    captured = io.StringIO()
    with mock.patch("subprocess.Popen", side_effect=RuntimeError("CPU probe forbids subprocesses")), \
            mock.patch("socket.socket", side_effect=RuntimeError("CPU probe forbids sockets")), contextlib.redirect_stdout(captured):
        from serve import server
        server.drafting_card_class = lambda: "blackwell"  # injected class, not a hardware measurement
        args = startup_args(server, cfg)
        env = server.child_env(cfg)
        gpu_order = server.ordered_gpus(cfg)
    library_dirs = [str(d) for d in cfg.get("lib_dirs", []) if Path(d).is_dir()]
    path = env.get("PATH", "")
    prefix = os.pathsep.join(library_dirs)
    if prefix:
        if not path.startswith(prefix + os.pathsep) and path != prefix:
            raise ValueError("unexpected library search path shaping")
        path = path[len(prefix):].lstrip(os.pathsep)
    result = {"scope": "CPU pure-function comparison; injected blackwell class; NOT a live process/environment observation",
              "configured_args": cfg["args"], "effective_args": args, "effort_suffix": [], "gpu_order": gpu_order,
              "tuning_env": {k: v for k, v in env.items() if k.startswith(("STRATA_", "CUDA_", "HIP_"))},
              "path_suffix_sha256": hashlib.sha256(path.encode()).hexdigest(), "lib_dirs": library_dirs,
              "diagnostics": captured.getvalue()}
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "probe":
        raise SystemExit("usage: contracts.py probe RUNTIME CONFIG OUTPUT")
    probe_main(*sys.argv[2:])
