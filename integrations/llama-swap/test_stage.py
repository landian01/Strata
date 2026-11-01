"""CPU-only integration contracts. No gateway, real engine or model is launched."""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import candidate_stage
import contracts
import stage


class ConfigContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = {"exe": "old/strata.exe", "args": ["--mtp", "weights/mtp", "--layer-split", ""],
                    "env": {"STRATA_CACHE": "123"}, "gpu": [0, 1], "gpu_order": "as_given",
                    "gpu_roles": {"0": "main", "1": "helper"}, "cwd": "old", "log": "old.log", "lib_dirs": ["old"]}

    def test_only_path_relocations(self):
        candidate = {**self.cfg, "exe": "new/strata.exe", "cwd": "new", "log": "new.log", "lib_dirs": ["new"]}
        contracts.assert_config(self.cfg, candidate)

    def test_tuning_changes_and_unknown_declarations_are_refused(self):
        for key, value in (("args", ["--max-context", "1"]), ("env", {}), ("gpu", [1, 0]),
                           ("gpu_roles", {}), ("parallel", 8), ("tokenizer", "other")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.assert_config(self.cfg, {**self.cfg, key: value})
        with self.assertRaises(ValueError):
            contracts.assert_config(self.cfg, {**self.cfg, "parallel": 8}, {"parallel": 8})

    def test_explicit_mtp_declaration_never_replaces_a_choice(self):
        candidate = {**self.cfg, "drafting": "mtp"}
        contracts.assert_config(self.cfg, candidate, {"drafting": "mtp"})
        with self.assertRaises(ValueError):
            contracts.assert_config(candidate, candidate, {"drafting": "mtp"})

    def test_effective_args_env_and_gpu_are_exact(self):
        before = {"configured_args": self.cfg["args"], "effective_args": self.cfg["args"],
                  "tuning_env": self.cfg["env"], "gpu_order": [0, 1], "path_suffix_sha256": "same"}
        contracts.assert_effective(before, copy.deepcopy(before))
        for key, value in (("effective_args", []), ("tuning_env", {}), ("gpu_order", [1, 0]), ("path_suffix_sha256", "different")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.assert_effective(before, {**before, key: value})

    def test_implicit_rewrite_is_not_accepted_even_when_both_runtimes_do_it(self):
        evidence = {"configured_args": [], "effective_args": ["--new-default"], "tuning_env": {},
                    "gpu_order": [], "path_suffix_sha256": "same"}
        with self.assertRaises(ValueError):
            contracts.assert_effective(evidence, evidence)

    def test_default_and_explicit_start_verify_empty_real_suffix(self):
        server = SimpleNamespace(engine_args=lambda cfg: list(cfg["args"]), effort_end_args=mock.Mock(return_value=None))
        for cfg in (self.cfg, {**self.cfg, "effort_position": "start"}):
            self.assertEqual(contracts.startup_args(server, cfg), cfg["args"])
            server.effort_end_args.assert_called_with(cfg, cfg["exe"], None)
        server.effort_end_args.return_value = ["--unexpected-tail", "1"]
        with self.assertRaisesRegex(ValueError, "unexpectedly adds"):
            contracts.startup_args(server, self.cfg)

    def test_end_effort_requires_separate_proof_not_a_guessed_suffix(self):
        server = SimpleNamespace(engine_args=mock.Mock(), effort_end_args=mock.Mock())
        with self.assertRaisesRegex(ValueError, "separate tokenizer/binary proof"):
            contracts.startup_args(server, {**self.cfg, "effort_position": "end"})
        server.effort_end_args.assert_not_called()
        server.engine_args.assert_not_called()

    def test_live_command_never_ignores_extra_tail_arguments(self):
        class AccessDenied(Exception):
            pass
        child = SimpleNamespace(pid=123, exe=lambda: self.cfg["exe"],
            cmdline=lambda: [self.cfg["exe"], "--serve", *self.cfg["args"], "--tail-role-token", "123"],
            environ=lambda: self.cfg["env"])
        with self.assertRaisesRegex(AssertionError, "live engine command"):
            contracts.observe_engine(SimpleNamespace(AccessDenied=AccessDenied), SimpleNamespace(children=lambda **kw: [child]), self.cfg, {})

    def test_metrics_version_comes_from_info_not_metadata(self):
        cfg = {"args": ["--pool-workers", "5", "--spec", "6", "--max-context", "32768", "--kv", "int8", "--kv-resident", "0"]}
        expected = contracts.expected_metrics({"model": "fixture", "expected_slots": [2664, 2200]}, cfg, "0.1.55")
        contracts.assert_metrics(expected, expected)
        with self.assertRaises(AssertionError):
            contracts.assert_metrics({**expected, "engine": "0.1.50"}, expected)
        with self.assertRaises(AssertionError):
            contracts.assert_metrics({k: v for k, v in expected.items() if k != "engine"}, expected)

    def test_flags_preserve_empty_values_and_skip_boolean_flags(self):
        self.assertEqual(contracts.flags(["--bool", "--layer-split", "", "--spec", "6"]), {"--layer-split": "", "--spec": "6"})

    def test_validation_root_legacy_and_candidate(self):
        self.assertEqual(contracts.validation_root({"root": str(Path.cwd())}), (Path.cwd() / "production/validation").resolve())
        self.assertEqual(contracts.validation_root({"root": "unused", "validation_root": str(Path.cwd())}), Path.cwd())

    def test_access_denied_is_not_verified(self):
        class AccessDenied(Exception):
            pass
        native = SimpleNamespace(children=mock.Mock(side_effect=AccessDenied("fixture")))
        result = contracts.observe_engine(SimpleNamespace(AccessDenied=AccessDenied), native, self.cfg, {})
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertIn("fixture", result["reason"])

    def test_live_observation_redacts_unrelated_environment(self):
        class AccessDenied(Exception):
            pass
        child = SimpleNamespace(pid=123, exe=lambda: self.cfg["exe"], cmdline=lambda: [self.cfg["exe"], "--serve", *self.cfg["args"]],
                                environ=lambda: {**self.cfg["env"], "UNRELATED_SECRET": "never record"})
        result = contracts.observe_engine(SimpleNamespace(AccessDenied=AccessDenied), SimpleNamespace(children=lambda **kw: [child]), self.cfg, {})
        self.assertEqual(result["status"], "VERIFIED")
        self.assertNotIn("UNRELATED_SECRET", json.dumps(result))


class EntryFailurePrivacy(unittest.TestCase):
    """Exercise the actual nested helpers without running the entry's main or a process."""
    def setUp(self):
        self.nonce = "c" * 64
        self.record = {"pid": 123, "token": "456", "nonce": self.nonce, "ready": True,
                       "state": "stopped", "graceful": True, "exit_code": 0}
        self.identity = SimpleNamespace(token="456", alive=lambda: False, exit_code=lambda: 0)
        filename = Path(__file__).with_name("verify_entry.py")
        tree = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        helpers = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in ("hold", "final_status")]
        self.assertEqual(len(helpers), 2)
        namespace = {"json": json, "hashlib": hashlib, "identities": [],
                     "winjob": SimpleNamespace(ProcessIdentity=lambda pid: self.identity),
                     "control": SimpleNamespace(read_bytes=lambda: json.dumps(self.record).encode("ascii"))}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(filename), "exec"), namespace)
        self.hold, self.final_status = namespace["hold"], namespace["final_status"]

    def assert_private_failure(self, operation):
        with self.assertRaises(AssertionError) as caught:
            operation()
        public_report = json.dumps({"ok": False, "error": repr(caught.exception)})
        self.assertNotIn(self.nonce, public_report)
        self.assertNotIn("d" * 64, public_report)

    def test_hold_assertion_error_never_contains_control_nonce(self):
        self.assert_private_failure(lambda: self.hold(self.record))  # matching token, dead handle
        self.identity.token = "different"
        self.identity.alive = lambda: True
        self.assert_private_failure(lambda: self.hold(self.record))

    def test_final_status_assertion_errors_never_contain_control_nonce(self):
        initial = dict(self.record)
        cases = [
            (True, {"nonce": "d" * 64}, False),
            (True, {}, True),
            (True, {"state": "forced"}, False),
            (False, {"ready": False, "state": "forced"}, False),
            (False, {"ready": False, "state": "aborted", "reason": "nonready-abort", "graceful": False, "exit_code": 0}, False),
        ]
        for ready, changes, alive in cases:
            with self.subTest(ready=ready, state=changes.get("state"), alive=alive):
                self.record = {**initial, **changes}
                self.identity.alive = lambda: alive
                self.assert_private_failure(lambda: self.final_status(initial, self.identity, ready=ready))

    def test_successful_final_status_exposes_only_hashed_instance(self):
        result = self.final_status(dict(self.record), self.identity, ready=True)
        self.assertNotIn(self.nonce, json.dumps(result))
        self.assertNotIn("nonce", result)
        self.assertEqual(result["instance"], hashlib.sha256(self.nonce.encode("ascii")).hexdigest()[:24])


class StageFixtures(unittest.TestCase):
    """Stage mechanics use fake payloads and mocked ACL/probe calls, NOT ACL acceptance."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "upgrade/source"
        self.output = self.root / "upgrade/deployment"
        self.terminal = self.root / "control"
        self.engine = self.root / "engine"
        self.baseline = self.root / "production/old"
        for path in (self.source, self.terminal, self.engine, self.baseline):
            path.mkdir(parents=True)
        def put(path, text):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.put = put
        for relative in ("serve/server.py", "serve/control.py", "serve/winjob.py", "strata_control/private_files.py", "tools/strata_tokenizer.py"):
            put(self.source / relative, "# fixture --control-file\n")
        put(self.source / "serve/another_import.py", "# must also be hashed\n")
        put(self.terminal / "python/python.exe", "fixture: never executed")
        put(self.terminal / "VERSION", "0.1.55\n")
        candidate_stage.save(self.terminal / "BUNDLE.json", {"component": "strata-control", "version": "0.1.55"})
        put(self.engine / "strata.exe", "fixture: never executed")
        put(self.engine / "strata_kq.dll", "fixture library")
        candidate_stage.save(self.engine / "BUILD.json", {"version": "0.1.55", "commit": "fixture-commit"})
        put(self.baseline / "gateway.exe", "fixture: never executed")
        bindings = []
        for i in range(2):
            cfg = {"exe": "old-strata.exe", "args": ["--mtp", "original-model/mtp", "--layer-split", ""],
                   "env": {"STRATA_FIXED": "1"}, "gpu": [0, 1], "cwd": "old", "log": "old.log", "lib_dirs": ["old"]}
            config = self.baseline / f"model-{i}.json"
            candidate_stage.save(config, cfg)
            bindings.append({"name": f"Fixture{i}", "model": f"model-{i}", "aliases": [f"iq{i+3}"], "config": str(config),
                             "command": [str(self.terminal / "python/python.exe")], "expected_slots": [1, 2], "port": 18181 + i})
        gateway = self.baseline / "gateway.json"
        candidate_stage.save(gateway, {"models": {b["model"]: {"env": ["STRATA_NO_PRESET=1", "STRATA_RUNTIME_DIR=old"]} for b in bindings},
                                      "security": {"cors": {"allowedOrigins": ["http://127.0.0.1:8081"]}}})
        self.manifest = self.baseline / "manifest.json"
        candidate_stage.save(self.manifest, {"mode": "direct-native", "root": str(self.root), "runtime": str(self.baseline / "native"),
            "bindings": bindings, "gateway": str(self.baseline / "gateway.exe"), "gateway_version": "fixture", "gateway_config": str(gateway),
            "port": 8081, "protected": [contracts.file_record(self.baseline / "gateway.exe")],
            "assets": [contracts.file_record(gateway), *[contracts.file_record(b["config"]) for b in bindings]]})
        self.args = argparse.Namespace(root=self.root, output=self.output, terminal_root=self.terminal, engine_dir=self.engine,
            native_source=self.source, baseline_manifest=self.manifest, state_directory=None, expected_engine_version="0.1.55",
            engine_commit="custom-commit", declare_mtp=False, port=9081, backend_ports=[19181, 19182])
        self.probes = []
        def fake_probe(python, runtime, config, output):
            cfg = contracts.load(config)
            result = {"scope": "MOCKED unit fixture, no model or ACL acceptance", "configured_args": cfg["args"], "effective_args": cfg["args"],
                      "tuning_env": cfg["env"], "gpu_order": cfg["gpu"], "path_suffix_sha256": "same"}
            candidate_stage.save(output, result)
            self.probes.append(result)
            return result
        self.addCleanup(mock.patch.stopall)
        self.probe_mock = mock.patch.object(candidate_stage, "probe", side_effect=fake_probe).start()
        self.acl = mock.patch.object(candidate_stage.subprocess, "run").start()
        mock.patch.object(candidate_stage.subprocess, "check_output", return_value="native-head\n").start()

    def run_stage(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(candidate_stage.stage_candidate(self.args), 0)
        return contracts.load(self.output / "manifest.json")

    def test_complete_snapshot_baseline_protection_and_config_relocation(self):
        manifest = self.run_stage()
        self.assertFalse(manifest["native_patch_applied"])
        self.assertFalse(manifest["models_and_engine_unchanged"])
        self.assertEqual(manifest["source_engine_commit"], "custom-commit")
        self.assertEqual(manifest["source_native_revision"], "native-head")
        self.assertEqual(manifest["control_version"], "0.1.55")
        self.assertEqual((Path(manifest["runtime"]) / "VERSION").read_text().strip(), "0.1.55")
        self.assertEqual(manifest["runtime_metadata_sources"][0]["source"]["path"], str(self.terminal / "VERSION"))
        self.assertEqual(len(manifest["native_tree_sha256"]), 64)
        assets = {Path(r["path"]).name for r in manifest["assets"]}
        self.assertIn("another_import.py", assets)
        self.assertIn("strata_kq.dll", {Path(r["path"]).name for r in manifest["protected"]})
        self.assertEqual(len(self.probes), 4)
        self.assertEqual(self.acl.call_count, 1)
        for binding in manifest["bindings"]:
            before, after = contracts.load(binding["baseline_config"]), contracts.load(binding["config"])
            contracts.assert_config(before, after)
            self.assertEqual(after["args"], before["args"])
            self.assertEqual(after["env"], before["env"])
            self.assertEqual(after["cwd"], manifest["runtime"])
        gateway = contracts.load(manifest["gateway_config"])
        self.assertEqual(gateway["security"]["cors"]["allowedOrigins"], ["http://127.0.0.1:9081"])
        contracts.assert_files(manifest)

    def test_changed_baseline_refuses_before_output(self):
        self.put(self.baseline / "model-0.json", "{}")
        with self.assertRaises(ValueError):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_build_version_conflict_refuses_before_output(self):
        self.args.expected_engine_version = "wrong"
        with self.assertRaises(ValueError):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_partial_or_source_output_is_refused(self):
        for output in (self.source / "must-not-write", self.baseline):
            self.args.output = output
            with self.assertRaises(ValueError):
                self.run_stage()

    def test_asset_tampering_is_detected(self):
        manifest = self.run_stage()
        self.put(self.output / "native/serve/another_import.py", "changed")
        with self.assertRaises(ValueError):
            contracts.assert_files(manifest)

    def test_nested_package_libraries_are_pinned(self):
        self.put(self.engine / "runtime/extra.dll", "fixture nested library")
        candidate_stage.save(self.engine / "BUILD.json", {"version": "0.1.55", "lib_dirs": ["runtime"]})
        manifest = self.run_stage()
        cfg = contracts.load(manifest["bindings"][0]["config"])
        self.assertEqual(cfg["lib_dirs"], [str(self.engine), str(self.engine / "runtime")])
        self.assertIn("extra.dll", {Path(item["path"]).name for item in manifest["protected"]})

    def test_external_package_library_path_is_refused(self):
        candidate_stage.save(self.engine / "BUILD.json", {"version": "0.1.55", "lib_dirs": ["../outside"]})
        with self.assertRaises(ValueError):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_source_mutation_during_snapshot_refuses_manifest(self):
        original_probe = self.probe_mock.side_effect
        def mutate_source(*args):
            result = original_probe(*args)
            self.put(self.source / "serve/another_import.py", "changed during staging")
            return result
        self.probe_mock.side_effect = mutate_source
        with self.assertRaisesRegex(ValueError, "native-source changed"):
            self.run_stage()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_declare_mtp_is_explicit_and_recorded(self):
        self.args.declare_mtp = True
        manifest = self.run_stage()
        for binding in manifest["bindings"]:
            self.assertEqual(binding["config_declarations"], {"drafting": "mtp"})


class CommandLineCompatibility(unittest.TestCase):
    def test_four_upgrade_paths_are_atomic(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            stage.main(["--root", "unused", "--output", "unused", "--terminal-root", "control"])
        self.assertEqual(raised.exception.code, 2)

    def test_legacy_still_reads_original_pinned_inputs(self):
        with mock.patch.object(stage, "read_json", side_effect=RuntimeError("legacy sentinel")) as read:
            with self.assertRaisesRegex(RuntimeError, "legacy sentinel"):
                stage.main(["--root", str(Path.cwd()), "--output", str(Path.cwd() / "production/NEVER-CREATED")])
        self.assertEqual(read.call_args.args[0], Path.cwd() / "production/release.json")


if __name__ == "__main__":
    unittest.main()
