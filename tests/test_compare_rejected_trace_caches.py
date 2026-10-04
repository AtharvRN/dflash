"""Strict serial/parallel cache parity rejects numerically meaningful drift."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from scripts.audit_rejected_trace_cache import ARRAY_SCHEMA, sha256
from scripts.compare_rejected_trace_caches import compare, save_comparison
from tests.test_rejected_trace_cache import make_cache


def rebind(root):
    receipts = json.loads((root/"receipts.json").read_text())
    for receipt in receipts:
        receipt["files"] = {name: sha256(root/name) for name in receipt["files"]}
        (root/f"receipt_{receipt['prompt_id']}.json").write_text(json.dumps(receipt))
    (root/"receipts.json").write_text(json.dumps(receipts))
    complete = json.loads((root/"COMPLETE.json").read_text())
    complete["binding"] = {name: sha256(root/name) for name in complete["binding"]}
    (root/"COMPLETE.json").write_text(json.dumps(complete))


def add_protocol(root):
    config = json.loads((root/"config.json").read_text())
    config.update(seed=1001, states_per_prompt=2, max_new_tokens=64, max_prompt_tokens=2048,
        dtype="bfloat16", attention="sdpa", thinking=False, temperature=0, tf32=False,
        manifest="/serial_inputs/manifest.jsonl", pilot_manifest="/serial_inputs/pilot.json",
        split_dir="/serial_inputs/split", eval_cache="/serial_inputs/eval", output=str(root),
        backup="/serial_backup", workers=1, gpu=0, max_seconds=3600,
        source_sha256="a"*64, reference_completion_sha256="b"*64,
        dependency_sha256={"dflash/model.py": "c"*64},
        models={name: {"repo": "synthetic/"+name, "revision": digit*40,
                       "path": "/serial_inputs/models/"+name}
                for name, digit in (("target", "1"), ("draft", "2"))},
        input_hashes={"/serial_inputs/"+name: digit*64 for name, digit in (
            ("manifest.jsonl", "3"), ("pilot.json", "4"), ("split/train_prompt_ids.json", "5"),
            ("split/val_prompt_ids.json", "6"), ("models.json", "7"))})
    (root/"config.json").write_text(json.dumps(config))
    rebind(root)


class RejectedTraceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.serial = make_cache(self.root/"serial")
        add_protocol(self.serial)
        self.parallel = self.root/"parallel"
        shutil.copytree(self.serial, self.parallel)

    def tearDown(self):
        self.directory.cleanup()

    def change_config(self, key, value):
        config = json.loads((self.parallel/"config.json").read_text())
        config[key] = value
        (self.parallel/"config.json").write_text(json.dumps(config))
        rebind(self.parallel)

    def change_shard(self, operation):
        path = self.parallel/"prompt_1.json"
        shard = json.loads(path.read_text())
        operation(shard)
        path.write_text(json.dumps(shard))
        rebind(self.parallel)

    def test_equal_caches_report_all_arrays_counts_and_bound_hashes(self):
        result = compare(self.serial, self.parallel)
        self.assertTrue(result["passed"])
        self.assertEqual(result["counts"]["rows"], 12)
        self.assertEqual(result["counts"]["prompts"], 6)
        self.assertEqual(set(result["arrays"]), set(ARRAY_SCHEMA) | {"prompt_id", "group", "eligible"})
        self.assertEqual(result["source_complete_sha256"]["left"], sha256(self.serial/"COMPLETE.json"))

    def test_paths_workers_and_elapsed_times_can_differ(self):
        config = json.loads((self.parallel/"config.json").read_text())
        for field in ("manifest", "pilot_manifest", "split_dir", "eval_cache"):
            config[field] = config[field].replace("serial_inputs", "parallel_inputs")
        config["input_hashes"] = {path.replace("serial_inputs", "parallel_inputs"): value
                                  for path, value in config["input_hashes"].items()}
        for model in config["models"].values():
            model["path"] = model["path"].replace("serial_inputs", "parallel_inputs")
        config.update(workers=4, gpu=3, output=str(self.parallel), backup="/parallel_backup", max_seconds=7200)
        (self.parallel/"config.json").write_text(json.dumps(config))
        for root, elapsed in ((self.serial, 2.0), (self.parallel, 1.0)):
            summary = json.loads((root/"collection_summary.json").read_text())
            summary.update(elapsed_seconds=elapsed, trace_capture_cuda_ms={"count": 4, "mean": elapsed, "p50": elapsed})
            (root/"collection_summary.json").write_text(json.dumps(summary))
            shard = json.loads((root/"prompt_1.json").read_text())
            shard["prompt_summary"]["trace_capture_cuda_ms"] = [elapsed, elapsed]
            (root/"prompt_1.json").write_text(json.dumps(shard))
            rebind(root)
        result = compare(self.serial, self.parallel)
        self.assertEqual(result["execution_differences"]["workers"], {"left": 1, "right": 4})

    def test_source_and_required_protocol_changes_rejected(self):
        original = json.loads((self.parallel/"config.json").read_text())
        for key, value, pattern in (("source_sha256", "d"*64, "source_sha256"),
                                    ("seed", 1002, "config.seed"),
                                    ("max_new_tokens", 128, "config.max_new_tokens")):
            with self.subTest(key=key):
                self.change_config(key, value)
                with self.assertRaisesRegex(ValueError, pattern):
                    compare(self.serial, self.parallel)
                (self.parallel/"config.json").write_text(json.dumps(original))
                rebind(self.parallel)
        models = copy.deepcopy(original["models"])
        models["target"]["revision"] = "3"*40
        self.change_config("models", models)
        with self.assertRaisesRegex(ValueError, "models.target.revision"):
            compare(self.serial, self.parallel)

    def test_gpu_execution_provenance_does_not_change_collection_protocol(self):
        self.change_config("gpu", None)
        self.change_config("use_visible_gpu", True)
        runtime = {"mode": "slurm_visible", "device": "cuda:0", "slurm_job_id": "123",
                   "occupancy_before_load": {"memory_mib": 24, "utilization_percent": 0}}
        self.change_config("gpu_runtime", runtime)
        result = compare(self.serial, self.parallel)
        self.assertTrue(result["passed"])
        self.assertEqual(result["execution_differences"]["gpu_runtime"]["right"], runtime)

    def test_dependency_hash_and_input_hash_changes_rejected(self):
        config = json.loads((self.parallel/"config.json").read_text())
        self.change_config("dependency_sha256", {"dflash/model.py": "d"*64})
        with self.assertRaisesRegex(ValueError, "dependency_sha256"):
            compare(self.serial, self.parallel)
        self.change_config("dependency_sha256", config["dependency_sha256"])
        hashes = config["input_hashes"]
        hashes[config["manifest"]] = "e"*64
        self.change_config("input_hashes", hashes)
        with self.assertRaisesRegex(ValueError, "input_hashes.manifest"):
            compare(self.serial, self.parallel)

    def test_float16_feature_change_rejected_even_with_valid_receipts(self):
        path = self.parallel/"prompt_1.npz"
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["trace_target"][0, 0, 0] += np.float16(1)
        np.savez(path, **arrays)
        rebind(self.parallel)
        with self.assertRaisesRegex(ValueError, "Array mismatch: trace_target"):
            compare(self.serial, self.parallel)

    def test_actual_labels_and_previous_proofs_are_compared(self):
        self.change_shard(lambda shard: shard["states"][0]["previous"]["posterior_ids"].__setitem__(-1, 90))
        with self.assertRaisesRegex(ValueError, "previous.posterior_ids"):
            compare(self.serial, self.parallel)
        self.change_shard(lambda shard: shard["states"][0]["previous"]["posterior_ids"].__setitem__(-1, 77))
        self.change_shard(lambda shard: shard["states"][0]["outcomes"]["2"].__setitem__("accepted", 0))
        path = self.parallel/"prompt_1.npz"
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["actual"][0, 0] = 0
        np.savez(path, **arrays)
        rebind(self.parallel)
        with self.assertRaisesRegex(ValueError, "outcomes.2.accepted"):
            compare(self.serial, self.parallel)

    def test_cycle_identity_and_check_coverage_are_compared(self):
        def move_cycle(shard):
            row = shard["states"][0]
            row.update(cycle=3, previous_cycle=2)
            row["previous"]["cycle"] = 2
        self.change_shard(move_cycle)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            compare(self.serial, self.parallel)

    def test_missing_provenance_or_incomplete_cache_rejected(self):
        config = json.loads((self.parallel/"config.json").read_text())
        del config["source_sha256"]
        (self.parallel/"config.json").write_text(json.dumps(config))
        rebind(self.parallel)
        with self.assertRaisesRegex(ValueError, "Missing comparison provenance"):
            compare(self.serial, self.parallel)
        (self.parallel/"COMPLETE.json").unlink()
        with self.assertRaisesRegex(ValueError, "not complete"):
            compare(self.serial, self.parallel)

    def test_fresh_output_only_and_failed_comparison_writes_nothing(self):
        output = self.root/"comparison.json"
        save_comparison(self.serial, self.parallel, output)
        before = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            save_comparison(self.serial, self.parallel, output)
        self.assertEqual(output.read_bytes(), before)
        self.change_config("seed", 1002)
        failed = self.root/"failed.json"
        with self.assertRaisesRegex(ValueError, "config.seed"):
            save_comparison(self.serial, self.parallel, failed)
        self.assertFalse(failed.exists())


if __name__ == "__main__":
    unittest.main()
