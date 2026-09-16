import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import scripts.audit_prefusion_cache as auditor
from scripts.train_prefusion_acceptance import validate_audit


class PrefusionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "shards").mkdir()
        (self.root / "provenance").mkdir()
        source = self.root / "provenance/messages.jsonl"
        source.write_text("fixture source\n")
        self.original_source = str(self.root / "removed/messages.jsonl")
        self.info = {
            "format": "dflash_prefusion_cache_v1", "input_kind": "paired_raw_and_fused",
            "input_dim": 2, "fused_dim": 2, "num_slots": 15, "context_window": 1,
            "train_rows": 2, "val_rows": 4,
            "collection_config": {
                "hashes": {self.original_source: auditor.sha256(source)},
                "selected_prompts": [],
            },
            "shards": [],
        }
        weights = {"fc": {"weight": torch.eye(2, dtype=torch.bfloat16)},
                   "hidden_norm": {"weight": torch.ones(2, dtype=torch.bfloat16)},
                   "rms_norm_eps": 1e-6}
        torch.save(weights, self.root / "fusion.pt")
        x = torch.tensor([[1., 2.], [2., 1.]])
        fused = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(torch.float16)
        split_data = {"train": [], "val": []}
        for pid, group in enumerate(("train", "calibration", "assessment")):
            trajectory = np.arange(10, 20, dtype=np.int64)
            starts = np.array([2, 4], dtype=np.int64)
            data = {"raw_features": x.numpy().astype(np.float16)[:, None],
                    "features": fused.numpy()[:, None],
                    "accepted_len": np.array([1, 0], dtype=np.int64),
                    "cycle_id": np.array([0, 1], dtype=np.int64),
                    "prefix_length": starts, "anchor_id": trajectory[starts],
                    "prefix_sha256": self.prefix_hashes(trajectory, starts),
                    "trajectory_token_ids": trajectory}
            path = self.root / "shards" / f"{pid}.npz"
            np.savez(path, **data)
            self.info["shards"].append({"prompt_id": pid, "group": group, "rows": 2,
                "path": str(self.root / "removed" / path.name), "sha256": auditor.sha256(path)})
            self.info["collection_config"]["selected_prompts"].append({"prompt_id": pid, "group": group})
            split_data["train" if group == "train" else "val"].append((pid, data))
        for split, entries in split_data.items():
            directory = self.root / split
            directory.mkdir()
            for name in ("raw_features", "features", "accepted_len"):
                np.save(directory / f"{name}.npy", np.concatenate([data[name] for _, data in entries]))
            n = len(entries) * 2
            rows = np.column_stack([np.zeros(n, dtype=np.int64), np.arange(n),
                np.repeat([pid for pid, _ in entries], 2), np.tile([0, 1], len(entries)),
                np.tile([1, 0], len(entries))])
            np.save(directory / "row_index.npy", rows)
            np.save(directory / "mask.npy", np.ones((n, 1), dtype=np.uint8))
        np.save(self.root / "val/calibration.npy", np.array([True, True, False, False]))
        self.save_manifest()

    @staticmethod
    def prefix_hashes(trajectory, starts):
        return np.array([hashlib.sha256(trajectory[:int(start)+1].tobytes()).hexdigest()
                         for start in starts])

    def save_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.info))

    def rewrite_shard(self, change):
        path = self.root / "shards/0.npz"
        with np.load(path) as stored:
            data = {name: stored[name] for name in stored.files}
        change(data)
        np.savez(path, **data)
        self.info["shards"][0]["sha256"] = auditor.sha256(path)
        self.save_manifest()

    def test_success_binds_every_array_and_auditor_source(self):
        np.save(self.root / "train/extra.npy", np.array([7]))
        result = auditor.audit(self.root)
        expected = {"manifest.json", "fusion.pt"}
        expected.update(p.relative_to(self.root).as_posix()
                        for split in ("train", "val") for p in (self.root / split).glob("*.npy"))
        self.assertEqual(set(result["binding"]), expected)
        self.assertEqual(result["binding"], auditor.cache_binding(self.root))
        self.assertEqual(result["auditor_source_sha256"], auditor.sha256(Path(auditor.__file__)))
        self.assertEqual(result["rows"], 6)
        self.assertLess(result["worst_row_relative_rmse"], .015)
        self.assertTrue(result["exact_cache_shard_alignment_verified"])
        self.assertEqual(validate_audit(self.root), result)

    def test_truncated_prefix_arrays_fail(self):
        path = self.root / "shards/0.npz"
        original = path.read_bytes()
        for field in ("prefix_length", "anchor_id", "prefix_sha256"):
            with self.subTest(field=field):
                path.write_bytes(original)
                self.rewrite_shard(lambda data: data.update({field: data[field][:0]}))
                with self.assertRaisesRegex(ValueError, "per-row entries"):
                    auditor.audit(self.root)

    def test_prefix_bounds_and_integer_types_fail(self):
        path = self.root / "shards/0.npz"
        original = path.read_bytes()
        for starts, message in ((np.array([2, 10]), "bounds"),
                                (np.array([-1, 4]), "nonnegative integer"),
                                (np.array([2., 4.]), "nonnegative integer")):
            with self.subTest(starts=starts):
                path.write_bytes(original)
                self.rewrite_shard(lambda data: data.update(prefix_length=starts))
                with self.assertRaisesRegex(ValueError, message):
                    auditor.audit(self.root)

    def test_consecutive_prefix_increment(self):
        def change(data):
            data["prefix_length"] = np.array([2, 5])
            data["anchor_id"] = data["trajectory_token_ids"][[2, 5]]
            data["prefix_sha256"] = self.prefix_hashes(data["trajectory_token_ids"], [2, 5])
        self.rewrite_shard(change)
        with self.assertRaisesRegex(ValueError, "accepted_len \\+ 1"):
            auditor.audit(self.root)

    def test_cycles_strictly_increase_but_may_have_gaps(self):
        for cycles, succeeds in (([1, 0], False), ([0, 2], True)):
            self.rewrite_shard(lambda data: data.update(cycle_id=np.array(cycles)))
            rows = np.load(self.root / "train/row_index.npy")
            rows[:, 3] = cycles
            np.save(self.root / "train/row_index.npy", rows)
            if succeeds:
                auditor.audit(self.root)
            else:
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    auditor.audit(self.root)

    def test_stale_arrays_change_binding_and_fail_alignment(self):
        result = auditor.audit(self.root)
        path = self.root / "train/raw_features.npy"
        np.save(path, np.load(path)[::-1].copy())
        self.assertNotEqual(result["binding"]["train/raw_features.npy"], auditor.sha256(path))
        with self.assertRaisesRegex(ValueError, "Stale audit binding"):
            validate_audit(self.root)
        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            auditor.audit(self.root)

    def test_explicit_provenance_mapping_overrides_basename(self):
        mapped = self.root / "provenance/target_config.json"
        mapped.write_text("pinned target config")
        (self.root / "provenance/config.json").write_text("wrong colliding basename")
        original = str(self.root / "removed/model/config.json")
        self.info["collection_config"]["hashes"][original] = auditor.sha256(mapped)
        self.info["collection_config"]["provenance_files"] = {original: "provenance/target_config.json"}
        self.save_manifest()
        auditor.audit(self.root)
        mapped.unlink()
        with self.assertRaisesRegex(ValueError, "Missing mapped provenance"):
            auditor.audit(self.root)

    def test_source_signatures_and_missing_sources_fail_closed(self):
        config = self.info["collection_config"]
        source = self.root / "provenance/collector.py"
        source.write_text("collector fixture")
        config["source_sha256"] = {str(self.root / "removed/collector.py"): auditor.sha256(source)}
        self.save_manifest()
        auditor.audit(self.root)
        source.write_text("changed collector")
        with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
            auditor.audit(self.root)
        source.unlink()
        with self.assertRaisesRegex(ValueError, "missing source"):
            auditor.audit(self.root)

    def test_missing_shard_provenance_is_rejected(self):
        del self.info["shards"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "requires paired collection shards"):
            auditor.audit(self.root)

    def test_repository_relative_source_signature_and_original_fallback(self):
        relative = "scripts/audit_prefusion_cache.py"
        repository = Path(auditor.__file__).resolve().parents[1]
        self.assertEqual(auditor.source_path(self.root, relative, {}, source_root=repository),
                         repository / relative)
        self.info["collection_config"]["source_sha256"] = {
            relative: auditor.sha256(repository / relative)}
        model = self.root / "weights.safetensors"
        model.write_bytes(b"fixture model signature")
        self.info["collection_config"]["model_files_sha256"] = {str(model): auditor.sha256(model)}
        self.save_manifest()
        auditor.audit(self.root)
        model.write_bytes(b"modified model")
        with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
            auditor.audit(self.root)

    def test_mutation_during_audit_does_not_publish_binding(self):
        real_binding = auditor.cache_binding
        calls = 0

        def mutate(cache):
            nonlocal calls
            calls += 1
            if calls == 2:
                np.save(cache / "train/extra.npy", np.array([1]))
            return real_binding(cache)

        with patch.object(auditor, "cache_binding", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during audit"):
                auditor.audit(self.root)
        self.assertFalse((self.root / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
