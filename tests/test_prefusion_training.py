import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.train_prefusion_acceptance import (
    MODEL_NAMES, PrefusionDataset, acceptance_nll, make_loader, make_model,
    main, model_spec, parameter_matched_proj_dim, parse_args, sha256, validate_audit, validate_cache,
)


def write_cache(root):
    info = {"format": "dflash_prefusion_cache_v1", "input_dim": 12, "fused_dim": 4,
            "num_slots": 15, "context_window": 1}
    (root / "manifest.json").write_text(json.dumps(info))
    for split, offset in (("train", 0), ("val", 100)):
        path = root / split
        path.mkdir()
        labels = np.array([0, 1, 2, 3, 4, 5, 14, 15], dtype=np.int64)
        rows = np.column_stack((np.zeros(8, dtype=np.int64), np.arange(8) + offset,
                                np.arange(8) // 2 + offset, np.arange(8) % 2, labels))
        for filename, width in (("raw_features", 12), ("features", 4)):
            features = np.broadcast_to(np.arange(8)[:, None, None], (8, 1, width)).astype(np.float16)
            np.save(path / f"{filename}.npy", features)
        np.save(path / "accepted_len.npy", labels)
        np.save(path / "mask.npy", np.ones((8, 1), dtype=np.uint8))
        np.save(path / "row_index.npy", rows)
    np.save(root / "val/calibration.npy", np.array([True] * 4 + [False] * 4))
    return info


class ParameterMatchingTest(unittest.TestCase):
    def test_width_is_nearest_actual_parameter_count(self):
        for raw_dim, fused_dim in ((48, 8), (16, 16), (8, 24)):
            with self.subTest(raw_dim=raw_dim, fused_dim=fused_dim):
                info = {"input_dim": raw_dim, "fused_dim": fused_dim}
                raw = make_model("raw", info)
                matched = make_model("fused_parameter_matched", info)
                raw_count = sum(p.numel() for p in raw.parameters())
                matched_count = sum(p.numel() for p in matched.parameters())
                width = parameter_matched_proj_dim(raw_dim, fused_dim)
                self.assertEqual(matched.input_proj[0].out_features, width)
                self.assertLessEqual(abs(raw_count - matched_count), (fused_dim + 259) / 2)
                constant = 256 + 256 * 128 + 128 + 128 * 15 + 15
                self.assertEqual(matched_count, width * (fused_dim + 259) + constant)
                for neighbor in (width - 1, width + 1):
                    self.assertLessEqual(abs(raw_count - matched_count),
                                         abs(raw_count - (neighbor * (fused_dim + 259) + constant)))
                if raw_dim > fused_dim:
                    self.assertGreater(width, 512)

    def test_realistic_width_without_allocating_large_models(self):
        width = parameter_matched_proj_dim(12800, 2560)
        self.assertEqual(width, 2372)
        self.assertEqual(512 * (12800 + 259) + 35087, 6721295)
        self.assertEqual(width * (2560 + 259) + 35087, 6721755)
        for dimensions in ((0, 4), (4, -1), (2.5, 4), (True, 4)):
            with self.assertRaises(ValueError):
                parameter_matched_proj_dim(*dimensions)

    def test_fixed_downstream_head_shapes_and_gradients(self):
        info = {"input_dim": 12, "fused_dim": 4}
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = make_model(name, info, dropout=0)
                spec = model_spec(name, info, 0)
                self.assertEqual((model.head[0].out_features, model.head[3].in_features,
                                  model.head[3].out_features, model.head[6].out_features),
                                 (256, 256, 128, 15))
                features = torch.randn(3, 1, spec["input_dim"], requires_grad=True)
                logits = model(features, torch.ones(3, 1))
                self.assertEqual(logits.shape, (3, 15))
                acceptance_nll(logits, torch.tensor([0, 7, 15])).backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(float(features.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            make_model("unknown", info)


class AuditValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        write_cache(self.cache)
        torch.save({"fixture": True}, self.cache / "fusion.pt")
        paths = [self.cache / "manifest.json", self.cache / "fusion.pt"]
        paths += list((self.cache / "train").glob("*.npy"))
        paths += list((self.cache / "val").glob("*.npy"))
        self.report = {
            "rows": 16,
            "binding": {path.relative_to(self.cache).as_posix(): sha256(path) for path in paths},
            "auditor_source_sha256": sha256(Path(__file__).resolve().parents[1] /
                                            "scripts/audit_prefusion_cache.py"),
        }
        self.write_report(self.report)
        self.output = self.root / "run"
        self.argv = ["--cache", str(self.cache), "--output", str(self.output), "--device", "cpu"]

    def write_report(self, report):
        (self.cache / "audit.json").write_text(json.dumps(report))

    def test_valid_audit(self):
        self.assertEqual(validate_audit(self.cache), self.report)

    def test_missing_audit_blocks_main_without_creating_output(self):
        (self.cache / "audit.json").unlink()
        # Structural validation remains available to synthetic/unit-test callers.
        validate_cache(self.cache)
        with self.assertRaisesRegex(ValueError, "Missing audit.json"):
            validate_audit(self.cache)
        with self.assertRaisesRegex(ValueError, "Missing audit.json"):
            main(self.argv)
        self.assertFalse(self.output.exists())

    def test_each_bound_file_change_invalidates_audit(self):
        for relative in self.report["binding"]:
            with self.subTest(relative=relative):
                path = self.cache / relative
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "Stale audit binding SHA-256 mismatch"):
                    validate_audit(self.cache)
                path.write_bytes(original)

    def test_stale_audit_blocks_main_without_creating_output(self):
        self.write_report({**self.report, "auditor_source_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "auditor_source_sha256 mismatch"):
            main(self.argv)
        self.assertFalse(self.output.exists())

    def test_binding_set_is_required_and_exact(self):
        for relative in self.report["binding"]:
            with self.subTest(missing=relative):
                binding = dict(self.report["binding"])
                del binding[relative]
                self.write_report({**self.report, "binding": binding})
                with self.assertRaisesRegex(ValueError, "binding file set mismatch"):
                    validate_audit(self.cache)
        for invalid in ({}, {"binding": []}, []):
            self.write_report(invalid)
            with self.assertRaisesRegex(ValueError, "binding dictionary"):
                validate_audit(self.cache)
        for extra in ("../outside.npy", "/outside.npy", "train/../manifest.json"):
            self.write_report({**self.report, "binding": {**self.report["binding"], extra: "0" * 64}})
            with self.assertRaisesRegex(ValueError, "binding file set mismatch"):
                validate_audit(self.cache)

    def test_new_cache_arrays_must_be_bound(self):
        self.write_report(self.report)
        path = self.cache / "train/extra.npy"
        np.save(path, np.arange(8))
        with self.assertRaisesRegex(ValueError, "binding file set mismatch"):
            validate_audit(self.cache)
        report = {**self.report, "binding": {**self.report["binding"], "train/extra.npy": sha256(path)}}
        self.write_report(report)
        self.assertEqual(validate_audit(self.cache), report)

    def test_missing_bound_file(self):
        (self.cache / "fusion.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Audit-bound file missing: fusion.pt"):
            validate_audit(self.cache)

    def test_report_rows_and_source_hash_required(self):
        for rows in (0, 8, 17, 16.0, "16", True, None):
            self.write_report({**self.report, "rows": rows})
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "row count mismatch"):
                validate_audit(self.cache)
        for key, message in (("rows", "row count mismatch"),
                             ("auditor_source_sha256", "auditor_source_sha256 mismatch")):
            report = dict(self.report)
            del report[key]
            self.write_report(report)
            with self.assertRaisesRegex(ValueError, message):
                validate_audit(self.cache)

    def test_nonempty_or_file_persistent_path_rejected_before_output_creation(self):
        persistent = self.root / "persistent"
        persistent.mkdir()
        previous = persistent / ".prior-run"
        previous.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "Persistent directory must be absent or empty"):
            main(self.argv + ["--persistent", str(persistent)])
        self.assertFalse(self.output.exists())
        self.assertEqual(previous.read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "Persistent directory must be absent or empty"):
            main(self.argv + ["--persistent", str(previous)])
        self.assertFalse(self.output.exists())

    def test_empty_or_absent_persistent_directory_passes_preflight(self):
        persistent = self.root / "persistent"
        for existing in (False, True):
            if existing:
                persistent.mkdir()
            # Stop after real preflight checks, before creating output or training.
            with patch("scripts.train_prefusion_acceptance.torch.set_num_threads",
                       side_effect=RuntimeError("preflight complete")):
                with self.assertRaisesRegex(RuntimeError, "preflight complete"):
                    main(self.argv + ["--persistent", str(persistent)])
            self.assertFalse(self.output.exists())


class PrefusionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.info = write_cache(self.root)

    def replace(self, path, change):
        full = self.root / path
        np.save(full, change(np.load(full)))

    def add_shards(self):
        shards = self.root / "shards"
        shards.mkdir()
        records, selected = [], []
        for split in ("train", "val"):
            rows = np.load(self.root / split / "row_index.npy")
            rows[:, 1] = np.arange(len(rows))
            np.save(self.root / split / "row_index.npy", rows)
            for start in range(0, len(rows), 2):
                pid = int(rows[start, 2])
                group = "train" if split == "train" else "calibration" if start < 4 else "assessment"
                path = shards / f"{pid}.npz"
                data = {name: np.load(self.root / split / f"{name}.npy")[start:start + 2]
                        for name in ("raw_features", "features", "accepted_len")}
                np.savez(path, **data, cycle_id=rows[start:start + 2, 3])
                # Simulate a relocated PVC cache whose original absolute paths are gone.
                record = {"path": f"/missing/collection/shards/{pid}.npz", "rows": 2,
                          "prompt_id": pid, "group": group, "sha256": sha256(path)}
                path.with_suffix(".json").write_text(json.dumps(record))
                records.append(record)
                selected.append({"prompt_id": pid, "group": group, "content_sha256": f"content-{pid}"})
        info = {**self.info, "shards": records, "collection_config": {"selected_prompts": selected}}
        (self.root / "manifest.json").write_text(json.dumps(info))
        return info

    def test_valid_cache_and_paired_dataset_alignment(self):
        info, calibration = validate_cache(self.root)
        self.assertEqual(info, self.info)
        np.testing.assert_array_equal(np.flatnonzero(calibration), [0, 1, 2, 3])
        for feature_file in ("raw_features.npy", "features.npy"):
            ds = PrefusionDataset(self.root / "val", feature_file, [6, 2])
            self.assertEqual(len(ds), 2)
            for index, row, label in ((0, 6, 14), (1, 2, 2)):
                features, mask, accepted = ds[index]
                self.assertTrue((features == row).all())
                self.assertTrue((mask == 1).all())
                self.assertEqual(int(accepted), label)

    def test_shard_hashes_pair_alignment_and_relocation(self):
        self.add_shards()
        validate_cache(self.root)
        for filename in ("raw_features.npy", "features.npy"):
            path = self.root / "train" / filename
            original = np.load(path)
            np.save(path, original[::-1])
            with self.assertRaisesRegex(ValueError, "Shard/cache .* alignment mismatch"):
                validate_cache(self.root)
            np.save(path, original)
        path = self.root / "train/accepted_len.npy"
        labels = np.load(path)
        labels[:2] = labels[:2][::-1]
        np.save(path, labels)
        rows = np.load(self.root / "train/row_index.npy")
        rows[:, 4] = labels
        np.save(self.root / "train/row_index.npy", rows)
        with self.assertRaisesRegex(ValueError, "accepted_len alignment mismatch"):
            validate_cache(self.root)

    def test_shard_checksum_and_receipt_tampering(self):
        self.add_shards()
        path = self.root / "shards/0.npz"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_cache(self.root)
        path.write_bytes(original)
        receipt = path.with_suffix(".json")
        record = json.loads(receipt.read_text())
        receipt.write_text(json.dumps({**record, "rows": 3}))
        with self.assertRaisesRegex(ValueError, "receipt metadata mismatch"):
            validate_cache(self.root)

    def test_shard_cycles_and_calibration_groups(self):
        self.add_shards()
        path = self.root / "val/row_index.npy"
        rows = np.load(path)
        rows[:, 3] += 10
        np.save(path, rows)
        with self.assertRaisesRegex(ValueError, "cycle alignment mismatch"):
            validate_cache(self.root)
        rows[:, 3] -= 10
        np.save(path, rows)
        self.replace("val/calibration.npy", lambda a: ~a)
        with self.assertRaisesRegex(ValueError, "calibration group alignment mismatch"):
            validate_cache(self.root)

    def test_selected_prompt_content_leakage_rejected(self):
        info = self.add_shards()
        selected = info["collection_config"]["selected_prompts"]
        selected[-1]["content_sha256"] = selected[0]["content_sha256"]
        (self.root / "manifest.json").write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "content hash crosses"):
            validate_cache(self.root)

    def test_shapes_dtypes_and_values_rejected(self):
        cases = (
            ("train/raw_features.npy", lambda a: a[:-1], "raw_features"),
            ("val/features.npy", lambda a: a.astype(np.float32), "features"),
            ("val/raw_features.npy", lambda a: np.full_like(a, np.nan), "nonfinite"),
            ("train/mask.npy", lambda a: a[:, 0], "mask"),
            ("val/mask.npy", lambda a: np.zeros_like(a), "mask"),
            ("train/accepted_len.npy", lambda a: a.astype(np.int32), "accepted_len"),
            ("train/accepted_len.npy", lambda a: a + 1, "outside"),
            ("val/calibration.npy", lambda a: a.astype(np.uint8), "bool"),
            ("val/calibration.npy", lambda a: a[:-1], "bool"),
            ("val/calibration.npy", lambda a: np.ones_like(a), "nonempty"),
            ("val/calibration.npy", lambda a: np.zeros_like(a), "nonempty"),
            ("train/row_index.npy", lambda a: a[:, :4], "row_index"),
            ("train/row_index.npy", lambda a: a.astype(np.float64), "integer"),
        )
        for path, change, message in cases:
            with self.subTest(path=path, message=message):
                original = np.load(self.root / path)
                self.replace(path, change)
                with self.assertRaisesRegex(ValueError, message):
                    validate_cache(self.root)
                np.save(self.root / path, original)

    def test_label_alignment_and_duplicate_identities_rejected(self):
        path = self.root / "train/row_index.npy"
        original = np.load(path)
        for mutation, message in (("label", "alignment"), ("row", "duplicate"),
                                  ("cycle", "duplicate"), ("source", "identity")):
            rows = original.copy()
            if mutation == "label":
                rows[:, 4] = rows[::-1, 4]
            elif mutation == "row":
                rows[1, 1] = rows[0, 1]
            elif mutation == "cycle":
                rows[1, 2:4] = rows[0, 2:4]
            else:
                rows[0, 0] = 1
            np.save(path, rows)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                validate_cache(self.root)
        np.save(path, original)

    def test_prompt_leakage_rejected(self):
        path = self.root / "val/row_index.npy"
        original = np.load(path)
        rows = original.copy()
        rows[:2, 2] = 0
        np.save(path, rows)
        with self.assertRaisesRegex(ValueError, "Training/validation prompt leakage"):
            validate_cache(self.root)
        np.save(path, original)
        # Split rows from the same prompt across calibration and assessment.
        np.save(self.root / "val/calibration.npy", np.arange(8) % 2 == 0)
        with self.assertRaisesRegex(ValueError, "Calibration/assessment prompt leakage"):
            validate_cache(self.root)

    def test_invalid_manifest_rejected(self):
        for key, value in (("format", "other"), ("input_dim", 0), ("fused_dim", True),
                           ("num_slots", 16), ("context_window", 2), ("train_rows", 9)):
            with self.subTest(key=key):
                (self.root / "manifest.json").write_text(json.dumps({**self.info, key: value}))
                with self.assertRaises(ValueError):
                    validate_cache(self.root)

    def test_identical_epoch_shuffle_and_evaluation_order(self):
        args = SimpleNamespace(batch_size=3, eval_batch_size=3, workers=0, seed=913)
        orders = []
        for name in MODEL_NAMES:
            # Deliberately consume different amounts of model/global RNG state.
            make_model(name, self.info)
            feature_file = "raw_features.npy" if name == "raw" else "features.npy"
            dataset = PrefusionDataset(self.root / "train", feature_file)
            loader = make_loader(dataset, args, torch.device("cpu"), training=True)
            orders.append([torch.cat([x[:, 0, 0] for x, _, _ in loader]).tolist() for _ in range(2)])
            evaluation = make_loader(dataset, args, torch.device("cpu"))
            self.assertEqual(torch.cat([x[:, 0, 0] for x, _, _ in evaluation]).tolist(), list(range(8)))
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[1], orders[2])
        self.assertNotEqual(orders[0][0], orders[0][1])

    def test_cli_defaults(self):
        args = parse_args(["--cache", str(self.root), "--output", str(self.root / "run")])
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.dropout, args.seed,
                          args.retention, args.workers, args.cpu_threads),
                         (6, 128, 3e-4, .05, 913, .96, 2, 4))


if __name__ == "__main__":
    unittest.main()
