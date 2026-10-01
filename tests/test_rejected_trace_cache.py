"""Causal alignment and tamper-detection tests for rejected-trace caches."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_rejected_trace_cache import ARRAY_SCHEMA, BLOCKS, load_cache, sha256, token_hash, validate_row


def fixture_row(accepted=2, cycle=4, pid=1, group="train"):
    old_prefix = [10, 11, 12]
    drafted = list(range(20, 35))
    posterior = drafted.copy() + [77]
    if accepted < 15:
        posterior[accepted] = 99
    previous = {"cycle": cycle - 1, "block_size": 16, "prefix_length": len(old_prefix) - 1,
                "prefix_token_ids": old_prefix, "prefix_sha256": token_hash(old_prefix),
                "draft_ids": drafted, "posterior_ids": posterior, "accepted": accepted,
                "correction_token_id": posterior[accepted], "trace_capture": "same_forward", "terminal": False}
    prefix = old_prefix + drafted[:accepted] + [posterior[accepted]]
    row = {"prompt_id": str(pid), "source": "synthetic", "group": group, "cycle": cycle,
           "prefix_length": len(prefix) - 1, "prefix_token_ids": prefix, "prefix_sha256": token_hash(prefix),
           "eligible": True, "input_capture": "before_current_draft", "has_previous": True, "previous_cycle": cycle - 1,
           "previous_B": 16, "previous_A": accepted, "previous": previous,
           "outcomes": {str(b): {"accepted": min(2, b - 1), "draft_ids": list(range(40, 40 + b - 1))}
                        for b in BLOCKS}}
    arrays = {key: np.zeros((1, *shape), dtype=dtype) for key, (shape, dtype) in ARRAY_SCHEMA.items()}
    arrays["trace_token_ids"][:] = -1
    arrays["features"][:] = 1
    arrays["anchor_embedding"][:] = 2
    arrays["previous_meta"][0] = [1, 1, accepted / 15]
    if accepted < 15:
        arrays["rejected_anchor_embedding"][:] = 3
    suffix = drafted[accepted + 1:]
    count = len(suffix)
    arrays["trace_mask"][0, :count] = 1
    arrays["trace_token_ids"][0, :count] = suffix
    arrays["trace_offsets"][0, :count] = np.arange(1, count + 1)
    arrays["trace_draft"][0, :count] = 4
    arrays["trace_target"][0, :count] = 5
    arrays["actual"][0] = [row["outcomes"][str(b)]["accepted"] for b in BLOCKS]
    return row, arrays


def write_fixture(root, *, accepted=2):
    row, arrays = fixture_row(accepted)
    config = {"schema_version": "dflash_rejected_trace_v1", "blocks": BLOCKS,
              "prompt_ids": [1], "prompt_groups": {"1": "train"},
              "canonical_train_prompt_ids": [1, 2], "canonical_val_prompt_ids": [3, 4],
              "frozen_calibration_prompt_ids": [3], "frozen_assessment_prompt_ids": [4],
              "prompt_content_hashes": {"1": "a" * 64}, "training_rows": 1, "smoke": True}
    (root / "config.json").write_text(json.dumps(config))
    (root / "prompt_1.json").write_text(json.dumps({"prompt_id": 1, "group": "train", "states": [row], "prompt_summary": {}}))
    np.savez(root / "prompt_1.npz", **arrays)
    bind_fixture(root)
    return row, arrays


def bind_fixture(root):
    receipts = [{"prompt_id": 1, "group": "train", "source": "synthetic", "states": 1,
                 "eligible_states": 1, "files": {name: sha256(root / name) for name in ("prompt_1.json", "prompt_1.npz")}}]
    (root / "receipt_1.json").write_text(json.dumps(receipts[0]))
    (root / "receipts.json").write_text(json.dumps(receipts))
    (root / "collection_summary.json").write_text(json.dumps({"states": 1, "eligible_states": 1}))
    (root / "COMPLETE.json").write_text(json.dumps({"states": 1, "eligible_states": 1, "sample_complete": True,
        "binding": {name: sha256(root / name) for name in ("config.json", "receipts.json", "collection_summary.json")}}))


def make_cache(root):
    """Write a complete CPU smoke cache: two prompts/two rows per partition."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    groups = {str(pid): group for group, ids in (("train", (1, 2)), ("calibration", (3, 4)), ("assessment", (5, 6)))
              for pid in ids}
    config = {"schema_version": "dflash_rejected_trace_v1", "blocks": BLOCKS,
              "prompt_ids": list(range(1, 7)), "prompt_groups": groups,
              "canonical_train_prompt_ids": [1, 2], "canonical_val_prompt_ids": [3, 4, 5, 6],
              "frozen_calibration_prompt_ids": [3, 4], "frozen_assessment_prompt_ids": [5, 6],
              "prompt_content_hashes": {str(pid): token_hash([pid]) for pid in range(1, 7)},
              "training_rows": 4, "smoke": True}
    (root / "config.json").write_text(json.dumps(config))
    receipts = []
    for pid in config["prompt_ids"]:
        group = groups[str(pid)]
        first, first_arrays = fixture_row(cycle=4, pid=pid, group=group)
        second, second_arrays = fixture_row(cycle=6, pid=pid, group=group)
        # The row for cycle 5 is deliberately not selected. Its actual immediate
        # trace still supplies cycle 6; cycle 4 remains the previous sampled row.
        parent_prefix = first["prefix_token_ids"] + first["outcomes"]["16"]["draft_ids"][:2] + [88]
        second["previous"].update(prefix_token_ids=parent_prefix, prefix_length=len(parent_prefix) - 1,
                                  prefix_sha256=token_hash(parent_prefix))
        prefix = parent_prefix + second["previous"]["draft_ids"][:2] + [99]
        second.update(prefix_token_ids=prefix, prefix_length=len(prefix) - 1, prefix_sha256=token_hash(prefix))
        arrays = {key: np.concatenate([first_arrays[key], second_arrays[key]]) for key in ARRAY_SCHEMA}
        arrays["features"][:, 0] = pid
        arrays["trace_target"][:, :12, 0] = pid
        (root / f"prompt_{pid}.json").write_text(json.dumps({"prompt_id": pid, "group": group,
                                                            "states": [first, second], "prompt_summary": {}}))
        np.savez(root / f"prompt_{pid}.npz", **arrays)
        receipts.append({"prompt_id": pid, "group": group, "source": "synthetic", "states": 2,
                         "eligible_states": 2, "files": {name: sha256(root / name)
                            for name in (f"prompt_{pid}.json", f"prompt_{pid}.npz")}})
        (root / f"receipt_{pid}.json").write_text(json.dumps(receipts[-1]))
    (root / "receipts.json").write_text(json.dumps(receipts))
    (root / "collection_summary.json").write_text(json.dumps({"states": 12, "eligible_states": 12}))
    (root / "COMPLETE.json").write_text(json.dumps({"states": 12, "eligible_states": 12, "sample_complete": True,
        "binding": {name: sha256(root / name) for name in ("config.json", "receipts.json", "collection_summary.json")}}))
    return root


class RejectedTraceCacheTests(unittest.TestCase):
    def test_six_prompt_smoke_cache_keeps_partitions_and_sparse_cycle_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, arrays, audit = load_cache(make_cache(Path(directory)))
            self.assertEqual(audit["eligible_by_group"], {"train": 4, "calibration": 4, "assessment": 4})
            self.assertEqual(len(rows), 12)
            self.assertEqual(arrays["features"].shape, (12, 2560))
            self.assertEqual(rows[1]["previous_cycle"], 5)

    def test_all_previous_acceptance_lengths_align_strict_suffix(self):
        for accepted in range(16):
            with self.subTest(accepted=accepted):
                row, arrays = fixture_row(accepted)
                validate_row(row, arrays, 0)
                self.assertEqual(int(arrays["trace_mask"].sum()), max(14 - accepted, 0))

    def test_first_cycle_has_no_predecessor_or_memory(self):
        row, arrays = fixture_row(15)
        row.update(cycle=0, has_previous=False, previous=None, previous_cycle=-1, previous_B=0, previous_A=-1)
        arrays["previous_meta"][:] = 0
        validate_row(row, arrays, 0)
        arrays["trace_target"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_row(row, arrays, 0)

    def test_immediate_predecessor_not_previous_sample(self):
        row, arrays = fixture_row(cycle=17)
        validate_row(row, arrays, 0)
        row["previous"]["cycle"] = 12
        row["previous_cycle"] = 12
        with self.assertRaisesRegex(ValueError, "immediately preceding"):
            validate_row(row, arrays, 0)

    def test_prefix_must_include_actual_correction(self):
        row, arrays = fixture_row()
        row["prefix_token_ids"][-1] = row["previous"]["draft_ids"][2]
        row["prefix_sha256"] = token_hash(row["prefix_token_ids"])
        with self.assertRaisesRegex(ValueError, "committed tokens and correction"):
            validate_row(row, arrays, 0)

    def test_previous_acceptance_checked_against_all_posterior_ids(self):
        row, arrays = fixture_row()
        row["previous"]["posterior_ids"][0] = 888
        with self.assertRaisesRegex(ValueError, "inconsistent with verifier"):
            validate_row(row, arrays, 0)

    def test_source_prefix_drift_cannot_remain_eligible(self):
        row, arrays = fixture_row()
        row.update(source_prefix_sha256=token_hash([88]), reference_prefix_match=False, source_eligible=True)
        with self.assertRaisesRegex(ValueError, "Ineligible reference replay"):
            validate_row(row, arrays, 0)
        row.update(eligible=False, exclusion="reference_prefix_drift")
        validate_row(row, arrays, 0)

    def test_current_input_capture_must_precede_current_draft(self):
        row, arrays = fixture_row()
        row["input_capture"] = "after_current_draft"
        with self.assertRaisesRegex(ValueError, "before drafting"):
            validate_row(row, arrays, 0)

    def test_trace_cannot_include_first_rejected_token(self):
        row, arrays = fixture_row(accepted=0)
        arrays["trace_token_ids"][0, 0] = row["previous"]["draft_ids"][0]
        with self.assertRaisesRegex(ValueError, "misaligned"):
            validate_row(row, arrays, 0)

    def test_cache_load_preserves_alignment_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            config, rows, arrays, audit = load_cache(root)
            self.assertEqual(rows[0]["previous_cycle"], 3)
            self.assertEqual(arrays["prompt_id"].tolist(), [1])
            self.assertEqual(arrays["group"].tolist(), ["train"])
            self.assertEqual(arrays["eligible"].tolist(), [True])
            self.assertTrue(audit["passed"])
            self.assertIn(str((root / "prompt_1.npz").resolve()), audit["input_file_sha256"])

    def test_hash_tampering_detected_before_semantic_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, arrays = write_fixture(root)
            arrays["features"][0, 0] = 99
            np.savez(root / "prompt_1.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_cache(root)

    def test_actual_labels_cannot_be_joined_from_other_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, arrays = write_fixture(root)
            arrays["actual"][0, 14] = 15
            np.savez(root / "prompt_1.npz", **arrays)
            bind_fixture(root)
            with self.assertRaisesRegex(ValueError, "fresh outcome records"):
                load_cache(root)

    def test_orphan_shard_and_missing_completion_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            (root / "prompt_2.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Orphan"):
                load_cache(root)
            (root / "prompt_2.json").unlink()
            (root / "COMPLETE.json").unlink()
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_cache(root)
            self.assertFalse(load_cache(root, require_complete=False)[3]["complete"])

    def test_wrong_dtype_and_nonfinite_rejected(self):
        for bad in ("dtype", "nan"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, arrays = write_fixture(root)
                if bad == "dtype":
                    arrays["trace_mask"] = arrays["trace_mask"].astype(bool)
                else:
                    arrays["features"][0, 0] = np.nan
                np.savez(root / "prompt_1.npz", **arrays)
                bind_fixture(root)
                with self.assertRaisesRegex(ValueError, "dtype|Nonfinite"):
                    load_cache(root)

    def test_canonical_prompt_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            config = json.loads((root / "config.json").read_text())
            config["canonical_val_prompt_ids"].append(1)
            (root / "config.json").write_text(json.dumps(config))
            bind_fixture(root)
            with self.assertRaisesRegex(ValueError, "split leakage"):
                load_cache(root)

    def test_duplicate_content_cannot_cross_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_cache(Path(directory))
            config = json.loads((root / "config.json").read_text())
            config["prompt_content_hashes"]["3"] = config["prompt_content_hashes"]["1"]
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Exact-content leakage"):
                load_cache(root)

    def test_non_smoke_cannot_drop_frozen_evaluation_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            config = json.loads((root / "config.json").read_text())
            config["smoke"] = False
            (root / "config.json").write_text(json.dumps(config))
            bind_fixture(root)
            with self.assertRaisesRegex(ValueError, "retain all frozen"):
                load_cache(root)

    def test_training_completion_needs_eligible_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            config = json.loads((root / "config.json").read_text())
            config["training_rows"] = 2
            (root / "config.json").write_text(json.dumps(config))
            bind_fixture(root)
            with self.assertRaisesRegex(ValueError, "insufficient eligible training"):
                load_cache(root)


if __name__ == "__main__":
    unittest.main()
