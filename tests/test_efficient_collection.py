import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_block_headroom import sha256
from scripts.efficient_collection import inventory, choose_workers, compare_replay, verified_copy


def fixture(root):
    config = {"collection_kind": "training", "blocks": list(range(2, 17)),
        "prompt_ids": [10, 11, 12], "prompt_groups": {str(p): "train" for p in [10, 11, 12]}}
    (root/"config.json").write_text(json.dumps(config))
    prefix = [1, 2, 3]
    state = {"prompt_id": "10", "group": "train", "source": "nemotron", "cycle": 0,
        "generated_before_anchor": 0, "prefix_token_ids": prefix, "prefix_length": 2,
        "prefix_sha256": hashlib.sha256(np.array(prefix, dtype=np.int64).tobytes()).hexdigest(),
        "eligible": True, "canonical_disagreements": 0,
        "outcomes": {str(b): {"draft_ids": [7]*(b-1), "accepted": 0} for b in range(2, 17)}}
    progress = {"prompt_id": "10", "states": 1}
    (root/"prompt_10.json").write_text(json.dumps({"states": [state], "progress": progress}))
    x = np.ones((1, 2560), dtype=np.float16)
    np.save(root/"prompt_10_fused.npy", x)
    receipt = {"prompt_id": 10, "group": "train", "source": "nemotron", "states": 1,
        "files": {n: sha256(root/n) for n in ("prompt_10.json", "prompt_10_fused.npy")}}
    (root/"receipt_10.json").write_text(json.dumps(receipt))
    return state, progress, x, receipt


class EfficientCollectionTests(unittest.TestCase):
    def test_partial_recovery_needs_no_final_manifest_and_preserves_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            orphan = root/"incomplete.tmp"
            orphan.write_text("preserve me")
            config, receipts, rows, progress, files, audit = inventory(root)
            self.assertEqual([r["prompt_id"] for r in receipts], [10])
            self.assertEqual(audit["eligible_rows"], 1)
            self.assertEqual(audit["ignored_source_files"], ["incomplete.tmp"])
            self.assertEqual(orphan.read_text(), "preserve me")
            self.assertNotIn("incomplete.tmp", files)

    def test_hash_and_noncontiguous_recovery_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            original = (root/"prompt_10.json").read_bytes()
            (root/"prompt_10.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash"):
                inventory(root)
            (root/"prompt_10.json").write_bytes(original)
            config = json.loads((root/"config.json").read_text())
            config["prompt_ids"] = [11, 10, 12]
            (root/"config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                inventory(root)

    def test_replay_checks_full_labels_tokens_features_and_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, progress, x, _ = fixture(root)
            self.assertEqual(compare_replay(root, 10, [(state, x[0])], progress), 1)
            with self.assertRaisesRegex(ValueError, "feature"):
                compare_replay(root, 10, [(state, x[0]*2)], progress)
            state["outcomes"]["4"]["draft_ids"][1] = 8
            with self.assertRaisesRegex(ValueError, "outcome"):
                compare_replay(root, 10, [(state, x[0])], progress)

    def test_worker_choice_requires_speed_correctness_memory_margin(self):
        def result(workers, speed, memory, passed=True):
            return {"workers": workers, "eligible_rows_per_s": speed, "peak_used_mib": memory,
                "gpu_total_mib": 96000, "exact_replay_passed": passed}
        self.assertEqual(choose_workers([result(1, 1, 11000), result(2, 1.05, 23000), result(4, 1.06, 44000)]), 1)
        self.assertEqual(choose_workers([result(1, 1, 11000), result(2, 1.6, 23000), result(4, 1.64, 44000)]), 2)
        self.assertEqual(choose_workers([result(1, 1, 11000), result(2, 1.5, 23000), result(4, 2, 44000)]), 4)
        self.assertEqual(choose_workers([result(1, 1, 11000), result(2, 1.5, 23000), result(4, 2, 90000)]), 2)
        self.assertEqual(choose_workers([result(1, 1, 11000), result(2, 1.5, 23000), result(4, 2, 44000, False)]), 2)

    def test_copy_hash_verification_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, b = root/"source", root/"destination"
            a.write_bytes(b"original immutable evidence")
            verified_copy(a, b, sha256(a))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with self.assertRaisesRegex(ValueError, "overwrite"):
                verified_copy(a, b)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verified_copy(a, root/"bad", "0"*64)
            self.assertFalse((root/"bad").exists())


if __name__ == "__main__":
    unittest.main()
