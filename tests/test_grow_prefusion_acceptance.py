import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.grow_prefusion_acceptance import assert_preserved, cycle_limit, new_training_prompts


class GrowPrefusionTests(unittest.TestCase):
    def test_only_unseen_training_prompts(self):
        selected = [(g, {"manifest_index": i}) for g, i in
                    [("train", 1), ("train", 2), ("calibration", 3), ("assessment", 4), ("train", 2)]]
        self.assertEqual(new_training_prompts(selected, [{"prompt_id": 1}]), [{"manifest_index": 2}])

    def test_exact_remaining_budget(self):
        self.assertEqual(cycle_limit(13935, 100000, 32), 32)
        self.assertEqual(cycle_limit(99997, 100000, 32), 3)
        with self.assertRaises(ValueError):
            cycle_limit(100000, 100000, 32)

    def test_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, grown = Path(tmp) / "base", Path(tmp) / "grown"
            for root in (base, grown):
                for split in ("train", "val"):
                    (root / split).mkdir(parents=True)
                    np.save(root / split / "accepted_len.npy", np.array([0, 3, 15]))
            np.save(grown / "train/accepted_len.npy", np.array([0, 3, 15, 7]))
            assert_preserved(base, grown)
            np.save(grown / "val/accepted_len.npy", np.array([0, 3, 14]))
            with self.assertRaisesRegex(ValueError, "Validation changed"):
                assert_preserved(base, grown)

    def test_training_prefix_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, grown = Path(tmp) / "base", Path(tmp) / "grown"
            for root in (base, grown):
                (root / "train").mkdir(parents=True)
            np.save(base / "train/features.npy", np.array([1, 2]))
            np.save(grown / "train/features.npy", np.array([2, 1, 3]))
            with self.assertRaisesRegex(ValueError, "training prefix changed"):
                assert_preserved(base, grown)
