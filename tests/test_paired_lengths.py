import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_dflash_paired_lengths import prefix_matches, select_prompts, summarize


class PairedLengthsTest(unittest.TestCase):
    def test_first_rejection_ends_acceptance(self):
        self.assertEqual(prefix_matches([1, 9, 3], [1, 2, 3]), 1)
        self.assertEqual(prefix_matches([1, 2], [1, 2, 3]), 2)
        self.assertEqual(prefix_matches([], [1]), 0)

    def test_actual_outcomes_are_not_clipped_labels(self):
        rows = []
        for prompt, actual, base in [("a", 1, 5), ("b", 3, 2)]:
            rows.append({"prompt_id": prompt, "eligible": True,
                         "outcomes": {"4": {"accepted": actual, "draft_ids": [1, 2, 3]},
                                      "16": {"accepted": base, "draft_ids": list(range(15))}},
                         "policies": {"test": 4}, "reverse_order_checked": False,
                         "canonical_checked": 0, "canonical_disagreements": 0,
                         "truncated_verify_checks": 0})
        excluded = dict(rows[0], eligible=False)
        result = summarize(rows + [excluded], bootstrap=100)
        self.assertEqual(result["eligible_states"], 2)
        self.assertEqual(result["blocks"]["4"]["label_mae"], 1.5)
        self.assertEqual(result["blocks"]["4"]["signed_error"], -.5)
        self.assertAlmostEqual(result["policies"]["test"]["actual_retention"], 4/7)
        self.assertAlmostEqual(result["policies"]["test"]["proxy_retention"], 5/7)

    def test_same_width_control_separates_verifier_change(self):
        row = {"prompt_id": "a", "eligible": True,
               "outcomes": {"4": {"accepted": 1, "truncated_accepted": 2,
                                    "draft_ids": [1, 2, 3]},
                            "16": {"accepted": 5, "draft_ids": list(range(15))}},
               "policies": {"test": 4}, "reverse_order_checked": False,
               "canonical_checked": 0, "canonical_disagreements": 0,
               "truncated_verify_checks": 1}
        block = summarize([row], bootstrap=10)["blocks"]["4"]
        self.assertEqual(block["label_mae"], 2)
        self.assertEqual(block["short_vs_same_width_truncated_mae"], 1)
        self.assertEqual(block["target_width_control_mae"], 1)

    def test_validation_ids_filter_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps({"manifest_index": i}) for i in range(10)))
            split = root / "split.json"
            split.write_text(json.dumps({"val_prompt_ids": ["2", "7"]}))
            selected = select_prompts(manifest, split, 10, 1)
            self.assertEqual({r["manifest_index"] for r in selected}, {2, 7})


if __name__ == "__main__":
    unittest.main()
