import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_policy_granularity import budgets_from_survival, decisions, metrics, paired_bootstrap, select_operating_points
from scripts.collect_policy_granularity import select_groups


class GranularityTests(unittest.TestCase):
    def test_integer_policy_and_full_budget_boundary(self):
        s = np.array([[.9**k for k in range(1, 16)], [.2**k for k in range(1, 16)]])
        d = budgets_from_survival(s, .96)
        self.assertTrue(np.all((d >= 1) & (d <= 15)))
        for row, budget in zip(s, d):
            self.assertGreaterEqual(row[:budget].sum(), .96*row.sum())
            if budget > 1:
                self.assertLess(row[:budget-1].sum(), .96*row.sum())
        np.testing.assert_array_equal(budgets_from_survival(s, 1), [15, 15])

    def test_actual_outcomes_not_clipped_and_ratio_is_aggregate(self):
        a = np.zeros((2, 15), dtype=int)
        a[:, -1] = [10, 2]
        a[0, 1], a[1, 3] = 0, 4  # Actual B5 can beat B16; do not monotonicize.
        m = metrics(a, np.array([2, 4]))
        self.assertEqual(m["total_accepted"], 4)
        self.assertAlmostEqual(m["aggregate_accept_ratio"], 4/6)
        self.assertAlmostEqual(m["retention"], 4/12)

    def test_request_policy_repeats_initial_prediction(self):
        cycle = np.array([[.95**k for k in range(1, 16)], [.2**k for k in range(1, 16)]])
        request = cycle[[0, 0]]
        d = decisions("request", .96, cycle, request)
        self.assertEqual(d[0], d[1])
        self.assertNotEqual(*decisions("cycle", .96, cycle, request))

    def test_calibration_selection_satisfies_constraint(self):
        a = np.minimum(np.arange(1, 16)[None], np.array([3, 7, 15])[:, None])
        s = np.array([[.6**k for k in range(1, 16)], [.85**k for k in range(1, 16)], [.99**k for k in range(1, 16)]])
        selected, curves = select_operating_points(a, s, s)
        for target, choices in selected.items():
            for name, point in choices.items():
                self.assertGreaterEqual(point["retention"]+1e-12, float(target))
                feasible = [p["mean_budget"] for p in curves[name] if p["retention"] >= float(target)-1e-12]
                self.assertEqual(point["mean_budget"], min(feasible))

    def test_paired_identical_policies_have_zero_difference(self):
        a = np.minimum(np.arange(1, 16)[None], np.array([3, 5, 7, 15])[:, None])
        d = np.full(4, 14)
        result = paired_bootstrap(a, np.array([1, 1, 2, 2]), {n: d for n in ("fixed", "request", "cycle")}, count=100)
        self.assertEqual(result["differences"]["cycle_minus_fixed"]["relative_budget_saving_ci95"], [0, 0])

    def test_preserve_groups_and_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"manifest_index": i, "messages": [{"role": "user", "content": str(i)}]} for i in range(3)]
            (root/"messages.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (root/"train_prompt_ids.json").write_text(json.dumps({"train_prompt_ids": [0]}))
            (root/"val_prompt_ids.json").write_text(json.dumps({"val_prompt_ids": [1, 2]}))
            selected = [{"prompt_id": i, "group": g, "content_sha256": hashlib.sha256(json.dumps(rows[i]["messages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
                        for i, g in ((1, "calibration"), (2, "assessment"))]
            pilot = {"shards": [{**r, "rows": 2} for r in selected], "collection_config": {"selected_prompts": selected}}
            (root/"pilot.json").write_text(json.dumps(pilot))
            out = select_groups(root/"messages.jsonl", root/"pilot.json", root, 928)
            self.assertEqual([(r["manifest_index"], r["group"]) for r in out], [(1, "calibration"), (2, "assessment")])
            rows[1]["messages"][0]["content"] = "tampered"
            (root/"messages.jsonl").write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "content"):
                select_groups(root/"messages.jsonl", root/"pilot.json", root, 928)


if __name__ == "__main__":
    unittest.main()
