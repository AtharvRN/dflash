"""Information ablation, donor isolation, and complete CPU training smoke."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dflash.rejected_trace import (COMMON_FIELDS, MEMORY_FIELDS, RejectedTraceResponseModel,
    audit_donor_mapping, make_donor_mapping, model_batch)
from scripts.audit_rejected_trace_cache import load_cache
from scripts.train_rejected_trace_predictor import (memory_coverage, paired_prompt_bootstrap,
    predict, run, select_rows)
from tests.test_rejected_trace_cache import make_cache


class RejectedTraceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parameter_budget_and_independent_actual_block_bounds(self):
        model = RejectedTraceResponseModel().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1546767)
        with tempfile.TemporaryDirectory() as directory:
            _, rows, arrays, _ = load_cache(make_cache(Path(directory)))
            donors = make_donor_mapping(rows)
            batch = model_batch(arrays, [0, 1], "aligned", donors)
            with torch.no_grad():
                model.head[-1].weight.zero_()
                model.head[-1].bias.fill_(-10)
                model.head[-1].bias[3] = 10
                y = model(**batch)
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(((y >= 0) & (y <= torch.arange(1, 16))).all())
            self.assertGreater(float(y[0, 3]), float(y[0, 14]))

    def test_control_masks_every_trace_value_and_null_memory_is_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, arrays, _ = load_cache(make_cache(Path(directory)))
            donors = make_donor_mapping(rows)
            model = RejectedTraceResponseModel().eval()
            batch = model_batch(arrays, [0, 1], "control", donors)
            altered = copy.deepcopy(batch)
            altered["trace_draft"].fill_(float("nan"))
            altered["trace_target"].fill_(float("inf"))
            altered["trace_offsets"].fill_(32767)
            with torch.no_grad():
                first, second = model(**batch), model(**altered)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(first).all())

    def test_donors_different_prompt_same_partition_previous_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, _, _ = load_cache(make_cache(Path(directory)))
            donors = make_donor_mapping(rows, 913)
            np.testing.assert_array_equal(donors, make_donor_mapping(rows, 913))
            self.assertTrue((donors >= 0).all())
            audit_donor_mapping(rows, donors)
            bad = donors.copy()
            bad[0] = 1
            with self.assertRaisesRegex(ValueError, "Same-prompt"):
                audit_donor_mapping(rows, bad)
            bad[0] = 4
            with self.assertRaisesRegex(ValueError, "partition"):
                audit_donor_mapping(rows, bad)
            rows[0]["previous_A"] = 5
            isolated = make_donor_mapping(rows)
            self.assertEqual(isolated[0], -1)

    def test_arms_share_common_inputs_and_no_donor_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, arrays, _ = load_cache(make_cache(Path(directory)))
            rows[0]["previous_A"] = 5  # unique stratum, hence no safe donor
            donors = make_donor_mapping(rows)
            batches = {arm: model_batch(arrays, [0, 2], arm, donors)
                       for arm in ("control", "aligned", "shuffled")}
            for field in COMMON_FIELDS:
                for batch in batches.values():
                    torch.testing.assert_close(batch[field], batches["control"][field])
            for batch in batches.values():
                self.assertFalse(batch["trace_mask"][0].any())
            donor = donors[2]
            np.testing.assert_array_equal(batches["shuffled"]["trace_target"][1].numpy(), arrays["trace_target"][donor])
            model = RejectedTraceResponseModel().eval()
            with torch.no_grad():
                outputs = [model(**batch)[0] for batch in batches.values()]
            for output in outputs[1:]:
                torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
            coverage = memory_coverage(rows, arrays, donors)
            self.assertEqual(coverage["train"]["no_different_prompt_donor_rows"], 1)

    def test_first_eligible_training_rows_and_frozen_eval_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows, arrays, _ = load_cache(make_cache(Path(directory)))
            rows[0]["eligible"] = False
            selected, _, masks, index = select_rows(rows, arrays, 2)
            np.testing.assert_array_equal(index[:2], [1, 2])
            self.assertEqual(int(masks["calibration"].sum()), 4)
            self.assertEqual(int(masks["assessment"].sum()), 4)
            self.assertEqual(selected[0]["cycle"], rows[1]["cycle"])
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                select_rows(rows, arrays, 2000)
            rows[4]["prompt_id"] = rows[1]["prompt_id"]
            with self.assertRaisesRegex(ValueError, "crosses"):
                select_rows(rows, arrays, 2)

    def test_paired_bootstrap_identical_decisions_zero_differences(self):
        actual = np.tile(np.minimum(np.arange(1, 16), 5), (4, 1))
        budgets = np.array([3, 5, 10, 15])
        result = paired_prompt_bootstrap(actual, [1, 1, 2, 2],
            {arm+"_seed_913": budgets for arm in ("control", "aligned", "shuffled")}, count=50)
        for comparison in result["paired_arm_differences"].values():
            for interval in comparison.values():
                self.assertEqual(interval, [0., 0.])

    def test_complete_cpu_training_pipeline_all_nine_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root/"cache")
            args = argparse.Namespace(cache=cache, output=root/"trained", gpu=None, training_rows=4,
                seeds=[913, 914, 915], donor_seed=913, epochs=2, batch_size=2, cpu_threads=2,
                bootstrap=50, benchmark_repeats=0, smoke=True)
            summary = run(args)
            self.assertEqual(len(summary["models"]), 9)
            self.assertTrue(summary["reload_and_matching_checks_passed"])
            self.assertEqual(summary["counts"]["train"]["rows"], 4)
            frozen = json.loads((args.output/"selection_frozen.json").read_text())
            for seed in args.seeds:
                values = [frozen["matched_randomness"][f"{arm}_seed_{seed}"]
                          for arm in ("control", "aligned", "shuffled")]
                self.assertEqual(values[0], values[1])
                self.assertEqual(values[0], values[2])
            complete = json.loads((args.output/"COMPLETE.json").read_text())
            self.assertTrue(complete["success"] and complete["smoke"])
            self.assertIn("assessment_predictions.npz", complete["binding"])
            self.assertEqual(len(summary["uncertainty"]["paired_arm_differences"]), 6)


if __name__ == "__main__":
    unittest.main()
