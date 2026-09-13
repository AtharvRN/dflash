import unittest
import json

import numpy as np

from scripts.analyze_context_attention_comparison import paired_intervals


class ContextComparisonTest(unittest.TestCase):
    def test_identical_predictions_have_zero_paired_intervals(self):
        x=np.array([[4,8,10,20,12],[2,2,5,9,6]],dtype=float)
        for result in paired_intervals(x,x,draws=200).values():
            self.assertEqual(result,{"difference":0.0,"ci95":[0.0,0.0],"valid_bootstrap_draws":200})

    def test_zero_acceptance_draws_do_not_break_serialization(self):
        x=np.array([[1,0,0,1,0],[1,1,1,1,1]],dtype=float)
        result=paired_intervals(x,x,draws=200)
        self.assertGreater(result["retention"]["valid_bootstrap_draws"],0)
        self.assertLess(result["retention"]["valid_bootstrap_draws"],200)
        all_zero=paired_intervals(x[:1],x[:1],draws=200)
        self.assertIsNone(all_zero["retention"]["difference"])
        self.assertIsNone(all_zero["retention"]["ci95"])
        json.dumps(all_zero,allow_nan=False)

    def test_alignment_is_required(self):
        x=np.array([[4,8,10,20,12],[2,2,5,9,6]],dtype=float)
        with self.assertRaises(ValueError):
            paired_intervals(x,x[::-1],draws=200)


if __name__=="__main__":
    unittest.main()
