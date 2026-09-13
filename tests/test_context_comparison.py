import unittest

import numpy as np

from scripts.analyze_context_attention_comparison import paired_intervals


class ContextComparisonTest(unittest.TestCase):
    def test_identical_predictions_have_zero_paired_intervals(self):
        x=np.array([[4,8,10,20,12],[2,2,5,9,6]],dtype=float)
        for result in paired_intervals(x,x,draws=200).values():
            self.assertEqual(result,{"difference":0.0,"ci95":[0.0,0.0]})

    def test_alignment_is_required(self):
        x=np.array([[4,8,10,20,12],[2,2,5,9,6]],dtype=float)
        with self.assertRaises(ValueError):
            paired_intervals(x,x[::-1],draws=200)


if __name__=="__main__":
    unittest.main()
