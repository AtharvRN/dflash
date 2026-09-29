import numpy as np
import pytest

from scripts.matched_policy_oracle import oracles


def test_actual_outcomes_cycle_oracle_dominates_request():
    a = np.tile(np.arange(1, 16), (4, 1))
    a[0, 1:] = 0
    a[2, 1:] = 0
    result = oracles(a, np.array([0, 0, 1, 1]), (.96, 1.0))
    for r in ("0.96", "1.0"):
        cycle, request = result["cycle_hindsight"][r], result["request_hindsight"][r]
        assert cycle["total_budget"] <= request["total_budget"]
        assert cycle["retention"] >= float(r)
        assert request["retention"] >= float(r)


def test_reject_impossible_acceptance():
    a = np.ones((2, 15), dtype=int)
    a[0, 0] = 2
    with pytest.raises(ValueError, match="accepted lengths"):
        oracles(a, np.array([0, 1]), (.96,))
