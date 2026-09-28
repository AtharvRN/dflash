import pytest
from scripts.analyze_latency_headroom import interpolate_cost


def test_uniform_cost_interpolation_rejects_extrapolation():
    costs = {8: 52., 12: 59., 16: 68.}
    assert interpolate_cost(14, costs) == 63.5
    assert interpolate_cost(8, costs) == 52.
    assert interpolate_cost(16, costs) == 68.
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_cost(24, costs)
