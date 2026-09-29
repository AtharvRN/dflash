import itertools

import numpy as np
import pytest

from dflash.midverify import calibrate, apply_setting, kept_rows
from dflash.midverify_cascade import apply_cascade, cascade_metrics, calibrate_cascade, paired_bootstrap


@pytest.mark.parametrize("layer", [0, 6, 35])
def test_exact_joint_search_matches_brute_force(layer):
    rng = np.random.default_rng(929)
    c = rng.integers(-3, 4, size=(5, 15)).astype(np.float32)
    q = rng.integers(-3, 4, size=(5, 15)).astype(np.float32)
    a = np.array([0, 1, 3, 7, 15])
    settings = calibrate_cascade(c, q, a, (.5, .96, 1.), layer)
    ts = lambda x: np.r_[-np.inf, np.nextafter(np.unique(np.minimum.accumulate(x.astype(float), axis=1)), np.inf)]
    for target, setting in settings.items():
        f, k = apply_cascade(c, q, setting)
        feasible = []
        for t0, t1 in itertools.product(ts(c), ts(q)):
            front, end = kept_rows(c, t0), np.minimum(kept_rows(c, t0), kept_rows(q, t1))
            accepted = int(np.minimum(a, end - 1).sum())
            if accepted / a.sum() >= float(target) - 1e-12:
                feasible.append((int((layer * front + (36 - layer) * end).sum()), -accepted))
        assert (int((layer * f + (36 - layer) * k).sum()), -int(np.minimum(a, k - 1).sum())) == min(feasible)


def test_suffix_cannot_be_resurrected_or_influence_kept_prefix():
    c = np.ones((3, 15), dtype=np.float32)
    c[0, 0] = 0
    c[1, 4] = 0
    q = np.ones_like(c)
    setting = {"stage0_threshold": .5, "stage1_threshold": .5}
    front, end = apply_cascade(c, q, setting)
    np.testing.assert_array_equal(front, [1, 5, 16])
    changed = q.copy()
    changed[0, :] = 0
    changed[1, 4:] = 0
    np.testing.assert_array_equal(apply_cascade(c, changed, setting)[1], end)
    q[1, 2] = 0
    assert apply_cascade(c, q, setting)[1][1] == 3
    with pytest.raises(ValueError, match="resurrected"):
        cascade_metrics(np.array([1, 5, 15]), front, [2, 5, 16])


def test_exact_float32_thresholds_and_bypass():
    c = np.full((2, 15), .5, dtype=np.float32)
    q = c.copy()
    just_above = float(np.nextafter(np.float64(.5), np.inf))
    f, k = apply_cascade(c, q, {"stage0_threshold": None, "stage1_threshold": just_above})
    np.testing.assert_array_equal(f, [16, 16])
    np.testing.assert_array_equal(k, [1, 1])
    setting = calibrate_cascade(c, q, np.array([3, 15]), (1.,))["1.0"]
    assert setting["stage1_threshold"] is None
    assert setting["calibration"]["retention"] == 1


def test_joint_calibration_includes_raw_confidence_policy():
    rng = np.random.default_rng(31)
    c, q = rng.normal(size=(2, 24, 15))
    a = rng.integers(0, 16, size=24)
    direct = calibrate(c, a, (.96,))["0.96"]
    joint = calibrate_cascade(c, q, a, (.96,))["0.96"]
    assert joint["calibration"]["equivalent_full_depth_rows"] <= apply_setting(c, direct).mean()


def test_stage_accounting_and_identical_bootstrap():
    a = np.array([0, 6, 15])
    f, k = np.array([4, 10, 16]), np.array([1, 7, 16])
    m = cascade_metrics(a, f, k)
    assert m["retention"] == 1
    assert m["equivalent_full_depth_rows"] == pytest.approx(((6*f + 30*k)/36).mean())
    b = paired_bootstrap(a, f, k, f, k, np.array(["a", "a", "b"]), reference_layer=6, draws=100)
    for key, value in b.items():
        if key.endswith("ci95"):
            np.testing.assert_array_equal(value, [0., 0.])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        calibrate_cascade(np.ones((2, 15)), np.full((2, 15), np.nan), np.array([3, 4]))
    with pytest.raises(ValueError):
        calibrate_cascade(np.ones((2, 15)), np.ones((2, 15)), np.array([3, 4]), targets=(1.1,))


def test_feature_ablations_do_not_leak_hidden_states_into_free_controls():
    torch = pytest.importorskip("torch")
    from scripts.train_midverify_cascade import case_inputs, build_probe, fit_confidence_normalizer
    from scripts.train_midverify_probe import make_features
    h, e = torch.randn(4, 15, 8), torch.randn(4, 15, 8)
    confidence = torch.randn(4, 15, 3)
    first, changed = make_features(h, e), make_features(h + 10, e)
    for case, width in (("confidence_only", 3), ("candidate_confidence", 11), ("target_candidate_confidence", 27)):
        x = case_inputs(first, confidence, case)
        assert x.shape == (4, 15, width)
        assert build_probe(width)(x).shape == (4, 15, 1)
        if case != "target_candidate_confidence":
            assert torch.equal(x, case_inputs(changed, confidence, case))
    stats = np.ones((4, 15, 3)); mask = np.zeros((4, 15), dtype=bool); mask[:, :3] = True
    mean, std = fit_confidence_normalizer(stats, mask)
    stats[:, 3:] = 999
    mean2, std2 = fit_confidence_normalizer(stats, mask)
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(std, std2)
