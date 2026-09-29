import numpy as np
import pytest

from dflash.midverify import (accepted_lengths, risk_mask, kept_rows, metrics,
                             calibrate, apply_setting)
from scripts.collect_midverify_probe import pack_rows


def test_predecessor_alignment_and_left_padding():
    rows = [{"prefix_length": 2, "prefix_token_ids": [3, 4, 5], "draft_ids": list(range(20, 35))},
            {"prefix_length": 4, "prefix_token_ids": [6, 7, 8, 9, 10], "draft_ids": list(range(40, 55))}]
    ids, mask, pos, block, lengths = pack_rows(rows)
    assert ids.tolist() == [[0, 0, 3, 4], [6, 7, 8, 9]]
    assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert pos.tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]
    assert block[:, 0].tolist() == [5, 10]
    assert block[:, 1].tolist() == [20, 40]
    assert lengths.tolist() == [2, 4]


def test_rejection_does_not_count_later_matches():
    x = np.ones((3, 15), dtype=int)
    x[0, 0] = 0
    x[1, 3] = 0
    np.testing.assert_array_equal(accepted_lengths(x), [0, 3, 15])
    assert risk_mask(np.array([0, 3, 15])).sum(1).tolist() == [1, 4, 15]


def test_anchor_and_bonus_convention():
    a = np.array([0, 3, 15])
    result = metrics(a, a + 1, layer=12)
    assert result["retention"] == 1
    assert result["mean_committed_tokens"] == 7
    scores = np.ones((3, 15))
    scores[0, 0] = -1
    scores[1, 3] = -1
    np.testing.assert_array_equal(kept_rows(scores, 0), a + 1)
    assert metrics(a, np.ones(3, dtype=int))["mean_accepted"] == 0


def test_clipping_applies_to_fixed_candidates_only():
    a = np.array([0, 6, 15])
    r = metrics(a, np.array([8, 8, 8]))
    assert r["mean_accepted"] == 13 / 3
    assert r["retention"] == 13 / 21
    assert r["mean_kept_rows"] == 8


def test_exact_calibration_matches_brute_force():
    rng = np.random.default_rng(9)
    values = rng.normal(size=(24, 15))
    a = rng.integers(0, 16, size=24)
    points = calibrate(values, a)
    for target, setting in points.items():
        got = apply_setting(values, setting)
        assert metrics(a, got)["retention"] >= float(target) - 1e-12
        thresholds = np.r_[-np.inf, np.nextafter(np.unique(np.minimum.accumulate(values, axis=1)), np.inf)]
        feasible = [kept_rows(values, t).mean() for t in thresholds
                    if metrics(a, kept_rows(values, t))["retention"] >= float(target) - 1e-12]
        assert got.mean() == min(feasible)


def test_nonfinite_scores_and_invalid_kept_rejected():
    with pytest.raises(ValueError):
        kept_rows(np.full((2, 15), np.nan), .5)
    with pytest.raises(ValueError):
        metrics(np.array([3]), np.array([0]))
    with pytest.raises(ValueError):
        metrics(np.array([3]), np.array([17]))


def test_candidate_conditioned_head_and_features():
    torch = pytest.importorskip("torch")
    from scripts.train_midverify_probe import make_features, build_probe
    h = torch.randn(2, 15, 8)
    e = torch.randn(2, 15, 8)
    x = make_features(h, e)
    assert x.shape == (2, 15, 24)
    assert torch.isfinite(x).all()
    for kind in ("linear", "mlp128"):
        assert build_probe(kind, width=8)(x).shape == (2, 15, 1)
