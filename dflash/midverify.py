"""Candidate-preserving verification trimming (greedy, offline diagnostics).

K counts target query rows INCLUDING the known anchor. Row j-1 predicts
candidate j. All retained rows must still execute every target layer.
"""
from __future__ import annotations

import numpy as np


def accepted_lengths(matches):
    matches = np.asarray(matches)
    if matches.ndim != 2 or matches.shape[1] != 15:
        raise ValueError("Expected fifteen candidate matches per B16 state")
    if not np.isin(matches, [0, 1]).all():
        raise ValueError("Nonbinary matches")
    return np.cumprod(matches.astype(np.int64), axis=1).sum(1)


def risk_mask(accepted):
    a = np.asarray(accepted)
    if a.ndim != 1 or np.any((a < 0) | (a > 15) | (a != a.astype(int))):
        raise ValueError("Invalid accepted lengths")
    # Include the first observed rejection, but never invent one beyond B16.
    return np.arange(15)[None] <= a[:, None]


def kept_rows(scores, threshold):
    # Match calibration's FP64 comparisons. A Python-float nextafter threshold
    # otherwise rounds back to float32 under NumPy weak-scalar promotion.
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 15 or not np.isfinite(values).all():
        raise ValueError("Invalid per-candidate scores")
    # Even a rejection of y1 requires the anchor's final logit for the bonus.
    return 1 + np.cumprod(values >= threshold, axis=1).sum(1)


def metrics(accepted, kept, layer=0, depth=36):
    a, k = np.asarray(accepted), np.asarray(kept)
    risk_mask(a)
    if k.shape != a.shape or np.any((k < 1) | (k > 16) | (k != k.astype(int))):
        raise ValueError("K must be an integer in [1,16], including the anchor")
    if not 0 <= layer <= depth or not len(a) or a.sum() <= 0:
        raise ValueError("Invalid layer or empty acceptance reference")
    got = np.minimum(a, k - 1)
    work = (layer * 16 + (depth - layer) * k) / depth
    return {"rows": len(a), "mean_accepted": float(got.mean()),
            "retention": float(got.sum() / a.sum()), "mean_kept_rows": float(k.mean()),
            "mean_committed_tokens": float((got + 1).mean()),
            "aggregate_accept_ratio": float(got.sum() / np.maximum((k - 1).sum(), 1)),
            "equivalent_full_depth_rows": float(work.mean()),
            "row_layer_work_per_committed_token": float(work.sum() / (got + 1).sum()),
            "row_layer_work_saving_vs_b16": float(1 - work.mean() / 16),
            "scope": "same B16 candidates; row-layer work proxy, NOT latency or throughput"}


def calibrate(scores, accepted, targets=(.90, .95, .96, .98, .99)):
    """Exact threshold transitions, using calibration rows only.

    Thresholded prefix minima yield monotone retained work and acceptance.
    Select the MOST aggressive feasible threshold for each target retention.
    """
    values = np.asarray(scores, dtype=np.float64)
    kept_rows(values, -np.inf)
    prefix_min = np.minimum.accumulate(values, axis=1)
    thresholds = np.r_[-np.inf, np.nextafter(np.unique(prefix_min), np.inf)]
    a = np.asarray(accepted)
    if a.shape != (len(values),) or a.sum() <= 0:
        raise ValueError("Invalid calibration acceptance")
    selected = {}
    for target in targets:
        if not 0 < target <= 1:
            raise ValueError("Invalid target retention")
        low, high = 0, len(thresholds)
        while low + 1 < high:
            mid = (low + high) // 2
            k = 1 + (prefix_min >= thresholds[mid]).sum(1)
            if np.minimum(a, k - 1).sum() / a.sum() >= target - 1e-12:
                low = mid
            else:
                high = mid
        setting = float(thresholds[low])
        # JSON has no Infinity. None means an explicit keep-all fallback.
        selected[str(target)] = {"threshold": setting if np.isfinite(setting) else None,
                                **metrics(a, kept_rows(values, setting))}
    return selected


def apply_setting(scores, setting):
    threshold = setting["threshold"]
    return kept_rows(scores, -np.inf if threshold is None else threshold)


def prompt_bootstrap(accepted, kept, prompt_ids, layer=0, draws=1000, seed=929):
    a, k = np.asarray(accepted), np.asarray(kept)
    _, inv = np.unique(prompt_ids, return_inverse=True)
    got = np.minimum(a, k - 1)
    work = (layer * 16 + (36 - layer) * k) / 36
    totals = np.column_stack([np.bincount(inv, weights=x) for x in
                             (got, a, k, np.ones(len(a)), work, got + 1)])
    rng = np.random.default_rng(seed)
    sample = totals[rng.integers(len(totals), size=(draws, len(totals)))].sum(1)
    ci = lambda x: np.quantile(x, [.025, .975]).tolist()
    return {"retention_ci95": ci(sample[:, 0] / sample[:, 1]),
            "mean_kept_rows_ci95": ci(sample[:, 2] / sample[:, 3]),
            "row_layer_work_per_committed_token_ci95": ci(sample[:, 4] / sample[:, 5]),
            "resamples": draws, "unit": "whole prompts; calibration decisions held fixed"}


def ideal_work_at_progress(mean_accepted, layer, depth=36):
    """Lower bound at a specified accepted-token count on unchanged candidates.

    Each accepted token needs one retained query row, plus one anchor row per
    state. This allows clairvoyance and arbitrary integer K in [1,16]. It is
    a row-layer-work bound only: it is NOT a lower bound on measured latency.
    """
    if not 0 <= mean_accepted <= 15 or not 0 <= layer <= depth or depth <= 0:
        raise ValueError("Invalid progress/depth")
    return (layer * 16 + (depth - layer) * (mean_accepted + 1)) / depth


def break_even_oracle_depth(mean_accepted, mean_kept_rows, depth=36):
    if not 0 <= mean_accepted <= 15 or not mean_accepted + 1 <= mean_kept_rows <= 16:
        raise ValueError("Control violates retained-prefix work bound")
    if mean_accepted == 15:
        return float(depth)
    return depth * (mean_kept_rows - mean_accepted - 1) / (15 - mean_accepted)
