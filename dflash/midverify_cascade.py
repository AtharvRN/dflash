"""Offline two-stage, candidate-preserving verification prefixes.

Both thresholds are global calibration choices. Stage two can only shorten
stage one's prefix. Query rows include the anchor; greedy, nonterminal states.
"""
from __future__ import annotations

import math
import numpy as np

from dflash.midverify import kept_rows, metrics, risk_mask


def apply_cascade(confidence, scores, setting):
    t0, t1 = setting["stage0_threshold"], setting["stage1_threshold"]
    front = kept_rows(confidence, -np.inf if t0 is None else t0)
    end = np.minimum(front, kept_rows(scores, -np.inf if t1 is None else t1))
    return front, end


def cascade_metrics(accepted, front, end, layer=6):
    a, f, k = map(np.asarray, (accepted, front, end))
    out = metrics(a, k)
    metrics(a, f)
    if not isinstance(layer, int) or not 0 <= layer < 36 or np.any(k > f):
        raise ValueError("Invalid cascade depth or resurrected suffix")
    work = (layer * f + (36 - layer) * k) / 36
    got, front_got = np.minimum(a, k - 1), np.minimum(a, f - 1)
    out.update({"layer": layer, "mean_front_rows": float(f.mean()),
        "stage0_retention": float(front_got.sum() / a.sum()),
        "stage1_incremental_retention_loss": float((front_got - got).sum() / a.sum()),
        "stage1_mean_rows_removed": float((f - k).mean()),
        "stage1_trimmed_state_fraction": float(np.mean(k < f)),
        "equivalent_full_depth_rows": float(work.mean()),
        "row_layer_work_per_committed_token": float(work.sum() / (got + 1).sum()),
        "row_layer_work_saving_vs_b16": float(1 - work.mean() / 16),
        "scope": "same B16 candidates, [L*K0+(36-L)*K1]/36; offline work only, not latency; full B16 draft cost unchanged"})
    return out


def calibrate_cascade(confidence, scores, accepted, targets=(.96, .99), layer=6):
    """Exact joint search over both prefix-score threshold transitions.

Sweep stage-zero transitions while maintaining histograms indexed by stage-one
prefix scores. A survival-count quantile gives the most aggressive feasible
stage-one threshold for each front prefix. No coarse four-arm/grid restriction.
"""
    confidence, scores = np.asarray(confidence, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    kept_rows(confidence, -np.inf)
    kept_rows(scores, -np.inf)
    a = np.asarray(accepted)
    risk_mask(a)
    if confidence.shape != scores.shape or a.shape != (len(scores),) or a.sum() <= 0:
        raise ValueError("Invalid cascade calibration data")
    if not isinstance(layer, int) or not 0 <= layer < 36 or any(not 0 < t <= 1 for t in targets):
        raise ValueError("Invalid layer or retention targets")
    p0 = np.minimum.accumulate(confidence, axis=1).ravel()
    p1 = np.minimum.accumulate(scores, axis=1).ravel()
    truth = (np.arange(15)[None, :] < a[:, None]).ravel()
    levels0, ranks0 = np.unique(p0, return_inverse=True)
    levels1, ranks1 = np.unique(p1, return_inverse=True)
    order = np.argsort(ranks0, kind="stable")
    boundaries = np.r_[0, np.cumsum(np.bincount(ranks0))]
    all_hist = np.bincount(ranks1, minlength=len(levels1))
    good_hist = np.bincount(ranks1[truth], minlength=len(levels1))
    active, good = len(p0), int(a.sum())
    required = {str(t): math.ceil(float(t) * int(a.sum()) - 1e-10) for t in targets}
    best, best_key = {}, {}
    n, evaluated = len(a), 0
    for gate in range(len(levels0) + 1):
        if gate:
            removed = order[boundaries[gate - 1]:boundaries[gate]]
            all_hist -= np.bincount(ranks1[removed], minlength=len(levels1))
            good_hist -= np.bincount(ranks1[removed[truth[removed]]], minlength=len(levels1))
            active -= len(removed)
            good -= int(truth[removed].sum())
        if good < min(required.values()):
            break
        cg, ca = np.cumsum(good_hist), np.cumsum(all_hist)
        t0 = None if gate == 0 else float(np.nextafter(levels0[gate - 1], np.inf))
        for target, need in required.items():
            if good < need:
                continue
            idx = int(np.searchsorted(cg, good - need, side="right") - 1)
            end_count = active - (int(ca[idx]) if idx >= 0 else 0)
            accepted_count = good - (int(cg[idx]) if idx >= 0 else 0)
            # Prefer an explicit bypass if stage two removes nothing on calibration.
            bypass = end_count == active
            t1 = None if idx < 0 or bypass else float(np.nextafter(levels1[idx], np.inf))
            work_numerator = 36 * n + layer * active + (36 - layer) * end_count
            key = (work_numerator, -accepted_count, int(not bypass), active, gate, idx)
            if target not in best_key or key < best_key[target]:
                best_key[target] = key
                best[target] = {"stage0_threshold": t0, "stage1_threshold": t1, "layer": layer,
                                "selection_work_numerator": work_numerator,
                                "selection_accepted_total": accepted_count}
        evaluated += 1
    for target, setting in best.items():
        f, k = apply_cascade(confidence, scores, setting)
        if int((layer * f + (36 - layer) * k).sum()) != setting["selection_work_numerator"]:
            raise ValueError("Joint calibration work-count mismatch")
        if int(np.minimum(a, k - 1).sum()) != setting["selection_accepted_total"]:
            raise ValueError("Joint calibration acceptance mismatch")
        setting["calibration"] = cascade_metrics(a, f, k, layer)
        setting["front_thresholds_evaluated"] = evaluated
        setting["search"] = "exact joint prefix-score transitions; ties higher acceptance, bypass, fewer front rows, earlier transitions"
    if set(best) != set(required):
        raise ValueError("Missing feasible keep-all calibration fallback")
    return best


def paired_bootstrap(accepted, front, end, reference_front, reference_end, prompts,
                     layer=6, reference_layer=0, draws=2000):
    a = np.asarray(accepted)
    cascade_metrics(a, front, end, layer)
    cascade_metrics(a, reference_front, reference_end, reference_layer)
    got, ref_got = np.minimum(a, np.asarray(end) - 1), np.minimum(a, np.asarray(reference_end) - 1)
    work = (layer * np.asarray(front) + (36 - layer) * np.asarray(end)) / 36
    ref_work = (reference_layer * np.asarray(reference_front) + (36 - reference_layer) * np.asarray(reference_end)) / 36
    _, inv = np.unique(prompts, return_inverse=True)
    totals = np.column_stack([np.bincount(inv, weights=v) for v in
                             (got, ref_got, work, ref_work, a, np.ones(len(a)))])
    rng = np.random.default_rng(929)
    sampled = totals[rng.integers(len(totals), size=(draws, len(totals)))].sum(1)
    ci = lambda v: np.quantile(v, [.025, .975]).tolist()
    return {"retention_delta_ci95": ci((sampled[:, 0] - sampled[:, 1]) / sampled[:, 4]),
            "work_rows_delta_ci95": ci((sampled[:, 2] - sampled[:, 3]) / sampled[:, 5]),
            "work_per_committed_delta_ci95": ci(sampled[:, 2] / (sampled[:, 0] + sampled[:, 5])
                                                - sampled[:, 3] / (sampled[:, 1] + sampled[:, 5])),
            "relative_work_saving_ci95": ci(1 - sampled[:, 2] / sampled[:, 3]),
            "resamples": draws, "scope": "policy minus reference; whole prompts; fitted decisions fixed; not calibration or training uncertainty"}
