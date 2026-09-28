"""Compact pre-draft predictors of actual integer-block acceptance responses."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


BUDGETS = np.arange(1, 16, dtype=np.int64)
TARGETS = (.90, .95, .96, .98, 1.0)


class BlockResponseMLP(nn.Module):
    """Same 1.48M-parameter trunk as the control; 15 independent bounded means.

    Output d estimates E[A_(d+1) | latest fused feature], not a survival curve.
    There is deliberately no monotonicity constraint across different drafts.
    """
    def __init__(self, input_dim=2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(.05),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(.05),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(.05), nn.Linear(128, 15))
        self.register_buffer("budgets", torch.arange(1, 16, dtype=torch.float32))

    def forward(self, features):
        return self.net(features.float()).sigmoid()*self.budgets


def training_targets(actual, mode):
    actual = np.asarray(actual)
    if actual.ndim != 2 or actual.shape[1] != 15 or not np.isfinite(actual).all():
        raise ValueError("Expected finite B2--B16 outcomes")
    if ((actual < 0) | (actual > BUDGETS) | (actual != np.floor(actual))).any():
        raise ValueError("Invalid integer acceptance outcomes")
    if mode == "actual":
        return actual.astype(np.float32, copy=True)
    if mode == "clipped":
        return np.minimum(actual[:, -1:], BUDGETS).astype(np.float32)
    raise ValueError("Unknown supervision mode")


def choose_response_budget(mu, setting):
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[1] != 15 or not np.isfinite(mu).all():
        raise ValueError("Expected finite block-response matrix")
    if setting["kind"] == "full":
        return np.full(len(mu), 15, dtype=np.int64)
    if setting["kind"] != "penalty" or not np.isfinite(setting["value"]) or setting["value"] < 0:
        raise ValueError("Invalid budget-penalty setting")
    # This is the Lagrangian of minimum budget under an accepted-token constraint.
    # The multiplier is calibrated on actual outcomes; it is not a timing reward.
    return (mu - setting["value"]*BUDGETS).argmax(1)+1


def penalty_candidates(mu):
    """Calibration-only upper-envelope transitions and their open intervals.

    For each row, take the upper concave hull of (budget, predicted acceptance).
    Positive hull slopes are the breakpoints of argmax(mu_d - lambda*d).
    Include exact boundaries and interval representatives to avoid a coarse grid.
    """
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[1] != 15 or not len(mu) or not np.isfinite(mu).all():
        raise ValueError("Expected nonempty finite calibration responses")
    critical = {0.0}
    for row in mu:
        hull = []
        for j in range(15):
            while len(hull) >= 2:
                a, b = hull[-2:]
                if (row[b]-row[a])/(b-a) > (row[j]-row[b])/(j-b):
                    break
                hull.pop()
            hull.append(j)
        for a, b in zip(hull, hull[1:]):
            slope = float((row[b]-row[a])/(b-a))
            if slope > 0:
                critical.add(slope)
    edges = np.array(sorted(critical))
    return np.unique(np.r_[edges, (edges[:-1]+edges[1:])/2, edges[-1]+1])


def response_metrics(actual, budgets):
    actual, budgets = np.asarray(actual), np.asarray(budgets)
    if actual.shape != (len(budgets), 15) or not len(budgets):
        raise ValueError("Invalid metric arrays")
    if ((budgets < 1) | (budgets > 15) | (budgets != np.floor(budgets))).any():
        raise ValueError("Invalid integer budget")
    selected = actual[np.arange(len(actual)), budgets.astype(np.int64)-1]
    accepted, proposed, reference = int(selected.sum()), int(budgets.sum()), int(actual[:, -1].sum())
    if reference <= 0:
        raise ValueError("Retention needs a positive B16 reference")
    return {"rows": len(actual), "total_accepted": accepted, "total_budget": proposed,
            "mean_accepted": accepted/len(actual), "mean_budget": proposed/len(actual),
            "aggregate_accept_ratio": accepted/proposed, "retention": accepted/reference}


def calibration_curve(mu, actual):
    """Call with calibration rows only. Always retain fixed B16 as a fallback."""
    curves = [{"setting": {"kind": "full"},
               **response_metrics(actual, np.full(len(actual), 15))}]
    previous = None
    for value in penalty_candidates(mu):
        setting = {"kind": "penalty", "value": float(value)}
        budget = choose_response_budget(mu, setting)
        if previous is not None and np.array_equal(budget, previous):
            continue
        curves.append({"setting": setting, **response_metrics(actual, budget)})
        previous = budget
    return curves


def select_response_points(curves, targets=TARGETS):
    selected = {}
    for target in targets:
        feasible = [p for p in curves if p["retention"] >= target-1e-12]
        if not feasible:
            raise ValueError("No calibration-feasible policy")
        selected[str(target)] = min(feasible, key=lambda p: (p["mean_budget"], -p["retention"]))
    return selected
