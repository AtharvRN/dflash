"""RecGuide-inspired pre-draft policy; no torch dependency or hidden-state head.

B includes the anchor, A counts accepted proposals, and V_B predicts A + 1.
The histogram estimator/backoff below is our explicit implementation choice,
not a claim to reproduce unspecified RecGuide calibration details.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import math
from statistics import fmean


def blocks_checked(blocks):
    blocks = tuple(blocks)
    if not blocks or any(type(b) is not int or b < 2 for b in blocks):
        raise ValueError("blocks must contain integers >= 2 (including anchor)")
    if blocks != tuple(sorted(set(blocks))):
        raise ValueError("blocks must be sorted and unique")
    return blocks


@dataclass
class History:
    entropies: list[float] = field(default_factory=list)
    previous_block: int | None = None
    previous_full: bool | None = None

    def snapshot(self):
        return {"entropy_count": len(self.entropies),
                "certainty": -fmean(self.entropies) if self.entropies else None,
                "previous_block": self.previous_block, "previous_full": self.previous_full}

    def observe(self, block, accepted, entropy):
        blocks_checked((block,))
        if type(accepted) is not int or not 0 <= accepted < block:
            raise ValueError("accepted proposals must be in [0, B-1]")
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError("entropy must be finite, nonnegative, in nats")
        self.entropies = (self.entropies + [float(entropy)])[-2:]
        self.previous_block, self.previous_full = block, accepted == block - 1


def validate_history(h):
    n, s = h["entropy_count"], h["certainty"]
    if type(n) is not int or n not in (0, 1, 2):
        raise ValueError("invalid history length")
    if n == 0:
        if any(h[k] is not None for k in ("certainty", "previous_block", "previous_full")):
            raise ValueError("cold-start history cannot contain past outcomes")
    else:
        blocks_checked((h["previous_block"],))
        if type(h["previous_full"]) is not bool or s is None or not math.isfinite(s) or s > 0:
            raise ValueError("invalid completed-cycle history")


def validate_row(row, blocks):
    validate_history(row["history"])
    if row.get("schema_version") != 1 or row.get("outcome_kind") != "actual_redraft":
        raise ValueError("requires versioned actual-redraft labels, never clipped B16")
    if not row.get("eligible") or row.get("terminal_or_capped"):
        raise ValueError("terminal/capped or ineligible row")
    if set(row["outcomes"]) != {str(b) for b in blocks}:
        raise ValueError("every candidate B must have an actual matched-state outcome")
    for b in blocks:
        a = row["outcomes"][str(b)]["accepted"]
        if type(a) is not int or not 0 <= a < b:
            raise ValueError("invalid actual acceptance")


class HistoryValueTable:
    def __init__(self, payload):
        self.payload = payload
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported table version")
        self.blocks = blocks_checked(payload["blocks"])
        self.edges = payload["certainty_edges"]
        if self.edges != sorted(set(self.edges)) or not all(math.isfinite(x) for x in self.edges):
            raise ValueError("invalid certainty bin edges")
        for values in [payload["global_progress"], *payload["values"].values()]:
            if len(values) != len(self.blocks) or any(
                not math.isfinite(v) or not 1 <= v <= b for v, b in zip(values, self.blocks)
            ):
                raise ValueError("invalid expected progress")

    def key(self, history):
        validate_history(history)
        if history["entropy_count"] == 0:
            return "cold"
        return ":".join(map(str, (history["entropy_count"],
            bisect_right(self.edges, history["certainty"]), history["previous_block"],
            int(history["previous_full"]))))

    def predict(self, history):
        key = self.key(history)
        return list(self.payload["values"].get(key, self.payload["global_progress"]))

    @classmethod
    def fit(cls, rows, blocks, *, bins=8, prior_count=20.0, provenance=None):
        blocks = blocks_checked(blocks)
        if type(bins) is not int or bins < 1 or not math.isfinite(prior_count) or prior_count < 0:
            raise ValueError("invalid estimator settings")
        if not rows:
            raise ValueError("no eligible training rows")
        seen, prompts, certainty = set(), set(), []
        for row in rows:
            validate_row(row, blocks)
            if row["group"] != "train":
                raise ValueError("fit accepts canonical training prompts only")
            identity = (str(row["prompt_id"]), row["cycle"])
            if identity in seen:
                raise ValueError("duplicate prompt/cycle")
            seen.add(identity)
            prompts.add(identity[0])
            if row["history"]["certainty"] is not None:
                certainty.append(row["history"]["certainty"])
        certainty.sort()
        edges = sorted({certainty[min(len(certainty)-1, i*len(certainty)//bins)]
                        for i in range(1, bins)}) if certainty else []
        global_values = [fmean(r["outcomes"][str(b)]["accepted"] + 1 for r in rows) for b in blocks]
        payload = {"schema_version": 1, "blocks": list(blocks), "certainty_edges": edges,
                   "global_progress": global_values, "values": {}, "counts": {},
                   "prior_count": prior_count, "fit_prompt_ids": sorted(prompts),
                   "fit_rows": len(rows), "provenance": provenance or {},
                   "definition": "B includes anchor; V_B = E[actual A_B + 1]; entropy in nats"}
        table = cls(payload)
        groups = {}
        for row in rows:
            groups.setdefault(table.key(row["history"]), []).append(row)
        for key, cohort in groups.items():
            n = len(cohort)
            payload["counts"][key] = n
            payload["values"][key] = [
                (sum(r["outcomes"][str(b)]["accepted"] + 1 for r in cohort) + prior_count*g)
                / (n + prior_count) for b, g in zip(blocks, global_values)]
        return cls(payload)


def select_block(table, history, *, mode, alpha=None, costs_ms=None, rho=None, history_free=False):
    """No fabricated cost curve. Ties select the smallest B deterministically."""
    validate_history(history)
    values = list(table.payload["global_progress"]) if history_free else table.predict(history)
    if mode == "priced":
        if rho is None or not math.isfinite(rho) or rho < 0 or costs_ms is None:
            raise ValueError("priced mode requires explicit rho (tokens/ms) and measured costs")
        if set(costs_ms) != set(table.blocks) or any(not math.isfinite(t) or t <= 0 for t in costs_ms.values()):
            raise ValueError("one finite positive cost per candidate B required")
        scores = [v - rho*costs_ms[b] for b, v in zip(table.blocks, values)]
        chosen = table.blocks[max(range(len(scores)), key=lambda i: scores[i])]
    elif mode == "retention":
        if alpha is None or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("retention mode requires alpha in (0,1]")
        # Expected proposed-token retention relative to the largest B, not progress retention.
        threshold = alpha * (values[-1] - 1)
        chosen = next(b for b, v in zip(table.blocks, values) if v - 1 >= threshold - 1e-12)
    else:
        raise ValueError("unknown policy mode")
    return chosen, values


def evaluate_rows(table, rows, **selection):
    """Matched-state assessment, NOT closed-loop throughput or a retention guarantee."""
    if not rows:
        raise ValueError("no assessment rows")
    fit_ids = set(table.payload["fit_prompt_ids"])
    seen, selected, counts = set(), [], {}
    for row in rows:
        validate_row(row, table.blocks)
        if str(row["prompt_id"]) in fit_ids or row["group"] not in ("calibration", "assessment"):
            raise ValueError("evaluation must be prompt-disjoint from fitting")
        identity = (str(row["prompt_id"]), row["cycle"])
        if identity in seen:
            raise ValueError("duplicate evaluation state")
        seen.add(identity)
        b, _ = select_block(table, row["history"], **selection)
        selected.append(b)
        counts[str(b)] = counts.get(str(b), 0) + 1
    baseline = sum(r["outcomes"][str(table.blocks[-1])]["accepted"] for r in rows)

    def metrics(choices):
        accepted = sum(r["outcomes"][str(b)]["accepted"] for r, b in zip(rows, choices))
        budget = sum(b - 1 for b in choices)
        return {"mean_accepted": accepted/len(rows), "mean_block": fmean(choices),
                "aggregate_accept_ratio": accepted/budget,
                "retention_vs_largest": accepted/baseline if baseline else None}

    return {"scope": "matched-state replay; history follows collection behavior, NOT adaptive rollout",
            "rows": len(rows), "prompts": len({r["prompt_id"] for r in rows}),
            "adaptive": metrics(selected), "block_counts": counts,
            "fixed": {str(b): metrics([b]*len(rows)) for b in table.blocks}}
