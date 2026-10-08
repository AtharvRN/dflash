"""Retention-free RecGuide score with calibration-only cost-weight selection.

The score V_B - rho*T_C(B) follows RecGuide. Selecting rho by calibration
aggregate progress / modeled cost is our choice, not a paper reproduction.
Uniform-width costs are only an offline proxy for a mixed-width batch.
"""
from __future__ import annotations

import math
from itertools import combinations

from dflash.history_policy import evaluate_rows, select_block


def validate_profile(profile, table, concurrency):
    if (profile.get("schema_version") != 1 or profile.get("units") != "ms"
            or profile.get("scope") != "whole_cycle"
            or profile.get("cost_basis") != "uniform_batch_cycle"
            or type(concurrency) is not int or concurrency < 1):
        raise ValueError("requires a versioned uniform-batch whole-cycle ms profile")
    if (not isinstance(profile.get("model_identity"), dict) or not profile["model_identity"]
            or profile["model_identity"] != table.payload["provenance"].get("model_identity")):
        raise ValueError("cost profile/model identity mismatch")
    for field in ("engine", "engine_revision", "gpu", "dtype", "attention_backend",
                  "graph_mode", "context_workload", "timing_boundaries"):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            raise ValueError(f"missing cost-profile provenance: {field}")
    sources = profile.get("source_hashes")
    if not isinstance(sources, dict) or not sources or any(
        not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
        for h in sources.values()
    ):
        raise ValueError("cost profile needs SHA256-bound measurement sources")
    costs = {int(b): t for b, t in profile["costs_ms"][str(concurrency)].items()}
    # Reuse selector validation; never interpolate a missing block measurement.
    select_block(table, dict(entropy_count=0, certainty=None, previous_block=None,
                             previous_full=None), mode="priced", costs_ms=costs, rho=0.)
    return costs


def candidate_rhos(vectors, blocks, costs):
    """All nonnegative pairwise crossings plus representatives of open intervals.

    Include both neighbors of each crossing for floating-point score ties.
    Equal-cost arms never cross; no monotonicity assumption on V or costs.
    """
    edges = {0.}
    for values in vectors:
        for i, j in combinations(range(len(blocks)), 2):
            delta = costs[blocks[i]] - costs[blocks[j]]
            if delta:
                cross = (values[i] - values[j]) / delta
                if math.isfinite(cross) and cross >= 0:
                    edges.add(cross)
    edges = sorted(edges)
    candidates = set(edges)
    for x in edges:
        candidates.update(y for y in (math.nextafter(x, -math.inf), math.nextafter(x, math.inf))
                          if math.isfinite(y) and y >= 0)
    candidates.update(a + (b-a)/2 for a, b in zip(edges, edges[1:]))
    last = edges[-1] + max(1., abs(edges[-1]))
    if math.isfinite(last):
        candidates.add(last)
    return sorted(candidates)


def priced_report(table, rows, costs, rho, *, history_free=False):
    options = dict(mode="priced", costs_ms=costs, rho=rho, history_free=history_free)
    result = evaluate_rows(table, rows, **options)
    choices = [select_block(table, r["history"], **options)[0] for r in rows]
    baseline = sum(r["outcomes"][str(table.blocks[-1])]["accepted"] + 1 for r in rows)
    baseline_rate = baseline / (len(rows) * costs[table.blocks[-1]])

    def add_costs(metrics, selected):
        progress = sum(r["outcomes"][str(b)]["accepted"] + 1 for r, b in zip(rows, selected))
        total_cost = sum(costs[b] for b in selected)
        metrics.update(mean_progress=progress/len(rows), mean_uniform_batch_cost_ms=total_cost/len(rows),
                       progress_per_modeled_ms=progress/total_cost,
                       modeled_rate_ratio_vs_largest=(progress/total_cost)/baseline_rate)

    add_costs(result["adaptive"], choices)
    for b in table.blocks:
        add_costs(result["fixed"][str(b)], [b]*len(rows))
    result["cost_scope"] = ("Offline uniform-batch-cost proxy, NOT measured adaptive throughput. "
        "Sum A+1 / sum T_C(B) excludes mixed-batch packing/buckets, policy overhead, "
        "prefill, scheduling and adaptive trajectory changes; not absolute tokens/s.")
    return result


def calibrate_priced(table, rows, costs, *, history_free=False):
    if not rows or any(r["group"] != "calibration" for r in rows):
        raise ValueError("rho tuning accepts calibration rows only")
    # Validate duplicate states, prompt separation, actual arms and finite costs first.
    evaluate_rows(table, rows, mode="priced", costs_ms=costs, rho=0., history_free=history_free)
    groups = {}
    for row in rows:
        values = tuple(table.payload["global_progress"] if history_free else table.predict(row["history"]))
        entry = groups.setdefault(values, [row["history"], 0, [0]*len(table.blocks)])
        entry[1] += 1
        for i, b in enumerate(table.blocks):
            entry[2][i] += row["outcomes"][str(b)]["accepted"] + 1
    candidates = candidate_rhos(groups, table.blocks, costs)
    sweep = []
    for rho in candidates:
        progress, total_cost = 0, 0.
        for history, count, totals in groups.values():
            b, _ = select_block(table, history, mode="priced", costs_ms=costs, rho=rho,
                                history_free=history_free)
            progress += totals[table.blocks.index(b)]
            total_cost += count * costs[b]
        sweep.append(dict(rho=rho, progress_per_modeled_ms=progress/total_cost))
    # Candidates are sorted: lower rho wins an exact objective tie.
    best = max(sweep, key=lambda r: r["progress_per_modeled_ms"])
    report = priced_report(table, rows, costs, best["rho"], history_free=history_free)
    best_fixed = max(table.blocks, key=lambda b: report["fixed"][str(b)]["progress_per_modeled_ms"])
    return dict(rho=best["rho"], history_free=history_free, best_fixed_on_calibration=best_fixed,
                objective="maximize calibration sum(actual A_B + 1) / sum(measured T_C(B)); no retention constraint",
                rho_units="tokens per uniform-batch millisecond", tie_break="lowest rho; smallest B",
                selection_rule="argmax_B V_B(history) - rho * T_C(B)",
                calibration=report, sweep=sweep)
