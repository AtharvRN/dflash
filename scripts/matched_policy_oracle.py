"""Exact hindsight budget oracles on the frozen policy-granularity assessment.

Never use oracle choices as deployed decisions or assessment-tuned thresholds.
No GPU, model fitting, or throughput extrapolation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.block_headroom import exact_frontier
from scripts.analyze_policy_granularity import decisions, metrics
from dflash.block_response import choose_response_budget


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(root, names):
    completion = json.loads((root/"COMPLETE.json").read_text())
    for name in names:
        if completion["binding"].get(name) != digest(root/name):
            raise ValueError("Artifact hash mismatch: "+str(root/name))
    return {name: digest(root/name) for name in names}


def oracles(actual, prompt, retentions):
    n = len(actual)
    reference = int(actual[:, -1].sum())
    if actual.shape != (n, 15) or reference <= 0 or len(prompt) != n:
        raise ValueError("Invalid paired matrix")
    if not np.isfinite(actual).all() or ((actual < 0) | (actual > np.arange(1, 16)) | (actual != np.floor(actual))).any():
        raise ValueError("Invalid accepted lengths")
    targets = [math.ceil(r*reference-1e-10) for r in retentions]
    budgets = np.broadcast_to(np.arange(1, 16), actual.shape).copy()
    groups, inverse = np.unique(prompt, return_inverse=True)
    grouped = np.zeros((len(groups), 15), dtype=np.int64)
    np.add.at(grouped, inverse, actual)
    counts = np.bincount(inverse)
    result = {}
    for name, a, c in (("cycle_hindsight", actual, budgets),
                        ("request_hindsight", grouped, counts[:, None]*np.arange(1, 16))):
        points, maximum = exact_frontier(a, c, targets)
        result[name] = {}
        for retention, point in zip(retentions, points):
            if not point["feasible"]:
                raise ValueError("B16 itself should be feasible")
            choices = np.asarray(point["selected_actions"])+1
            if name == "request_hindsight":
                choices = choices[inverse]
            measured = metrics(actual, choices)
            assert measured["total_accepted"] == point["total_accepted"]
            assert measured["total_budget"] == point["total_budget"]
            result[name][str(retention)] = {"target_accepted": point["target_accepted"], **measured}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior reports; use a new destination")
    binding = {"granularity": verified(args.analysis, ["predictions.npz", "summary.json", "calibration.json", "assessment_curves.json", "report.md"]),
               "heads": verified(args.heads, ["assessment_predictions.npz", "summary.json", "config.json", "audit.json", "report.md"])}
    arrays = np.load(args.analysis/"predictions.npz", allow_pickle=False)
    mask = arrays["eligible"] & (arrays["group"] == "assessment")
    actual, prompt = arrays["actual"][mask], arrays["prompt_id"][mask]
    head_arrays = np.load(args.heads/"assessment_predictions.npz", allow_pickle=False)
    for key, expected in (("actual", actual), ("prompt_id", prompt), ("frozen_survival", arrays["survival"][mask])):
        if not np.array_equal(head_arrays[key], expected):
            raise ValueError("Head/granularity row or prediction mismatch: "+key)
    old = json.loads((args.analysis/"summary.json").read_text())
    heads = json.loads((args.heads/"summary.json").read_text())
    retentions = (.9, .95, .96, .98, 1.0)
    hindsight = oracles(actual, prompt, retentions)
    policies = {}
    for name, point in old["assessment"]["0.96"].items():
        budget = decisions(name, point["setting"], arrays["survival"][mask], arrays["request_survival"][mask])
        measured = metrics(actual, budget)
        if any(measured[k] != point[k] for k in ("total_accepted", "total_budget")):
            raise ValueError("Frozen-policy result did not reproduce")
        policies[name] = measured
    for name, model in heads["models"].items():
        point = model["operating_points"]["0.96"]
        measured = metrics(actual, choose_response_budget(head_arrays[name], point["setting"]))
        if any(measured[k] != point[k] for k in ("total_accepted", "total_budget")):
            raise ValueError("Trained-head result did not reproduce")
        policies[name] = measured
    summary = {"scope": "Hindsight minimum proposed work on COMMON B16 reference states; not learnability or closed-loop throughput",
        "rows": len(actual), "prompts": len(np.unique(prompt)), "reference": metrics(actual, np.full(len(actual), 15)),
        "input_binding": binding, "oracle_frontiers": hindsight, "frozen_calibration_selected_policies": policies,
        "exact_row_match": True, "no_retraining_or_retuning": True,
        "limits": ["Previously inspected development assessment, not a fresh final test",
            "Oracle sees actual outcomes for every B2--B16 at every state, unavailable before drafting",
            "Request oracle sees all sampled future cycles, not just the initial request",
            "All budgets are at least one proposed token; B includes the known anchor",
            "Minimizes sum(B-1), not latency; mixed batches, graph buckets and trajectory shifts are not modeled",
            "96% calibration-selected learned policies need not achieve 96% on assessment",
            "Only included artifact bindings reverified; head checkpoints not reloaded in this CPU report"]}
    args.output.mkdir(parents=True)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# Matched-state budget oracle", "", f"{len(actual):,} assessment cycles / {len(np.unique(prompt)):,} development prompts. All rows and actual B2--B16 outcomes match exactly.", "",
        "Learned policies keep their existing calibration-selected operating points. Hindsight oracles optimize on assessment labels by definition; they are not deployable policies.", "",
        "| Policy | Mean accepted A | Mean proposed budget B−1 | Aggregate ratio | Actual retention |",
        "|---|---:|---:|---:|---:|"]
    table = {"fixed_B16": summary["reference"], **policies,
             **{name: result["0.96"] for name, result in hindsight.items()}}
    for name, point in table.items():
        lines.append(f"| {name} | {point['mean_accepted']:.4f} | {point['mean_budget']:.4f} | {point['aggregate_accept_ratio']:.5f} | {point['retention']:.5f} |")
    lines += ["", "The 96% cycle oracle uses mean block size **7.4633**, including the anchor. This is a token-work bound, not a measured or rigorous throughput ceiling.", "",
              "At 100% aggregate retention, cycle/request hindsight mean budgets are "+
              f"{hindsight['cycle_hindsight']['1.0']['mean_budget']:.4f} / {hindsight['request_hindsight']['1.0']['mean_budget']:.4f}.", ""]
    lines += ["- "+s for s in summary["limits"]]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    (args.output/"COMPLETE.json").write_text(json.dumps({"binding": {n: digest(args.output/n) for n in ("summary.json", "report.md")},
                                                       "source_sha256": digest(Path(__file__))}, indent=2)+"\n")
    print("\n".join(lines[:len(table)+8]))


if __name__ == "__main__":
    main()
