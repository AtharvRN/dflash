"""Frozen-predictor decision-frequency ablation on actual paired outcomes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256, validate_states
from scripts.collect_policy_granularity import CHECKPOINT_SHA, atomic_json


def budgets_from_survival(survival, alpha):
    survival = np.asarray(survival, dtype=np.float64)
    if survival.ndim != 2 or survival.shape[1] != 15 or not np.isfinite(survival).all():
        raise ValueError("Expected finite 15-position survival curves")
    if not 0 < alpha <= 1:
        raise ValueError("Invalid alpha")
    if alpha == 1:
        return np.full(len(survival), 15, dtype=np.int64)
    cumulative = survival.cumsum(1)
    return (cumulative >= alpha*cumulative[:, -1:]).argmax(1)+1


def metrics(actual, budgets):
    actual, budgets = np.asarray(actual), np.asarray(budgets)
    if actual.shape != (len(budgets), 15) or not len(budgets):
        raise ValueError("Invalid outcome matrix")
    if np.any(budgets != budgets.astype(int)) or np.any((budgets < 1) | (budgets > 15)):
        raise ValueError("Invalid integer budgets")
    if np.any((actual < 0) | (actual > np.arange(1, 16))):
        raise ValueError("Invalid accepted lengths")
    a = actual[np.arange(len(actual)), budgets.astype(int)-1]
    baseline = actual[:, -1].sum()
    if baseline <= 0:
        raise ValueError("Empty accepted-token reference")
    return {"rows": len(budgets), "total_accepted": int(a.sum()), "total_budget": int(budgets.sum()),
        "mean_accepted": float(a.mean()), "mean_budget": float(budgets.mean()), "mean_block": float(budgets.mean()+1),
        "aggregate_accept_ratio": float(a.sum()/budgets.sum()), "retention": float(a.sum()/baseline),
        "budget_saving_vs_b16": float(1-budgets.sum()/(15*len(budgets))),
        "verify_positions_per_advanced_token": float((budgets.sum()+len(budgets))/(a.sum()+len(budgets))),
        "block_histogram": {str(b): int((budgets == b-1).sum()) for b in range(2, 17)}}


def calibration_breakpoints(survival):
    """Every decision transition of the FP64 alpha rule on calibration inputs.

    Avoid a coarse alpha grid spuriously jumping straight from .999 to B16.
    No acceptance labels or assessment inputs enter candidate construction.
    """
    cumulative = np.asarray(survival, dtype=np.float64).cumsum(1)
    total = cumulative[:, -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        transitions = cumulative[:, :-1]/total
    # Find the first representable alpha for which prefix expectation is smaller
    # than alpha * total, accounting for rounding of that multiplication.
    for _ in range(4):
        transitions = np.where(transitions*total <= cumulative[:, :-1],
                               np.nextafter(transitions, np.inf), transitions)
    take = np.isfinite(transitions) & (transitions >= .5) & (transitions < 1)
    if np.any(take & (transitions*total <= cumulative[:, :-1])):
        raise ValueError("Unresolved numerical alpha transition")
    return np.unique(np.concatenate((np.linspace(.5, 1, 501), transitions[take])))


def select_operating_points(actual, cycle_s, request_s, targets=(.90, .95, .96, .98, 1.0), *, exact=False):
    """Only calibration rows may be passed to this function."""
    curves = {}
    for name, s in (("fixed", None), ("request", request_s), ("cycle", cycle_s)):
        points = []
        settings = range(1, 16) if name == "fixed" else (calibration_breakpoints(s) if exact else np.linspace(.5, 1, 501))
        for setting in settings:
            d = np.full(len(actual), setting, dtype=int) if s is None else budgets_from_survival(s, float(setting))
            points.append({"setting": int(setting) if s is None else float(setting), **metrics(actual, d)})
        curves[name] = points
    selected = {}
    for target in targets:
        selected[str(target)] = {}
        for name, points in curves.items():
            feasible = [p for p in points if p["retention"] >= target-1e-12]
            if not feasible:
                raise ValueError("No calibration-feasible policy, including B16")
            selected[str(target)][name] = min(feasible, key=lambda p: (p["mean_budget"], -p["retention"], p["setting"]))
    return selected, curves


def decisions(name, setting, cycle_s, request_s):
    if name == "fixed":
        return np.full(len(cycle_s), int(setting), dtype=int)
    return budgets_from_survival(request_s if name == "request" else cycle_s, setting)


def paired_bootstrap(actual, prompt_ids, choices, count=2000, seed=928):
    prompts, inverse = np.unique(prompt_ids, return_inverse=True)
    totals = {}
    for name, d in choices.items():
        a = actual[np.arange(len(actual)), d-1]
        totals[name] = np.column_stack([np.bincount(inverse, weights=v, minlength=len(prompts))
                                      for v in (a, d, actual[:, -1], np.ones(len(a)))])
    draws = np.random.default_rng(seed).integers(len(prompts), size=(count, len(prompts)))
    sampled = {name: value[draws].sum(1) for name, value in totals.items()}
    ci = lambda value: np.quantile(value, [.025, .975]).tolist()
    out = {"resamples": count, "prompts": len(prompts),
           "scope": "paired whole-prompt bootstrap; weights and calibration settings held fixed", "policies": {}, "differences": {}}
    for name, value in sampled.items():
        out["policies"][name] = {"retention_ci95": ci(value[:, 0]/value[:, 2]),
            "ratio_ci95": ci(value[:, 0]/value[:, 1]), "mean_budget_ci95": ci(value[:, 1]/value[:, 3])}
    for other in ("fixed", "request"):
        c, b = sampled["cycle"], sampled[other]
        out["differences"]["cycle_minus_"+other] = {
            "mean_budget_ci95": ci((c[:, 1]-b[:, 1])/c[:, 3]),
            "retention_ci95": ci((c[:, 0]-b[:, 0])/c[:, 2]),
            "relative_budget_saving_ci95": ci(1-c[:, 1]/b[:, 1]),
            "aggregate_ratio_ci95": ci(c[:, 0]/c[:, 1]-b[:, 0]/b[:, 1])}
    return out


def load_pairs(root):
    config = json.loads((root/"config.json").read_text())
    complete = json.loads((root/"COMPLETE.json").read_text())
    if not complete["sample_complete"] or config["blocks"] != list(range(2, 17)):
        raise ValueError("Incomplete or wrong block-size collection")
    for name, digest in complete["binding"].items():
        if sha256(root/name) != digest:
            raise ValueError("Completion hash mismatch")
    rows, feature_chunks, initial, seen = [], [], [], set()
    for receipt in json.loads((root/"receipts.json").read_text()):
        pid = receipt["prompt_id"]
        if pid in seen or str(pid) not in config["prompt_groups"]:
            raise ValueError("Unexpected/duplicate receipt")
        seen.add(pid)
        if json.loads((root/f"receipt_{pid}.json").read_text()) != receipt:
            raise ValueError("Receipt disagreement")
        for name, digest in receipt["files"].items():
            if Path(name).name != name or sha256(root/name) != digest:
                raise ValueError("Shard hash mismatch")
        states = json.loads((root/f"prompt_{pid}.json").read_text())["states"]
        validate_states(states, config["blocks"])
        if len(states) != receipt["states"]:
            raise ValueError("Row-count mismatch")
        if not states:
            continue
        if states[0]["cycle"] != 0 or states[0]["generated_before_anchor"] != 0:
            raise ValueError("Missing initial request feature")
        if any(int(r["prompt_id"]) != pid or r["group"] != config["prompt_groups"][str(pid)] for r in states):
            raise ValueError("Prompt/group alignment failure")
        features = np.load(root/f"prompt_{pid}_fused.npy", allow_pickle=False)
        if features.shape != (len(states), 2560) or not np.isfinite(features).all():
            raise ValueError("Feature alignment failure")
        initial.extend([len(rows)]*len(states))
        rows.extend(states)
        feature_chunks.append(features)
    if seen != set(config["prompt_ids"]) or len(rows) != complete["states"]:
        raise ValueError("Missing prompts or states")
    return config, rows, np.concatenate(feature_chunks), np.asarray(initial)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exact-calibration-breakpoints", action="store_true",
                   help="Exploratory grid-resolution sensitivity; preserve the primary analysis separately")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Refusing existing analysis destination")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Unexpected checkpoint")
    config, rows, features, initial = load_pairs(args.cache)
    if config["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Collection/checkpoint mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from dflash.policy import HorizonPredictorRuntime
    from dflash.context_attention import acceptance_survival
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]
    original_cal = set(map(int, cfg["cache_manifest"]["calibration_prompt_ids"]))
    for pid, group in config["prompt_groups"].items():
        if (int(pid) in original_cal) != (group == "calibration"):
            raise ValueError("Original 100k checkpoint calibration membership changed")
    if checkpoint["architecture"] != "last_mlp" or checkpoint["epoch"] != 4 or cfg["length_weight"] != 0:
        raise ValueError("Checkpoint model configuration mismatch")
    model = HorizonPredictorRuntime(input_dim=2560, proj_dim=512, hidden_size=256, num_slots=15,
        architecture="last_mlp", num_layers=1, dropout=.05, context_window=16).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"])
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            x = torch.from_numpy(features[start:start+256].astype(np.float32))[:, None]
            chunks.append(acceptance_survival(model(x, torch.ones(x.shape[:2]))).numpy())
    survival = np.concatenate(chunks)
    request_s = survival[initial]
    eligible = np.array([r["eligible"] for r in rows], dtype=bool)
    group = np.array([r["group"] for r in rows])
    prompt = np.array([int(r["prompt_id"]) for r in rows])
    actual = np.array([[r["outcomes"][str(b)]["accepted"] for b in range(2, 17)] for r in rows])
    cal, assess = eligible & (group == "calibration"), eligible & (group == "assessment")
    if not cal.any() or not assess.any() or set(prompt[cal]) & set(prompt[assess]):
        raise ValueError("Invalid calibration/assessment partition")
    selected, cal_curves = select_operating_points(actual[cal], survival[cal], request_s[cal], exact=args.exact_calibration_breakpoints)
    args.output.mkdir(parents=True)
    # Persist frozen calibration choices before evaluating any assessment outcomes.
    atomic_json(args.output/"calibration.json", {"selected": selected, "curves": cal_curves})
    results, assessment_curves = {}, {}
    for target, policies in selected.items():
        results[target] = {}
        choices = {}
        for name, point in policies.items():
            d = decisions(name, point["setting"], survival[assess], request_s[assess])
            results[target][name] = {"setting": point["setting"], **metrics(actual[assess], d)}
            choices[name] = d
        if target == "0.96":
            uncertainty = paired_bootstrap(actual[assess], prompt[assess], choices)
    # Curves are descriptive. No assessment-based choice is fed back to selection.
    for name, points in cal_curves.items():
        assessment_curves[name] = [{"setting": q["setting"], **metrics(actual[assess],
            decisions(name, q["setting"], survival[assess], request_s[assess]))} for q in points]
    summary = {"scope": config["scope"], "checkpoint": str(args.checkpoint), "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_training": "100k historical cycle rows; frozen 1,478,415-parameter last-fused MLP, epoch 4",
        "head_inference": "CPU FP32; no PCA or feature normalization; dropout disabled",
        "calibration_search": "exploratory exact decision breakpoints from calibration inputs" if args.exact_calibration_breakpoints else "preregistered .001 alpha grid",
        "cache": str(args.cache), "cache_config_sha256": sha256(args.cache/"config.json"),
        "counts": {g: {"rows": int(m.sum()), "prompts": len(np.unique(prompt[m]))}
                   for g, m in (("calibration", cal), ("assessment", assess))},
        "excluded_rows": int((~eligible).sum()), "settings_from_calibration_only": True,
        "assessment_b16": metrics(actual[assess], np.full(int(assess.sum()), 15)),
        "assessment": results, "primary_prompt_bootstrap": uncertainty,
        "limitations": ["common reference states, not closed-loop trajectories", "previously inspected development prompts",
            "single frozen checkpoint; no training-seed uncertainty", "discrete fixed sizes need not match retention exactly",
            "at most eight progress-spaced states per prompt; not all decoding cycles", "not throughput"]}
    atomic_json(args.output/"summary.json", summary)
    atomic_json(args.output/"assessment_curves.json", assessment_curves)
    np.savez_compressed(args.output/"predictions.npz", survival=survival, request_survival=request_s,
        actual=actual, eligible=eligible, group=group, prompt_id=prompt, initial_row=initial)
    lines = ["# Frozen predictor: request versus cycle adaptation", "", summary["scope"], "",
        "Exact 100k checkpoint; no model training. Primary target: 96% calibration retention.", "",
        "| Policy | Setting | Mean accepted | Mean draft budget | Aggregate ratio | Retention |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, point in results["0.96"].items():
        setting = f"B{point['setting']+1}" if name == "fixed" else f"alpha={point['setting']:.3f}"
        lines.append(f"| {name} | {setting} | {point['mean_accepted']:.4f} | {point['mean_budget']:.4f} | {point['aggregate_accept_ratio']:.5f} | {100*point['retention']:.3f}% |")
    lines += ["", "## Uncertainty", "", "```json", json.dumps(uncertainty, indent=2), "```", "",
              "## Limitations", ""] + ["- "+s for s in summary["limitations"]]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    atomic_json(args.output/"COMPLETE.json", {"binding": {name: sha256(args.output/name) for name in
        ("calibration.json", "summary.json", "assessment_curves.json", "predictions.npz", "report.md")},
        "analysis_source_sha256": sha256(__file__)})
    print(json.dumps({"counts": summary["counts"], "primary": results["0.96"], "uncertainty": uncertainty}, indent=2))


if __name__ == "__main__":
    main()
