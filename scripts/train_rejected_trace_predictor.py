"""Matched control/aligned/shuffled previous-trace pre-draft predictor pilot.

Only calibration selects checkpoints and penalties.  Assessment is evaluated
after every model has been frozen.  This measures information value on fixed
trajectories; it does not claim closed-loop latency or throughput improvements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.block_response import (TARGETS, calibration_curve, choose_response_budget,
    response_metrics, select_response_points, training_targets)
from dflash.rejected_trace import ARMS, RejectedTraceResponseModel, make_donor_mapping, model_batch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def weight_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_rows(rows, arrays, training_rows, smoke=False):
    """Freeze first-N eligible train rows and every eligible evaluation row."""
    if training_rows < 1:
        raise ValueError("Positive training row target required")
    eligible = np.array([bool(row["eligible"]) for row in rows])
    group = np.array([row["group"] for row in rows])
    if not set(group) <= {"train", "calibration", "assessment"}:
        raise ValueError("Unknown prompt partition")
    train = np.flatnonzero(eligible & (group == "train"))
    if len(train) < training_rows and not smoke:
        raise ValueError("Insufficient eligible training rows")
    take = np.concatenate((train[:training_rows], np.flatnonzero(eligible & (group == "calibration")),
                           np.flatnonzero(eligible & (group == "assessment"))))
    selected = [rows[int(i)] for i in take]
    masks = {g: np.array([row["group"] == g for row in selected])
             for g in ("train", "calibration", "assessment")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Need eligible train, calibration, and assessment rows")
    pids = {g: {int(row["prompt_id"]) for row in selected if row["group"] == g} for g in masks}
    if any(pids[g] & pids[h] for g in pids for h in pids if g != h):
        raise ValueError("Prompt crosses partitions")
    values = {name: np.asarray(value)[take] for name, value in arrays.items()}
    training_targets(values["actual"], "actual")
    return selected, values, masks, take


def memory_coverage(rows, arrays, donors):
    result = {}
    original = np.asarray(arrays["trace_mask"]).astype(bool).any(1)
    effective = np.asarray(donors) >= 0
    for group in ("train", "calibration", "assessment"):
        mask = np.array([row["group"] == group for row in rows])
        result[group] = {"rows": int(mask.sum()), "original_memory_rows": int((mask & original).sum()),
            "effective_memory_rows": int((mask & effective).sum()),
            "no_different_prompt_donor_rows": int((mask & original & ~effective).sum()),
            "first_cycle_rows": sum(not row["has_previous"] for row in rows if row["group"] == group),
            "previous_full_accept_rows": sum(row["has_previous"] and row["previous_A"] == row["previous_B"]-1
                                            for row in rows if row["group"] == group)}
    return result


def predict(model, arrays, indices, arm, donors, batch_size=128):
    import torch
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return np.empty((0, 15), dtype=np.float32)
    model = model.cpu().eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = model_batch(arrays, indices[start:start+batch_size], arm, donors)
            chunks.append(model(**batch).numpy())
    return np.concatenate(chunks)


def safe_metrics(actual, budget):
    if not len(actual):
        return {"rows": 0, "reason": "empty subgroup"}
    if np.asarray(actual)[:, -1].sum() == 0:
        return {"rows": len(actual), "reason": "zero B16 acceptance; retention undefined"}
    return response_metrics(actual, budget)


def paired_prompt_bootstrap(actual, prompts, choices, count=2000, seed=929):
    """Same whole-prompt draws for every arm, with checkpoint/settings frozen."""
    if count < 1:
        raise ValueError("Positive bootstrap count required")
    groups, inverse = np.unique(prompts, return_inverse=True)
    draws = np.random.default_rng(seed).integers(len(groups), size=(count, len(groups)))
    sampled, intervals = {}, {}
    def interval(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        return np.quantile(finite, [.025, .975]).tolist() if len(finite) else None
    for name, budget in choices.items():
        values = (actual[np.arange(len(actual)), budget-1], budget, actual[:, -1], np.ones(len(actual)))
        totals = np.column_stack([np.bincount(inverse, weights=value, minlength=len(groups)) for value in values])
        sampled[name] = q = totals[draws].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            intervals[name] = {"ratio_ci95": interval(q[:, 0]/q[:, 1]),
                "retention_ci95": interval(q[:, 0]/q[:, 2]), "mean_budget_ci95": interval(q[:, 1]/q[:, 3]),
                "zero_reference_draws": int((q[:, 2] == 0).sum())}
    paired = {}
    for name, a in sampled.items():
        if not name.startswith("aligned_seed_"):
            continue
        for arm in ("control", "shuffled"):
            other = name.replace("aligned_", arm+"_", 1)
            if other not in sampled:
                continue
            b = sampled[other]
            with np.errstate(divide="ignore", invalid="ignore"):
                paired[name+"_minus_"+other] = {
                    "ratio_ci95": interval(a[:, 0]/a[:, 1]-b[:, 0]/b[:, 1]),
                    "retention_ci95": interval((a[:, 0]-b[:, 0])/a[:, 2]),
                    "mean_budget_delta_ci95": interval((a[:, 1]-b[:, 1])/a[:, 3]),
                    "relative_budget_saving_ci95": interval(1-a[:, 1]/b[:, 1])}
    return {"resamples": count, "prompts": len(groups), "policies": intervals,
        "paired_arm_differences": paired,
        "scope": "whole-prompt paired resampling, selected checkpoints/lambda and donor map held fixed; not training-seed uncertainty"}


def benchmark_predictor(model, arrays, indices, arm, donors, repeats, device="cpu"):
    """Preallocated B1/64/128 full-model timing, not an engine benchmark."""
    import torch
    indices = np.asarray(indices, dtype=np.int64)
    model = model.to(device).eval()
    measurements = {}
    with torch.inference_mode():
        for batch_rows in (1, 64, 128):
            ids = np.resize(indices, batch_rows)
            batch = model_batch(arrays, ids, arm, donors, device)
            for _ in range(10):
                model(**batch)
            if device == "cuda":
                torch.cuda.synchronize()
                events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
                for begin, end in events:
                    begin.record()
                    model(**batch)
                    end.record()
                torch.cuda.synchronize()
                elapsed = [begin.elapsed_time(end) for begin, end in events]
            else:
                elapsed = []
                for _ in range(repeats):
                    before = time.perf_counter()
                    model(**batch)
                    elapsed.append((time.perf_counter()-before)*1000)
            measurements[str(batch_rows)] = {"median_ms": float(np.median(elapsed)),
                "p05_ms": float(np.quantile(elapsed, .05)), "p95_ms": float(np.quantile(elapsed, .95)),
                "effective_memory_rows": int(batch["trace_mask"].any(1).sum()),
                "input_bytes": sum(value.numel()*value.element_size() for value in batch.values())}
    model.cpu()
    return {"device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "timing": "CUDA events" if device == "cuda" else "perf_counter",
        "warmup": 10, "repeats": repeats, "batches": measurements,
        "scope": "preallocated full predictor including trace encoding and dtype conversion; excludes input gather/H2D, target capture/retention, and serving integration"}


def run(args):
    if args.output.exists():
        raise ValueError("Output already exists; choose a new destination")
    if args.epochs < 1 or args.batch_size < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Invalid training limits or duplicate seeds")
    if args.bootstrap < 1 or args.cpu_threads < 1 or args.benchmark_repeats < 0:
        raise ValueError("Invalid bootstrap, CPU thread, or benchmark count")
    if not args.smoke and (args.training_rows != 2000 or args.epochs != 6 or args.batch_size != 128
                           or args.seeds != [913, 914, 915]):
        raise ValueError("Non-smoke run requires 2000 rows, seeds913/914/915, six epochs, batch128")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu is None else str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from scripts.audit_rejected_trace_cache import load_cache
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.gpu is not None:
        used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], text=True)
        if int(used.strip()) > 1024:
            raise RuntimeError("GPU is occupied; no training launched")
    cache_config, raw_rows, raw_arrays, audit = load_cache(args.cache, require_complete=not args.smoke)
    if not args.smoke and (cache_config.get("smoke", False) or cache_config["training_rows"] != 2000):
        raise ValueError("Non-smoke training needs a production 2000-cycle cache")
    rows, arrays, masks, raw_indices = select_rows(raw_rows, raw_arrays, args.training_rows, args.smoke)
    del raw_arrays
    donors = make_donor_mapping(rows, args.donor_seed)
    coverage = memory_coverage(rows, arrays, donors)
    actual = arrays["actual"].astype(np.int64)
    train, cal, assess = (np.flatnonzero(masks[g]) for g in ("train", "calibration", "assessment"))
    args.output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    model = RejectedTraceResponseModel()
    config.update({"model": model.model_config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "arms": list(ARMS), "loss": "MSE of actual accepted-token count across B2--B16, unweighted",
        "optimizer": "AdamW lr3e-4 wd0.01 clip1.0, no scheduler",
        "selection": "checkpoint and lambda minimize calibration mean budget at>=96% B16 acceptance retention; ties higher retention then earlier epoch",
        "policy": "argmax mu_B - lambda*(B-1); all integer blocks2--16; calibration upper-envelope breakpoints and fixedB16 fallback",
        "matching": "same architecture, initialization, epoch row orders and dropout RNG stream within seed; all scores CPUFP32",
        "donors": "fixed seed, within partition and previousB/A, different prompt, sampled with replacement; no donor means empty trace in ALL arms",
        "torch": torch.__version__, "cache_config": cache_config,
        "source_hashes": {str(path): sha256(path) for path in (Path(__file__),
            Path(__file__).resolve().parents[1]/"dflash/rejected_trace.py",
            Path(__file__).resolve().parents[1]/"dflash/block_response.py",
            Path(__file__).resolve().parents[1]/"scripts/audit_rejected_trace_cache.py")}})
    atomic_json(args.output/"config.json", config)
    atomic_json(args.output/"audit.json", audit)
    index = [{"selected_row": i, "cache_row": int(raw_indices[i]), "prompt_id": int(row["prompt_id"]),
        "cycle": int(row["cycle"]), "group": row["group"], "prefix_sha256": row["prefix_sha256"],
        "previous_B": row["previous_B"], "previous_A": row["previous_A"],
        "donor_row": int(donors[i]), "donor_prompt_id": int(rows[donors[i]]["prompt_id"]) if donors[i] >= 0 else None}
        for i, row in enumerate(rows)]
    atomic_json(args.output/"row_index_and_donors.json", index)
    atomic_json(args.output/"memory_coverage.json", coverage)
    print("AUDIT_PASSED", json.dumps({"selected": coverage, "parameter_count": config["parameter_count"]}), flush=True)
    device = "cuda" if args.gpu is not None else "cpu"
    started = time.monotonic()
    selected, randomness = {}, {}
    for seed in args.seeds:
        initial_hashes, order_hashes = [], []
        for arm in ARMS:
            name = f"{arm}_seed_{seed}"
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = RejectedTraceResponseModel().to(device)
            initial = weight_digest(model.state_dict())
            initial_hashes.append(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            generator = torch.Generator().manual_seed(seed)
            order_hash = hashlib.sha256()
            history, best, best_score = [], None, None
            for epoch in range(1, args.epochs+1):
                model.train()
                order = train[torch.randperm(len(train), generator=generator).numpy()]
                order_hash.update(order.tobytes())
                losses = 0.
                for start in range(0, len(order), args.batch_size):
                    ids = order[start:start+args.batch_size]
                    batch = model_batch(arrays, ids, arm, donors, device)
                    target = torch.as_tensor(actual[ids], dtype=torch.float32, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(**batch)-target).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    losses += float(loss.detach())*len(ids)
                mu = predict(copy.deepcopy(model), arrays, cal, arm, donors, args.batch_size)
                curves = calibration_curve(mu, actual[cal])
                points = select_response_points(curves, TARGETS)
                primary = points["0.96"]
                record = {"epoch": epoch, "train_mse": losses/len(train),
                    "calibration_mse": float(np.mean((mu-actual[cal])**2)), "primary_calibration": primary}
                history.append(record)
                print("EPOCH", name, json.dumps(record), flush=True)
                score = (primary["mean_budget"], -primary["retention"], epoch)
                if best_score is None or score < best_score:
                    best_score = score
                    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    best = {"name": name, "arm": arm, "seed": seed, "epoch": epoch, "model": state,
                        "model_config": model.model_config, "calibration_points": points,
                        "initial_state_sha256": initial, "state_sha256": weight_digest(state),
                        "training_config_sha256": sha256(args.output/"config.json")}
                    path = args.output/f"{name}.pt"
                    torch.save(best, path.with_suffix(".pt.tmp"))
                    path.with_suffix(".pt.tmp").replace(path)
                    atomic_json(args.output/f"{name}_calibration.json", {"epoch": epoch, "selected": points, "curves": curves})
                atomic_json(args.output/f"{name}_history.json", history)
            order_hashes.append(order_hash.hexdigest())
            randomness[name] = {"initial_state_sha256": initial, "epoch_orders_sha256": order_hash.hexdigest()}
            selected[name] = {key: value for key, value in best.items() if key != "model"}
            del model, optimizer
        if len(set(initial_hashes)) != 1 or len(set(order_hashes)) != 1:
            raise AssertionError("Arms did not share initialization and orders")
    atomic_json(args.output/"selection_frozen.json", {"models": selected, "matched_randomness": randomness})
    results, choices, saved_arrays, all_curves, timings = {}, {}, {}, {}, {}
    for name, selection in selected.items():
        saved = torch.load(args.output/f"{name}.pt", map_location="cpu", weights_only=False)
        if weight_digest(saved["model"]) != selection["state_sha256"]:
            raise ValueError("Checkpoint reload hash mismatch")
        model = RejectedTraceResponseModel(**saved["model_config"]).eval()
        model.load_state_dict(saved["model"])
        arm = selection["arm"]
        # Recomputed calibration selections must reproduce the saved settings.
        reloaded_cal = predict(model, arrays, cal, arm, donors, args.batch_size)
        if select_response_points(calibration_curve(reloaded_cal, actual[cal]), TARGETS) != selection["calibration_points"]:
            raise ValueError("Reloaded calibration policy mismatch")
        mu = predict(model, arrays, assess, arm, donors, args.batch_size)
        saved_arrays[name+"_mu"] = mu
        results[name] = {"arm": arm, "seed": selection["seed"], "epoch": selection["epoch"],
            "actual_mae_by_block": np.abs(mu-actual[assess]).mean(0).tolist(),
            "actual_mse_by_block": ((mu-actual[assess])**2).mean(0).tolist(), "operating_points": {}}
        for target, point in selection["calibration_points"].items():
            budget = choose_response_budget(mu, point["setting"])
            results[name]["operating_points"][target] = {"setting": point["setting"],
                "calibration_retention": point["retention"], **response_metrics(actual[assess], budget)}
            if target == "0.96":
                choices[name] = budget
                saved_arrays[name+"_budget"] = budget
                subgroups = {"effective_memory": donors[assess] >= 0,
                    "empty_effective_memory": donors[assess] < 0,
                    "original_memory_without_donor": np.asarray(arrays["trace_mask"])[assess].astype(bool).any(1) & (donors[assess] < 0)}
                results[name]["primary_subgroups"] = {key: safe_metrics(actual[assess][mask], budget[mask]) for key, mask in subgroups.items()}
        calibrated = json.loads((args.output/f"{name}_calibration.json").read_text())["curves"]
        all_curves[name] = [{"setting": point["setting"], **response_metrics(actual[assess],
            choose_response_budget(mu, point["setting"]))} for point in calibrated]
        if args.benchmark_repeats and selection["seed"] == args.seeds[0]:
            timings[name] = benchmark_predictor(model, arrays, assess, arm, donors, args.benchmark_repeats, device)
    fixed_cal = [{"setting": {"kind": "fixed", "budget": budget},
                  **response_metrics(actual[cal], np.full(len(cal), budget))} for budget in range(1, 16)]
    fixed_points = select_response_points(fixed_cal, TARGETS)
    controls = {target: {"setting": point["setting"], "calibration_retention": point["retention"],
        **response_metrics(actual[assess], np.full(len(assess), point["setting"]["budget"]))}
        for target, point in fixed_points.items()}
    choices["fixed_calibrated"] = np.full(len(assess), fixed_points["0.96"]["setting"]["budget"])
    choices["fixed_B16"] = np.full(len(assess), 15)
    prompts = np.array([int(rows[i]["prompt_id"]) for i in assess])
    uncertainty = paired_prompt_bootstrap(actual[assess], prompts, choices, args.bootstrap)
    across_seeds = {}
    for arm in ARMS:
        points = [result["operating_points"]["0.96"] for result in results.values() if result["arm"] == arm]
        across_seeds[arm] = {key: {"mean": float(np.mean([point[key] for point in points])),
            "std": float(np.std([point[key] for point in points], ddof=1)) if len(points)>1 else None,
            "min": min(point[key] for point in points), "max": max(point[key] for point in points)}
            for key in ("aggregate_accept_ratio", "retention", "mean_budget", "mean_accepted")}
    # Every input file checked at load time must remain immutable through training.
    for name, expected in audit.get("input_file_sha256", {}).items():
        path = Path(name)
        if not path.is_absolute():
            path = args.cache/path
        if sha256(path) != expected:
            raise ValueError("Cache changed during training: "+str(path))
    summary = {"scope": "SMOKE ONLY" if args.smoke else "2000-cycle rejected-trace information pilot",
        "counts": {group: {"rows": int(mask.sum()), "prompts": len({int(row["prompt_id"])
            for row, keep in zip(rows, mask) if keep})} for group, mask in masks.items()},
        "models": results, "fixed_calibrated": controls, "across_seeds": across_seeds,
        "uncertainty": uncertainty, "memory_coverage": coverage,
        "reload_and_matching_checks_passed": True, "elapsed_s": time.monotonic()-started,
        "timings": timings, "limitations": [
            "Matched states on fixed B16 trajectories; no closed-loop adaptive policy rollout or throughput claim.",
            "Previously inspected development assessment; calibration retention does not guarantee assessment retention.",
            "Previous block size is fixed by collection; cannot establish information value from deliberately changing previous B.",
            "Shuffled traces are sampled with replacement within previous B/A and partition, always from another prompt.",
            "Rows lacking a donor retain common inputs and labels but have empty trace in all three arms.",
            "Bootstrap fixes checkpoints, settings and donor mapping; seed results are reported individually."]}
    atomic_json(args.output/"summary.json", summary)
    atomic_json(args.output/"assessment_curves.json", all_curves)
    np.savez_compressed(args.output/"assessment_predictions.npz", **saved_arrays,
        actual=actual[assess], prompt_id=prompts, effective_memory=donors[assess] >= 0)
    lines = ["# Previous rejected-trace pre-draft predictor", "", summary["scope"], "",
        "Checkpoint and penalty selection use calibration only, targeting >=96% actual B16 accepted-token retention.", "",
        "| Model | Epoch | Cal retention | Assessment ratio | Assessment retention | Mean budget |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        q = result["operating_points"]["0.96"]
        lines.append(f"| {name} | {result['epoch']} | {q['calibration_retention']:.5f} | {q['aggregate_accept_ratio']:.5f} | {q['retention']:.5f} | {q['mean_budget']:.4f} |")
    lines += ["", "All seeds and all arms are reported. Full retention/work curves, subgroup metrics, donor identities, and paired whole-prompt intervals are saved alongside this report.",
        "", "## Memory coverage", "", "| Partition | Rows | Original memory | Effective memory | No donor |", "|---|---:|---:|---:|---:|"]
    for group, item in coverage.items():
        lines.append(f"| {group} | {item['rows']} | {item['original_memory_rows']} | {item['effective_memory_rows']} | {item['no_different_prompt_donor_rows']} |")
    lines += ["", "## Limitations", ""]+["- "+item for item in summary["limitations"]]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    atomic_json(args.output/"COMPLETE.json", {"success": True, "smoke": args.smoke,
        "binding": {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()}})
    print("COMPLETE", json.dumps({"counts": summary["counts"], "across_seeds": across_seeds,
                                  "elapsed_s": summary["elapsed_s"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--training-rows", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[913, 914, 915])
    parser.add_argument("--donor-seed", type=int, default=913)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
