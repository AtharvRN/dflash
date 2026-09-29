"""Nested data scaling with equal-update controls and frozen validation."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import accepted_lengths, risk_mask, calibrate, apply_setting, metrics, prompt_bootstrap
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.train_midverify_probe import make_features, build_probe, score, TARGETS


def batch_stream(size, batch_size, seed):
    """Shuffled epochs, carrying remainders so all updates have equal work."""
    import torch
    if min(size, batch_size) < 1:
        raise ValueError("Positive stream sizes required")
    generator = torch.Generator().manual_seed(seed)
    order, cursor = torch.randperm(size, generator=generator), 0
    while True:
        pieces, need = [], batch_size
        while need:
            if cursor == size:
                order, cursor = torch.randperm(size, generator=generator), 0
            count = min(need, size - cursor)
            pieces.append(order[cursor:cursor + count])
            cursor += count
            need -= count
        yield torch.cat(pieces)


def parameter_sha(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def calibration_bce(q, labels, mask):
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-7, 1 - 1e-7)
    losses = -(labels * np.log(q) + (1 - labels) * np.log1p(-q))
    return float((losses * mask).sum() / mask.sum())


def load_verified(path):
    complete = json.loads((path / "COMPLETE.json").read_text())
    if not complete["passed"]:
        raise ValueError("Incomplete artifact")
    for name, digest in complete["binding"].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError(f"Artifact checksum failure: {name}")
    return complete


def paired_intervals(accepted, small_k, large_k, prompts, layer, draws=1000):
    a, x, y = np.asarray(accepted), np.asarray(small_k), np.asarray(large_k)
    _, inverse = np.unique(prompts, return_inverse=True)
    ax, ay = np.minimum(a, x - 1), np.minimum(a, y - 1)
    wx, wy = (layer * 16 + (36 - layer) * x) / 36, (layer * 16 + (36 - layer) * y) / 36
    totals = np.column_stack([np.bincount(inverse, weights=v) for v in (ax, ay, x, y, wx, wy, a, np.ones(len(a)))])
    indices = np.random.default_rng(929).integers(len(totals), size=(draws, len(totals)))
    v = totals[indices].sum(1)
    ci = lambda z: np.quantile(z, [.025, .975]).tolist()
    return {"retention_delta_ci95": ci((v[:, 1] - v[:, 0]) / v[:, 6]),
            "kept_rows_delta_ci95": ci((v[:, 3] - v[:, 2]) / v[:, 7]),
            "work_rows_delta_ci95": ci((v[:, 5] - v[:, 4]) / v[:, 7]),
            "work_per_committed_delta_ci95": ci(v[:, 5] / (v[:, 1] + v[:, 7]) - v[:, 4] / (v[:, 0] + v[:, 7])),
            "scope": "larger minus smaller training size; paired whole-prompt bootstrap, fixed fitted policies",
            "resamples": draws}


def name_for(kind, layer, seed, size, budget):
    return f"{kind}_L{layer}_seed{seed}_n{size}_{budget}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu", type=int)
    p.add_argument("--cpu", action="store_true", help="Only for the bounded driver smoke")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-seconds", type=int, default=3600)
    args = p.parse_args()
    if args.output.exists() or args.max_seconds < 1 or (args.cpu and not args.smoke) or (not args.cpu and args.gpu is None):
        raise ValueError("Require fresh output and explicit authorized device")
    sizes, layers, seeds = [2000, 5000, 10000], [6, 9, 12, 18, 24], [913, 914, 915]
    budgets, max_updates, eval_every, patience = [128, 1024], 2048, 64, 8
    batch_size, kinds = 128, ["linear", "mlp128"]
    if args.smoke:
        sizes, layers, seeds = [16, 32], [6], [913]
        budgets, max_updates, eval_every, patience = [2, 4], 6, 2, 1
    cache_complete = load_verified(args.cache)
    load_verified(args.baseline)
    collection = json.loads((args.cache / "config.json").read_text())
    audit = json.loads((args.cache / "audit.json").read_text())
    reference_config = json.loads((args.baseline / "config.json").read_text())
    reference_summary = json.loads((args.baseline / "summary.json").read_text())
    if not audit["passed"] or not audit["seed_array_content_bytes_preserved"]:
        raise ValueError("Expansion did not preserve frozen pilot data")
    if collection["seed_completion_sha256"] != reference_config["collection_completion_sha256"]:
        raise ValueError("Baselines and preserved validation come from different caches")
    original_cache = Path(reference_config["cache"])
    if sha256(original_cache / "COMPLETE.json") != collection["seed_completion_sha256"]:
        raise ValueError("Original pilot identity changed")
    original_complete = json.loads((original_cache / "COMPLETE.json").read_text())
    if sha256(original_cache / "rows.json") != original_complete["binding"]["rows.json"]:
        raise ValueError("Original row metadata changed")
    rows = json.loads((args.cache / "rows.json").read_text())
    old_rows = json.loads((original_cache / "rows.json").read_text())
    train_idx = np.asarray([i for i, r in enumerate(rows) if r["group"] == "train"])
    eval_idx = np.asarray([i for i, r in enumerate(rows) if r["group"] != "train"])
    old_eval_idx = np.asarray([i for i, r in enumerate(old_rows) if r["group"] != "train"])
    if len(eval_idx) != len(old_eval_idx) or len(train_idx) < max(sizes):
        raise ValueError("Wrong dataset sizes")
    for i, j in zip(eval_idx, old_eval_idx):
        if {k: v for k, v in rows[i].items() if k != "row"} != {k: v for k, v in old_rows[j].items() if k != "row"}:
            raise ValueError("Frozen validation metadata changed")
    expected_groups = {"train": 10000, "calibration": 342, "assessment": 1416}
    if not args.smoke and dict(Counter(r["group"] for r in rows)) != expected_groups:
        raise ValueError("Unexpected full scaling dataset")
    if not args.cpu:
        used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(used.strip()) > 1024:
            raise RuntimeError("Requested GPU is occupied")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cpu" if args.cpu else "cuda"
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"sizes": sizes, "layers": layers, "seeds": seeds, "kinds": kinds, "batch_size": batch_size,
        "update_budgets": budgets, "extended_max_updates": max_updates, "calibration_every_updates": eval_every,
        "early_stopping_patience_evaluations": patience, "minimum_updates": max(budgets),
        "collection_completion_sha256": sha256(args.cache / "COMPLETE.json"),
        "baseline_completion_sha256": sha256(args.baseline / "COMPLETE.json"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {p: sha256(p) for p in ("scripts/train_midverify_scaling.py", "scripts/train_midverify_probe.py", "dflash/midverify.py", "docs/midverify_scaling_20260929.md")},
        "architecture": "unchanged: RMS(h), RMS(candidate LM-head row e), h*e -> linear or 7680-128-1 GELU/dropout.05",
        "loss": "binary cross-entropy on the at-risk prefix through first rejection; right-censored at 15",
        "optimizer": "AdamW lr3e-4 weight_decay.01, clip1; FP32, TF32 off",
        "selection": "calibration min kept rows at >=96% retention, then higher retention, then earlier step; same rule within each update budget",
        "minibatches": "full128-state shuffled stream, carry epoch tails; exactly matched exposures at equal updates; differs from original pilot partial tail batches",
        "targets": TARGETS, "torch": torch.__version__,
        "scope": "offline greedy same-candidate work proxies, not measured throughput; development assessment, not an untouched test"})
    atomic_json(args.output / "config.json", config)
    labels = np.load(args.cache / "matches.npy")
    accepted = accepted_lengths(labels)
    if not np.array_equal(accepted, np.load(args.cache / "accepted_len.npy")):
        raise ValueError("Acceptance labels misaligned")
    masks = risk_mask(accepted)
    eaccepted = accepted[eval_idx]
    egroup = np.asarray([rows[i]["group"] for i in eval_idx])
    calibration, assessment = egroup == "calibration", egroup == "assessment"
    prompts = np.asarray([rows[i]["prompt_id"] for i in eval_idx])
    if [int(calibration.sum()), int(assessment.sum())] != [342, 1416]:
        raise ValueError("Fixed validation counts changed")
    atomic_json(args.output / "evaluation_rows.json", [rows[i] for i in eval_idx])
    subsets = {str(n): {"states": n, "prompts": len({rows[i]["prompt_id"] for i in train_idx[:n]}),
                       "at_risk_positions": int(masks[train_idx[:n]].sum()),
                       "row_identity_sha256": hashlib.sha256(json.dumps([rows[i] for i in train_idx[:n]], sort_keys=True).encode()).hexdigest()} for n in sizes}
    atomic_json(args.output / "subsets.json", subsets)
    policies, all_scores, histories, initialization = {}, {}, {}, {}
    completed_models, checkpoint_names = 0, []
    started = time.monotonic()

    def evaluate(name, q, layer, metadata):
        settings = calibrate(q[calibration], eaccepted[calibration], TARGETS)
        points = {}
        for target, setting in settings.items():
            k = apply_setting(q[assessment], setting)
            points[target] = {"calibration": setting, "assessment": metrics(eaccepted[assessment], k, layer),
                "uncertainty": prompt_bootstrap(eaccepted[assessment], k, prompts[assessment], layer)}
        policies[name] = {"layer": layer, **metadata, "points": points}
        all_scores[name] = q.astype(np.float32)
        return settings

    with np.load(args.baseline / "scores.npz", allow_pickle=False) as saved:
        for name, info in reference_summary["policies"].items():
            if name.startswith(("linear_", "mlp128_")):
                continue
            evaluate(name, saved[name][old_eval_idx], info["layer"], {"kind": "fixed_reference"})
            for target in map(str, TARGETS):
                for field in ("calibration", "assessment"):
                    if policies[name]["points"][target][field] != info["points"][target][field]:
                        raise ValueError("Frozen reference policy changed")
    candidates = torch.from_numpy(np.load(args.cache / "candidate_vectors.npy")).to(device)
    for layer in layers:
        hidden = torch.from_numpy(np.load(args.cache / f"hidden_L{layer}.npy")).to(device)
        x = make_features(hidden, candidates)
        del hidden
        ex = x[eval_idx]
        cx = ex[calibration]
        for size in sizes:
            selected = train_idx[:size]
            tx = x[selected]
            ty = torch.from_numpy(labels[selected].astype(np.float32)).to(device)
            tm = torch.from_numpy(masks[selected].astype(np.float32)).to(device)
            for kind in kinds:
                for seed in seeds:
                    torch.manual_seed(seed)
                    model = build_probe(kind).to(device)
                    init_key = f"{kind}_L{layer}_seed{seed}"
                    initial = parameter_sha(model.state_dict())
                    if init_key in initialization and initialization[init_key] != initial:
                        raise ValueError("Across-size initialization differs")
                    initialization[init_key] = initial
                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
                    batches = batch_stream(size, batch_size, seed)
                    best, best_key, history, snapshots = None, None, [], {}
                    stale, total, denominator = 0, 0., 0.
                    run_name = f"{kind}_L{layer}_seed{seed}_n{size}"
                    for step in range(1, max_updates + 1):
                        if time.monotonic() - started > args.max_seconds:
                            raise RuntimeError("Training time bound reached; partial evidence preserved")
                        model.train()
                        batch = next(batches).to(device)
                        optimizer.zero_grad(set_to_none=True)
                        z = model(tx[batch]).squeeze(-1)
                        losses = F.binary_cross_entropy_with_logits(z, ty[batch], reduction="none")
                        loss = (losses * tm[batch]).sum() / tm[batch].sum()
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        total += float((losses.detach() * tm[batch]).sum())
                        denominator += float(tm[batch].sum())
                        if step % eval_every:
                            continue
                        q = score(model, cx)
                        point = calibrate(q, eaccepted[calibration], (.96,))["0.96"]
                        key = (point["mean_kept_rows"], -point["retention"], step)
                        if best_key is None or key < best_key:
                            best, best_key = copy.deepcopy(model.state_dict()), key
                            stale = 0
                        else:
                            stale += 1
                        history.append({"update": step, "state_exposures": step * batch_size,
                            "equivalent_epochs": step * batch_size / size, "train_window_risk_bce": total / denominator,
                            "calibration_risk_bce": calibration_bce(q, labels[eval_idx][calibration], masks[eval_idx][calibration]),
                            "calibration_96": point, "best_update": best_key[2]})
                        total, denominator = 0., 0.
                        if step in budgets:
                            snapshots[f"updates_{step}"] = (copy.deepcopy(best), best_key[2])
                        if step >= max(budgets) and stale >= patience:
                            break
                    snapshots["extended"] = (best, best_key[2])
                    histories[run_name] = history
                    # All model/checkpoint selection above uses calibration only.
                    for budget, (weights, chosen_step) in snapshots.items():
                        model.load_state_dict(weights)
                        q = score(model, ex)
                        name = name_for(kind, layer, seed, size, budget)
                        metadata = {"kind": kind, "seed": seed, "n_train": size, "budget": budget,
                                    "selected_update": chosen_step, "trained_updates": step,
                                    "initial_parameter_sha256": initial}
                        settings = evaluate(name, q, layer, metadata)
                        path = args.output / f"{name}.pt"
                        state = {k: v.cpu() for k, v in weights.items()}
                        torch.save({"model": state, **metadata, "layer": layer, "config": config,
                                    "settings": settings, "parameter_sha256": parameter_sha(state)}, path)
                        reloaded = torch.load(path, map_location=device, weights_only=False)
                        model.load_state_dict(reloaded["model"])
                        q_again = score(model, ex)
                        if not np.array_equal(q, q_again):
                            raise ValueError("Reloaded checkpoint changed same-shape predictions")
                        policies[name]["checkpoint_reload_exact"] = True
                        checkpoint_names.append(path.name)
                    completed_models += 1
                    progress = {"models_finished": completed_models, "models_planned": len(layers) * len(sizes) * len(kinds) * len(seeds),
                                "last": run_name, "last_trained_updates": step, "last_best_update": best_key[2],
                                "elapsed_s": time.monotonic() - started}
                    atomic_json(args.output / "progress.json", progress)
                    atomic_json(args.output / "histories.json", histories)
                    print(json.dumps(progress), flush=True)
                    del model, optimizer, best, snapshots, weights, reloaded
            del tx, ty, tm
        del x, ex, cx
        if not args.cpu:
            torch.cuda.empty_cache()
    paired = {}
    for layer in layers:
        for kind in kinds:
            for seed in seeds:
                for budget in [*[f"updates_{v}" for v in budgets], "extended"]:
                    small = name_for(kind, layer, seed, min(sizes), budget)
                    large = name_for(kind, layer, seed, max(sizes), budget)
                    k_small = apply_setting(all_scores[small][assessment], policies[small]["points"]["0.96"]["calibration"])
                    k_large = apply_setting(all_scores[large][assessment], policies[large]["points"]["0.96"]["calibration"])
                    paired[large + "_minus_" + str(min(sizes))] = paired_intervals(eaccepted[assessment], k_small, k_large, prompts[assessment], layer)
    summary = {"scope": config["scope"], "config": config, "subsets": subsets, "policies": policies,
               "paired_largest_minus_smallest_at_calibration96": paired, "models_trained": completed_models,
               "checkpoint_policies": len(checkpoint_names), "initialization_sha256": initialization,
               "frozen_validation_bytes_verified": True, "baseline_metrics_exactly_unchanged": True,
               "assessment_b16_mean_accepted": float(eaccepted[assessment].mean()),
               "elapsed_s": time.monotonic() - started}
    atomic_json(args.output / "summary.json", summary)
    np.savez(args.output / "scores.npz", **all_scores)
    write_report(args.output, summary, histories)
    names = ["config.json", "evaluation_rows.json", "subsets.json", "histories.json", "summary.json", "scores.npz",
             "report.md", "data_scaling.png", "optimization_L6.png", *checkpoint_names]
    atomic_json(args.output / "COMPLETE.json", {"passed": True, "models_trained": completed_models,
                "binding": {name: sha256(args.output / name) for name in names}})
    print("COMPLETE", json.dumps({"models": completed_models, "policies": len(checkpoint_names),
                                  "elapsed_s": time.monotonic() - started}), flush=True)


def write_report(output, summary, histories):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    c, policies = summary["config"], summary["policies"]
    main_budget = f"updates_{max(c['update_budgets'])}"
    lines = ["# Mid-verification data/update scaling", "", summary["scope"], "",
             "Equal full128-state updates; all original pilot and validation feature bytes preserved. All checkpoints/thresholds selected on calibration only.", "",
             "## Primary equal-update MLP comparison", "",
             f"Budget: {max(c['update_budgets'])} optimizer updates. Three-seed means (one seed in smoke). Calibration target96%; actual assessment retentions below.", "",
             "| Layer | Train states | Retention | Kept rows | Full-depth-row proxy |", "| --- | ---: | ---: | ---: | ---: |"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for layer in c["layers"]:
        means, bounds = [], []
        for size in c["sizes"]:
            points = [policies[name_for("mlp128", layer, seed, size, main_budget)]["points"]["0.96"]["assessment"] for seed in c["seeds"]]
            v = {field: float(np.mean([p[field] for p in points])) for field in ("retention", "mean_kept_rows", "equivalent_full_depth_rows")}
            means.append(v)
            bounds.append({field: (min(p[field] for p in points), max(p[field] for p in points)) for field in v})
            lines.append(f"| {layer} | {size} | {v['retention']:.5f} | {v['mean_kept_rows']:.3f} | {v['equivalent_full_depth_rows']:.3f} |")
        for ax, field in zip(axes, ("equivalent_full_depth_rows", "retention")):
            plotted = ax.plot(c["sizes"], [m[field] for m in means], "o-", label=f"L{layer}")
            ax.fill_between(c["sizes"], [b[field][0] for b in bounds], [b[field][1] for b in bounds], color=plotted[0].get_color(), alpha=.12)
    control = policies["draft_candidate_logprob"]["points"]["0.96"]["assessment"]
    for ax, field, ylabel in zip(axes, ("equivalent_full_depth_rows", "retention"), ("Full-depth-equivalent rows (NOT latency)", "Assessment accepted-token retention")):
        ax.axhline(control[field], color="black", linestyle="--", label="Frozen draft-confidence control")
        ax.set(xlabel="Unique training cycle states", ylabel=ylabel, xticks=c["sizes"])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"MLP scaling at {max(c['update_budgets'])} updates; bands are seed min/max, not confidence intervals")
    fig.tight_layout()
    fig.savefig(output / "data_scaling.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, size in enumerate(c["sizes"]):
        color = plt.get_cmap("tab10")(i)
        for j, seed in enumerate(c["seeds"]):
            h = histories[f"mlp128_L6_seed{seed}_n{size}"]
            updates = [v["update"] for v in h]
            label = f"n={size}" if j == 0 else None
            axes[0].plot(updates, [v["calibration_96"]["mean_kept_rows"] for v in h], color=color, alpha=.6, label=label)
            axes[1].plot(updates, [v["train_window_risk_bce"] for v in h], color=color, alpha=.5)
            axes[1].plot(updates, [v["calibration_risk_bce"] for v in h], color=color, linestyle="--", alpha=.7, label=label)
    axes[0].set(ylabel="Calibration retained rows at >=96% retention")
    axes[1].set(ylabel="Masked BCE: train solid / calibration dashed")
    for ax in axes:
        for budget in c["update_budgets"]:
            ax.axvline(budget, linestyle=":", color="gray", alpha=.5)
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Layer-6 MLP optimization; every seed shown")
    fig.tight_layout()
    fig.savefig(output / "optimization_L6.png", dpi=150)
    plt.close(fig)
    lines.extend(["", "## Frozen draft-confidence control", "", f"Retention {control['retention']:.5f}; kept/work rows {control['mean_kept_rows']:.3f}.", "",
                  "See summary.json for both architectures, all seeds, three update budgets, five retention settings, bootstrap intervals and paired largest-minus-smallest differences.", ""])
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
