"""Bounded confidence-first, L6 acceptance-correction ablation."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import accepted_lengths, risk_mask, calibrate, apply_setting
from dflash.midverify_cascade import apply_cascade, cascade_metrics, calibrate_cascade, paired_bootstrap
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.train_midverify_probe import make_features, score
from scripts.train_midverify_scaling import batch_stream, calibration_bce, load_verified, parameter_sha

CASES = ("confidence_only", "candidate_confidence", "target_candidate_confidence")
TARGETS = (.96, .99)


def fit_confidence_normalizer(training_stats, training_mask):
    values = np.asarray(training_stats, dtype=np.float64)[np.asarray(training_mask, dtype=bool)]
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid training-only confidence statistics")
    return values.mean(0), np.maximum(values.std(0), 1e-6)


def case_inputs(base, confidence, case):
    import torch
    width = base.shape[-1] // 3
    if base.shape[-1] != 3 * width or confidence.shape != (*base.shape[:-1], 3):
        raise ValueError("Invalid feature shape")
    if case == "confidence_only":
        return confidence
    if case == "candidate_confidence":
        return torch.cat([base[..., width:2 * width], confidence], -1)
    if case == "target_candidate_confidence":
        return torch.cat([base, confidence], -1)
    raise ValueError("Unknown feature ablation")


def build_probe(input_width):
    from torch import nn
    return nn.Sequential(nn.Linear(input_width, 128), nn.GELU(), nn.Dropout(.05), nn.Linear(128, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "previous", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu", type=int)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-seconds", type=int, default=1800)
    args = p.parse_args()
    if args.output.exists() or args.max_seconds < 1 or (args.cpu and not args.smoke) or (not args.cpu and args.gpu is None):
        raise ValueError("Require fresh output and explicit authorized device")
    load_verified(args.cache)
    load_verified(args.previous)
    prior_config = json.loads((args.previous / "config.json").read_text())
    prior_summary = json.loads((args.previous / "summary.json").read_text())
    if prior_config["collection_completion_sha256"] != sha256(args.cache / "COMPLETE.json"):
        raise ValueError("Prior controls and current cache differ")
    rows = json.loads((args.cache / "rows.json").read_text())
    if dict(Counter(r["group"] for r in rows)) != {"train": 9999, "calibration": 342, "assessment": 1416}:
        raise ValueError("Frozen groups changed")
    group = np.array([r["group"] for r in rows])
    train, evaluation = np.flatnonzero(group == "train"), np.flatnonzero(group != "train")
    eval_rows = [rows[i] for i in evaluation]
    if eval_rows != json.loads((args.previous / "evaluation_rows.json").read_text()):
        raise ValueError("Frozen evaluation row order or labels changed")
    groups = {g: {r["prompt_id"] for r in rows if r["group"] == g} for g in set(group)}
    if any(groups[a] & groups[b] for a in groups for b in groups if a < b):
        raise ValueError("Prompt leakage")
    labels = np.load(args.cache / "matches.npy")
    accepted = accepted_lengths(labels)
    if not np.array_equal(accepted, np.load(args.cache / "accepted_len.npy")):
        raise ValueError("Labels are misaligned")
    masks = risk_mask(accepted)
    stats = np.load(args.cache / "draft_stats.npy")
    if stats.shape != (len(rows), 15, 3) or not np.isfinite(stats).all():
        raise ValueError("Invalid confidence cache")
    seeds, updates, every = [913, 914, 915], 1024, 64
    if args.smoke:
        seeds, updates, every, train = [913], 4, 2, train[:128]
    mean, std = fit_confidence_normalizer(stats[train], masks[train])
    normalized_stats = ((stats.astype(np.float64) - mean) / std).astype(np.float32)
    eaccepted, elabels, emasks = accepted[evaluation], labels[evaluation], masks[evaluation]
    ca = group[evaluation] == "calibration"
    ass = group[evaluation] == "assessment"
    prompts = np.array([r["prompt_id"] for r in eval_rows])
    confidence = stats[evaluation, :, 0]
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
    config.update({"seeds": seeds, "cases": CASES, "updates": updates, "calibration_every": every,
        "targets": TARGETS, "train_states": len(train), "train_prompts": len({rows[i]["prompt_id"] for i in train}),
        "at_risk_training_positions": int(masks[train].sum()), "layer": 6, "batch_size_states": 128,
        "architecture": "input -> Linear128 -> GELU -> Dropout.05 -> Linear1; hidden width matched, parameter counts NOT matched",
        "feature_widths": {"confidence_only": 3, "candidate_confidence": 2563, "target_candidate_confidence": 7683},
        "confidence_mean": mean.tolist(), "confidence_std": std.tolist(),
        "normalization_fit": "training at-risk positions only; no clipping or fitted validation statistics",
        "optimizer": "AdamW3e-4, weight_decay.01, clip1; FP32/TF32 off", "loss": "same first-rejection masked BCE",
        "selection": "separate best checkpoint per retention target; exact joint calibration min work, higher retained acceptance, earlier update",
        "source_hashes": {name: sha256(name) for name in [__file__, "dflash/midverify_cascade.py", "dflash/midverify.py",
            "scripts/train_midverify_probe.py", "scripts/train_midverify_scaling.py", "docs/midverify_cascade_20260929.md"]},
        "cache_completion_sha256": sha256(args.cache / "COMPLETE.json"),
        "prior_completion_sha256": sha256(args.previous / "COMPLETE.json"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "scope": "offline same-candidate greedy cascade; proxy work, NOT latency; cached full-width L6 features, no segmented-forward parity claim; development assessment"})
    atomic_json(args.output / "config.json", config)
    atomic_json(args.output / "evaluation_rows.json", eval_rows)
    policies, scores_saved, histories, checkpoints = {}, {}, {}, []
    started = time.monotonic()

    def record(name, values, setting, case, target, seed=None, extra=None):
        front, end = apply_cascade(confidence[ass], values[ass], setting)
        policies[name] = {"case": case, "target": target, "seed": seed, "setting": setting,
            "calibration": setting["calibration"], "assessment": cascade_metrics(eaccepted[ass], front, end, setting["layer"]),
            **(extra or {})}
        scores_saved[name] = values.astype(np.float32)

    with np.load(args.previous / "scores.npz", allow_pickle=False) as old:
        if not np.array_equal(confidence, old["draft_candidate_logprob"]):
            raise ValueError("Frozen confidence scores changed")
        settings = calibrate(confidence[ca], eaccepted[ca], TARGETS)
        for target, point in settings.items():
            if point != prior_summary["policies"]["draft_candidate_logprob"]["points"][target]["calibration"]:
                raise ValueError("Frozen confidence calibration changed")
            k = apply_setting(confidence[ca], point)
            setting = {"stage0_threshold": point["threshold"], "stage1_threshold": None, "layer": 0,
                "calibration": cascade_metrics(eaccepted[ca], k, k, 0)}
            name = f"raw_confidence_r{target}"
            record(name, confidence, setting, "raw_confidence", float(target))
            expected = prior_summary["policies"]["draft_candidate_logprob"]["points"][target]["assessment"]
            for key in ("retention", "mean_kept_rows", "equivalent_full_depth_rows", "aggregate_accept_ratio"):
                if policies[name]["assessment"][key] != expected[key]:
                    raise ValueError("Frozen baseline assessment changed")
        for seed in seeds:
            key = f"mlp128_L6_seed{seed}_n9999_updates_1024"
            q = old[key]
            settings = calibrate_cascade(confidence[ca], q[ca], eaccepted[ca], TARGETS, 6)
            for target, setting in settings.items():
                record(f"frozen_target_only_seed{seed}_r{target}", q, setting, "frozen_target_only", float(target), seed,
                       {"prior_policy": key, "weights_changed": False})
    candidate = torch.from_numpy(np.load(args.cache / "candidate_vectors.npy")).to(device)
    hidden = torch.from_numpy(np.load(args.cache / "hidden_L6.npy")).to(device)
    base = make_features(hidden, candidate)
    del hidden, candidate
    conf = torch.from_numpy(normalized_stats).to(device)
    ty = torch.from_numpy(labels[train].astype(np.float32)).to(device)
    tm = torch.from_numpy(masks[train].astype(np.float32)).to(device)
    finished = 0
    for case in CASES:
        x = case_inputs(base, conf, case)
        tx, ex = x[train], x[evaluation]
        cx, ax = ex[ca], ex[ass]
        layer = 6 if case == "target_candidate_confidence" else 0
        for seed in seeds:
            torch.manual_seed(seed)
            model = build_probe(x.shape[-1]).to(device)
            initial = parameter_sha(model.state_dict())
            params = sum(v.numel() for v in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            stream = batch_stream(len(train), 128, seed)
            best, best_key, history = {}, {}, []
            numerator, denominator = 0., 0.
            for step in range(1, updates + 1):
                if time.monotonic() - started > args.max_seconds:
                    raise RuntimeError("Bounded runtime exceeded; partial evidence preserved")
                model.train()
                b = next(stream).to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(tx[b]).squeeze(-1)
                losses = F.binary_cross_entropy_with_logits(logits, ty[b], reduction="none")
                loss = (losses * tm[b]).sum() / tm[b].sum()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                numerator += float((losses.detach() * tm[b]).sum())
                denominator += float(tm[b].sum())
                if step % every:
                    continue
                qc = score(model, cx)
                settings = calibrate_cascade(confidence[ca], qc, eaccepted[ca], TARGETS, layer)
                for target, setting in settings.items():
                    key = (setting["selection_work_numerator"], -setting["selection_accepted_total"], step)
                    if target not in best or key < best_key[target]:
                        best[target] = (copy.deepcopy(model.state_dict()), copy.deepcopy(setting), qc.copy())
                        best_key[target] = key
                history.append({"update": step, "train_window_risk_bce": numerator / denominator,
                    "calibration_risk_bce": calibration_bce(qc, elabels[ca], emasks[ca]),
                    "settings": settings, "best_updates": {t: v[2] for t, v in best_key.items()}})
                numerator, denominator = 0., 0.
            # Assessment is touched only after all checkpoint/threshold selection.
            for target, (weights, setting, selected_cal_scores) in best.items():
                model.load_state_dict(weights)
                q = np.empty((len(evaluation), 15), dtype=np.float32)
                q[ca], q[ass] = score(model, cx), score(model, ax)
                if not np.array_equal(q[ca], selected_cal_scores):
                    raise ValueError("Selected calibration scores changed")
                name = f"{case}_seed{seed}_r{target}"
                extra = {"selected_update": best_key[target][2], "parameters": params, "input_width": x.shape[-1],
                    "initial_parameter_sha256": initial,
                    "calibration_risk_bce": calibration_bce(q[ca], elabels[ca], emasks[ca]),
                    "assessment_risk_bce": calibration_bce(q[ass], elabels[ass], emasks[ass])}
                record(name, q, setting, case, float(target), seed, extra)
                path = args.output / f"{name}.pt"
                state = {k: v.cpu() for k, v in weights.items()}
                torch.save({"model": state, "case": case, "seed": seed, "target": target, "setting": setting,
                            "config": config, **extra, "parameter_sha256": parameter_sha(state)}, path)
                loaded = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(loaded["model"])
                if not np.array_equal(q[ca], score(model, cx)) or not np.array_equal(q[ass], score(model, ax)):
                    raise ValueError("Checkpoint reload predictions changed")
                policies[name]["checkpoint_reload_exact"] = True
                checkpoints.append(path.name)
            histories[f"{case}_seed{seed}"] = history
            finished += 1
            progress = {"models_finished": finished, "models_planned": len(CASES) * len(seeds), "last": case,
                "seed": seed, "selected_updates": {t: v[2] for t, v in best_key.items()}, "elapsed_s": time.monotonic() - started}
            atomic_json(args.output / "progress.json", progress)
            atomic_json(args.output / "histories.json", histories)
            print(json.dumps(progress), flush=True)
            del model, opt, best, weights, loaded
        del x, tx, ex, cx, ax
    references, comparisons = {}, {}
    for target in map(str, TARGETS):
        for seed in seeds:
            choices = [f"raw_confidence_r{target}", *[f"{case}_seed{seed}_r{target}" for case in CASES[:2]]]
            chosen = min(choices, key=lambda n: (policies[n]["calibration"]["equivalent_full_depth_rows"],
                                                -policies[n]["calibration"]["retention"], n))
            references[f"seed{seed}_r{target}"] = chosen
    for name, policy in policies.items():
        if policy["case"] == "raw_confidence":
            continue
        target, seed = str(policy["target"]), policy["seed"]
        front, end = apply_cascade(confidence[ass], scores_saved[name][ass], policy["setting"])
        references_to_use = {"raw_confidence": f"raw_confidence_r{target}",
                             "calibration_chosen_target_free": references[f"seed{seed}_r{target}"]}
        comparisons[name] = {}
        for label, ref_name in references_to_use.items():
            ref = policies[ref_name]
            rf, rk = apply_cascade(confidence[ass], scores_saved[ref_name][ass], ref["setting"])
            comparisons[name][label] = {"reference": ref_name, **paired_bootstrap(eaccepted[ass], front, end, rf, rk,
                prompts[ass], policy["setting"]["layer"], ref["setting"]["layer"], draws=100 if args.smoke else 2000)}
    oracles = {}
    for target in map(str, TARGETS):
        ref = policies[f"raw_confidence_r{target}"]
        front, _ = apply_cascade(confidence[ass], confidence[ass], ref["setting"])
        end = np.minimum(eaccepted[ass], front - 1) + 1
        oracles[target] = cascade_metrics(eaccepted[ass], front, end, 6)
    summary = {"config": config, "policies": policies, "comparisons": comparisons,
        "calibration_chosen_target_free_references": references, "perfect_second_stage_after_raw_confidence": oracles,
        "models_trained": finished, "checkpoints": len(checkpoints), "frozen_confidence_baseline_exact": True,
        "assessment_b16_mean_accepted": float(eaccepted[ass].mean()), "elapsed_s": time.monotonic() - started}
    atomic_json(args.output / "summary.json", summary)
    np.savez(args.output / "scores.npz", **scores_saved)
    write_report(args.output, summary, histories)
    names = ["config.json", "evaluation_rows.json", "histories.json", "summary.json", "scores.npz", "report.md",
             "cascade_comparison.png", "learning_curves.png", *checkpoints]
    atomic_json(args.output / "COMPLETE.json", {"passed": True, "models_trained": finished,
        "checkpoints": len(checkpoints), "binding": {name: sha256(args.output / name) for name in names}})
    print("COMPLETE", json.dumps({"models": finished, "checkpoints": len(checkpoints), "elapsed_s": time.monotonic() - started}), flush=True)


def write_report(output, summary, histories):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    ps = summary["policies"]
    names = ["raw_confidence", "confidence_only", "candidate_confidence", "frozen_target_only", "target_candidate_confidence"]
    labels = ["Raw confidence", "Learned confidence", "Candidate + confidence", "Confidence → frozen L6", "Confidence → fused L6"]
    lines = ["# Confidence-first / L6 cascade", "", summary["config"]["scope"], "",
        "All checkpoint and joint-threshold choices use calibration only. Seed means below; actual assessment retention is NOT constrained to equal calibration retention.", "",
        "| Calibration target | Policy | Assessment retention | Front rows | Final rows | Work proxy | Work/committed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, target in zip(axes, TARGETS):
        for i, (case, label) in enumerate(zip(names, labels)):
            values = [v["assessment"] for v in ps.values() if v["case"] == case and v["target"] == target]
            mean = {k: float(np.mean([v[k] for v in values])) for k in
                    ("retention", "mean_front_rows", "mean_kept_rows", "equivalent_full_depth_rows", "row_layer_work_per_committed_token")}
            lines.append(f"| {target:.0%} | {label} | {mean['retention']:.5f} | {mean['mean_front_rows']:.3f} | {mean['mean_kept_rows']:.3f} | {mean['equivalent_full_depth_rows']:.3f} | {mean['row_layer_work_per_committed_token']:.4f} |")
            color = plt.get_cmap("tab10")(i)
            ax.scatter([v["retention"] for v in values], [v["equivalent_full_depth_rows"] for v in values], color=color, s=22, alpha=.6)
            ax.scatter(mean["retention"], mean["equivalent_full_depth_rows"], color=color, marker="D", s=60, label=label)
        ax.axvline(target, linestyle=":", color="gray", alpha=.6)
        ax.set(title=f"Calibrated for {target:.0%} retention", xlabel="Actual assessment accepted-token retention",
               ylabel="Full-depth-equivalent rows (NOT latency)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Cascade comparison: dots are individual seeds, diamonds are means")
    fig.tight_layout()
    fig.savefig(output / "cascade_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, case in enumerate(CASES):
        color = plt.get_cmap("tab10")(i + 1)
        for j, seed in enumerate(summary["config"]["seeds"]):
            h = histories[f"{case}_seed{seed}"]
            x = [r["update"] for r in h]
            axes[0].plot(x, [r["settings"]["0.96"]["calibration"]["equivalent_full_depth_rows"] for r in h], color=color, alpha=.6, label=case if j == 0 else None)
            axes[1].plot(x, [r["train_window_risk_bce"] for r in h], color=color, alpha=.5)
            axes[1].plot(x, [r["calibration_risk_bce"] for r in h], color=color, linestyle="--", alpha=.7, label=case if j == 0 else None)
    axes[0].set(ylabel="Calibration work at >=96% retention")
    axes[1].set(ylabel="Masked BCE: train solid, calibration dashed")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    lines.extend(["", "Target-free learned controls charge final rows only; L6 policies charge [6*front+30*final]/36. Full B16 drafting and all systems overhead remain outside this proxy.", "",
                  "See summary.json for all seed/checkpoint/threshold choices, paired prompt-bootstrap intervals, and the calibration-chosen strongest target-free reference.", ""])
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
