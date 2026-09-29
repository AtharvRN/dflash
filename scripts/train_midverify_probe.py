"""Small candidate-conditioned intermediate probes; calibration-only selection."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import (accepted_lengths, risk_mask, calibrate, apply_setting,
                             metrics, prompt_bootstrap)
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json

TARGETS = (.90, .95, .96, .98, .99)


def make_features(hidden, candidate):
    import torch
    # RMS-normalize separately, with no fitted assessment statistics.
    h, e = hidden.float(), candidate.float()
    h = h / h.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    e = e / e.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return torch.cat([h, e, h * e], dim=-1)


def build_probe(kind, width=2560):
    from torch import nn
    if kind == "linear":
        return nn.Linear(width * 3, 1)
    if kind == "mlp128":
        return nn.Sequential(nn.Linear(width * 3, 128), nn.GELU(), nn.Dropout(.05), nn.Linear(128, 1))
    raise ValueError(kind)


def score(model, x, batch=128):
    import torch
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(x[s:s + batch]).squeeze(-1).sigmoid().cpu().numpy()
                               for s in range(0, len(x), batch)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[913, 914, 915])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128, help="States, each containing fifteen positions")
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists() or min(args.epochs, args.batch_size, args.max_seconds) < 1:
        raise ValueError("Fresh output and positive limits required")
    complete = json.loads((args.cache / "COMPLETE.json").read_text())
    for name, expected in complete["binding"].items():
        if Path(name).name != name or sha256(args.cache / name) != expected:
            raise ValueError("Cache completion/hash audit failed")
    collection = json.loads((args.cache / "config.json").read_text())
    audit = json.loads((args.cache / "audit.json").read_text())
    if not complete["passed"] or not audit["passed"] or collection["smoke"] != args.smoke:
        raise ValueError("Wrong or failed dataset")
    rows = json.loads((args.cache / "rows.json").read_text())
    groups = np.array([r["group"] for r in rows])
    take = {g: np.flatnonzero(groups == g) for g in ("train", "calibration", "assessment")}
    if not args.smoke and [len(take[g]) for g in take] != [2000, 342, 1416]:
        raise ValueError("Unexpected bounded-pilot split")
    used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError("Requested GPU is occupied")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"collection_completion_sha256": sha256(args.cache / "COMPLETE.json"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(__file__), "module_sha256": sha256("dflash/midverify.py"),
        "input": "RMS-normalized predecessor h^L, candidate LM-head weight vector e, and elementwise h*e; 7680 dims",
        "models": "linear (7681 params); MLP 7680-128-1 with GELU/dropout0.05 (983297 params)",
        "output": "candidate match logit at each position; stop prefix at first score below calibration threshold",
        "loss": "binary cross entropy on at-risk positions through first rejection; no failure invented at A=15",
        "optimizer": "AdamW lr=3e-4 weight_decay=.01; grad_norm_clip=1; FP32, TF32 off",
        "selection": "per layer/model/seed, checkpoint minimizes calibration kept rows at >=96% retention; ties higher retention then earlier epoch",
        "targets": TARGETS, "assessment": "never used for checkpoint, threshold, feature, or layer selection",
        "scope": "candidate-preserving offline prefixes; row-layer proxies NOT measured speedups"})
    atomic_json(args.output / "config.json", config)
    matches = np.load(args.cache / "matches.npy")
    accepted = accepted_lengths(matches)
    if not np.array_equal(accepted, np.load(args.cache / "accepted_len.npy")):
        raise ValueError("Saved label/match alignment failure")
    masks = risk_mask(accepted)
    prompt = np.array([r["prompt_id"] for r in rows])
    ca, ass, tr = take["calibration"], take["assessment"], take["train"]
    candidate = torch.from_numpy(np.load(args.cache / "candidate_vectors.npy")).cuda()
    all_scores, summaries, histories, checkpoints = {}, {}, {}, []
    started = time.monotonic()

    def assess(name, values, layer):
        settings = calibrate(values[ca], accepted[ca], TARGETS)
        points = {}
        for target, setting in settings.items():
            k = apply_setting(values[ass], setting)
            points[target] = {"calibration": setting, "assessment": metrics(accepted[ass], k, layer),
                             "uncertainty": prompt_bootstrap(accepted[ass], k, prompt[ass], layer)}
        summaries[name] = {"layer": layer, "points": points}
        all_scores[name] = values.astype(np.float32)

    # Replayed draft confidence is scored on the exact same saved candidates.
    confidence = np.load(args.cache / "draft_stats.npy")
    assess("draft_candidate_logprob", confidence[:, :, 0], 0)
    assess("draft_negative_entropy", -confidence[:, :, 1], 0)
    for layer in collection["layers"]:
        assess(f"lens_topk_L{layer}", -np.load(args.cache / f"lens_rank_L{layer}.npy").astype(float), layer)
        assess(f"lens_margin_L{layer}", np.load(args.cache / f"lens_margin_L{layer}.npy"), layer)
        hidden = torch.from_numpy(np.load(args.cache / f"hidden_L{layer}.npy")).cuda()
        x = make_features(hidden, candidate)
        del hidden
        tx, cx = x[tr], x[ca]
        ty = torch.from_numpy(matches[tr].astype(np.float32)).cuda()
        tm = torch.from_numpy(masks[tr].astype(np.float32)).cuda()
        for kind in ("linear", "mlp128"):
            for seed in args.seeds:
                if time.monotonic() - started > args.max_seconds:
                    raise RuntimeError("Training time bound reached; partial outputs preserved")
                torch.manual_seed(seed)
                model = build_probe(kind).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
                generator = torch.Generator().manual_seed(seed)
                best, best_key, history = None, None, []
                name = f"{kind}_L{layer}_seed{seed}"
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    order = torch.randperm(len(tr), generator=generator)
                    total, denominator = 0., 0.
                    for start in range(0, len(tr), args.batch_size):
                        batch = order[start:start + args.batch_size].cuda()
                        optimizer.zero_grad(set_to_none=True)
                        z = model(tx[batch]).squeeze(-1)
                        losses = F.binary_cross_entropy_with_logits(z, ty[batch], reduction="none")
                        loss = (losses * tm[batch]).sum() / tm[batch].sum()
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite probe loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        total += float((losses.detach() * tm[batch]).sum())
                        denominator += float(tm[batch].sum())
                    values = score(model, cx)
                    point = calibrate(values, accepted[ca], (.96,))["0.96"]
                    key = (point["mean_kept_rows"], -point["retention"], epoch)
                    history.append({"epoch": epoch, "train_risk_bce": total / denominator, "calibration_96": point})
                    if best_key is None or key < best_key:
                        best, best_key = copy.deepcopy(model.state_dict()), key
                model.load_state_dict(best)
                # Assessment first enters here, after this model's selection is frozen.
                values = score(model, x)
                assess(name, values, layer)
                histories[name] = history
                checkpoint = args.output / f"{name}.pt"
                torch.save({"model": {k: v.cpu() for k, v in best.items()}, "layer": layer, "kind": kind,
                            "seed": seed, "selected_epoch": best_key[2], "input_width": 2560,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "config": config, "calibration_settings": {k: v["calibration"] for k, v in summaries[name]["points"].items()}}, checkpoint)
                checkpoints.append(checkpoint.name)
                summaries[name]["selected_epoch"] = best_key[2]
                print(json.dumps({"finished": name, "epoch": best_key[2],
                                  "calibration_mean_kept_96": best_key[0], "elapsed_s": time.monotonic() - started}), flush=True)
                atomic_json(args.output / "progress.json", {"models_finished": len(checkpoints), "last": name,
                                                            "elapsed_s": time.monotonic() - started})
                del model, optimizer, best
        del x, tx, cx, ty, tm
        torch.cuda.empty_cache()
    fixed = {str(k): metrics(accepted[ass], np.full(len(ass), k)) for k in range(1, 17)}
    oracle = {str(layer): metrics(accepted[ass], accepted[ass] + 1, layer) for layer in [0, *collection["layers"]]}
    # Global choice for each target uses calibration proxy only, including depth cost.
    selected = {}
    for retention in map(str, TARGETS):
        candidates = []
        for name, info in summaries.items():
            if not (name.startswith("linear_") or name.startswith("mlp128_")):
                continue
            cal = info["points"][retention]["calibration"]
            L = info["layer"]
            work = (L * 16 + (36 - L) * cal["mean_kept_rows"]) / 36
            candidates.append((work, name))
        selected[retention] = min(candidates)[1]
    summary = {"scope": config["scope"], "groups": {g: len(take[g]) for g in take}, "collection_audit": audit,
               "assessment_b16_mean_accepted": float(accepted[ass].mean()),
               "policies": summaries, "fixed_candidate_truncation": fixed,
               "candidate_preserving_100pct_oracle_by_layer": oracle,
               "calibration_selected_learned_policy_by_retention": selected,
               "elapsed_s": time.monotonic() - started,
               "limitations": ["small development assessment, not an untouched final test",
                               "draft confidence is recomputed at saved prefixes, not original cached logits",
                               "three seeds reported separately; no assessment-based winner selection",
                               "row-layer proxies omit nonlinear kernels, probe, compaction, KV and graph overhead"]}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "histories.json", histories)
    np.savez(args.output / "scores.npz", **all_scores)
    plot_report(args.output, summary)
    names = ["config.json", "summary.json", "histories.json", "scores.npz", "report.md",
             "retention_vs_work.png", "retention_vs_kept.png", *checkpoints]
    atomic_json(args.output / "COMPLETE.json", {"passed": True, "binding": {n: sha256(args.output / n) for n in names}})
    print("COMPLETE", json.dumps({"models": len(checkpoints), "elapsed_s": time.monotonic() - started}), flush=True)


def plot_report(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    policies = summary["policies"]
    for field, filename, xlabel in (("equivalent_full_depth_rows", "retention_vs_work.png", "Equivalent full-depth rows (proxy, NOT latency)"),
                                    ("mean_kept_rows", "retention_vs_kept.png", "Mean retained target rows, including anchor")):
        fig, ax = plt.subplots(figsize=(9, 6))
        for name in ("draft_candidate_logprob", "draft_negative_entropy"):
            points = list(policies[name]["points"].values())
            ax.plot([p["assessment"][field] for p in points], [p["assessment"]["retention"] for p in points], "o--", label=name)
        colors = plt.get_cmap("tab10")
        for i, layer in enumerate((6, 9, 12, 18, 24)):
            names = [n for n in policies if n.startswith(f"mlp128_L{layer}_")]
            if not names:
                continue
            x = [[policies[n]["points"][str(t)]["assessment"][field] for t in TARGETS] for n in names]
            y = [[policies[n]["points"][str(t)]["assessment"]["retention"] for t in TARGETS] for n in names]
            ax.plot(np.mean(x, axis=0), np.mean(y, axis=0), "o-", color=colors(i), label=f"MLP L{layer} (seed mean)")
        fixed = list(summary["fixed_candidate_truncation"].values())
        ax.plot([p[field] for p in fixed], [p["retention"] for p in fixed], color="gray", label="Fixed trim of B16 candidates")
        ax.set(xlabel=xlabel, ylabel="Assessment accepted-token retention", ylim=(.85, 1.005))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    lines = ["# Mid-verification probe pilot", "", "Fixed B16 candidates; fresh same-forward labels. These are offline work proxies, not speedups.", "",
             "## Calibration-96% operating points", "",
             "| Policy | Assessment retention | Kept rows | Full-depth-row proxy |", "| --- | ---: | ---: | ---: |"]
    for name, policy in policies.items():
        point = policy["points"]["0.96"]["assessment"]
        lines.append(f"| {name} | {point['retention']:.4f} | {point['mean_kept_rows']:.3f} | {point['equivalent_full_depth_rows']:.3f} |")
    lines.extend(["", "All threshold/checkpoint selection uses calibration only. See summary.json for five operating points, prompt-bootstrap intervals, replay audits and oracle/fixed controls.", ""])
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
