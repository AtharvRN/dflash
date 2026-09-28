"""Matched causal-signal probes and probability geometry from audited replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_acceptance_geometry import digest, assessment_metrics, ridge_probe, grouped_indices


def save(fig, output, name):
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Use a new destination")
    audit = json.loads((args.replay / "COMPLETE.json").read_text())
    for name, expected in audit["binding"].items():
        if digest(args.replay / name) != expected:
            raise ValueError("Replay audit mismatch: " + name)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, normalize
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    original = [json.loads(line) for line in (args.replay / "rows.jsonl").read_text().splitlines()]
    keep = np.array([r["eligible"] for r in original], dtype=bool)
    rows = [r for r in original if r["eligible"]]
    y = np.array([r["accepted_len"] for r in rows], dtype=float)
    groups = np.array([r["group"] for r in rows])
    prompts = np.array([r["prompt_id"] for r in rows])
    tr, assess = groups == "train", groups == "assessment"
    if min(tr.sum(), (groups == "calibration").sum(), assess.sum()) < 32:
        raise ValueError("Too few rows for the prespecified probes")
    group_prompts = {g: set(prompts[groups == g]) for g in np.unique(groups)}
    for a, b in (("train", "calibration"), ("train", "assessment"), ("calibration", "assessment")):
        if group_prompts[a] & group_prompts[b]:
            raise ValueError("Prompt split leakage")
    args.output.mkdir(parents=True)
    summary = {"replay": str(args.replay), "audit": {k:v for k,v in audit.items() if k != "binding"},
               "counts": {g: {"rows": int((groups == g).sum()), "prompts": len(group_prompts[g])} for g in group_prompts},
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "script_sha256": digest(__file__), "sklearn": sklearn.__version__,
               "probes": {}, "geometry": {}, "contrasts": {},
               "note": "Fresh B16 replay labels on a fixed descriptive subset. Separate from old/full pilot MAEs. POST diagnostics are not pre-draft inputs."}
    data = {"y": y, "group": groups}
    def load(name):
        return np.load(args.replay / f"{name}.npy", mmap_mode="r")[keep].astype(np.float32)
    def scaled(x):
        return StandardScaler().fit(x[tr]).transform(x)
    def project(name, x, dimensions=64):
        print("PCA", name, x.shape, flush=True)
        pca = PCA(n_components=dimensions, svd_solver="randomized", iterated_power=4, random_state=927).fit(x[tr])
        z = pca.transform(x)
        summary["geometry"][name] = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
        return z
    sources = np.array([r["source"] for r in rows])
    meta = np.column_stack([*(sources == s for s in sorted(set(sources))),
        np.log1p([r["prompt_tokens"] for r in rows]), np.log1p([r["prefix_length"] for r in rows]),
        [r["cycle_id"] for r in rows], [r["prefix_length"] - r["prompt_tokens"] for r in rows]])
    meta = scaled(meta)
    fused = project("fused", normalize(load("fused")))
    final = project("target_final", normalize(load("target_final")))
    anchor = project("anchor_embedding", normalize(load("anchor_embedding")), 32)
    history = load("predraft_history")
    history_mask = load("predraft_history_mask")
    latest = history[:, -1]
    hist_x = scaled(np.column_stack([history.reshape(len(rows), -1), history_mask]))
    # Identity-preserving full distribution versus identity-free concentration shape.
    probs = load("predraft_probabilities")
    sorted_top = np.empty((len(rows), 32), np.float32)
    for start in range(0, len(rows), 64):
        batch = probs[start:start + 64]
        sorted_top[start:start + 64] = np.sort(np.partition(batch, -32, axis=1)[:, -32:], axis=1)[:, ::-1]
    shape = scaled(np.column_stack([np.log(np.maximum(sorted_top, 1e-30)), latest]))
    probability_pc = project("verifier_sqrt_probability", np.sqrt(probs))
    del probs
    base = np.column_stack([meta, scaled(fused)])
    predictions = {}
    def probe(name, x, timing="pre-draft"):
        prediction, selection = ridge_probe(x, data)
        predictions[name] = prediction
        result = assessment_metrics(y[assess], prediction[assess], prompts[assess], 927)
        result.update({"availability": timing, "selection": selection})
        summary["probes"][name] = result
        print(name, json.dumps({k: result[k] for k in ("mae", "r2", "availability")}), flush=True)
    probe("Source + progress", meta)
    probe("Fused", base)
    probe("Verifier confidence only (+ controls)", np.column_stack([meta, scaled(latest)]))
    probe("Verifier probability PCs (+ controls)", np.column_stack([meta, scaled(probability_pc)]))
    probe("Anchor embedding (+ controls)", np.column_stack([meta, scaled(anchor)]))
    additions = {"latest verifier confidence": scaled(latest), "verifier history": hist_x,
                 "verifier probability PCs": scaled(probability_pc), "sorted probability shape": shape,
                 "anchor embedding": scaled(anchor), "final target state": scaled(final)}
    for name, x in additions.items():
        probe("Fused + " + name, np.column_stack([base, x]))
    probe("Fused + anchor + confidence", np.column_stack([base, scaled(anchor), scaled(latest)]))
    probe("Fused + anchor + final target state", np.column_stack([base, scaled(anchor), scaled(final)]))
    for name, file, timing in (("POST-DRAFT: current draft confidence", "draft_confidence", "post-draft, before verification"),
                               ("POST-VERIFY: current verifier confidence", "current_verifier_confidence", "post-verification; label-related diagnostic only")):
        x = load(file).reshape(len(rows), -1)
        probe(name, np.column_stack([meta, scaled(x)]), timing)
    baseline = np.abs(y[assess] - predictions["Fused"][assess])
    indices = grouped_indices(prompts[assess])
    counts = np.array([len(i) for i in indices])
    draws = np.random.default_rng(927).integers(len(indices), size=(1000, len(indices)))
    for name, pred in predictions.items():
        delta = np.abs(y[assess] - pred[assess]) - baseline
        sums = np.array([delta[i].sum() for i in indices])
        bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
        summary["probes"][name]["delta_mae_vs_fused"] = float(delta.mean())
        summary["probes"][name]["paired_delta_95"] = np.quantile(bootstrap, [.025, .975]).tolist()
    for i, name in enumerate(("entropy", "anchor_probability", "top1_top2_margin", "anchor_logprob")):
        summary["contrasts"]["spearman_" + name] = float(spearmanr(latest[assess, i], y[assess]).statistic)
    embeddings = {"verifier_pca": probability_pc[:, :2], "fused_pca": fused[:, :2]}
    for name, x in (("verifier_tsne", probability_pc[:, :50]), ("fused_tsne", fused[:, :50])):
        print("t-SNE", name, flush=True)
        embeddings[name] = TSNE(perplexity=50, random_state=0, init="pca", learning_rate="auto", max_iter=1000, n_jobs=4).fit_transform(x)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for ax, name, title in zip(axes.flat, ("verifier_pca", "fused_pca", "verifier_tsne", "fused_tsne"),
                             ("Pre-draft target probabilities: PCA", "Matched fused vectors: PCA",
                              "Pre-draft target probabilities: t-SNE", "Matched fused vectors: t-SNE")):
        dots = ax.scatter(*embeddings[name].T, c=y, cmap="viridis", vmin=0, vmax=15, s=8, alpha=.75, rasterized=True)
        ax.set(title=title, xlabel="PC1" if name.endswith("pca") else "t-SNE 1",
               ylabel="PC2" if name.endswith("pca") else "t-SNE 2")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(dots, ax=axes, label="Observed accepted proposed tokens A (0–15)", shrink=.8)
    fig.suptitle(f"Same {len(rows):,} replay states and fresh B16 labels\nTarget distribution predicts the known anchor, before drafting; PCA uses sqrt(probabilities)")
    save(fig, args.output, "verifier_vs_fused_geometry")
    for kind in ("pca", "tsne"):
        xy = embeddings["verifier_" + kind]
        fig, axes = plt.subplots(4, 4, figsize=(12, 11), sharex=True, sharey=True)
        for a, ax in enumerate(axes.flat):
            mask = y == a
            ax.scatter(*xy[~mask].T, c="#BBBBBB", s=4, alpha=.2, rasterized=True)
            ax.scatter(*xy[mask].T, c="#0072B2", s=12, alpha=.8, rasterized=True)
            ax.set_title(f"A={a} | {mask.sum()} cycles", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if a >= 12: ax.set_xlabel("PC1" if kind == "pca" else "t-SNE 1")
            if a % 4 == 0: ax.set_ylabel("PC2" if kind == "pca" else "t-SNE 2")
        fig.suptitle(f"Pre-draft verifier probability vectors grouped by acceptance — {kind.upper()}\nShared coordinates; colored = selected A, gray = all other states")
        fig.tight_layout(rect=(0, 0, 1, .94))
        save(fig, args.output, "verifier_" + kind + "_by_acceptance")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, j, label in zip(axes, (0, 1, 2), ("Entropy (nats)", "Probability of known anchor", "Top-1 minus top-2 probability")):
        ax.boxplot([latest[y == a, j] for a in range(16)], positions=np.arange(16), showfliers=False,
                   medianprops={"color": "#0072B2"}, whiskerprops={"color": "#777777"})
        ax.set(xlabel="Fresh observed acceptance A", ylabel=label)
    fig.suptitle("Causally available target confidence by acceptance length (all replay splits)")
    save(fig, args.output, "causal_verifier_confidence_by_acceptance")
    names = list(summary["probes"])
    mae = np.array([summary["probes"][n]["mae"] for n in names])
    intervals = np.array([summary["probes"][n]["mae_prompt_bootstrap_95"] for n in names])
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    ax.barh(np.arange(len(names)), mae, color=["#D55E00" if n.startswith("POST") else "#0072B2" for n in names], alpha=.8)
    ax.errorbar(mae, np.arange(len(names)), xerr=np.maximum(0, np.stack([mae - intervals[:, 0], intervals[:, 1] - mae])), fmt="none", color="#222222", capsize=2)
    ax.set_yticks(np.arange(len(names)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="Assessment MAE (tokens; lower is better)", title="Matched fresh-label diagnostic probes; 95% prompt-bootstrap intervals\nBlue: pre-draft. Orange: future information, NOT a pre-draft policy.")
    ax.set_xlim(0, intervals[:, 1].max() + .35)
    for i, value in enumerate(mae): ax.text(intervals[i, 1] + .035, i, f"{value:.3f}", va="center", fontsize=9)
    save(fig, args.output, "incremental_signal_probes")
    np.savez_compressed(args.output / "coordinates.npz", accepted_len=y, group=groups, prompt_id=prompts,
                        anchor_id=np.array([r["anchor_id"] for r in rows]), **embeddings)
    np.savez_compressed(args.output / "predictions.npz", y=y, group=groups, prompt_id=prompts,
                        **{f"pred_{i}":v for i,v in enumerate(predictions.values())})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Verifier probability and causal signal analysis", "", summary["note"], "",
             "## Matched assessment results", "", "| Probe | MAE | R² | Δ MAE vs fused | 95% paired interval |",
             "|---|---:|---:|---:|---|" ]
    for name, value in summary["probes"].items():
        lines.append(f"| {name} | {value['mae']:.4f} | {value['r2']:.4f} | {value['delta_mae_vs_fused']:+.4f} | {value['paired_delta_95']} |")
    lines += ["", "## Figures", ""]
    for name in ("verifier_vs_fused_geometry", "verifier_pca_by_acceptance", "verifier_tsne_by_acceptance",
                 "causal_verifier_confidence_by_acceptance", "incremental_signal_probes"):
        lines += [f"![{name}]({name}.png)", ""]
    lines += ["## Limits", "", "- Previously inspected development prompts, not a final test.",
              "- Predictions target fresh replay labels, never old labels joined to new features.",
              "- Simple linear probes may miss feature interactions; a negative result is not proof of no information.",
              "- The last target distribution describes the known anchor, not the first proposed token.",
              "- Current draft/verification confidence is future information and only a diagnostic.",
              "- B16 labels do not identify actual acceptance at other draft block sizes.",
              "- Intervals reflect prompt sampling conditional on this fit; not multi-seed uncertainty or multiplicity-adjusted inference."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    (args.output / "COMPLETE.json").write_text(json.dumps({"summary_sha256": digest(args.output / "summary.json")}) + "\n")
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
