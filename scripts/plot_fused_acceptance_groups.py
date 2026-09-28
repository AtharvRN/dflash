"""Replot collected fused-vector embeddings by exact observed acceptance.

Only presentation changes: use the saved joint PCA/t-SNE coordinates, without
refitting embeddings, predicting labels, balancing classes, or averaging vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def save(fig, root, name):
    for extension in ("png", "pdf"):
        fig.savefig(root / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory; do not overwrite evidence")
    summary = json.loads((args.analysis / "summary.json").read_text())
    path = args.analysis / "plot_coordinates.npz"
    with np.load(path, allow_pickle=False) as data:
        labels = data["accepted_len"].copy()
        rows = data["rows"].copy()
        prompts = data["prompt_id"].copy()
        embeddings = {"PCA": data["fused_pca"].copy(),
                      "t-SNE": data["fused_tsne_p50_s0"].copy()}
    if not np.all((labels >= 0) & (labels <= 15) & (labels == labels.astype(int))):
        raise ValueError("Expected integer acceptance labels in 0..15")
    with np.load(args.analysis / "probe_predictions.npz", allow_pickle=False) as data:
        if not np.array_equal(labels, data["y"][rows]):
            raise ValueError("Plot labels do not match original observations")
        if not np.array_equal(prompts, data["prompt_id"][rows]):
            raise ValueError("Plot prompt identities do not match observations")
    for xy in embeddings.values():
        if xy.shape != (len(labels), 2) or not np.isfinite(xy).all():
            raise ValueError("Invalid saved embedding")
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=16)
    assert counts.sum() == len(rows) == summary["plot_sample"]["rows"]
    args.output.mkdir(parents=True)
    palette = list(plt.get_cmap("tab20").colors)
    colors = np.array([palette[i] for i in (0, 2, 4, 6, 8, 10, 12, 14,
                                          16, 18, 1, 3, 5, 7, 9, 11)])
    variance = summary["geometry"]["fused"]["explained_variance_ratio"]
    axis_labels = {"PCA": (f"PC1 ({100*variance[0]:.1f}% variance)",
                           f"PC2 ({100*variance[1]:.1f}% variance)"),
                   "t-SNE": ("t-SNE 1 (arbitrary units)", "t-SNE 2 (arbitrary units)")}
    order = np.random.default_rng(927).permutation(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    for ax, (method, xy) in zip(axes, embeddings.items()):
        ax.scatter(*xy[order].T, c=colors[labels[order]], s=9, alpha=.75,
                   linewidths=0, rasterized=True)
        ax.set(title=method, xlabel=axis_labels[method][0], ylabel=axis_labels[method][1])
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[a],
                      label=f"A = {a} (n={counts[a]})", markersize=6) for a in range(16)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Collected fused vectors, grouped by observed acceptance length\n"
                 f"One point = one cycle; {len(labels):,} cycles; A counts 0–15 proposed tokens")
    fig.tight_layout(rect=(0, .17, 1, .91))
    save(fig, args.output, "fused_acceptance_groups_overlay")
    for method, xy in embeddings.items():
        fig, axes = plt.subplots(4, 4, figsize=(13, 12), sharex=True, sharey=True)
        for a, ax in enumerate(axes.flat):
            selected = labels == a
            ax.scatter(*xy[~selected].T, c="#B5B5B5", s=4, alpha=.22,
                       linewidths=0, rasterized=True)
            ax.scatter(*xy[selected].T, c=[colors[a]], s=13, alpha=.9,
                       linewidths=.25, edgecolors="#333333", rasterized=True)
            n_prompts = len(np.unique(prompts[selected]))
            ax.set_title(f"A = {a}  |  {counts[a]} cycles / {n_prompts} prompts", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if a >= 12:
                ax.set_xlabel(axis_labels[method][0])
            if a % 4 == 0:
                ax.set_ylabel(axis_labels[method][1])
        fig.suptitle(f"Collected fused vectors by acceptance length — {method}\n"
                     "Shared embedding and axes; colored = selected A, gray = all other cycles", fontsize=14)
        fig.text(.5, .013, f"Same {len(labels):,} sampled cycles in every panel; at most 4 per prompt. "
                 "Labels only select/color points, never fit the embedding.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .035, 1, .94))
        save(fig, args.output, f"fused_{'pca' if method == 'PCA' else 'tsne'}_by_acceptance")
    metadata = {"source_analysis": str(args.analysis),
                "source_coordinates_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(labels), "prompts": len(np.unique(prompts)),
                "counts_by_accepted_len": counts.tolist(),
                "same_saved_embeddings": True, "embeddings_refit": False,
                "vectors_averaged": False, "class_balancing": False,
                "feature_input": "Collected 2560-d fused features, row-L2-normalized; train-fitted PCA. t-SNE uses first 50 PCs.",
                "sampling": "Original label-independent sample, maximum 4 cycles/prompt, seed 927.",
                "tsne": {"perplexity": 50, "seed": 0}}
    (args.output / "grouping_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
