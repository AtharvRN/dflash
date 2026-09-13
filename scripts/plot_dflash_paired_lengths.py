"""Plot the paired-length diagnostic without importing model dependencies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.summary.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    blocks = [b for b in ("4", "8", "12", "16") if b in data["blocks"]]
    x = np.arange(len(blocks))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for offset, key, label, color in [(-.18, "mean_proxy_accepted", "Clipped B16 estimate", "#8497a9"),
                                     (.18, "mean_actual_accepted", "Actual shorter draft", "#16846c")]:
        bars = axes[0].bar(x+offset, [data["blocks"][b][key] for b in blocks], width=.34, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    axes[0].set(ylabel="Mean accepted draft tokens", xticks=x, xticklabels=["B"+b for b in blocks])
    axes[0].legend(frameon=False, loc="upper left")
    for offset, key, label, color in [(-.18, "mismatch_rate", "Actual vs clipped label", "#c75a59"),
                                     (.18, "target_width_control_mismatch_rate", "Target-width control", "#8d78ad")]:
        bars = axes[1].bar(x+offset, [100*data["blocks"][b][key] for b in blocks], width=.34, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.1f%%", fontsize=9, padding=3)
    axes[1].set(ylabel="States with different acceptance (%)", xticks=x, xticklabels=["B"+b for b in blocks])
    axes[1].legend(frameon=False, loc="upper left")
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.2)
        axis.margins(y=.25)
    fig.suptitle(f"Qwen3-4B DFlash: actual draft length vs B16 truncation\n"
                 f"{data['eligible_states']:,} nonterminal paired states; {data['prompts']} validation prompts", fontsize=14)
    fig.savefig(args.output_dir / "paired_length_diagnostic.png", dpi=180)
    plt.close(fig)
    if not data["policies"]:
        return
    names = list(data["policies"])
    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    x = np.arange(len(names))
    for offset, key, label, color in [(-.18, "proxy_retention", "Clipped B16 estimate", "#8497a9"),
                                     (.18, "actual_retention", "Actual shorter draft", "#16846c")]:
        bars = axis.bar(x+offset, [100*data["policies"][name][key] for name in names], width=.34, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f%%", fontsize=11, padding=3)
    axis.axhline(95, linestyle="--", color="#777777", linewidth=1)
    axis.set(xticks=x, xticklabels=[name.title()+" policy" for name in names],
             ylabel="Accepted draft tokens relative to B16 (%)",
             title="Frozen policies on the same B16 reference states")
    axis.margins(y=.2)
    axis.legend(frameon=True, facecolor="white", framealpha=1, loc="lower left")
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=.2)
    fig.savefig(args.output_dir / "paired_policy_retention.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
