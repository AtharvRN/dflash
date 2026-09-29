"""Rescore frozen predictions with FP64 threshold comparisons; never retrain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import apply_setting, metrics, prompt_bootstrap
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.train_midverify_probe import plot_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing existing rescore destination")
    complete = json.loads((args.training / "COMPLETE.json").read_text())
    for name, expected in complete["binding"].items():
        if sha256(args.training / name) != expected:
            raise ValueError("Original result hash mismatch")
    config = json.loads((args.training / "config.json").read_text())
    cache = Path(config["cache"])
    if sha256(cache / "COMPLETE.json") != config["collection_completion_sha256"]:
        raise ValueError("Source cache changed")
    cache_complete = json.loads((cache / "COMPLETE.json").read_text())
    if sha256(cache / "rows.json") != cache_complete["binding"]["rows.json"]:
        raise ValueError("Row metadata hash mismatch")
    rows = json.loads((cache / "rows.json").read_text())
    a = np.asarray([r["accepted_len"] for r in rows])
    prompt = np.asarray([r["prompt_id"] for r in rows])
    ca = np.asarray([r["group"] == "calibration" for r in rows])
    ass = np.asarray([r["group"] == "assessment" for r in rows])
    summary = json.loads((args.training / "summary.json").read_text())
    changes = {}
    with np.load(args.training / "scores.npz", allow_pickle=False) as scores:
        for name, policy in summary["policies"].items():
            changes[name] = {}
            for target, value in policy["points"].items():
                setting = value["calibration"]
                # Check calibration/selection really are unchanged by this fix.
                kc = apply_setting(scores[name][ca], setting)
                checked = metrics(a[ca], kc)
                if checked["mean_kept_rows"] != setting["mean_kept_rows"] or checked["retention"] != setting["retention"]:
                    raise ValueError("Unexpected calibration change: rescore cannot repair selection")
                q = scores[name][ass]
                old_q = q.astype(np.float64) if name.startswith("lens_topk_") else q
                threshold = -np.inf if setting["threshold"] is None else setting["threshold"]
                old_k = 1 + np.cumprod(old_q >= threshold, axis=1).sum(1)
                old_metric = metrics(a[ass], old_k, policy["layer"])
                if old_metric["mean_kept_rows"] != value["assessment"]["mean_kept_rows"]:
                    raise ValueError("Cannot reconstruct original scoring behavior")
                k = apply_setting(q, setting)
                changes[name][target] = int((old_k != k).sum())
                value["assessment"] = metrics(a[ass], k, policy["layer"])
                value["uncertainty"] = prompt_bootstrap(a[ass], k, prompt[ass], policy["layer"])
    reason = "Preserve float64 nextafter threshold transitions when applying to float32 saved scores. Calibration and checkpoint selection were already float64 and are verified unchanged."
    summary["rescoring"] = {"reason": reason, "new_training": False, "new_gpu_work": False,
                             "source_training": str(args.training),
                             "source_training_completion_sha256": sha256(args.training / "COMPLETE.json"),
                             "assessment_decisions_changed": changes}
    config["rescore"] = summary["rescoring"]
    config["rescore_module_sha256"] = sha256("dflash/midverify.py")
    config["rescore_script_sha256"] = sha256(__file__)
    args.output.mkdir(parents=True)
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "config.json", config)
    shutil.copyfile(args.training / "scores.npz", args.output / "scores.npz")
    plot_report(args.output, summary)
    names = ["summary.json", "config.json", "scores.npz", "report.md", "retention_vs_work.png", "retention_vs_kept.png"]
    atomic_json(args.output / "COMPLETE.json", {"passed": True, "binding": {n: sha256(args.output / n) for n in names}})
    print(json.dumps({"max_assessment_rows_changed_per_point": max(c for v in changes.values() for c in v.values()),
                      "changed_policy_points": sum(c > 0 for v in changes.values() for c in v.values()),
                      "calibration_and_checkpoints_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
