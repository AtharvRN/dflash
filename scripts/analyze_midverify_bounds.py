"""Same-candidate ideal mid-verifier bound at each frozen control's progress."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import ideal_work_at_progress, break_even_oracle_depth
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh analysis output")
    complete = json.loads((args.training / "COMPLETE.json").read_text())
    for name, digest in complete["binding"].items():
        if sha256(args.training / name) != digest:
            raise ValueError("Training artifact hash mismatch")
    summary = json.loads((args.training / "summary.json").read_text())
    results = {}
    for target, policy in summary["policies"]["draft_candidate_logprob"]["points"].items():
        point = policy["assessment"]
        a, k = point["mean_accepted"], point["mean_kept_rows"]
        per_layer = {str(L): {"ideal_full_depth_rows": ideal_work_at_progress(a, L),
                             "maximum_proxy_work_saving_vs_control": 1 - ideal_work_at_progress(a, L) / k}
                     for L in (0, 6, 9, 12, 18, 24)}
        results[target] = {"control_assessment": point,
                           "oracle_break_even_completed_layers": break_even_oracle_depth(a, k),
                           "oracle_at_IDENTICAL_accepted_count": per_layer}
    args.output.mkdir(parents=True)
    report = {"training_completion_sha256": sha256(args.training / "COMPLETE.json"),
              "scope": "Greedy unchanged B16 candidates; clairvoyant integer K; same assessment states AND same accepted-token total as each frozen control",
              "derivation": "sum(K) >= N + sum(A_trim); ideal work = [16*L + (36-L)*(1+mean(A_trim))]/36",
              "not_latency_bound": True, "new_gpu_work": False, "controls": results}
    atomic_json(args.output / "summary.json", report)
    lines = ["# Matched-progress oracle work bound", "",
             "At each frozen draft-confidence control's actual assessment retention, assume a perfect mid-target probe. No new threshold was tuned on assessment.", "",
             "| Calibration target | Assessment retention | Confidence rows | Perfect L6 rows | Perfect L12 rows | Break-even depth |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for target, r in results.items():
        c = r["control_assessment"]
        v = r["oracle_at_IDENTICAL_accepted_count"]
        lines.append(f"| {target} | {c['retention']:.5f} | {c['mean_kept_rows']:.3f} | {v['6']['ideal_full_depth_rows']:.3f} | {v['12']['ideal_full_depth_rows']:.3f} | {r['oracle_break_even_completed_layers']:.2f} |")
    lines.extend(["", "Rows are full-depth-equivalent work proxies. This is not a measured latency or throughput bound. It ignores probe/compaction cost and does not establish behavior on other workloads or models.", ""])
    (args.output / "report.md").write_text("\n".join(lines))
    atomic_json(args.output / "COMPLETE.json", {"passed": True,
                "binding": {name: sha256(args.output / name) for name in ("summary.json", "report.md")}})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
