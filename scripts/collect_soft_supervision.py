"""Fresh exact-prefix replay for matched hard/TV/margin supervision.

All old labels are replay controls only. Primary eligibility excludes changed
anchors and accepted EOS uniformly across arms; exclusions remain in the cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_soft_acceptance import select_training_rows, initialize, work, summarize
from scripts.analyze_policy_granularity import load_pairs
from scripts.audit_block_headroom import sha256
from scripts.collect_midverify_probe import backup_file
from scripts.collect_policy_granularity import atomic_json
from scripts.collect_rejected_trace import prefix_hash, validate_models
from scripts.gpu_runtime import configure_gpu_runtime


def select_rows(source, evaluation, smoke=False):
    rows, binding = select_training_rows(source, 2000)
    source_config = json.loads((source / "config.json").read_text())
    ev_config, ev_rows, _, _ = load_pairs(evaluation)
    validate_models(ev_config["models"])
    val_ids = set(map(int, source_config["canonical_val_prompt_ids"]))
    for group, expected_rows, expected_prompts in (("calibration", 342, 47), ("assessment", 1416, 187)):
        selected = [r for r in ev_rows if r["group"] == group and r["eligible"]]
        if len(selected) != expected_rows or len({r["prompt_id"] for r in selected}) != expected_prompts:
            raise ValueError("Fixed source evaluation membership changed")
        frozen = set(map(int, source_config[f"frozen_{group}_prompt_ids"]))
        if {int(r["prompt_id"]) for r in selected} != frozen:
            raise ValueError("Evaluation groups disagree with training source")
        for r in selected:
            if int(r["prompt_id"]) not in val_ids or prefix_hash(r["prefix_token_ids"]) != r["prefix_sha256"]:
                raise ValueError("Invalid validation prefix or canonical membership")
            rows.append({k: r[k] for k in ("prompt_id", "cycle", "group", "source", "prefix_length",
                         "prefix_token_ids", "prefix_sha256")} | {"row": len(rows),
                         "draft_ids": r["outcomes"]["16"]["draft_ids"],
                         "old_accepted_len": r["outcomes"]["16"]["accepted"]})
    if smoke:
        rows = [r for group in ("train", "calibration", "assessment")
                for r in [r for r in rows if r["group"] == group][:8]]
    seen, groups = set(), {}
    for i, row in enumerate(rows):
        row["row"] = i
        key = (int(row["prompt_id"]), row["cycle"])
        if key in seen:
            raise ValueError("Repeated row identity")
        seen.add(key)
        groups.setdefault(row["group"], set()).add(int(row["prompt_id"]))
    if any(groups[a] & groups[b] for a in groups for b in groups if a < b):
        raise ValueError("Prompt leakage")
    binding["evaluation_completion_sha256"] = sha256(evaluation / "COMPLETE.json")
    return rows, binding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "evaluation", "models", "output", "backup"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, choices=(1, 4), default=4)
    args = p.parse_args()
    if args.output.exists() or args.backup.exists() or args.output.resolve() == args.backup.resolve():
        raise ValueError("Fresh independent destinations required")
    rows, binding = select_rows(args.source, args.evaluation, args.smoke)
    models = json.loads(args.models.read_text())
    validate_models(models)
    runtime = configure_gpu_runtime(use_container_gpu=True, require_gpu=True)
    batches = []
    for group in ("train", "calibration", "assessment"):
        subset = [r for r in rows if r["group"] == group]
        subset[0]["canonical_check"] = True
        batches.extend(subset[i:i+4] for i in range(0, len(subset), 4))
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    config = {"schema": "soft_supervision_v1", "rows": rows, "source_binding": binding,
              "models": models, "runtime": runtime, "workers": args.workers, "batch_size": 4,
              "smoke": args.smoke, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {name: sha256(Path(__file__).parent/name) for name in
                  ("collect_soft_supervision.py", "audit_soft_acceptance.py")},
              "primary_eligibility": "same saved anchor argmax AND no accepted EOS; exclusions kept, not replaced",
              "protocol": "Fresh B16 candidates, features and labels in same exact-prefix replay; original labels only controls",
              "temperature_soft": 1., "decoding": "greedy", "dtype": "BF16", "attention": "SDPA", "tf32": False}
    atomic_json(args.output / "config.json", config)
    backup_file(args.output / "config.json", args.backup / "config.json")
    receipts, counts, started = [], Counter(), time.monotonic()
    # Bound in-flight results to eight batches, avoiding a full-cache result queue.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(models,)) as pool:
        pending = {}
        for i in range(min(8, len(batches))):
            pending[i] = pool.submit(work, batches[i])
        for i, batch in enumerate(batches):
            arrays = pending.pop(i).result()
            if i+8 < len(batches):
                pending[i+8] = pool.submit(work, batches[i+8])
            arrays["eligible"] = arrays["anchor_match"] & ~arrays["accepted_eos"]
            path = args.output / f"batch_{i:04d}.npz"
            np.savez(path, **arrays)
            backup_file(path, args.backup/path.name)
            receipt = {"file": path.name, "sha256": sha256(path), "rows": [r["row"] for r in batch],
                       "group": batch[0]["group"], "summary": summarize(arrays, batch),
                       "eligible": int(arrays["eligible"].sum())}
            receipts.append(receipt)
            counts[receipt["group"]] += receipt["eligible"]
            atomic_json(args.output / "receipts.json", receipts)
            backup_file(args.output / "receipts.json", args.backup / "receipts.json")
            print("PROGRESS", json.dumps({"batches": i+1, "total_batches": len(batches), "eligible": counts,
                                          "elapsed_seconds": time.monotonic()-started}), flush=True)
            if receipt["summary"].get("canonical_disagreements", 0):
                raise ValueError("Canonical check failed; preserved partial evidence, no completion marker")
    summary = {"complete": True, "eligible": dict(counts), "coverage": {}, "elapsed_seconds": time.monotonic()-started}
    for group in counts:
        subset = [r for r in receipts if r["group"] == group]
        summary["coverage"][group] = {"planned_rows": sum(len(r["rows"]) for r in subset),
            "planned_prompts": len({r["prompt_id"] for r in rows if r["group"] == group}),
            "eligible_rows": counts[group], "anchor_mismatch_rows": sum(len(r["rows"])-r["summary"]["anchor_matches"] for r in subset),
            "accepted_eos_rows": sum(r["summary"]["accepted_eos_rows"] for r in subset),
            "old_acceptance_match_rows": sum(r["summary"]["acceptance_matches_source_rows"] for r in subset)}
    atomic_json(args.output / "summary.json", summary)
    backup_file(args.output / "summary.json", args.backup / "summary.json")
    atomic_json(args.output / "COMPLETE.json", {"complete": True, "binding": {
        name: sha256(args.output/name) for name in ("config.json", "receipts.json", "summary.json")}})
    backup_file(args.output / "COMPLETE.json", args.backup / "COMPLETE.json")
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
