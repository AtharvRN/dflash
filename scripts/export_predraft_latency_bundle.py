"""Export an audited CPU-only replay bundle; never select on assessment outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json

POLICIES = {"predraft_hard": "hard_seed913", "predraft_mixed_tv": "mixed_tv_bce_seed913"}


def verify(directory):
    manifest = json.loads((directory / "COMPLETE.json").read_text())
    if not manifest.get("complete") or not manifest.get("binding"):
        raise ValueError("Incomplete artifact")
    for name, digest in manifest["binding"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError("Artifact binding mismatch: " + name)
    return manifest


def select(rows, eligible, count=128):
    chosen, seen = [], set()
    for row, valid in zip(rows, eligible, strict=True):
        pid = str(row["prompt_id"])
        if row["group"] != "assessment" or not valid or pid in seen:
            continue
        tokens = row["prefix_token_ids"]
        if len(tokens) != row["prefix_length"] + 1 or row["prefix_length"] < 1:
            raise ValueError("Prefix/anchor alignment mismatch")
        digest = hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()
        if digest != row["prefix_sha256"]:
            raise ValueError("Prefix binding mismatch")
        chosen.append(dict(row))
        seen.add(pid)
        if len(chosen) == count:
            return chosen
    raise ValueError("Insufficient distinct eligible assessment prompts")


def select_all_assessment(rows, eligible):
    """Retain every eligible cycle, including later cycles and repeated prompts."""
    selected, seen = [], set()
    for row, valid in zip(rows, eligible, strict=True):
        if row['group'] != 'assessment' or not valid:
            continue
        key = (str(row['prompt_id']), row['cycle'])
        if key in seen:
            raise ValueError('Duplicate assessment cycle')
        seen.add(key)
        # Reuse the exact prefix/anchor integrity checks, not its sampling rule.
        selected.extend(select([row], [True], count=1))
    if not selected or len(selected) > 4096:
        raise ValueError('Full-cycle replay requires 1..4096 eligible assessment states')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument('--all-assessment', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Fresh export destination required")
    from scripts.train_soft_supervision import load_cache
    verify(args.training)
    collection, rows, arrays = load_cache(args.cache, view="predraft")
    if collection["smoke"]:
        raise ValueError("Require full pilot, not smoke data")
    selected = (select_all_assessment(rows, arrays['eligible']) if args.all_assessment
                else select(rows, arrays["eligible"]))
    indices = [r["row"] for r in selected]
    lookup = {r["row"]: r for r in selected}
    # Fresh same-forward candidates belong with this cache's fused features and
    # labels, not the older saved candidate IDs in collection/config.json.
    for receipt in json.loads((args.cache / "receipts.json").read_text()):
        if not any(i in lookup for i in receipt["rows"]):
            continue
        with np.load(args.cache / receipt["file"], allow_pickle=False) as shard:
            for j, i in enumerate(receipt["rows"]):
                if i in lookup:
                    row = lookup[i]
                    row["draft_ids"] = shard["candidate_ids"][j].astype(int).tolist()
                    row["cached_accepted_len"] = int(arrays["accepted_len"][i])
    summary = json.loads((args.training / "summary.json").read_text())
    args.output.mkdir(parents=True)
    atomic_json(args.output / "states.json", selected)
    np.save(args.output / "cached_fused.npy", arrays["fused"][indices], allow_pickle=False)
    policies = {}
    for case, name in POLICIES.items():
        checkpoint = name + ".pt"
        shutil.copy2(args.training / checkpoint, args.output / checkpoint)
        entry = summary["models"][name]
        policies[case] = {"checkpoint": checkpoint, "arm": name.removesuffix("_seed913"),
                          "seed": 913, "selected_update": entry["selected_update"],
                          "threshold": entry["primary"]["calibration"]["threshold"]}
    atomic_json(args.output / "bundle.json", {
        "schema": "predraft_latency_bundle_v1", "policies": policies,
        "models": collection["models"], "rows": len(selected),
        "cohort_mode": "all_assessment_cycles" if args.all_assessment else "first_128_prompts",
        "selection": ("All eligible assessment cycles in canonical order; no additional sampling or acceptance filtering"
                      if args.all_assessment else "First eligible cycle per distinct assessment prompt in canonical order; first 128 prompts; no acceptance-based selection"),
        "source_cache": str(args.cache), "source_training": str(args.training),
        "source_cache_complete_sha256": sha256(args.cache / "COMPLETE.json"),
        "source_training_complete_sha256": sha256(args.training / "COMPLETE.json"),
        "setting": "Seed913 fixed, primary calibration96 threshold, no assessment retuning",
        "candidate_scope": "Cached B16 candidates; runtime may regenerate native B16 once and freezes it across all candidate-preserving cases"})
    files = sorted(p.name for p in args.output.iterdir() if p.is_file())
    atomic_json(args.output / "COMPLETE.json", {"complete": True,
                "binding": {name: sha256(args.output / name) for name in files}})
    print(json.dumps({"complete": str(args.output), "rows": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
