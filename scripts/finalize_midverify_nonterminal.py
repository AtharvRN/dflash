"""Audit completed replay evidence and exclude new terminal TRAINING states.

Never changes the failed source cache, frozen pilot, or validation membership.
This is an explicit censoring review, not a bypass of a failed completion flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import accepted_lengths
from scripts.audit_block_headroom import sha256
from scripts.collect_midverify_probe import seed_mapping, array_row_digest
from scripts.collect_policy_granularity import atomic_json


def nonterminal_selection(rows, eos_ids, protected, protected_train_prefix=5000):
    protected = set(protected)
    protected.update([i for i, r in enumerate(rows) if r["group"] == "train"][:protected_train_prefix])
    keep, excluded = [], []
    for i, row in enumerate(rows):
        terminal = any(t in eos_ids for t in row["draft_ids"][:row["accepted_len"]])
        if terminal != row["replayed_accepted_eos"]:
            raise ValueError("Terminal metadata mismatch")
        if terminal:
            if row["group"] != "train" or i in protected:
                raise ValueError("Cannot exclude validation or protected training states")
            excluded.append(i)
        else:
            keep.append(i)
    return np.asarray(keep), excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-exclusions", type=int, default=1)
    args = p.parse_args()
    if args.output.exists() or (args.cache / "COMPLETE.json").exists():
        raise ValueError("Require fresh destination and an uncompleted replay cache")
    config = json.loads((args.cache / "config.json").read_text())
    audit = json.loads((args.cache / "audit.json").read_text())
    records = json.loads((args.cache / "rows.json").read_text())
    source = json.loads((args.cache / "source_rows.json").read_text())
    receipts = json.loads((args.cache / "chunks.json").read_text())
    if audit["passed"] or audit["fresh_accepted_eos_states"] != args.expected_exclusions:
        raise ValueError("Unexpected source audit condition")
    for path, digest in config["source_bindings"].items():
        if sha256(path) != digest:
            raise ValueError("Original source binding changed")
    seed = Path(config["seed_cache"])
    complete = json.loads((seed / "COMPLETE.json").read_text())
    if not complete["passed"] or sha256(seed / "COMPLETE.json") != config["seed_completion_sha256"]:
        raise ValueError("Wrong frozen seed")
    for name, digest in complete["binding"].items():
        if Path(name).name != name or sha256(seed / name) != digest:
            raise ValueError("Frozen seed hash changed")
    seed_rows = json.loads((seed / "source_rows.json").read_text())
    seed_records = json.loads((seed / "rows.json").read_text())
    reuse = seed_mapping(seed_rows, source)
    if reuse.tolist() != config["seed_row_mapping"]:
        raise ValueError("Reuse identity changed")
    arrays = {Path(name).stem: np.load(args.cache / name, mmap_mode="r", allow_pickle=False)
              for name in complete["binding"] if name.endswith(".npy")}
    seen = np.zeros(len(source), dtype=int)
    seen[reuse] = 1
    for key, value in arrays.items():
        original = np.load(seed / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != (len(source), *original.shape[1:]) or value.dtype != original.dtype:
            raise ValueError("Expanded array schema changed")
        for group in ("train", "calibration", "assessment"):
            idx = np.array([i for i, r in enumerate(seed_rows) if r["group"] == group])
            expected = array_row_digest(original, idx)
            if expected != audit["seed_array_group_sha256"][key][group] or array_row_digest(value, reuse[idx]) != expected:
                raise ValueError("Frozen feature bytes changed")
    for receipt in receipts:
        path = args.cache / receipt["path"]
        if path.parent != args.cache / "chunks" or sha256(path) != receipt["sha256"]:
            raise ValueError("New replay chunk hash changed")
        with np.load(path, allow_pickle=False) as chunk:
            idx = chunk["row_ids"]
            if idx.tolist() != receipt["row_ids"] or set(chunk.files) - {"row_ids"} != set(arrays):
                raise ValueError("New replay chunk identity/schema changed")
            seen[idx] += 1
            for key, value in arrays.items():
                if chunk[key].dtype != value.dtype or not np.isfinite(chunk[key]).all() or not np.array_equal(chunk[key], value[idx]):
                    raise ValueError("Materialized array differs from replay evidence")
    if not np.all(seen == 1) or len(records) != len(source):
        raise ValueError("Missing or duplicated row evidence")
    a = accepted_lengths(arrays["matches"])
    if not np.array_equal(a, arrays["accepted_len"]) or not np.array_equal(arrays["matches"], arrays["target_argmax"] == arrays["candidate_ids"]):
        raise ValueError("Fresh labels are not aligned to same-forward logits")
    for i, (row, record) in enumerate(zip(source, records)):
        expected = {k: v for k, v in row.items() if k != "prefix_token_ids"}
        if any(record[k] != v for k, v in expected.items()) or row["row"] != i:
            raise ValueError("Row provenance changed")
        if hashlib.sha256(np.asarray(row["prefix_token_ids"], dtype=np.int64).tobytes()).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Prefix hash changed")
        if row["draft_ids"] != arrays["candidate_ids"][i].tolist() or record["accepted_len"] != int(a[i]):
            raise ValueError("Candidate/label metadata changed")
    for old, i in zip(seed_records, reuse):
        if {k: v for k, v in old.items() if k != "row"} != {k: v for k, v in records[i].items() if k != "row"}:
            raise ValueError("Frozen metadata changed")
    generation = json.loads((Path(config["models"]["target"]["path"]) / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    keep, excluded = nonterminal_selection(records, eos, reuse)
    if len(excluded) != args.expected_exclusions:
        raise ValueError("Unexpected terminal exclusion count")
    args.output.mkdir(parents=True)
    new_source = [{**source[i], "row": j} for j, i in enumerate(keep)]
    new_records = [{**records[i], "row": j} for j, i in enumerate(keep)]
    new_reuse = seed_mapping(seed_rows, new_source)
    config.update({"rows": len(keep), "groups": dict(Counter(r["group"] for r in new_records)),
                   "train_rows": sum(r["group"] == "train" for r in new_records),
                   "seed_row_mapping": new_reuse.tolist(), "finalization_source": str(args.cache),
                   "terminal_training_exclusions": [records[i] for i in excluded],
                   "finalization_script_sha256": sha256(__file__),
                   "selection_amendment": "Exclude only newly replayed accepted-EOS training states; original evidence and 2k/5k subsets unchanged"})
    binding = {str(args.cache / name): sha256(args.cache / name) for name in
               ["config.json", "audit.json", "rows.json", "source_rows.json", "chunks.json", *[f"{key}.npy" for key in arrays]]}
    for key, value in arrays.items():
        output = np.lib.format.open_memmap(args.output / f"{key}.npy", mode="w+", dtype=value.dtype, shape=(len(keep), *value.shape[1:]))
        for start in range(0, len(keep), 64):
            idx = keep[start:start + 64]
            output[start:start + len(idx)] = value[idx]
        output.flush()
        if array_row_digest(value, keep) != array_row_digest(output, np.arange(len(keep))):
            raise ValueError("Filtered array copy changed bytes")
        for group in ("train", "calibration", "assessment"):
            idx = np.array([i for i, r in enumerate(seed_rows) if r["group"] == group])
            if array_row_digest(output, new_reuse[idx]) != audit["seed_array_group_sha256"][key][group]:
                raise ValueError("Filtered copy changed frozen bytes")
    final_audit = {**audit, "passed": True, "rows": len(keep), "groups": config["groups"],
        "fresh_accepted_eos_states": 0, "excluded_terminal_training_states": len(excluded),
        "label_replay_matches": sum(r["label_replay_match"] for r in new_records),
        "anchor_replay_matches": sum(r["anchor_replay_match"] for r in new_records),
        "draft_argmax_match_fraction": sum(r["draft_argmax_matches_saved"] for r in new_records) / (15 * len(keep)),
        "newly_replayed_rows": len(keep) - len(reuse), "source_all_chunks_hash_verified": True,
        "all_arrays_match_seed_or_replay_chunks": True,
        "batch_single_control_row_ids_refer_to_unfiltered_cache": True,
        "finalization_note": "Source audit failed only because of accepted EOS. Original source and excluded state remain intact."}
    for name, content in (("config.json", config), ("audit.json", final_audit), ("rows.json", new_records),
                          ("source_rows.json", new_source), ("source_verification.json", binding)):
        atomic_json(args.output / name, content)
    names = ["config.json", "audit.json", "rows.json", "source_rows.json", "source_verification.json", *[f"{key}.npy" for key in arrays]]
    atomic_json(args.output / "COMPLETE.json", {"passed": True, "rows": len(keep), "binding": {name: sha256(args.output / name) for name in names}})
    print(json.dumps({"passed": True, "groups": config["groups"], "excluded": [records[i] for i in excluded]}), flush=True)


if __name__ == "__main__":
    main()
