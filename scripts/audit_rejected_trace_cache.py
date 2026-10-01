"""Audit and load the causal, actual-block rejected-trace pilot cache.

This checks the serialized execution evidence, not GPU re-execution. In
particular, a hash cannot establish that an activation came from the claimed
forward pass; the collector and its indexing tests establish that contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "dflash_rejected_trace_v1"
BLOCKS = list(range(2, 17))
GROUPS = ("train", "calibration", "assessment")
ARRAY_SCHEMA = {
    "features": ((2560,), np.dtype("float16")),
    "anchor_embedding": ((2560,), np.dtype("float16")),
    "rejected_anchor_embedding": ((2560,), np.dtype("float16")),
    "previous_meta": ((3,), np.dtype("float32")),
    "trace_draft": ((15, 2560), np.dtype("float16")),
    "trace_target": ((15, 2560), np.dtype("float16")),
    "trace_mask": ((15,), np.dtype("uint8")),
    "trace_token_ids": ((15,), np.dtype("int64")),
    "trace_offsets": ((15,), np.dtype("int16")),
    "actual": ((15,), np.dtype("int16")),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _tokens(value, message):
    _require(isinstance(value, list) and all(_int(t) and t >= 0 for t in value), message)


def _ids(values, name):
    _require(isinstance(values, list) and all(_int(p) and p >= 0 for p in values), f"Invalid {name}")
    _require(len(set(values)) == len(values), f"Duplicate {name}")
    return set(values)


def _prefix(record, context):
    tokens = record["prefix_token_ids"]
    _tokens(tokens, f"Invalid {context} prefix token IDs")
    _require(_int(record["prefix_length"]) and record["prefix_length"] >= 0
             and len(tokens) == record["prefix_length"] + 1, f"Invalid {context} prefix length")
    _require(token_hash(tokens) == record["prefix_sha256"], f"{context} prefix hash mismatch")
    return tokens


def _validate_config(config):
    _require(config.get("schema_version") == SCHEMA_VERSION, "Unsupported cache schema")
    _require(config.get("blocks") == BLOCKS, "Expected actual B2 through B16 outcomes")
    planned = _ids(config["prompt_ids"], "planned prompt IDs")
    groups = config["prompt_groups"]
    _require(set(groups) == set(map(str, planned)), "Planned prompt/group mismatch")
    _require(set(groups.values()) <= set(GROUPS), "Unknown prompt group")
    train = _ids(config["canonical_train_prompt_ids"], "canonical training IDs")
    val = _ids(config["canonical_val_prompt_ids"], "canonical validation IDs")
    calibration = _ids(config["frozen_calibration_prompt_ids"], "frozen calibration IDs")
    assessment = _ids(config["frozen_assessment_prompt_ids"], "frozen assessment IDs")
    _require(not train & val and not calibration & assessment, "Prompt split leakage")
    _require(calibration | assessment <= val, "Frozen evaluation prompts outside canonical validation")
    for pid in planned:
        group = groups[str(pid)]
        allowed = train if group == "train" else calibration if group == "calibration" else assessment
        _require(pid in allowed, "Planned prompt outside its canonical/frozen split")
    hashes = config["prompt_content_hashes"]
    _require(set(hashes) == set(map(str, planned)), "Planned prompt/content hash mismatch")
    digest_groups = {}
    for pid, digest in hashes.items():
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(c in "0123456789abcdef" for c in digest), "Invalid prompt content hash")
        _require(digest not in digest_groups or digest_groups[digest] == groups[pid],
                 "Exact-content leakage across groups")
        digest_groups[digest] = groups[pid]
    _require(_int(config["training_rows"]) and config["training_rows"] > 0, "Invalid training row target")
    return planned, groups


def validate_row(row, arrays, index):
    """Check immediate predecessor evidence and causal feature alignment."""
    prefix = _prefix(row, "Current")
    _require(_int(row["cycle"]) and row["cycle"] >= 0, "Invalid current cycle")
    _require(isinstance(row["eligible"], bool), "Invalid eligibility flag")
    _require(row["input_capture"] == "before_current_draft", "Current inputs were not recorded before drafting")
    _require(row["eligible"] == (not bool(row.get("exclusion"))), "Eligibility/exclusion mismatch")
    _require(not row.get("canonical_disagreements", 0), "Canonical greedy disagreement requires investigation")
    if "source_prefix_sha256" in row:
        source_hash = row["source_prefix_sha256"]
        matches = source_hash is None or source_hash == row["prefix_sha256"]
        _require(row["reference_prefix_match"] is matches, "Reference prefix match/hash mismatch")
        _require(not row["eligible"] or (matches and row["source_eligible"]), "Ineligible reference replay retained")
    _require(set(row["outcomes"]) == set(map(str, BLOCKS)), "Incomplete actual block matrix")
    actual = []
    for block in BLOCKS:
        outcome = row["outcomes"][str(block)]
        _tokens(outcome["draft_ids"], "Invalid actual draft IDs")
        accepted = outcome["accepted"]
        _require(len(outcome["draft_ids"]) == block - 1 and _int(accepted)
                 and 0 <= accepted < block, "Invalid actual acceptance/draft length")
        actual.append(accepted)
    _require(np.array_equal(arrays["actual"][index], actual), "Actual labels disagree with fresh outcome records")
    _require(isinstance(row["has_previous"], bool), "Invalid previous-cycle availability")
    previous = row["previous"]
    if not row["has_previous"]:
        _require(previous is None and row["cycle"] == 0 and row["previous_cycle"] == -1
                 and row["previous_B"] == 0 and row["previous_A"] == -1, "Invalid absent predecessor")
        expected_meta = np.zeros(3, np.float32)
        trace_tokens = []
        _require(not np.any(arrays["rejected_anchor_embedding"][index]), "Absent predecessor has rejected anchor")
    else:
        _require(isinstance(previous, dict), "Missing predecessor proof")
        old_prefix = _prefix(previous, "Previous")
        _require(previous["cycle"] == row["cycle"] - 1 == row["previous_cycle"],
                 "Previous trace is not the immediately preceding cycle")
        _require(previous["block_size"] == row["previous_B"] == 16, "Unexpected predecessor block size")
        accepted = previous["accepted"]
        _require(_int(accepted) and 0 <= accepted <= 15 and accepted == row["previous_A"],
                 "Invalid predecessor acceptance")
        draft_ids, posterior_ids = previous["draft_ids"], previous["posterior_ids"]
        _tokens(draft_ids, "Invalid predecessor draft IDs")
        _tokens(posterior_ids, "Invalid predecessor posterior IDs")
        _require(len(draft_ids) == 15 and len(posterior_ids) == 16, "Invalid predecessor token lengths")
        observed = 0
        for drafted, posterior in zip(draft_ids, posterior_ids):
            if drafted != posterior:
                break
            observed += 1
        _require(observed == accepted, "Predecessor acceptance inconsistent with verifier posterior")
        correction = previous["correction_token_id"]
        _require(_int(correction) and correction == posterior_ids[accepted], "Incorrect predecessor correction token")
        _require(previous["terminal"] is False, "Terminal predecessor cannot supply a next state")
        _require(previous["trace_capture"] == "same_forward", "Missing same-forward predecessor provenance")
        _require(prefix == old_prefix + draft_ids[:accepted] + [correction],
                 "Current prefix does not follow predecessor committed tokens and correction")
        expected_meta = np.asarray([1, 1, accepted / 15], np.float32)
        # Draft index A is the rejected token replaced by the new anchor. Its
        # successors are indices A+1 onward; verifier prediction rows A+1..14
        # and draft output rows A+2..15 align to those successor tokens.
        trace_tokens = draft_ids[accepted + 1:]
        if accepted == 15:
            _require(not np.any(arrays["rejected_anchor_embedding"][index]),
                     "Fully accepted predecessor has no rejected anchor")
    _require(np.array_equal(arrays["previous_meta"][index], expected_meta), "Incorrect normalized previous metadata")
    count = len(trace_tokens)
    mask = arrays["trace_mask"][index]
    _require(np.array_equal(mask, np.asarray([1] * count + [0] * (15 - count), np.uint8)),
             "Trace mask must pack only tokens strictly after first rejection")
    _require(np.array_equal(arrays["trace_token_ids"][index], trace_tokens + [-1] * (15 - count)),
             "Trace token IDs are misaligned with rejected suffix")
    _require(np.array_equal(arrays["trace_offsets"][index], list(range(1, count + 1)) + [0] * (15 - count)),
             "Trace offsets do not start after the corrected anchor")
    for key in ("trace_draft", "trace_target"):
        _require(not np.any(arrays[key][index, count:]), f"Nonzero padding in {key}")


def _validate_arrays(arrays, count):
    _require(set(arrays) == set(ARRAY_SCHEMA), "Unexpected or missing cache array")
    for name, (tail, dtype) in ARRAY_SCHEMA.items():
        value = arrays[name]
        _require(value.shape == (count, *tail) and value.dtype == dtype, f"Invalid shape/dtype for {name}")
        _require(np.isfinite(value).all(), f"Nonfinite values in {name}")


def load_cache(root, require_complete=True):
    """Return (config, rows, arrays, audit) in receipt order, with no filtering.

    Derived arrays ``prompt_id``, ``group`` and ``eligible`` align with all
    original feature arrays and actual outcomes. Training must filter these
    arrays together and fit preprocessing using training rows only.
    """
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    planned, groups = _validate_config(config)
    receipts = json.loads((root / "receipts.json").read_text())
    _require(isinstance(receipts, list), "Receipts must be a list")
    input_hashes = {str((root / name).resolve()): sha256(root / name) for name in ("config.json", "receipts.json")}
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
    if require_complete:
        _require(complete is not None and complete.get("sample_complete") is True, "Cache is not complete")
    if complete is not None:
        bound_names = {"config.json", "receipts.json", "collection_summary.json"}
        _require(set(complete["binding"]) == bound_names, "Unexpected completion binding inventory")
        for name in sorted(bound_names):
            digest = sha256(root / name)
            _require(complete["binding"][name] == digest, f"Completion binding mismatch for {name}")
            input_hashes[str((root / name).resolve())] = digest
        input_hashes[str(complete_path.resolve())] = sha256(complete_path)
    seen, rows, expected_shards, expected_receipts = set(), [], set(), set()
    chunks = {name: [] for name in ARRAY_SCHEMA}
    previous_order = -1
    order = {pid: index for index, pid in enumerate(config["prompt_ids"])}
    for receipt in receipts:
        pid = receipt["prompt_id"]
        _require(_int(pid) and pid in planned and pid not in seen, "Unexpected or duplicate receipt prompt")
        _require(order[pid] > previous_order, "Receipt ordering differs from planned prompt ordering")
        previous_order = order[pid]
        seen.add(pid)
        _require(receipt["group"] == groups[str(pid)], "Receipt group mismatch")
        names = {f"prompt_{pid}.json", f"prompt_{pid}.npz"}
        _require(set(receipt["files"]) == names, "Unexpected receipt file inventory")
        expected_shards.update(names)
        for name, expected_hash in receipt["files"].items():
            path = root / name
            _require(path.is_file(), f"Missing shard {name}")
            digest = sha256(path)
            _require(digest == expected_hash, f"Receipt file hash mismatch: {name}")
            input_hashes[str(path.resolve())] = digest
        receipt_path = root / f"receipt_{pid}.json"
        expected_receipts.add(receipt_path.name)
        _require(receipt_path.is_file(), "Missing per-prompt receipt")
        _require(json.loads(receipt_path.read_text()) == receipt, "Per-prompt receipt mismatch")
        input_hashes[str(receipt_path.resolve())] = sha256(receipt_path)
        shard = json.loads((root / f"prompt_{pid}.json").read_text())
        batch = shard["states"]
        _require(str(shard["prompt_id"]) == str(pid) and shard["group"] == receipt["group"], "Shard identity mismatch")
        _require(len(batch) == receipt["states"], "Receipt row-count mismatch")
        with np.load(root / f"prompt_{pid}.npz", allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        _validate_arrays(arrays, len(batch))
        previous_row = None
        for index, row in enumerate(batch):
            _require(str(row["prompt_id"]) == str(pid) and row["group"] == receipt["group"]
                     and row["source"] == receipt["source"], "Row identity/group/source mismatch")
            validate_row(row, arrays, index)
            if previous_row is not None:
                _require(row["cycle"] > previous_row["cycle"] and row["prefix_length"] > previous_row["prefix_length"],
                         "Duplicate or nonmonotone sampled states")
                outcome = previous_row["outcomes"]["16"]
                committed = previous_row["prefix_token_ids"] + outcome["draft_ids"][:outcome["accepted"]]
                _require(row["prefix_token_ids"][:len(committed)] == committed, "Sampled B16 trajectory prefix drift")
            previous_row = row
        _require(sum(row["eligible"] for row in batch) == receipt["eligible_states"], "Receipt eligible-count mismatch")
        rows.extend(batch)
        for key in ARRAY_SCHEMA:
            chunks[key].append(arrays[key])
    actual_shards = {p.name for pattern in ("prompt_*.json", "prompt_*.npz") for p in root.glob(pattern)}
    _require(actual_shards == expected_shards, "Orphan or missing prompt shard")
    _require({p.name for p in root.glob("receipt_*.json")} == expected_receipts, "Orphan or missing per-prompt receipt")
    eligible_counts = Counter(row["group"] for row in rows if row["eligible"])
    if complete is not None:
        summary = json.loads((root / "collection_summary.json").read_text())
        for record in (complete, summary):
            _require(record.get("states") == len(rows), "Completion/summary row-count mismatch")
            _require(record.get("eligible_states") == sum(eligible_counts.values()), "Completion/summary eligible-count mismatch")
        if "group_eligible" in summary:
            _require(all(summary["group_eligible"].get(group, 0) == eligible_counts[group] for group in GROUPS),
                     "Summary group eligible-count mismatch")
        if "prompts" in summary:
            _require(summary["prompts"] == len(seen), "Summary prompt-count mismatch")
        if "sample_complete" in summary:
            _require(summary["sample_complete"] == complete["sample_complete"], "Summary completion status mismatch")
        if complete["sample_complete"]:
            _require(not summary.get("failure") and not summary.get("canonical_disagreements", 0),
                     "Completed cache records an unresolved collection failure")
            _require(summary.get("reference_data_unchanged", True), "Evaluation reference changed during collection")
            planned_train = [pid for pid in config["prompt_ids"] if groups[str(pid)] == "train"]
            seen_train = [pid for pid in config["prompt_ids"] if pid in seen and groups[str(pid)] == "train"]
            _require(seen_train == planned_train[:len(seen_train)], "Training prompts are not an ordered planned prefix")
            _require(eligible_counts["train"] >= config["training_rows"], "Completed cache has insufficient eligible training rows")
            required_evaluation = {pid for pid in planned if groups[str(pid)] != "train"}
            _require(required_evaluation <= seen, "Completed cache is missing frozen evaluation prompts")
            if not config.get("smoke", False):
                frozen = set(config["frozen_calibration_prompt_ids"]) | set(config["frozen_assessment_prompt_ids"])
                _require(required_evaluation == frozen, "Full pilot must retain all frozen evaluation prompts")
    arrays = {key: np.concatenate(values) if values else np.empty((0, *ARRAY_SCHEMA[key][0]), ARRAY_SCHEMA[key][1])
              for key, values in chunks.items()}
    arrays.update(prompt_id=np.asarray([int(row["prompt_id"]) for row in rows], np.int64),
                  group=np.asarray([row["group"] for row in rows], dtype="U11"),
                  eligible=np.asarray([row["eligible"] for row in rows], bool))
    audit = {"passed": True, "schema_version": SCHEMA_VERSION, "complete": bool(complete and complete["sample_complete"]),
             "prompts": len(seen), "states": len(rows), "eligible_states": sum(eligible_counts.values()),
             "eligible_by_group": {group: eligible_counts[group] for group in GROUPS},
             "input_file_sha256": input_hashes, "immediate_predecessor_and_prefix_verified": True,
             "strict_rejected_suffix_alignment_verified": True, "actual_block_labels_match_records": True,
             "note": "Serialized integrity/provenance audit; not GPU replay or proof of hidden-state values."}
    return config, rows, arrays, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(load_cache(args.cache, require_complete=not args.allow_incomplete)[3], indent=2))


if __name__ == "__main__":
    main()
