"""Exact audited serial/parallel rejected-trace collection parity gate."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_rejected_trace_cache import GROUPS, load_cache, sha256


IGNORED_CONFIG_FIELDS = {
    "manifest", "pilot_manifest", "split_dir", "eval_cache", "output", "backup",
    "gpu", "workers", "max_seconds",
}
REQUIRED_PROTOCOL_FIELDS = {
    "models", "seed", "states_per_prompt", "max_new_tokens", "max_prompt_tokens",
    "source_sha256", "input_hashes", "reference_completion_sha256", "dtype", "attention",
    "thinking", "temperature", "tf32", "dependency_sha256",
}


def _digest(value, context):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid SHA256 for "+context)
    return value


def normalized_protocol(config):
    """Ignore execution locations/concurrency while binding input identities."""
    missing = REQUIRED_PROTOCOL_FIELDS-set(config)
    if missing:
        raise ValueError("Missing comparison provenance: "+", ".join(sorted(missing)))
    _digest(config["source_sha256"], "collector source")
    _digest(config["reference_completion_sha256"], "reference completion")
    if not isinstance(config["dependency_sha256"], dict) or not config["dependency_sha256"]:
        raise ValueError("Missing collector dependency hashes")
    for name, digest in config["dependency_sha256"].items():
        _digest(digest, "collector dependency "+name)
    if "collector_source_sha256" in config and config["collector_source_sha256"] != config["source_sha256"]:
        raise ValueError("Conflicting collector source hashes")
    models = copy.deepcopy(config["models"])
    if set(models) != {"draft", "target"}:
        raise ValueError("Require target and draft model identities")
    for name, model in models.items():
        if not isinstance(model, dict) or not model.get("repo") or not model.get("revision"):
            raise ValueError("Missing pinned model identity: "+name)
        revision = model["revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Model revision must be a pinned commit: "+name)
        model.pop("path", None)
    hashes = dict(config["input_hashes"])
    for path, digest in hashes.items():
        _digest(digest, "input "+path)
    # The collector records five input files. Resolve their roles before paths
    # are removed, so moving caches or model directories does not break parity.
    roles = {}
    for key in ("manifest", "pilot_manifest"):
        path = config.get(key)
        if path not in hashes:
            raise ValueError("Missing input hash for "+key)
        roles[key] = hashes.pop(path)
    if not isinstance(config.get("split_dir"), str):
        raise ValueError("Missing split input location")
    for group in ("train", "val"):
        path = str(Path(config["split_dir"])/f"{group}_prompt_ids.json")
        if path not in hashes:
            raise ValueError("Missing canonical split hash: "+group)
        roles[group+"_prompt_ids"] = hashes.pop(path)
    if len(hashes) != 1:
        raise ValueError("Expected exactly one remaining model-manifest input hash")
    roles["model_manifest"] = next(iter(hashes.values()))
    normalized = {key: copy.deepcopy(value) for key, value in config.items() if key not in IGNORED_CONFIG_FIELDS}
    normalized["models"] = models
    normalized["input_hashes"] = roles
    return normalized


def _first_difference(left, right, path="row"):
    """Return an actionable path without dumping generated user text/tokens."""
    if type(left) is not type(right):
        return path+" (type)"
    if isinstance(left, dict):
        if set(left) != set(right):
            return path+" (keys)"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], path+"."+str(key))
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path+" (length)"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else path


def compare(left_root, right_root):
    left_root, right_root = Path(left_root), Path(right_root)
    left_config, left_rows, left_arrays, left_audit = load_cache(left_root, require_complete=True)
    right_config, right_rows, right_arrays, right_audit = load_cache(right_root, require_complete=True)
    left_protocol, right_protocol = normalized_protocol(left_config), normalized_protocol(right_config)
    if left_config["source_sha256"] != right_config["source_sha256"]:
        raise ValueError("Collector source_sha256 mismatch")
    difference = _first_difference(left_protocol, right_protocol, "config")
    if difference:
        raise ValueError("Collection protocol mismatch: "+difference)
    received_left = [receipt["prompt_id"] for receipt in json.loads((left_root/"receipts.json").read_text())]
    received_right = [receipt["prompt_id"] for receipt in json.loads((right_root/"receipts.json").read_text())]
    if received_left != received_right:
        raise ValueError("Collected prompt IDs/order mismatch")
    def without_timing(record):
        value = copy.deepcopy(record)
        value.pop("elapsed_seconds", None)
        timing = value.get("trace_capture_cuda_ms")
        if isinstance(timing, list):
            value["trace_capture_cuda_ms"] = {"count": len(timing)}
        elif isinstance(timing, dict):
            value["trace_capture_cuda_ms"] = {"count": timing["count"]}
        return value
    for pid in received_left:
        left_shard = json.loads((left_root/f"prompt_{pid}.json").read_text())
        right_shard = json.loads((right_root/f"prompt_{pid}.json").read_text())
        difference = _first_difference(without_timing(left_shard.get("prompt_summary", {})),
            without_timing(right_shard.get("prompt_summary", {})), f"prompt[{pid}].summary")
        if difference:
            raise ValueError("Prompt summary mismatch: "+difference)
    difference = _first_difference(without_timing(json.loads((left_root/"collection_summary.json").read_text())),
        without_timing(json.loads((right_root/"collection_summary.json").read_text())), "collection_summary")
    if difference:
        raise ValueError("Collection summary mismatch: "+difference)
    if len(left_rows) != len(right_rows):
        raise ValueError("Collected row count mismatch")
    identities = []
    for index, (left, right) in enumerate(zip(left_rows, right_rows)):
        identity = (int(left["prompt_id"]), int(left["cycle"]))
        if identity != (int(right["prompt_id"]), int(right["cycle"])):
            raise ValueError(f"Row identity mismatch at row {index}")
        difference = _first_difference(left, right, f"row[{index}]")
        if difference:
            raise ValueError("Row/proof/outcome/check mismatch: "+difference)
        identities.append(identity)
    if set(left_arrays) != set(right_arrays):
        raise ValueError("Array inventory mismatch")
    array_report = {}
    for name in sorted(left_arrays):
        left, right = left_arrays[name], right_arrays[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("Array shape/dtype mismatch: "+name)
        if not np.array_equal(left, right):
            mismatch = np.argwhere(left != right)
            raise ValueError(f"Array mismatch: {name}; {len(mismatch)} unequal elements; first index {mismatch[0].tolist()}")
        array_report[name] = {"shape": list(left.shape), "dtype": str(left.dtype),
                              "numeric_exact_equal": True, "elements": int(left.size)}
    # Recheck the audited files to catch mutation between loading and reporting.
    for audit in (left_audit, right_audit):
        for path, expected in audit["input_file_sha256"].items():
            if sha256(path) != expected:
                raise ValueError("Cache changed during comparison: "+path)
    counts = Counter(row["group"] for row in left_rows)
    eligible = Counter(row["group"] for row in left_rows if row["eligible"])
    return {"passed": True, "scope": "Exact parity of two audited completed collections; no tolerance, training, or speedup claim",
        "left_cache": str(left_root.resolve()), "right_cache": str(right_root.resolve()),
        "source_complete_sha256": {"left": sha256(left_root/"COMPLETE.json"), "right": sha256(right_root/"COMPLETE.json")},
        "comparison_script_sha256": sha256(Path(__file__)), "collector_source_sha256": left_config["source_sha256"],
        "normalized_protocol_sha256": hashlib.sha256(json.dumps(left_protocol, sort_keys=True).encode()).hexdigest(),
        "models": left_protocol["models"], "seed": left_config["seed"],
        "counts": {"prompts": len(received_left), "rows": len(left_rows), "eligible_rows": sum(eligible.values()),
            "by_group": {group: {"rows": counts[group], "eligible_rows": eligible[group],
                "prompts": sum(left_config["prompt_groups"][str(pid)] == group for pid in received_left)} for group in GROUPS}},
        "row_identity_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
        "all_row_metadata_and_proofs_exact_equal": True, "arrays": array_report,
        "execution_differences": {key: {"left": left_config.get(key), "right": right_config.get(key)}
            for key in sorted(IGNORED_CONFIG_FIELDS) if left_config.get(key) != right_config.get(key)}}


def save_comparison(left, right, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Comparison output already exists; choose a fresh path")
    result = compare(left, right)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also closes the overwrite race after the first check.
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", "--left", dest="left", type=Path, required=True)
    parser.add_argument("--parallel", "--right", dest="right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = save_comparison(args.left, args.right, args.output)
    print(json.dumps({"passed": result["passed"], "counts": result["counts"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
