"""Summarize numeric discrepancies without conflating execution with parity."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def aggregate(comparisons):
    if not comparisons:
        return None
    return {"comparisons": len(comparisons),
            "all_bitwise_equal": all(r["bitwise_equal"] for r in comparisons),
            "max_abs": max(r["max_abs"] for r in comparisons),
            "max_relative_l2": max(r["relative_l2"] for r in comparisons),
            "top1_mismatches": sum(r.get("top1_mismatches", 0) for r in comparisons),
            "logit_token_positions": sum(r.get("tokens", 0) for r in comparisons)}


def summarize(root):
    groups = defaultdict(list)
    for path in sorted(root.glob("*/audit_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "same_state_forward":
                groups[(path.parent.name, row["role"])].append(row)
    result = {}
    for (mode, role), rows in sorted(groups.items()):
        key = mode + "/" + role
        result[key] = {
            "forwards": len(rows), "graph_forwards": sum(r["graph_used"] for r in rows),
            "batch_sizes": sorted(set(r["batch_size"] for r in rows)),
            "block_sizes": sorted(set(b for r in rows for b in r["real_lengths"])),
            "independent_hidden": aggregate([s["hidden"] for r in rows for s in r["requests"]]),
            "independent_logits": aggregate([s["logits"] for r in rows for s in r["requests"] if "logits" in s]),
        }
        for field in ("same_batch_eager_hidden", "same_batch_eager_logits", "restored_hidden", "restored_logits"):
            result[key][field] = aggregate([r[field] for r in rows if field in r])
    result["limits"] = ["not a complete correctness certificate", "not a throughput benchmark",
                        "no independent target-only transcript comparison yet",
                        "inspect top-1 differences even if relative hidden error is small"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("Preserve old summaries")
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
