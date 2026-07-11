from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DFlash per-cycle profile JSONL files.")
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    timing_keys = sorted(
        {key for rec in records for key in (rec.get("timings_ms") or {}).keys()}
    )
    timing_summary = {}
    for key in timing_keys:
        vals = [
            float((rec.get("timings_ms") or {}).get(key))
            for rec in records
            if key in (rec.get("timings_ms") or {})
        ]
        timing_summary[key] = {
            "mean_ms": mean(vals),
            "p50_ms": percentile(vals, 0.50),
            "p95_ms": percentile(vals, 0.95),
        }

    total_vals = [float(rec.get("total_profiled_ms", 0.0)) for rec in records]
    commit_means = [float(rec.get("commit_lens_mean", 0.0)) for rec in records]
    correct_means = [
        float(rec.get("num_correct_drafts_sum", 0.0)) / max(1, int(rec.get("batch_size", 1)))
        for rec in records
    ]
    blocks = [float(rec.get("runtime_block_size", rec.get("max_block_size", 0))) for rec in records]
    batch_sizes = [float(rec.get("batch_size", 0)) for rec in records]
    return {
        "cycles": len(records),
        "mean_total_profiled_ms": mean(total_vals),
        "p50_total_profiled_ms": percentile(total_vals, 0.50),
        "p95_total_profiled_ms": percentile(total_vals, 0.95),
        "mean_batch_size": mean(batch_sizes),
        "mean_runtime_block_size": mean(blocks),
        "mean_commit_len": mean(commit_means),
        "mean_correct_drafts": mean(correct_means),
        "timings": timing_summary,
    }


def main() -> None:
    args = parse_args()
    output: dict[str, Any] = {}
    for path in sorted(args.profile_dir.glob("*.jsonl")):
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not records:
            continue
        by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            block = record.get("runtime_block_size", record.get("max_block_size", "unknown"))
            by_block[str(block)].append(record)
        output[path.stem] = {
            "all": summarize_records(records),
            "by_runtime_block_size": {
                block: summarize_records(block_records)
                for block, block_records in sorted(by_block.items(), key=lambda item: item[0])
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
