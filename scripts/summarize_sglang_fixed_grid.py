from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(r"fixed_b(?P<block>\d+)_c(?P<concurrency>\d+)\.json$")
PROFILE_RE = re.compile(r"fixed_b(?P<block>\d+)\.jsonl$")


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[idx])


def _timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    timing_keys = sorted({key for rec in records for key in (rec.get("timings_ms") or {})})
    timings = {}
    for key in timing_keys:
        vals = [
            float((rec.get("timings_ms") or {}).get(key))
            for rec in records
            if key in (rec.get("timings_ms") or {})
        ]
        timings[key] = {
            "mean_ms": _mean(vals),
            "p50_ms": _percentile(vals, 0.50),
            "p95_ms": _percentile(vals, 0.95),
        }
    total = [float(rec.get("total_profiled_ms", 0.0)) for rec in records]
    batch_sizes = [float(rec.get("batch_size", 0.0)) for rec in records]
    commit_lens = [float(rec.get("commit_lens_mean", 0.0)) for rec in records]
    correct_drafts = [
        float(rec.get("num_correct_drafts_sum", 0.0)) / max(1, int(rec.get("batch_size", 1)))
        for rec in records
    ]
    return {
        "profile_cycles": len(records),
        "mean_observed_batch_size": _mean(batch_sizes),
        "mean_total_profiled_ms": _mean(total),
        "p50_total_profiled_ms": _percentile(total, 0.50),
        "p95_total_profiled_ms": _percentile(total, 0.95),
        "mean_commit_len": _mean(commit_lens),
        "mean_correct_drafts": _mean(correct_drafts),
        "timings": timings,
    }


def _load_profile_records(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    records_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((run_dir / "profiles").glob("fixed_b*.jsonl")):
        match = PROFILE_RE.match(path.name)
        if match is None:
            continue
        block = int(match.group("block"))
        with path.open() as f:
            for line in f:
                if line.strip():
                    records_by_block[block].append(json.loads(line))
    return records_by_block


def _nearest_profile_summary(
    records: list[dict[str, Any]],
    *,
    concurrency: int,
) -> dict[str, Any]:
    if not records:
        return {}
    exact = [rec for rec in records if int(rec.get("batch_size", -1)) == concurrency]
    if exact:
        return _timing_summary(exact)
    nearest_batch = min(
        {int(rec.get("batch_size", 0)) for rec in records},
        key=lambda batch_size: abs(batch_size - concurrency),
    )
    out = _timing_summary([rec for rec in records if int(rec.get("batch_size", -1)) == nearest_batch])
    out["profile_batch_size_note"] = f"no exact batch_size={concurrency}; used nearest observed batch_size={nearest_batch}"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fixed-block SGLang DFlash concurrency grid.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_records = _load_profile_records(args.run_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted((args.run_dir / "results").glob("fixed_b*_c*.json")):
        match = RESULT_RE.match(path.name)
        if match is None:
            continue
        block = int(match.group("block"))
        concurrency = int(match.group("concurrency"))
        result = json.loads(path.read_text())
        profile = _nearest_profile_summary(profile_records.get(block, []), concurrency=concurrency)
        timing = profile.get("timings", {})
        row = {
            "block": block,
            "concurrency": concurrency,
            "throughput_tok_s": result.get("throughput_tok_s"),
            "mean_latency_s": result.get("mean_latency_s"),
            "p50_latency_s": result.get("p50_latency_s"),
            "p95_latency_s": result.get("p95_latency_s"),
            "mean_completion_tokens": result.get("mean_completion_tokens"),
            "mean_spec_accept_length": result.get("mean_spec_accept_length"),
            "mean_spec_verify_ct": result.get("mean_spec_verify_ct"),
            "total_completion_tokens": result.get("total_completion_tokens"),
            "wall_time_s": result.get("wall_time_s"),
            "profile_cycles": profile.get("profile_cycles"),
            "profile_mean_batch_size": profile.get("mean_observed_batch_size"),
            "cycle_total_ms": profile.get("mean_total_profiled_ms"),
            "cycle_p50_total_ms": profile.get("p50_total_profiled_ms"),
            "cycle_p95_total_ms": profile.get("p95_total_profiled_ms"),
            "cycle_mean_commit_len": profile.get("mean_commit_len"),
            "cycle_mean_correct_drafts": profile.get("mean_correct_drafts"),
            "draft_forward_ms": (timing.get("draft_forward") or {}).get("mean_ms"),
            "target_verify_forward_ms": (timing.get("target_verify_forward") or {}).get("mean_ms"),
            "accept_verify_ms": (timing.get("accept_verify") or {}).get("mean_ms"),
            "draft_sample_ms": (timing.get("draft_sample") or {}).get("mean_ms"),
            "profile_note": profile.get("profile_batch_size_note"),
            "result_path": str(path),
        }
        rows.append(row)

    rows.sort(key=lambda row: (int(row["concurrency"]), int(row["block"])))
    payload = {
        "run_dir": str(args.run_dir),
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if rows:
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
