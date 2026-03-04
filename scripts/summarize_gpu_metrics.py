#!/usr/bin/env python3
"""Summarize GPU utilization/memory CSV emitted by scripts/record_gpu_metrics.sh."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def _to_float(value: str) -> float | None:
    v = value.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _parse_ts(value: str) -> datetime | None:
    v = value.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            pass
    return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def _fmt_num(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _build_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    per_gpu: dict[int, dict[str, Any]] = {}
    ts_vals: list[datetime] = []

    for row in rows:
        gpu_idx_f = _to_float(row.get("gpu_index", ""))
        if gpu_idx_f is None:
            continue
        gpu_idx = int(gpu_idx_f)
        ts = _parse_ts(row.get("timestamp", ""))
        if ts is not None:
            ts_vals.append(ts)

        gpu_name = row.get("gpu_name", "").strip()
        util_gpu = _to_float(row.get("utilization_gpu_pct", ""))
        util_mem = _to_float(row.get("utilization_memory_pct", ""))
        mem_total = _to_float(row.get("memory_total_mib", ""))
        mem_used = _to_float(row.get("memory_used_mib", ""))

        slot = per_gpu.setdefault(
            gpu_idx,
            {
                "gpu_index": gpu_idx,
                "gpu_name": gpu_name,
                "samples": 0,
                "utilization_gpu_pct": [],
                "utilization_memory_pct": [],
                "memory_total_mib": [],
                "memory_used_mib": [],
            },
        )
        slot["samples"] += 1
        if gpu_name and not slot["gpu_name"]:
            slot["gpu_name"] = gpu_name
        if util_gpu is not None:
            slot["utilization_gpu_pct"].append(util_gpu)
        if util_mem is not None:
            slot["utilization_memory_pct"].append(util_mem)
        if mem_total is not None:
            slot["memory_total_mib"].append(mem_total)
        if mem_used is not None:
            slot["memory_used_mib"].append(mem_used)

    gpu_summaries: list[dict[str, Any]] = []
    for gpu_idx in sorted(per_gpu.keys()):
        slot = per_gpu[gpu_idx]
        util_gpu = slot["utilization_gpu_pct"]
        util_mem = slot["utilization_memory_pct"]
        mem_total_vals = slot["memory_total_mib"]
        mem_used_vals = slot["memory_used_mib"]
        mem_total_ref = mem_total_vals[0] if mem_total_vals else None
        mem_peak = max(mem_used_vals) if mem_used_vals else None
        mem_peak_pct = (
            (mem_peak / mem_total_ref * 100.0)
            if mem_peak is not None and mem_total_ref not in (None, 0)
            else None
        )
        gpu_summaries.append(
            {
                "gpu_index": gpu_idx,
                "gpu_name": slot["gpu_name"] or f"GPU {gpu_idx}",
                "samples": slot["samples"],
                "util_gpu_avg_pct": (sum(util_gpu) / len(util_gpu) if util_gpu else None),
                "util_gpu_p95_pct": _percentile(util_gpu, 95),
                "util_gpu_max_pct": max(util_gpu) if util_gpu else None,
                "util_mem_avg_pct": (sum(util_mem) / len(util_mem) if util_mem else None),
                "util_mem_p95_pct": _percentile(util_mem, 95),
                "mem_total_mib": mem_total_ref,
                "mem_used_avg_mib": (sum(mem_used_vals) / len(mem_used_vals) if mem_used_vals else None),
                "mem_used_p95_mib": _percentile(mem_used_vals, 95),
                "mem_used_peak_mib": mem_peak,
                "mem_used_peak_pct_of_total": mem_peak_pct,
            }
        )

    start_ts = min(ts_vals) if ts_vals else None
    end_ts = max(ts_vals) if ts_vals else None
    duration_s = (end_ts - start_ts).total_seconds() if start_ts and end_ts else None
    total_samples = sum(g["samples"] for g in gpu_summaries)

    return {
        "rows": len(rows),
        "total_samples": total_samples,
        "start_timestamp": start_ts.isoformat(sep=" ") if start_ts else None,
        "end_timestamp": end_ts.isoformat(sep=" ") if end_ts else None,
        "duration_s": duration_s,
        "gpus": gpu_summaries,
    }


def _render_markdown(summary: dict[str, Any], source_path: Path) -> str:
    lines: list[str] = []
    lines.append("# GPU Metrics Summary")
    lines.append("")
    lines.append(f"- source: `{source_path}`")
    lines.append(f"- rows: `{summary['rows']}`")
    lines.append(f"- total_samples: `{summary['total_samples']}`")
    lines.append(f"- start_timestamp: `{summary.get('start_timestamp') or 'N/A'}`")
    lines.append(f"- end_timestamp: `{summary.get('end_timestamp') or 'N/A'}`")
    lines.append(f"- duration_s: `{_fmt_num(summary.get('duration_s'))}`")
    lines.append("")
    lines.append(
        "| gpu_index | gpu_name | samples | util_gpu_avg_pct | util_gpu_p95_pct | util_gpu_max_pct | util_mem_avg_pct | util_mem_p95_pct | mem_total_mib | mem_used_avg_mib | mem_used_p95_mib | mem_used_peak_mib | mem_used_peak_pct_of_total |"
    )
    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for g in summary["gpus"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(g["gpu_index"]),
                    g["gpu_name"],
                    str(g["samples"]),
                    _fmt_num(g["util_gpu_avg_pct"]),
                    _fmt_num(g["util_gpu_p95_pct"]),
                    _fmt_num(g["util_gpu_max_pct"]),
                    _fmt_num(g["util_mem_avg_pct"]),
                    _fmt_num(g["util_mem_p95_pct"]),
                    _fmt_num(g["mem_total_mib"], 1),
                    _fmt_num(g["mem_used_avg_mib"], 1),
                    _fmt_num(g["mem_used_p95_mib"], 1),
                    _fmt_num(g["mem_used_peak_mib"], 1),
                    _fmt_num(g["mem_used_peak_pct_of_total"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to GPU metrics CSV.")
    parser.add_argument(
        "--output-md",
        required=True,
        help="Path to output markdown summary.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_md_path = Path(args.output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    summary = _build_summary(rows)
    output_md_path.write_text(
        _render_markdown(summary, input_path), encoding="utf-8"
    )
    print(f"Wrote GPU summary markdown: {output_md_path}")

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote GPU summary JSON: {out_json_path}")


if __name__ == "__main__":
    main()
