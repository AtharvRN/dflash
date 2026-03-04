#!/usr/bin/env python3
"""Aggregate C=16 policy-matrix runs into a single markdown/csv summary."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


def _to_float(raw: str) -> Optional[float]:
    s = raw.strip().strip("`").replace(",", "")
    if not s or s.upper() == "N/A":
        return None
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _find_metric_table_value(md: str, heading: str) -> Optional[float]:
    lines = md.splitlines()
    marker = f"### {heading}"
    for i, line in enumerate(lines):
        if line.strip() != marker:
            continue
        for j in range(i + 1, min(i + 14, len(lines))):
            m = re.match(r"^\|\s*value\s*\|\s*([^|]+)\|", lines[j].strip())
            if m:
                return _to_float(m.group(1))
    return None


def _find_usage_line(md: str, heading: str) -> Optional[str]:
    lines = md.splitlines()
    marker = f"### {heading}"
    for i, line in enumerate(lines):
        if line.strip() != marker:
            continue
        for j in range(i + 1, min(i + 16, len(lines))):
            m = re.match(r"^\|\s*16\s*\|\s*([^|]+)\|", lines[j].strip())
            if m:
                return m.group(1).strip()
    return None


def _find_bullet(md: str, name: str) -> Optional[float]:
    pat = rf"-\s+{re.escape(name)}:\s+`([^`]+)`"
    m = re.search(pat, md)
    if not m:
        return None
    return _to_float(m.group(1))


def _extract_block_histogram(md: str) -> str:
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "### Block-Size Cycle Histogram":
            hist: List[str] = []
            for j in range(i + 1, min(i + 40, len(lines))):
                cur = lines[j].strip()
                if not cur.startswith("|"):
                    if hist:
                        break
                    continue
                m = re.match(r"^\|\s*(\d+)\s*\|\s*([0-9,]+)\s*\|\s*([0-9.]+%)\s*\|", cur)
                if m:
                    bs = m.group(1)
                    pct = m.group(3)
                    hist.append(f"{bs}:{pct}")
            if hist:
                return ", ".join(hist)
            break
    return ""


@dataclass
class Row:
    case: str
    mode: str
    baseline_toks_s: Optional[float]
    dflash_toks_s: Optional[float]
    speedup_reported: Optional[float]
    speedup_vs_global: Optional[float]
    tau: Optional[float]
    accept_rate: Optional[float]
    verify_per_s: Optional[float]
    draft_tok_per_s: Optional[float]
    draft_ms_per_cycle: Optional[float]
    verify_ms_per_cycle: Optional[float]
    runtime_bs_usage: str
    cycle_hist: str
    run_tag: str
    md_path: str
    call_summary_path: str


def _mode_from_case(case: str) -> str:
    if case == "baseline_and_static_bs8":
        return "baseline+static_bs8"
    if case.startswith("static_bs"):
        return case
    if case.startswith("adaptive_"):
        return case
    return "other"


def _collect_one(run_dir: str, prefix: str) -> Optional[Row]:
    run_tag = os.path.basename(run_dir.rstrip("/"))
    case = run_tag[len(prefix) + 1 :] if run_tag.startswith(prefix + "_") else run_tag

    md_path = os.path.join(run_dir, f"{run_tag}.md")
    call_summary_path = os.path.join(run_dir, f"{run_tag}_calls_summary.md")
    if not os.path.isfile(md_path):
        return None

    main_md = _read_text(md_path)
    call_md = _read_text(call_summary_path) if os.path.isfile(call_summary_path) else ""

    baseline_toks = _find_metric_table_value(main_md, "Baseline output tok/s")
    dflash_toks = _find_metric_table_value(main_md, "DFLASH output tok/s")
    speedup = _find_metric_table_value(main_md, "Speedup (DFLASH / baseline)")
    tau = _find_metric_table_value(main_md, "DFLASH tau (accept length)")
    accept_rate = _find_metric_table_value(main_md, "DFLASH acceptance rate")
    verify_per_s = _find_metric_table_value(main_md, "DFLASH verify calls per second")
    draft_tok_per_s = _find_metric_table_value(main_md, "DFLASH drafted tokens per second")
    runtime_bs_usage = _find_usage_line(main_md, "DFLASH dynamic block usage (chunk counts)") or ""

    draft_cycle_s = _find_bullet(call_md, "weighted_draft_time_per_cycle_s")
    verify_cycle_s = _find_bullet(call_md, "weighted_verify_time_per_cycle_s")
    cycle_hist = _extract_block_histogram(call_md)

    return Row(
        case=case,
        mode=_mode_from_case(case),
        baseline_toks_s=baseline_toks,
        dflash_toks_s=dflash_toks,
        speedup_reported=speedup,
        speedup_vs_global=None,
        tau=tau,
        accept_rate=accept_rate,
        verify_per_s=verify_per_s,
        draft_tok_per_s=draft_tok_per_s,
        draft_ms_per_cycle=(draft_cycle_s * 1000.0 if draft_cycle_s is not None else None),
        verify_ms_per_cycle=(verify_cycle_s * 1000.0 if verify_cycle_s is not None else None),
        runtime_bs_usage=runtime_bs_usage,
        cycle_hist=cycle_hist,
        run_tag=run_tag,
        md_path=md_path,
        call_summary_path=call_summary_path,
    )


def _fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _write_csv(rows: List[Row], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "mode",
                "run_tag",
                "baseline_toks_s",
                "dflash_toks_s",
                "speedup_reported",
                "speedup_vs_global",
                "tau",
                "accept_rate",
                "verify_per_s",
                "draft_tok_per_s",
                "draft_ms_per_cycle",
                "verify_ms_per_cycle",
                "runtime_bs_usage",
                "cycle_hist",
                "md_path",
                "call_summary_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.case,
                    r.mode,
                    r.run_tag,
                    r.baseline_toks_s,
                    r.dflash_toks_s,
                    r.speedup_reported,
                    r.speedup_vs_global,
                    r.tau,
                    r.accept_rate,
                    r.verify_per_s,
                    r.draft_tok_per_s,
                    r.draft_ms_per_cycle,
                    r.verify_ms_per_cycle,
                    r.runtime_bs_usage,
                    r.cycle_hist,
                    r.md_path,
                    r.call_summary_path,
                ]
            )


def _write_md(rows: List[Row], path: str, baseline_global: Optional[float], prefix: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    lines.append("# SGLang C=16 Policy Matrix Summary")
    lines.append("")
    lines.append(f"- run_tag_prefix: `{prefix}`")
    lines.append(f"- runs_found: `{len(rows)}`")
    lines.append(
        f"- baseline_toks_s_reference: `{_fmt_num(baseline_global, 2)}`"
        if baseline_global is not None
        else "- baseline_toks_s_reference: `N/A`"
    )
    lines.append("")
    lines.append(
        "| case | dflash tok/s | speedup (reported) | speedup vs baseline ref | tau | accept_rate | verify/s | draft_tok/s | draft ms/cycle | verify ms/cycle | runtime block usage | cycle hist |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.case,
                    _fmt_num(r.dflash_toks_s, 2),
                    _fmt_num(r.speedup_reported, 3),
                    _fmt_num(r.speedup_vs_global, 3),
                    _fmt_num(r.tau, 3),
                    _fmt_num(r.accept_rate, 3),
                    _fmt_num(r.verify_per_s, 2),
                    _fmt_num(r.draft_tok_per_s, 2),
                    _fmt_num(r.draft_ms_per_cycle, 3),
                    _fmt_num(r.verify_ms_per_cycle, 3),
                    (r.runtime_bs_usage or "N/A"),
                    (r.cycle_hist or "N/A"),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Artifacts")
    for r in rows:
        lines.append(f"- `{r.case}`")
        lines.append(f"  - md: `{r.md_path}`")
        lines.append(f"  - calls_summary: `{r.call_summary_path}`")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag-prefix", required=True, help="Prefix used by matrix launcher.")
    ap.add_argument("--logs-root", default="logs", help="Root logs directory.")
    ap.add_argument("--output-md", default=None, help="Output markdown path.")
    ap.add_argument("--output-csv", default=None, help="Output csv path.")
    args = ap.parse_args()

    pattern = os.path.join(args.logs_root, f"{args.run_tag_prefix}_*")
    run_dirs = sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])
    rows: List[Row] = []
    for d in run_dirs:
        row = _collect_one(d, args.run_tag_prefix)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No run dirs found for prefix '{args.run_tag_prefix}' under '{args.logs_root}'.")

    baseline_global: Optional[float] = None
    for r in rows:
        if r.case == "baseline_and_static_bs8" and r.baseline_toks_s is not None:
            baseline_global = r.baseline_toks_s
            break
    if baseline_global is None:
        for r in rows:
            if r.baseline_toks_s is not None:
                baseline_global = r.baseline_toks_s
                break

    for r in rows:
        if baseline_global and baseline_global > 0 and r.dflash_toks_s is not None:
            r.speedup_vs_global = r.dflash_toks_s / baseline_global

    rows.sort(key=lambda r: r.case)

    out_md = args.output_md or os.path.join(args.logs_root, f"{args.run_tag_prefix}_matrix_summary.md")
    out_csv = args.output_csv or os.path.join(args.logs_root, f"{args.run_tag_prefix}_matrix_summary.csv")
    _write_md(rows, out_md, baseline_global, args.run_tag_prefix)
    _write_csv(rows, out_csv)
    print(f"Wrote matrix markdown: {out_md}")
    print(f"Wrote matrix csv: {out_csv}")


if __name__ == "__main__":
    main()
