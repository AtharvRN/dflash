#!/usr/bin/env python3
"""Summarize static-only SGLang dataset x concurrency sweeps."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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


def _fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


@dataclass
class Row:
    dataset: str
    conc: int
    case: str
    block_size: int
    run_tag: str
    md_path: str
    baseline_toks_s: Optional[float]
    dflash_toks_s: Optional[float]
    speedup_reported: Optional[float]
    speedup_vs_ref: Optional[float]
    tau: Optional[float]
    accept_rate: Optional[float]
    verify_per_s: Optional[float]
    draft_tok_per_s: Optional[float]


def _parse_run_tag(run_tag: str, prefix: str) -> Optional[Tuple[str, int, str, int]]:
    # <prefix>_<dataset>_c<conc>_<case>, where case is one of:
    # - baseline_and_static_bs<k>
    # - static_bs<k>
    pat = re.compile(
        rf"^{re.escape(prefix)}_(?P<dataset>.+?)_c(?P<conc>\d+)_(?P<case>(?:baseline_and_static_bs|static_bs)(?P<bs>\d+))$"
    )
    m = pat.match(run_tag)
    if not m:
        return None
    dataset = m.group("dataset")
    conc = int(m.group("conc"))
    case = m.group("case")
    block_size = int(m.group("bs"))
    return dataset, conc, case, block_size


def _collect_rows(prefix: str, logs_root: str) -> List[Row]:
    pattern = os.path.join(logs_root, f"{prefix}_*")
    run_dirs = sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])
    rows: List[Row] = []
    for d in run_dirs:
        run_tag = os.path.basename(d.rstrip("/"))
        parsed = _parse_run_tag(run_tag, prefix)
        if parsed is None:
            continue
        dataset, conc, case, block_size = parsed
        md_path = os.path.join(d, f"{run_tag}.md")
        if not os.path.isfile(md_path):
            continue
        main_md = _read_text(md_path)
        rows.append(
            Row(
                dataset=dataset,
                conc=conc,
                case=case,
                block_size=block_size,
                run_tag=run_tag,
                md_path=md_path,
                baseline_toks_s=_find_metric_table_value(main_md, "Baseline output tok/s"),
                dflash_toks_s=_find_metric_table_value(main_md, "DFLASH output tok/s"),
                speedup_reported=_find_metric_table_value(main_md, "Speedup (DFLASH / baseline)"),
                speedup_vs_ref=None,
                tau=_find_metric_table_value(main_md, "DFLASH tau (accept length)"),
                accept_rate=_find_metric_table_value(main_md, "DFLASH acceptance rate"),
                verify_per_s=_find_metric_table_value(main_md, "DFLASH verify calls per second"),
                draft_tok_per_s=_find_metric_table_value(main_md, "DFLASH drafted tokens per second"),
            )
        )
    return rows


def _write_csv(rows: List[Row], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "concurrency",
                "case",
                "block_size",
                "run_tag",
                "baseline_toks_s",
                "dflash_toks_s",
                "speedup_reported",
                "speedup_vs_ref",
                "tau",
                "accept_rate",
                "verify_per_s",
                "draft_tok_per_s",
                "md_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.dataset,
                    r.conc,
                    r.case,
                    r.block_size,
                    r.run_tag,
                    r.baseline_toks_s,
                    r.dflash_toks_s,
                    r.speedup_reported,
                    r.speedup_vs_ref,
                    r.tau,
                    r.accept_rate,
                    r.verify_per_s,
                    r.draft_tok_per_s,
                    r.md_path,
                ]
            )


def _write_md(rows: List[Row], path: str, prefix: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    lines.append("# SGLang Static Dataset x Concurrency Summary")
    lines.append("")
    lines.append(f"- run_tag_prefix: `{prefix}`")
    lines.append(f"- runs_found: `{len(rows)}`")
    lines.append("")

    lines.append(
        "| dataset | conc | bs | dflash tok/s | speedup vs baseline-ref | tau | accept_rate | verify/s | draft_tok/s | run_tag |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.dataset,
                    str(r.conc),
                    str(r.block_size),
                    _fmt_num(r.dflash_toks_s, 2),
                    _fmt_num(r.speedup_vs_ref, 3),
                    _fmt_num(r.tau, 3),
                    _fmt_num(r.accept_rate, 3),
                    _fmt_num(r.verify_per_s, 2),
                    _fmt_num(r.draft_tok_per_s, 2),
                    f"`{r.run_tag}`",
                ]
            )
            + " |"
        )

    # Best-bs table
    best_rows: List[Tuple[str, int, int, Optional[float], Optional[float], Optional[float]]] = []
    grouped: Dict[Tuple[str, int], List[Row]] = {}
    for r in rows:
        grouped.setdefault((r.dataset, r.conc), []).append(r)
    for (dataset, conc), grp in sorted(grouped.items()):
        best = None
        for r in grp:
            if r.dflash_toks_s is None:
                continue
            if best is None or r.dflash_toks_s > best.dflash_toks_s:
                best = r
        if best is None:
            continue
        best_rows.append((dataset, conc, best.block_size, best.dflash_toks_s, best.speedup_vs_ref, best.tau))

    lines.append("")
    lines.append("## Best Block Size by Dataset/Concurrency")
    lines.append("")
    lines.append("| dataset | conc | best bs | best tok/s | speedup vs baseline-ref | tau |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset, conc, bs, toks, speedup, tau in best_rows:
        lines.append(
            f"| {dataset} | {conc} | {bs} | {_fmt_num(toks, 2)} | {_fmt_num(speedup, 3)} | {_fmt_num(tau, 3)} |"
        )

    # Variation summary: how best bs changes with conc per dataset.
    lines.append("")
    lines.append("## Best-BS Variation by Dataset")
    lines.append("")
    lines.append("| dataset | best bs by conc |")
    lines.append("|---|---|")
    by_ds: Dict[str, List[Tuple[int, int]]] = {}
    for dataset, conc, bs, *_ in best_rows:
        by_ds.setdefault(dataset, []).append((conc, bs))
    for ds in sorted(by_ds.keys()):
        parts = [f"c={c}:bs={bs}" for c, bs in sorted(by_ds[ds], key=lambda x: x[0])]
        lines.append(f"| {ds} | {', '.join(parts)} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag-prefix", required=True)
    ap.add_argument("--logs-root", default="logs")
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()

    rows = _collect_rows(args.run_tag_prefix, args.logs_root)
    if not rows:
        raise SystemExit(
            f"No run dirs found for prefix '{args.run_tag_prefix}' under '{args.logs_root}'."
        )

    # Baseline reference per (dataset, conc): from the baseline+static run.
    baseline_ref: Dict[Tuple[str, int], float] = {}
    for r in rows:
        key = (r.dataset, r.conc)
        if r.baseline_toks_s is not None and key not in baseline_ref:
            baseline_ref[key] = r.baseline_toks_s

    for r in rows:
        key = (r.dataset, r.conc)
        base = baseline_ref.get(key)
        if base is not None and base > 0 and r.dflash_toks_s is not None:
            r.speedup_vs_ref = r.dflash_toks_s / base

    rows.sort(key=lambda r: (r.dataset, r.conc, r.block_size))

    out_md = args.output_md or os.path.join(args.logs_root, f"{args.run_tag_prefix}_global_summary.md")
    out_csv = args.output_csv or os.path.join(args.logs_root, f"{args.run_tag_prefix}_global_summary.csv")
    _write_md(rows, out_md, args.run_tag_prefix)
    _write_csv(rows, out_csv)
    print(f"Wrote summary markdown: {out_md}")
    print(f"Wrote summary csv: {out_csv}")


if __name__ == "__main__":
    main()
