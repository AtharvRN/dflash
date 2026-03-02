#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _as_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_int(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _first_present(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _load_rows(path: Path):
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}") from e

            # Older traces may nest meta under meta_info_raw.
            meta_raw = obj.get("meta_info_raw")
            meta = meta_raw if isinstance(meta_raw, dict) else {}

            merged = {}
            merged.update(meta)
            merged.update(obj)

            verify_ct = _as_int(
                _first_present(merged, ["spec_verify_ct", "spec_verify_calls"])
            )
            draft_tokens = _as_int(
                _first_present(
                    merged, ["spec_draft_token_num", "spec_draft_tokens"]
                )
            )
            accept_tokens = _as_int(
                _first_present(
                    merged, ["spec_accept_token_num", "spec_accepted_tokens"]
                )
            )
            accept_rate = _as_float(_first_present(merged, ["spec_accept_rate"]))
            accept_len = _as_float(
                _first_present(merged, ["spec_accept_length", "tau"])
            )

            runtime_bs = _as_int(
                _first_present(
                    merged,
                    [
                        "dflash_runtime_block_size",
                        "runtime_block_size",
                        "spec_runtime_block_size",
                        "spec_effective_block_size",
                    ],
                )
            )

            row = {
                "verify_ct": verify_ct if verify_ct is not None else 0,
                "draft_tokens": draft_tokens if draft_tokens is not None else 0,
                "accept_tokens": accept_tokens if accept_tokens is not None else 0,
                "accept_rate": accept_rate,
                "accept_len": accept_len,
                "draft_time_s": _as_float(
                    _first_present(merged, ["draft_time_s", "spec_draft_time_s"])
                ),
                "verify_time_s": _as_float(
                    _first_present(merged, ["verify_time_s", "spec_verify_time_s"])
                ),
                "draft_time_per_cycle_s": _as_float(
                    _first_present(
                        merged,
                        ["draft_time_per_cycle_s", "spec_draft_time_per_cycle_s"],
                    )
                ),
                "verify_time_per_cycle_s": _as_float(
                    _first_present(
                        merged,
                        ["verify_time_per_cycle_s", "spec_verify_time_per_cycle_s"],
                    )
                ),
                "completion_tokens": _as_int(_first_present(merged, ["completion_tokens"]))
                or 0,
                "runtime_bs_exact": runtime_bs,
            }

            if row["runtime_bs_exact"] is None:
                if row["verify_ct"] > 0 and row["draft_tokens"] >= 0:
                    inferred = int(round(row["draft_tokens"] / row["verify_ct"] + 1.0))
                    inferred = max(1, inferred)
                else:
                    inferred = None
                row["runtime_bs_inferred"] = inferred
            else:
                row["runtime_bs_inferred"] = row["runtime_bs_exact"]

            rows.append(row)
    return rows


def _wmean(values, weights):
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        if v is None:
            continue
        ww = float(w)
        if ww <= 0:
            continue
        num += float(v) * ww
        den += ww
    if den <= 0:
        return None
    return num / den


def _fmt(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def summarize(rows):
    total_calls = len(rows)
    verify_calls = sum(r["verify_ct"] for r in rows)
    draft_tokens = sum(r["draft_tokens"] for r in rows)
    accept_tokens = sum(r["accept_tokens"] for r in rows)
    completion_tokens = sum(r["completion_tokens"] for r in rows)

    weighted_accept_rate = (
        (accept_tokens / draft_tokens) if draft_tokens > 0 else None
    )
    weighted_tau = (completion_tokens / verify_calls) if verify_calls > 0 else None

    accept_rate_samples = [r["accept_rate"] for r in rows if r["accept_rate"] is not None]
    accept_len_samples = [r["accept_len"] for r in rows if r["accept_len"] is not None]

    exact_bs_present = any(r["runtime_bs_exact"] is not None for r in rows)
    bs_cycle_hist = Counter()
    bs_request_hist = Counter()
    per_bs = defaultdict(
        lambda: {"verify_ct": 0, "draft_tokens": 0, "accept_tokens": 0, "requests": 0}
    )

    for r in rows:
        bs = r["runtime_bs_inferred"]
        if bs is None:
            continue
        bs_request_hist[bs] += 1
        if r["verify_ct"] > 0:
            bs_cycle_hist[bs] += r["verify_ct"]
            per_bs[bs]["verify_ct"] += r["verify_ct"]
            per_bs[bs]["draft_tokens"] += r["draft_tokens"]
            per_bs[bs]["accept_tokens"] += r["accept_tokens"]
            per_bs[bs]["requests"] += 1

    most_common_cycle_bs = bs_cycle_hist.most_common(1)[0] if bs_cycle_hist else None
    most_common_request_bs = (
        bs_request_hist.most_common(1)[0] if bs_request_hist else None
    )

    draft_cycle_wmean_s = _wmean(
        [r["draft_time_per_cycle_s"] for r in rows], [r["verify_ct"] for r in rows]
    )
    verify_cycle_wmean_s = _wmean(
        [r["verify_time_per_cycle_s"] for r in rows], [r["verify_ct"] for r in rows]
    )
    draft_req_mean_s = mean(
        [r["draft_time_s"] for r in rows if r["draft_time_s"] is not None]
    ) if any(r["draft_time_s"] is not None for r in rows) else None
    verify_req_mean_s = mean(
        [r["verify_time_s"] for r in rows if r["verify_time_s"] is not None]
    ) if any(r["verify_time_s"] is not None for r in rows) else None

    lines = []
    lines.append("# SGLang Call Trace Summary")
    lines.append("")
    lines.append("## Core")
    lines.append(f"- calls: `{total_calls}`")
    lines.append(f"- total_verify_cycles: `{verify_calls}`")
    lines.append(f"- total_draft_tokens: `{draft_tokens}`")
    lines.append(f"- total_accepted_draft_tokens: `{accept_tokens}`")
    lines.append(f"- weighted_accept_rate (accepted/drafted): `{_fmt(weighted_accept_rate, 4)}`")
    lines.append(f"- weighted_tau (completion_tokens/verify_cycles): `{_fmt(weighted_tau, 4)}`")
    lines.append(
        f"- mean_spec_accept_rate (row avg): `{_fmt(mean(accept_rate_samples), 4) if accept_rate_samples else 'N/A'}`"
    )
    lines.append(
        f"- mean_spec_accept_length (row avg): `{_fmt(mean(accept_len_samples), 4) if accept_len_samples else 'N/A'}`"
    )
    lines.append("")
    lines.append("## Block Size")
    lines.append(
        f"- block_size_source: `{'exact_runtime_block_size' if exact_bs_present else 'inferred_from_draft_tokens_per_verify'}`"
    )
    if most_common_cycle_bs:
        lines.append(
            f"- most_common_block_size (cycle-weighted): `{most_common_cycle_bs[0]}` "
            f"(cycles={most_common_cycle_bs[1]})"
        )
    else:
        lines.append("- most_common_block_size (cycle-weighted): `N/A`")
    if most_common_request_bs:
        lines.append(
            f"- most_common_block_size (request-count): `{most_common_request_bs[0]}` "
            f"(requests={most_common_request_bs[1]})"
        )
    else:
        lines.append("- most_common_block_size (request-count): `N/A`")
    lines.append("")
    lines.append("## Timing")
    lines.append(
        f"- weighted_draft_time_per_cycle_s: `{_fmt(draft_cycle_wmean_s, 6)}` "
        f"({ _fmt(draft_cycle_wmean_s * 1000.0, 3) if draft_cycle_wmean_s is not None else 'N/A' } ms)"
    )
    lines.append(
        f"- weighted_verify_time_per_cycle_s: `{_fmt(verify_cycle_wmean_s, 6)}` "
        f"({ _fmt(verify_cycle_wmean_s * 1000.0, 3) if verify_cycle_wmean_s is not None else 'N/A' } ms)"
    )
    lines.append(
        f"- mean_draft_time_per_request_s: `{_fmt(draft_req_mean_s, 6)}`"
    )
    lines.append(
        f"- mean_verify_time_per_request_s: `{_fmt(verify_req_mean_s, 6)}`"
    )
    lines.append("")
    lines.append("## Per-Block Breakdown")
    lines.append("| block_size | requests | verify_cycles | draft_tokens | accepted_tokens | accept_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for bs in sorted(per_bs):
        d = per_bs[bs]
        ar = (d["accept_tokens"] / d["draft_tokens"]) if d["draft_tokens"] > 0 else None
        lines.append(
            f"| {bs} | {d['requests']} | {d['verify_ct']} | {d['draft_tokens']} | "
            f"{d['accept_tokens']} | {_fmt(ar, 4)} |"
        )

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Summarize SGLang benchmark call-trace JSONL (acceptance, block sizes, timings)."
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to *_calls.jsonl written by benchmark_sglang.py",
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional markdown output path. If omitted, prints to stdout only.",
    )
    args = ap.parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    md = summarize(rows)
    print(md)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)
        print(f"Wrote summary markdown: {args.output_md}")


if __name__ == "__main__":
    main()
