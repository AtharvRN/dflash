#!/usr/bin/env python3
import argparse
import json
import re
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


def _normalize_hist(raw_hist):
    if not isinstance(raw_hist, dict):
        return {}
    out = {}
    for k, v in raw_hist.items():
        bs = _as_int(k)
        ct = _as_int(v)
        if bs is None or ct is None:
            continue
        if bs <= 0 or ct <= 0:
            continue
        out[int(bs)] = int(ct)
    return out


def _normalize_cycle_trace(raw_trace):
    if not isinstance(raw_trace, list):
        return []

    out = []
    for item in raw_trace:
        if not isinstance(item, dict):
            continue
        runtime_bs = _as_int(item.get("runtime_block_size"))
        if runtime_bs is not None and runtime_bs <= 0:
            runtime_bs = None
        out.append(
            {
                "cycle_idx": _as_int(item.get("cycle_idx")),
                "runtime_block_size": runtime_bs,
                "accepted_draft_tokens": _as_int(item.get("accepted_draft_tokens")),
                "accept_length": _as_float(item.get("accept_length")),
                "accept_rate": _as_float(item.get("accept_rate")),
                "draft_time_s": _as_float(item.get("draft_time_s")),
                "verify_time_s": _as_float(item.get("verify_time_s")),
            }
        )
    return out


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
                raise ValueError(
                    f"Invalid JSON at line {i}: {e}. "
                    f"Expected a JSONL call trace (e.g. *_calls.jsonl), got: {path}"
                ) from e

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
            runtime_bs_hist = _normalize_hist(
                _first_present(
                    merged,
                    [
                        "spec_runtime_bs_hist",
                        "dflash_runtime_bs_hist",
                        "runtime_bs_hist",
                    ],
                )
            )
            spec_cycle_trace = _normalize_cycle_trace(
                _first_present(merged, ["spec_cycle_trace"])
            )

            if not runtime_bs_hist and spec_cycle_trace:
                derived_hist = Counter()
                for c in spec_cycle_trace:
                    bs = c.get("runtime_block_size")
                    if bs is not None and int(bs) > 0:
                        derived_hist[int(bs)] += 1
                runtime_bs_hist = dict(derived_hist)

            if verify_ct is None and spec_cycle_trace:
                verify_ct = len(spec_cycle_trace)

            row = {
                "verify_ct": verify_ct if verify_ct is not None else 0,
                "draft_tokens": draft_tokens if draft_tokens is not None else 0,
                "accept_tokens": accept_tokens if accept_tokens is not None else 0,
                "accept_rate": accept_rate,
                "accept_len": accept_len,
                "request_idx": _as_int(
                    _first_present(
                        merged,
                        ["request_local_idx", "request_idx", "call_idx"],
                    )
                ),
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
                "runtime_bs_hist": runtime_bs_hist,
                "spec_cycle_trace": spec_cycle_trace,
            }

            if row["runtime_bs_exact"] is None:
                if runtime_bs_hist:
                    inferred = max(
                        sorted(runtime_bs_hist.items()),
                        key=lambda kv: kv[1],
                    )[0]
                elif row["verify_ct"] > 0 and row["draft_tokens"] >= 0:
                    inferred = int(round(row["draft_tokens"] / row["verify_ct"] + 1.0))
                    inferred = max(1, inferred)
                else:
                    inferred = None
                row["runtime_bs_inferred"] = inferred
            else:
                row["runtime_bs_inferred"] = row["runtime_bs_exact"]

            rows.append(row)
    return rows


def _resolve_candidate_path(candidate: str, *, log_path: Path) -> Path | None:
    cand = candidate.strip().strip("'").strip('"')
    if not cand:
        return None
    p = Path(cand)
    if p.is_absolute() and p.exists():
        return p

    rel1 = (log_path.parent / p).resolve()
    if rel1.exists():
        return rel1

    rel2 = (Path.cwd() / p).resolve()
    if rel2.exists():
        return rel2
    return None


def _extract_calls_jsonl_from_log(log_path: Path) -> Path | None:
    """
    Try to find the generated call-trace jsonl path from a benchmark run log.
    Supports lines like:
    - Wrote per-call JSONL trace to: ...
    - Call trace: ...
    - command: ... --save-call-trace-path <path> ...
    """
    patterns = [
        re.compile(r"Wrote per-call JSONL trace to:\s*(\S+)"),
        re.compile(r"Call trace:\s*(\S+)"),
        re.compile(r"--save-call-trace-path\s+(\S+)"),
    ]
    with log_path.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            for pat in patterns:
                m = pat.search(s)
                if not m:
                    continue
                resolved = _resolve_candidate_path(m.group(1), log_path=log_path)
                if resolved is not None:
                    return resolved
    return None


def _resolve_input_path(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() == ".log":
        traced = _extract_calls_jsonl_from_log(input_path)
        if traced is None:
            raise ValueError(
                "Input is a .log file, but no call-trace JSONL path was found in it. "
                "Pass --input <..._calls.jsonl> directly, or ensure log contains "
                "'Wrote per-call JSONL trace to: ...'."
            )
        print(f"Resolved call trace from log: {traced}")
        return traced

    return input_path


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

    exact_hist_present = any(bool(r.get("runtime_bs_hist")) for r in rows)
    exact_bs_present = exact_hist_present or any(
        r["runtime_bs_exact"] is not None for r in rows
    )
    cycle_trace_rows = [
        c
        for r in rows
        for c in (r.get("spec_cycle_trace") or [])
        if isinstance(c, dict)
    ]

    cycle_trace_bs_hist = Counter()
    cycle_trace_accept_len_vals = []
    cycle_trace_accept_rate_vals = []
    cycle_trace_draft_time_vals = []
    cycle_trace_verify_time_vals = []
    cycle_trace_per_bs = defaultdict(
        lambda: {
            "cycles": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "accept_len_sum": 0.0,
            "accept_len_cnt": 0,
            "accept_rate_sum": 0.0,
            "accept_rate_cnt": 0,
            "draft_time_sum": 0.0,
            "draft_time_cnt": 0,
            "verify_time_sum": 0.0,
            "verify_time_cnt": 0,
        }
    )

    for c in cycle_trace_rows:
        bs = _as_int(c.get("runtime_block_size"))
        if bs is None or bs <= 0:
            continue
        bs = int(bs)
        cycle_trace_bs_hist[bs] += 1
        accepted = _as_int(c.get("accepted_draft_tokens"))
        proposed = max(0, bs - 1)

        d = cycle_trace_per_bs[bs]
        d["cycles"] += 1
        d["draft_tokens"] += proposed
        if accepted is not None and accepted >= 0:
            d["accepted_tokens"] += int(accepted)

        accept_len = _as_float(c.get("accept_length"))
        if accept_len is not None:
            cycle_trace_accept_len_vals.append(accept_len)
            d["accept_len_sum"] += float(accept_len)
            d["accept_len_cnt"] += 1

        accept_rate = _as_float(c.get("accept_rate"))
        if accept_rate is not None:
            cycle_trace_accept_rate_vals.append(accept_rate)
            d["accept_rate_sum"] += float(accept_rate)
            d["accept_rate_cnt"] += 1

        draft_time_s = _as_float(c.get("draft_time_s"))
        if draft_time_s is not None:
            cycle_trace_draft_time_vals.append(draft_time_s)
            d["draft_time_sum"] += float(draft_time_s)
            d["draft_time_cnt"] += 1

        verify_time_s = _as_float(c.get("verify_time_s"))
        if verify_time_s is not None:
            cycle_trace_verify_time_vals.append(verify_time_s)
            d["verify_time_sum"] += float(verify_time_s)
            d["verify_time_cnt"] += 1

    bs_cycle_hist = Counter()
    bs_request_hist = Counter()
    per_bs = defaultdict(
        lambda: {"verify_ct": 0, "draft_tokens": 0, "accept_tokens": 0, "requests": 0}
    )

    for r in rows:
        if r["runtime_bs_hist"]:
            # Exact cycle counts by block-size for this request.
            for bs, ct in r["runtime_bs_hist"].items():
                bs_cycle_hist[bs] += int(ct)
                per_bs[bs]["verify_ct"] += int(ct)
                # Drafted tokens are exact given block size and cycle count.
                per_bs[bs]["draft_tokens"] += int(ct) * max(0, int(bs) - 1)
            req_mode_bs = max(
                sorted(r["runtime_bs_hist"].items()),
                key=lambda kv: kv[1],
            )[0]
            bs_request_hist[int(req_mode_bs)] += 1
            per_bs[int(req_mode_bs)]["requests"] += 1
            continue

        bs = r["runtime_bs_inferred"]
        if bs is None:
            continue
        bs_request_hist[int(bs)] += 1
        if r["verify_ct"] > 0:
            bs_cycle_hist[int(bs)] += r["verify_ct"]
            per_bs[int(bs)]["verify_ct"] += r["verify_ct"]
            per_bs[int(bs)]["draft_tokens"] += r["draft_tokens"]
            per_bs[int(bs)]["accept_tokens"] += r["accept_tokens"]
            per_bs[int(bs)]["requests"] += 1

    if cycle_trace_bs_hist:
        # Prefer exact cycle-level block histogram when available.
        bs_cycle_hist = cycle_trace_bs_hist

    most_common_cycle_bs = bs_cycle_hist.most_common(1)[0] if bs_cycle_hist else None
    most_common_request_bs = (
        bs_request_hist.most_common(1)[0] if bs_request_hist else None
    )

    requests_with_runtime_hist = [r for r in rows if r["runtime_bs_hist"]]
    mixed_bs_requests = sum(
        1 for r in requests_with_runtime_hist if len(r["runtime_bs_hist"]) > 1
    )
    unique_bs_per_request = (
        mean([len(r["runtime_bs_hist"]) for r in requests_with_runtime_hist])
        if requests_with_runtime_hist
        else None
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
        f"- block_size_source: `{'exact_runtime_bs_hist' if exact_hist_present else ('exact_runtime_block_size' if exact_bs_present else 'inferred_from_draft_tokens_per_verify')}`"
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
    if requests_with_runtime_hist:
        lines.append(
            f"- requests_with_exact_runtime_hist: `{len(requests_with_runtime_hist)}`"
        )
        lines.append(
            f"- requests_with_mixed_block_sizes: `{mixed_bs_requests}`"
        )
        lines.append(
            f"- mean_unique_block_sizes_per_request: `{_fmt(unique_bs_per_request, 3)}`"
        )
    else:
        lines.append("- requests_with_exact_runtime_hist: `0`")
    if cycle_trace_rows:
        lines.append(f"- cycle_trace_rows: `{len(cycle_trace_rows)}`")
        lines.append(
            f"- cycle_trace_rows_with_timing: `{len(cycle_trace_draft_time_vals)}` draft, `{len(cycle_trace_verify_time_vals)}` verify"
        )
    else:
        lines.append("- cycle_trace_rows: `0`")

    lines.append("")
    lines.append("### Block-Size Cycle Histogram")
    lines.append("| block_size | verify_cycles | cycle_pct |")
    lines.append("|---:|---:|---:|")
    total_cycle_hist = sum(bs_cycle_hist.values())
    for bs, ct in sorted(bs_cycle_hist.items()):
        pct = (float(ct) / float(total_cycle_hist)) if total_cycle_hist > 0 else 0.0
        lines.append(f"| {bs} | {ct} | {_fmt(pct * 100.0, 2)}% |")
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
    if cycle_trace_rows:
        cycle_draft_mean = (
            mean(cycle_trace_draft_time_vals)
            if cycle_trace_draft_time_vals
            else None
        )
        cycle_verify_mean = (
            mean(cycle_trace_verify_time_vals)
            if cycle_trace_verify_time_vals
            else None
        )
        lines.append(
            f"- mean_draft_time_per_cycle_from_trace_s: `{_fmt(cycle_draft_mean, 6)}` "
            f"({ _fmt(cycle_draft_mean * 1000.0, 3) if cycle_draft_mean is not None else 'N/A' } ms)"
        )
        lines.append(
            f"- mean_verify_time_per_cycle_from_trace_s: `{_fmt(cycle_verify_mean, 6)}` "
            f"({ _fmt(cycle_verify_mean * 1000.0, 3) if cycle_verify_mean is not None else 'N/A' } ms)"
        )
        lines.append(
            f"- mean_accept_length_per_cycle_from_trace: `{_fmt(mean(cycle_trace_accept_len_vals), 4) if cycle_trace_accept_len_vals else 'N/A'}`"
        )
        lines.append(
            f"- mean_accept_rate_per_cycle_from_trace: `{_fmt(mean(cycle_trace_accept_rate_vals), 4) if cycle_trace_accept_rate_vals else 'N/A'}`"
        )
    lines.append("")
    lines.append("## Per-Request Block Usage")
    if requests_with_runtime_hist:
        lines.append(
            "_Rows show how many verify cycles used each block size for that request._"
        )
        lines.append("| request_local_idx | verify_cycles | runtime_bs_hist | mode_bs | avg_bs |")
        lines.append("|---:|---:|---|---:|---:|")

        def _sort_key(rr):
            return (
                rr["request_idx"] if rr["request_idx"] is not None else 10**12,
                rr["verify_ct"],
            )

        preview_rows = sorted(requests_with_runtime_hist, key=_sort_key)[:32]
        for r in preview_rows:
            hist = r["runtime_bs_hist"]
            hist_str = ", ".join(f"{bs}:{ct}" for bs, ct in sorted(hist.items()))
            mode_bs = max(
                sorted(hist.items()),
                key=lambda kv: kv[1],
            )[0]
            total_ct = sum(hist.values())
            avg_bs = (
                sum(int(bs) * int(ct) for bs, ct in hist.items()) / float(total_ct)
                if total_ct > 0
                else None
            )
            req_idx = r["request_idx"] if r["request_idx"] is not None else -1
            lines.append(
                f"| {req_idx} | {r['verify_ct']} | `{hist_str}` | {mode_bs} | {_fmt(avg_bs, 3)} |"
            )
        if len(requests_with_runtime_hist) > len(preview_rows):
            lines.append("")
            lines.append(
                f"_Showing first {len(preview_rows)} requests out of {len(requests_with_runtime_hist)} with exact runtime block-size histograms._"
            )
    else:
        lines.append(
            "_No exact runtime block-size histogram in this trace; only inferred block size is available._"
        )
    if cycle_trace_rows:
        lines.append("")
        lines.append("## Per-Block Breakdown (Cycle Trace, Exact)")
        lines.append(
            "| block_size | verify_cycles | draft_tokens | accepted_tokens | accept_rate | mean_accept_length | mean_accept_rate | mean_draft_ms | mean_verify_ms |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for bs in sorted(cycle_trace_per_bs):
            d = cycle_trace_per_bs[bs]
            accept_rate = (
                (float(d["accepted_tokens"]) / float(d["draft_tokens"]))
                if d["draft_tokens"] > 0
                else None
            )
            mean_accept_length = (
                d["accept_len_sum"] / float(d["accept_len_cnt"])
                if d["accept_len_cnt"] > 0
                else None
            )
            mean_accept_rate = (
                d["accept_rate_sum"] / float(d["accept_rate_cnt"])
                if d["accept_rate_cnt"] > 0
                else None
            )
            mean_draft_ms = (
                (d["draft_time_sum"] / float(d["draft_time_cnt"])) * 1000.0
                if d["draft_time_cnt"] > 0
                else None
            )
            mean_verify_ms = (
                (d["verify_time_sum"] / float(d["verify_time_cnt"])) * 1000.0
                if d["verify_time_cnt"] > 0
                else None
            )
            lines.append(
                f"| {bs} | {d['cycles']} | {d['draft_tokens']} | {d['accepted_tokens']} | {_fmt(accept_rate, 4)} | "
                f"{_fmt(mean_accept_length, 4)} | {_fmt(mean_accept_rate, 4)} | {_fmt(mean_draft_ms, 3)} | {_fmt(mean_verify_ms, 3)} |"
            )

    lines.append("")
    lines.append("## Per-Block Breakdown")
    lines.append("| block_size | requests(mode) | verify_cycles | draft_tokens | accepted_tokens | accept_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for bs in sorted(per_bs):
        d = per_bs[bs]
        # accepted_tokens per block-size is exact only when rows don't mix block sizes.
        ar = (d["accept_tokens"] / d["draft_tokens"]) if d["draft_tokens"] > 0 else None
        accept_tokens_disp = (
            str(d["accept_tokens"]) if not exact_hist_present else "N/A"
        )
        accept_rate_disp = _fmt(ar, 4) if not exact_hist_present else "N/A"
        lines.append(
            f"| {bs} | {d['requests']} | {d['verify_ct']} | {d['draft_tokens']} | "
            f"{accept_tokens_disp} | {accept_rate_disp} |"
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
        help=(
            "Path to *_calls.jsonl written by benchmark_sglang.py. "
            "You may also pass a run .log file; the script will auto-resolve the calls JSONL."
        ),
    )
    ap.add_argument(
        "--output-md",
        "--output",
        type=Path,
        dest="output_md",
        default=None,
        help="Optional markdown output path. If omitted, prints to stdout only.",
    )
    args = ap.parse_args()

    input_path = _resolve_input_path(args.input)
    rows = _load_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    md = summarize(rows)
    print(md)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)
        print(f"Wrote summary markdown: {args.output_md}")


if __name__ == "__main__":
    main()
