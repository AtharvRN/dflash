#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunSummary:
    run_name: str
    throughput_tok_s: float
    total_completion_tokens: int
    wall_time_s: float
    cycles: int
    mean_verify_len: float
    mean_accept_length: float
    mean_verify_time_ms: float
    total_verify_time_s: float
    boundary_exact_rate: float
    boundary_within_1_rate: float
    boundary_mae: float
    boundary_signed_error: float
    boundary_under_rate: float
    boundary_over_rate: float
    verify_can_run_cuda_graph_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze c=1 DFLASH call traces, compare verify-length control quality, "
            "and estimate an oracle upper bound using the measured verify-time curve."
        )
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument(
        "--compare-run",
        action="append",
        default=[],
        help="Additional run names to compare against the baseline.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_md_metric(text: str, header: str) -> float | None:
    parts = text.split(header)
    if len(parts) < 2:
        return None
    for line in parts[1].strip().splitlines():
        if line.startswith("| value |"):
            raw = line.split("|")[2].strip().replace(",", "")
            return float(raw) if raw != "N/A" else None
    return None


def load_run(base_dir: Path, run_name: str) -> tuple[RunSummary, list[dict[str, float]]]:
    md_path = base_dir / run_name / f"{run_name}.md"
    calls_path = base_dir / run_name / f"{run_name}_calls.jsonl"
    md_text = md_path.read_text(encoding="utf-8")

    throughput_tok_s = parse_md_metric(md_text, "### DFLASH output tok/s")
    if throughput_tok_s is None:
        raise ValueError(f"Missing throughput in {md_path}")

    cycles: list[dict[str, float]] = []
    total_completion_tokens = 0
    with calls_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total_completion_tokens += int(row.get("completion_tokens", 0))
            for cyc in row.get("spec_cycle_trace") or []:
                dec = cyc.get("confidence_gate_decision") or {}
                verify_len = dec.get("verify_token_num")
                if verify_len is None:
                    verify_len = cyc.get("runtime_block_size", 0)
                accept_length = cyc.get("accept_length", 0)
                verify_time_s = cyc.get("verify_time_s") or 0.0
                cycle_e2e_s = cyc.get("cycle_e2e_s")
                draft_time_s = cyc.get("draft_time_s") or 0.0
                cycles.append(
                    {
                        "verify_len": float(verify_len),
                        "accept_length": float(accept_length),
                        "verify_time_s": float(verify_time_s),
                        "draft_time_s": float(draft_time_s),
                        "cycle_e2e_s": float(cycle_e2e_s) if cycle_e2e_s is not None else 0.0,
                        "verify_can_run_cuda_graph": 1.0
                        if cyc.get("verify_can_run_cuda_graph")
                        else 0.0,
                    }
                )

    if not cycles:
        raise ValueError(f"No cycle traces found in {calls_path}")

    wall_time_s = float(total_completion_tokens) / float(throughput_tok_s)
    total_verify_time_s = sum(c["verify_time_s"] for c in cycles)
    total_abs_err = 0.0
    total_signed_err = 0.0
    exact = 0
    within_1 = 0
    under = 0
    over = 0
    for c in cycles:
        err = float(c["verify_len"]) - float(c["accept_length"])
        total_abs_err += abs(err)
        total_signed_err += err
        exact += int(err == 0.0)
        within_1 += int(abs(err) <= 1.0)
        under += int(err < 0.0)
        over += int(err > 0.0)

    summary = RunSummary(
        run_name=run_name,
        throughput_tok_s=float(throughput_tok_s),
        total_completion_tokens=int(total_completion_tokens),
        wall_time_s=wall_time_s,
        cycles=len(cycles),
        mean_verify_len=sum(c["verify_len"] for c in cycles) / len(cycles),
        mean_accept_length=sum(c["accept_length"] for c in cycles) / len(cycles),
        mean_verify_time_ms=1000.0 * total_verify_time_s / len(cycles),
        total_verify_time_s=total_verify_time_s,
        boundary_exact_rate=float(exact) / len(cycles),
        boundary_within_1_rate=float(within_1) / len(cycles),
        boundary_mae=total_abs_err / len(cycles),
        boundary_signed_error=total_signed_err / len(cycles),
        boundary_under_rate=float(under) / len(cycles),
        boundary_over_rate=float(over) / len(cycles),
        verify_can_run_cuda_graph_rate=sum(
            c["verify_can_run_cuda_graph"] for c in cycles
        )
        / len(cycles),
    )
    return summary, cycles


def build_verify_time_curve(run_cycles: list[list[dict[str, float]]]) -> dict[int, float]:
    by_len: dict[int, list[float]] = defaultdict(list)
    for cycles in run_cycles:
        for c in cycles:
            by_len[int(round(c["verify_len"]))].append(float(c["verify_time_s"]))

    if not by_len:
        raise ValueError("No verify-time observations found.")

    observed = sorted(by_len)
    curve = {k: sum(v) / len(v) for k, v in by_len.items()}
    full_curve: dict[int, float] = {}
    for k in range(min(observed), max(observed) + 1):
        if k in curve:
            full_curve[k] = curve[k]
            continue
        lower = max((v for v in observed if v < k), default=None)
        upper = min((v for v in observed if v > k), default=None)
        if lower is None and upper is None:
            raise RuntimeError("Unreachable: missing neighbors for interpolation.")
        if lower is None:
            full_curve[k] = curve[upper]
        elif upper is None:
            full_curve[k] = curve[lower]
        else:
            ratio = float(k - lower) / float(upper - lower)
            full_curve[k] = curve[lower] + ratio * (curve[upper] - curve[lower])
    return full_curve


def oracle_from_baseline(
    baseline_summary: RunSummary,
    baseline_cycles: list[dict[str, float]],
    verify_time_curve: dict[int, float],
) -> dict[str, float]:
    oracle_total_verify_time_s = 0.0
    oracle_total_cycle_time_s = 0.0
    baseline_total_cycle_time_s = 0.0
    for c in baseline_cycles:
        oracle_verify_len = int(round(c["accept_length"]))
        est_verify_time_s = float(
            verify_time_curve.get(oracle_verify_len, verify_time_curve[max(verify_time_curve)])
        )
        overhead_s = max(
            float(c["cycle_e2e_s"]) - float(c["draft_time_s"]) - float(c["verify_time_s"]),
            0.0,
        )
        oracle_total_verify_time_s += est_verify_time_s
        oracle_total_cycle_time_s += float(c["draft_time_s"]) + est_verify_time_s + overhead_s
        baseline_total_cycle_time_s += float(c["cycle_e2e_s"])

    saved_verify_time_s = float(baseline_summary.total_verify_time_s) - oracle_total_verify_time_s
    optimistic_wall_time_s = max(float(baseline_summary.wall_time_s) - saved_verify_time_s, 1e-6)
    optimistic_throughput_tok_s = (
        float(baseline_summary.total_completion_tokens) / optimistic_wall_time_s
    )
    cycle_proxy_throughput_tok_s = (
        float(baseline_summary.total_completion_tokens) / max(oracle_total_cycle_time_s, 1e-6)
    )
    baseline_cycle_proxy_tok_s = (
        float(baseline_summary.total_completion_tokens) / max(baseline_total_cycle_time_s, 1e-6)
    )
    return {
        "oracle_mean_verify_len": sum(c["accept_length"] for c in baseline_cycles)
        / len(baseline_cycles),
        "oracle_total_verify_time_s": oracle_total_verify_time_s,
        "baseline_total_verify_time_s": float(baseline_summary.total_verify_time_s),
        "saved_verify_time_s": saved_verify_time_s,
        "saved_verify_time_pct": saved_verify_time_s
        / max(float(baseline_summary.total_verify_time_s), 1e-9),
        "optimistic_wall_time_s": optimistic_wall_time_s,
        "optimistic_throughput_tok_s": optimistic_throughput_tok_s,
        "optimistic_throughput_delta_pct": optimistic_throughput_tok_s
        / float(baseline_summary.throughput_tok_s)
        - 1.0,
        "baseline_cycle_proxy_tok_s": baseline_cycle_proxy_tok_s,
        "oracle_cycle_proxy_tok_s": cycle_proxy_throughput_tok_s,
        "oracle_cycle_proxy_delta_pct": cycle_proxy_throughput_tok_s
        / max(baseline_cycle_proxy_tok_s, 1e-9)
        - 1.0,
    }


def write_markdown(
    output_path: Path,
    summaries: list[RunSummary],
    verify_time_curve: dict[int, float],
    oracle: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("# DFLASH C=1 Oracle Analysis")
    lines.append("")
    lines.append("## Run Comparison")
    lines.append("")
    lines.append(
        "| run | tok/s | wall time (s) | mean verify len | mean accept len | verify time/cycle (ms) | exact | within-1 | MAE | signed err | under | over |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for s in summaries:
        lines.append(
            f"| `{s.run_name}` | "
            f"{s.throughput_tok_s:.2f} | "
            f"{s.wall_time_s:.2f} | "
            f"{s.mean_verify_len:.3f} | "
            f"{s.mean_accept_length:.3f} | "
            f"{s.mean_verify_time_ms:.3f} | "
            f"{s.boundary_exact_rate:.3f} | "
            f"{s.boundary_within_1_rate:.3f} | "
            f"{s.boundary_mae:.3f} | "
            f"{s.boundary_signed_error:.3f} | "
            f"{s.boundary_under_rate:.3f} | "
            f"{s.boundary_over_rate:.3f} |"
        )
    lines.append("")
    lines.append("## Empirical Verify-Time Curve")
    lines.append("")
    lines.append("| verify len | mean verify time (ms) |")
    lines.append("| ---: | ---: |")
    for k in sorted(verify_time_curve):
        lines.append(f"| {k} | {1000.0 * verify_time_curve[k]:.3f} |")
    lines.append("")
    lines.append("## Oracle Upper Bound")
    lines.append("")
    lines.append(
        "Oracle uses the true cycle boundary (`verify_len = accept_length`) on the baseline trace, "
        "while keeping the measured draft time and non-draft/non-verify overhead fixed. "
        "Verify time is estimated from the empirical c=1 verify-time curve above."
    )
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    for key, value in oracle.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_names = [args.baseline_run, *args.compare_run]
    summaries: list[RunSummary] = []
    cycles_by_run: dict[str, list[dict[str, float]]] = {}
    for run_name in run_names:
        summary, cycles = load_run(base_dir=base_dir, run_name=run_name)
        summaries.append(summary)
        cycles_by_run[run_name] = cycles

    verify_time_curve = build_verify_time_curve(
        [cycles_by_run[run_name] for run_name in run_names]
    )
    baseline_summary = summaries[0]
    baseline_cycles = cycles_by_run[args.baseline_run]
    oracle = oracle_from_baseline(
        baseline_summary=baseline_summary,
        baseline_cycles=baseline_cycles,
        verify_time_curve=verify_time_curve,
    )

    payload = {
        "runs": [asdict(s) for s in summaries],
        "verify_time_curve_s": verify_time_curve,
        "oracle_from_baseline": oracle,
    }
    (output_dir / "c1_oracle_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(
        output_path=output_dir / "c1_oracle_analysis.md",
        summaries=summaries,
        verify_time_curve=verify_time_curve,
        oracle=oracle,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
