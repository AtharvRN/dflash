#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _resolve_candidate_path(candidate: str, *, log_path: Path) -> Path | None:
    cand = candidate.strip().strip("'").strip('"')
    if not cand:
        return None
    path = Path(cand)
    if path.is_absolute() and path.exists():
        return path

    relative_to_log = (log_path.parent / path).resolve()
    if relative_to_log.exists():
        return relative_to_log

    relative_to_cwd = (Path.cwd() / path).resolve()
    if relative_to_cwd.exists():
        return relative_to_cwd
    return None


def _extract_calls_jsonl_from_log(log_path: Path) -> Path | None:
    patterns = [
        re.compile(r"Wrote per-call JSONL trace to:\s*(\S+)"),
        re.compile(r"Call trace:\s*(\S+)"),
        re.compile(r"--save-call-trace-path\s+(\S+)"),
    ]
    with log_path.open() as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                resolved = _resolve_candidate_path(match.group(1), log_path=log_path)
                if resolved is not None:
                    return resolved
    return None


def _normalize_cycle_trace(raw_trace: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_trace, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in raw_trace:
        if not isinstance(item, dict):
            continue
        adaptive = item.get("adaptive_decision")
        confidence = item.get("confidence_gate_decision")
        rows.append(
            {
                "cycle_idx": _as_int(item.get("cycle_idx")),
                "runtime_block_size": _as_int(item.get("runtime_block_size")),
                "accepted_draft_tokens": _as_int(item.get("accepted_draft_tokens")),
                "accept_length": _as_int(item.get("accept_length")),
                "accept_rate": _as_float(item.get("accept_rate")),
                "draft_time_s": _as_float(item.get("draft_time_s")),
                "verify_time_s": _as_float(item.get("verify_time_s")),
                "cycle_e2e_s": _as_float(item.get("cycle_e2e_s")),
                "cycle_e2e_batch_s": _as_float(item.get("cycle_e2e_batch_s")),
                "adaptive_decision": adaptive if isinstance(adaptive, dict) else None,
                "confidence_gate_decision": (
                    confidence if isinstance(confidence, dict) else None
                ),
            }
        )
    return rows


def _iter_input_paths(raw_inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_inputs:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("*_calls.jsonl"))
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def _resolve_calls_path(path: Path) -> Path:
    if path.suffix.lower() == ".log":
        resolved = _extract_calls_jsonl_from_log(path)
        if resolved is None:
            raise ValueError(
                f"Could not resolve a *_calls.jsonl path from log file: {path}"
            )
        return resolved
    return path


def _flatten_cycle_row(
    *,
    source_path: Path,
    row: dict[str, Any],
    cycle: dict[str, Any],
) -> dict[str, Any]:
    runtime_block_size = _as_int(cycle.get("runtime_block_size"))
    accepted_draft_tokens = _as_int(cycle.get("accepted_draft_tokens"))
    proposed_draft_tokens = (
        max(0, int(runtime_block_size) - 1) if runtime_block_size is not None else None
    )
    first_reject_pos = None
    all_draft_accepted = None
    if proposed_draft_tokens is not None and accepted_draft_tokens is not None:
        if accepted_draft_tokens < proposed_draft_tokens:
            first_reject_pos = int(accepted_draft_tokens) + 1
            all_draft_accepted = False
        else:
            all_draft_accepted = True

    adaptive = cycle.get("adaptive_decision") or {}
    confidence = cycle.get("confidence_gate_decision") or {}

    cycle_row = {
        "source_path": str(source_path),
        "run_tag": source_path.parent.name,
        "request_idx": _as_int(row.get("request_local_idx")),
        "cycle_idx": _as_int(cycle.get("cycle_idx")),
        "mode": row.get("mode"),
        "speculative_algorithm": row.get("speculative_algorithm"),
        "backend": row.get("backend"),
        "tp_size": _as_int(row.get("tp_size")),
        "concurrency": _as_int(row.get("concurrency")),
        "question_count": _as_int(row.get("question_count")),
        "batch_requests": bool(row.get("batch_requests", False)),
        "dynamic_mode": bool(row.get("dynamic_mode", False)),
        "dynamic_block_size": _as_int(row.get("dynamic_block_size")),
        "dynamic_chunk_idx": _as_int(row.get("dynamic_chunk_idx")),
        "dynamic_chunk_size": _as_int(row.get("dynamic_chunk_size")),
        "completion_tokens": _as_int(row.get("completion_tokens")),
        "request_e2e_latency_s": _as_float(row.get("e2e_latency_s")),
        "request_spec_verify_ct": _as_int(row.get("spec_verify_ct")),
        "request_spec_accept_token_num": _as_int(row.get("spec_accept_token_num")),
        "request_spec_draft_token_num": _as_int(row.get("spec_draft_token_num")),
        "request_spec_accept_length": _as_float(row.get("spec_accept_length")),
        "request_spec_accept_rate": _as_float(row.get("spec_accept_rate")),
        "runtime_block_size": runtime_block_size,
        "proposed_draft_tokens": proposed_draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "accept_length_with_bonus": _as_int(cycle.get("accept_length")),
        "accept_rate": _as_float(cycle.get("accept_rate")),
        "all_draft_accepted": all_draft_accepted,
        "first_reject_pos": first_reject_pos,
        "draft_time_s": _as_float(cycle.get("draft_time_s")),
        "verify_time_s": _as_float(cycle.get("verify_time_s")),
        "cycle_e2e_s": _as_float(cycle.get("cycle_e2e_s")),
        "cycle_e2e_batch_s": _as_float(cycle.get("cycle_e2e_batch_s")),
        "adaptive_algo": adaptive.get("algo"),
        "adaptive_prev_bs": _as_int(adaptive.get("prev_bs")),
        "adaptive_next_bs": _as_int(adaptive.get("next_bs")),
        "adaptive_action": adaptive.get("action"),
        "adaptive_reason": adaptive.get("reason"),
        "adaptive_accept_ratio": _as_float(adaptive.get("accept_ratio")),
        "adaptive_accept_ratio_ewma": _as_float(
            adaptive.get("accept_ratio_ewma")
        ),
        "confidence_gate_enabled": bool(confidence) if confidence else False,
        "confidence_gate_mode": confidence.get("mode"),
        "confidence_gate_selection_reason": confidence.get("selection_reason"),
        "confidence_gate_aggregate": confidence.get("aggregate"),
        "confidence_gate_verify_token_num": _as_int(
            confidence.get("verify_token_num")
        ),
        "confidence_gate_min_verify_tokens": _as_int(
            confidence.get("min_verify_tokens")
        ),
        "confidence_gate_score_metric": confidence.get("score_metric"),
        "confidence_gate_score_budget": _as_float(
            confidence.get("score_budget")
        ),
        "confidence_gate_grouped_verify_applied": bool(
            confidence.get("grouped_verify_applied", False)
        )
        if confidence
        else False,
    }
    return cycle_row


def _iter_position_rows(cycle_row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    proposed = _as_int(cycle_row.get("proposed_draft_tokens"))
    accepted = _as_int(cycle_row.get("accepted_draft_tokens"))
    if proposed is None or accepted is None or proposed <= 0:
        return

    for pos in range(1, int(proposed) + 1):
        token_accepted = int(accepted >= pos)
        first_reject_here = int(accepted == pos - 1)
        yield {
            **cycle_row,
            "draft_pos": int(pos),
            "token_accepted": token_accepted,
            "prefix_survives_to_pos": token_accepted,
            "first_reject_here": first_reject_here,
        }


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc

            meta_raw = obj.get("meta_info_raw")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            merged = {}
            merged.update(meta)
            merged.update(obj)
            merged["spec_cycle_trace"] = _normalize_cycle_trace(
                _first_present(merged, ["spec_cycle_trace"])
            )
            yield merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DFLASH predictor-training rows from SGLang *_calls.jsonl traces. "
            "Produces cycle-level rows and per-draft-position acceptance labels."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help=(
            "One or more *_calls.jsonl files, benchmark .log files, or directories "
            "to scan recursively for *_calls.jsonl traces."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "Prefix for extracted datasets. Writes <prefix>_cycles.jsonl and "
            "<prefix>_positions.jsonl."
        ),
    )
    args = parser.parse_args()

    input_paths = _iter_input_paths(args.input)
    resolved_paths = [_resolve_calls_path(path) for path in input_paths]

    request_count = 0
    cycle_count = 0
    position_count = 0

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cycles_path = output_prefix.with_name(output_prefix.name + "_cycles.jsonl")
    positions_path = output_prefix.with_name(output_prefix.name + "_positions.jsonl")
    manifest_path = output_prefix.with_name(output_prefix.name + "_manifest.json")

    with cycles_path.open("w") as cycle_fp, positions_path.open("w") as pos_fp:
        for path in resolved_paths:
            for row in _iter_rows(path):
                request_count += 1
                for cycle in row.get("spec_cycle_trace") or []:
                    cycle_row = _flatten_cycle_row(
                        source_path=path,
                        row=row,
                        cycle=cycle,
                    )
                    cycle_fp.write(
                        json.dumps(cycle_row, ensure_ascii=False, default=str) + "\n"
                    )
                    cycle_count += 1
                    for position_row in _iter_position_rows(cycle_row):
                        pos_fp.write(
                            json.dumps(
                                position_row, ensure_ascii=False, default=str
                            )
                            + "\n"
                        )
                        position_count += 1

    manifest = {
        "inputs": [str(path) for path in resolved_paths],
        "request_rows": int(request_count),
        "cycle_rows": int(cycle_count),
        "position_rows": int(position_count),
        "cycles_path": str(cycles_path),
        "positions_path": str(positions_path),
        "notes": [
            "Cycle rows are derived from spec_cycle_trace in *_calls.jsonl.",
            "Position rows expand each cycle into one row per drafted position t=1..k-1.",
            "No hidden-state or per-token probability features are present in current call traces.",
            "This extractor is intended as the label/schema stage before richer online feature capture.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Resolved {len(resolved_paths)} input trace files.")
    print(f"Wrote cycle rows: {cycle_count} -> {cycles_path}")
    print(f"Wrote position rows: {position_count} -> {positions_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
