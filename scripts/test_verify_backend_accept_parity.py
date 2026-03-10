#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _float_equal(a: float | None, b: float | None, tol: float) -> bool:
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    assert a is not None and b is not None
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


@dataclass(frozen=True)
class CycleTrace:
    cycle_idx: int | None
    runtime_block_size: int | None
    accepted_draft_tokens: int | None
    accept_length: int | None
    accept_rate: float | None
    verify_can_run_cuda_graph: bool | None
    num_candidates: int | None
    effective_verify_tokens_per_req: int | None


@dataclass(frozen=True)
class RequestTrace:
    request_idx: int
    completion_tokens: int
    spec_verify_ct: int
    spec_accept_token_num: int
    spec_draft_token_num: int
    spec_accept_length: float | None
    spec_accept_rate: float | None
    cycles: tuple[CycleTrace, ...]


def _normalize_cycle(raw: Any) -> CycleTrace:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid cycle entry type: {type(raw).__name__}")

    mc = raw.get("multi_candidate_decision")
    if not isinstance(mc, dict):
        mc = {}

    verify_can_run_cuda_graph = raw.get("verify_can_run_cuda_graph")
    if verify_can_run_cuda_graph is not None:
        verify_can_run_cuda_graph = bool(verify_can_run_cuda_graph)

    return CycleTrace(
        cycle_idx=_as_int(raw.get("cycle_idx")),
        runtime_block_size=_as_int(raw.get("runtime_block_size")),
        accepted_draft_tokens=_as_int(raw.get("accepted_draft_tokens")),
        accept_length=_as_int(raw.get("accept_length")),
        accept_rate=_as_float(raw.get("accept_rate")),
        verify_can_run_cuda_graph=verify_can_run_cuda_graph,
        num_candidates=_as_int(mc.get("num_candidates")),
        effective_verify_tokens_per_req=_as_int(
            mc.get("effective_verify_tokens_per_req")
        ),
    )


def _normalize_request(raw: Any) -> RequestTrace:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid request entry type: {type(raw).__name__}")

    req_idx = _as_int(raw.get("request_local_idx"))
    if req_idx is None:
        req_idx = _as_int(raw.get("request_idx"))
    if req_idx is None:
        raise ValueError("Missing request index (request_local_idx/request_idx).")

    raw_cycles = raw.get("spec_cycle_trace")
    if raw_cycles is None:
        raw_cycles = []
    if not isinstance(raw_cycles, list):
        raise ValueError(
            f"spec_cycle_trace must be a list, got {type(raw_cycles).__name__}."
        )

    return RequestTrace(
        request_idx=req_idx,
        completion_tokens=_as_int(raw.get("completion_tokens")) or 0,
        spec_verify_ct=_as_int(raw.get("spec_verify_ct")) or 0,
        spec_accept_token_num=_as_int(raw.get("spec_accept_token_num")) or 0,
        spec_draft_token_num=_as_int(raw.get("spec_draft_token_num")) or 0,
        spec_accept_length=_as_float(raw.get("spec_accept_length")),
        spec_accept_rate=_as_float(raw.get("spec_accept_rate")),
        cycles=tuple(_normalize_cycle(c) for c in raw_cycles),
    )


def _load_trace(path: Path) -> dict[int, RequestTrace]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    out: dict[int, RequestTrace] = {}
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            req = _normalize_request(raw)
            if req.request_idx in out:
                raise ValueError(
                    f"{path}:{line_no}: duplicate request index {req.request_idx}."
                )
            out[req.request_idx] = req
    return out


def _summarize(name: str, traces: dict[int, RequestTrace]) -> str:
    rows = [traces[k] for k in sorted(traces.keys())]
    n = len(rows)
    if n == 0:
        return f"{name}: n=0"
    tau_vals = [r.spec_accept_length for r in rows if r.spec_accept_length is not None]
    rate_vals = [r.spec_accept_rate for r in rows if r.spec_accept_rate is not None]
    verify_total = sum(r.spec_verify_ct for r in rows)
    accepted_total = sum(r.spec_accept_token_num for r in rows)
    drafted_total = sum(r.spec_draft_token_num for r in rows)
    return (
        f"{name}: n={n} "
        f"tau={mean(tau_vals):.4f} "
        f"accept_rate={mean(rate_vals):.4f} "
        f"verify_ct_sum={verify_total} "
        f"accept_tok_sum={accepted_total} "
        f"draft_tok_sum={drafted_total}"
    )


def _compare_traces(
    *,
    a_name: str,
    b_name: str,
    a_traces: dict[int, RequestTrace],
    b_traces: dict[int, RequestTrace],
    float_tol: float,
) -> None:
    a_ids = sorted(a_traces.keys())
    b_ids = sorted(b_traces.keys())
    if a_ids != b_ids:
        raise AssertionError(
            "Request index sets differ.\n"
            f"{a_name}: {a_ids}\n"
            f"{b_name}: {b_ids}"
        )

    for req_idx in a_ids:
        a = a_traces[req_idx]
        b = b_traces[req_idx]

        int_fields = [
            ("completion_tokens", a.completion_tokens, b.completion_tokens),
            ("spec_verify_ct", a.spec_verify_ct, b.spec_verify_ct),
            ("spec_accept_token_num", a.spec_accept_token_num, b.spec_accept_token_num),
            ("spec_draft_token_num", a.spec_draft_token_num, b.spec_draft_token_num),
        ]
        for field_name, av, bv in int_fields:
            if av != bv:
                raise AssertionError(
                    f"Request {req_idx} mismatch: {field_name}\n"
                    f"{a_name}: {av}\n"
                    f"{b_name}: {bv}"
                )

        float_fields = [
            ("spec_accept_length", a.spec_accept_length, b.spec_accept_length),
            ("spec_accept_rate", a.spec_accept_rate, b.spec_accept_rate),
        ]
        for field_name, av, bv in float_fields:
            if not _float_equal(av, bv, float_tol):
                raise AssertionError(
                    f"Request {req_idx} mismatch: {field_name}\n"
                    f"{a_name}: {av}\n"
                    f"{b_name}: {bv}"
                )

        if len(a.cycles) != len(b.cycles):
            raise AssertionError(
                f"Request {req_idx} mismatch: number of cycles\n"
                f"{a_name}: {len(a.cycles)}\n"
                f"{b_name}: {len(b.cycles)}"
            )

        for cycle_idx, (ac, bc) in enumerate(zip(a.cycles, b.cycles)):
            int_cycle_fields = [
                ("cycle_idx", ac.cycle_idx, bc.cycle_idx),
                ("runtime_block_size", ac.runtime_block_size, bc.runtime_block_size),
                ("accepted_draft_tokens", ac.accepted_draft_tokens, bc.accepted_draft_tokens),
                ("accept_length", ac.accept_length, bc.accept_length),
                ("num_candidates", ac.num_candidates, bc.num_candidates),
                (
                    "effective_verify_tokens_per_req",
                    ac.effective_verify_tokens_per_req,
                    bc.effective_verify_tokens_per_req,
                ),
            ]
            for field_name, av, bv in int_cycle_fields:
                if av != bv:
                    raise AssertionError(
                        f"Request {req_idx} cycle {cycle_idx} mismatch: {field_name}\n"
                        f"{a_name}: {av}\n"
                        f"{b_name}: {bv}\n"
                        f"{a_name} cycle raw: {ac}\n"
                        f"{b_name} cycle raw: {bc}"
                    )

            float_cycle_fields = [
                ("accept_rate", ac.accept_rate, bc.accept_rate),
            ]
            for field_name, av, bv in float_cycle_fields:
                if not _float_equal(av, bv, float_tol):
                    raise AssertionError(
                        f"Request {req_idx} cycle {cycle_idx} mismatch: {field_name}\n"
                        f"{a_name}: {av}\n"
                        f"{b_name}: {bv}\n"
                        f"{a_name} cycle raw: {ac}\n"
                        f"{b_name} cycle raw: {bc}"
                    )

            if ac.verify_can_run_cuda_graph != bc.verify_can_run_cuda_graph:
                raise AssertionError(
                    f"Request {req_idx} cycle {cycle_idx} mismatch: verify_can_run_cuda_graph\n"
                    f"{a_name}: {ac.verify_can_run_cuda_graph}\n"
                    f"{b_name}: {bc.verify_can_run_cuda_graph}"
                )


def _run_command_with_tee(*, cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, prefix: str) -> None:
    print(f"[{prefix}] command: {' '.join(shlex.quote(x) for x in cmd)}")
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            sys.stdout.write(f"[{prefix}] {line}")
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"[{prefix}] benchmark run failed with exit code {rc}. Log: {log_path}")


def _run_benchmark_case(
    *,
    repo_root: Path,
    python_exe: str,
    output_root: Path,
    label: str,
    verify_backend_override: str | None,
    cuda_visible_devices: str | None,
    dataset_name: str,
    target_model: str,
    draft_model: str,
    attention_backend: str,
    tp_size: int,
    concurrency: int,
    fixed_question_count: int,
    fixed_question_offset: int,
    max_new_tokens: int,
    timeout_s: int,
    block_size: int,
    max_candidates: int,
    deterministic_prefix_len: int,
    sample_temperature: float,
    random_seed: int,
    enable_stage_timing: bool,
    small_cuda_graph: bool,
    extra_server_args: str,
) -> Path:
    run_dir = output_root / label
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / f"{label}_calls.jsonl"
    md_path = run_dir / f"{label}.md"
    log_path = run_dir / f"{label}.log"

    server_extra_parts: list[str] = [
        "--random-seed",
        str(random_seed),
        "--speculative-dflash-multi-candidate",
        "--speculative-dflash-multi-candidate-max-candidates",
        str(max_candidates),
        "--speculative-dflash-multi-candidate-sample-temperature",
        str(sample_temperature),
        "--speculative-dflash-multi-candidate-deterministic-prefix-len",
        str(deterministic_prefix_len),
        "--speculative-dflash-multi-candidate-verify-mode",
        "packed_tree",
    ]
    if verify_backend_override is not None:
        server_extra_parts.extend(
            [
                "--speculative-dflash-multi-candidate-verify-attention-backend",
                verify_backend_override,
            ]
        )
    if small_cuda_graph:
        server_extra_parts.extend(["--cuda-graph-bs", "1", "--cuda-graph-max-bs", "1"])
    if extra_server_args.strip():
        server_extra_parts.extend(shlex.split(extra_server_args))

    cmd = [
        python_exe,
        "benchmark_sglang.py",
        "--dataset-name",
        dataset_name,
        "--target-model",
        target_model,
        "--draft-model",
        draft_model,
        "--tp-size",
        str(tp_size),
        "--attention-backends",
        attention_backend,
        "--concurrencies",
        str(concurrency),
        "--questions-per-concurrency-base",
        str(fixed_question_count),
        "--max-questions-per-config",
        str(fixed_question_count),
        "--fixed-question-count",
        str(fixed_question_count),
        "--fixed-question-offset",
        str(fixed_question_offset),
        "--max-new-tokens",
        str(max_new_tokens),
        "--timeout-s",
        str(timeout_s),
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-dflash-block-size",
        str(block_size),
        "--skip-baseline",
        "--enable-dflash-cycle-trace",
        "--save-call-trace-path",
        str(trace_path),
        "--output-md",
        str(md_path),
        "--server-extra-args",
        shlex.join(server_extra_parts),
    ]
    if enable_stage_timing:
        cmd.append("--enable-dflash-stage-timing")

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    _run_command_with_tee(
        cmd=cmd,
        cwd=repo_root,
        env=env,
        log_path=log_path,
        prefix=label,
    )
    return trace_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict acceptance parity test for DFLASH packed-tree multi-candidate verify "
            "between default backend and triton override."
        )
    )
    parser.add_argument("--trace-a", type=str, default=None, help="Optional existing calls.jsonl path for backend A.")
    parser.add_argument("--trace-b", type=str, default=None, help="Optional existing calls.jsonl path for backend B.")
    parser.add_argument("--trace-a-name", type=str, default="backend_a")
    parser.add_argument("--trace-b-name", type=str, default="backend_b")
    parser.add_argument("--float-tol", type=float, default=1e-8)

    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="aime25")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--attention-backend", type=str, default="flashinfer")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--fixed-question-count", type=int, default=6)
    parser.add_argument("--fixed-question-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--deterministic-prefix-len", type=int, default=2)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--enable-stage-timing", action="store_true")
    parser.add_argument(
        "--small-cuda-graph",
        action="store_true",
        help="Override server args to capture only bs=1 graph for faster tiny parity runs.",
    )
    parser.add_argument(
        "--extra-server-args",
        type=str,
        default="",
        help="Extra raw args appended to each server launch.",
    )
    args = parser.parse_args()

    if (args.trace_a is None) != (args.trace_b is None):
        raise ValueError("Either provide both --trace-a/--trace-b or neither.")

    if args.trace_a is not None:
        trace_a_path = Path(args.trace_a).resolve()
        trace_b_path = Path(args.trace_b).resolve()
        a_name = args.trace_a_name
        b_name = args.trace_b_name
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_root = args.output_root
        if output_root is None:
            output_root = args.repo_root / "logs" / f"verify_backend_parity_{ts}"
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"[setup] writing artifacts under {output_root}")
        trace_a_path = _run_benchmark_case(
            repo_root=args.repo_root,
            python_exe=args.python_exe,
            output_root=output_root,
            label="default_verify_backend",
            verify_backend_override=None,
            cuda_visible_devices=args.cuda_visible_devices,
            dataset_name=args.dataset_name,
            target_model=args.target_model,
            draft_model=args.draft_model,
            attention_backend=args.attention_backend,
            tp_size=args.tp_size,
            concurrency=args.concurrency,
            fixed_question_count=args.fixed_question_count,
            fixed_question_offset=args.fixed_question_offset,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout_s,
            block_size=args.block_size,
            max_candidates=args.max_candidates,
            deterministic_prefix_len=args.deterministic_prefix_len,
            sample_temperature=args.sample_temperature,
            random_seed=args.random_seed,
            enable_stage_timing=args.enable_stage_timing,
            small_cuda_graph=args.small_cuda_graph,
            extra_server_args=args.extra_server_args,
        )
        trace_b_path = _run_benchmark_case(
            repo_root=args.repo_root,
            python_exe=args.python_exe,
            output_root=output_root,
            label="triton_verify_backend",
            verify_backend_override="triton",
            cuda_visible_devices=args.cuda_visible_devices,
            dataset_name=args.dataset_name,
            target_model=args.target_model,
            draft_model=args.draft_model,
            attention_backend=args.attention_backend,
            tp_size=args.tp_size,
            concurrency=args.concurrency,
            fixed_question_count=args.fixed_question_count,
            fixed_question_offset=args.fixed_question_offset,
            max_new_tokens=args.max_new_tokens,
            timeout_s=args.timeout_s,
            block_size=args.block_size,
            max_candidates=args.max_candidates,
            deterministic_prefix_len=args.deterministic_prefix_len,
            sample_temperature=args.sample_temperature,
            random_seed=args.random_seed,
            enable_stage_timing=args.enable_stage_timing,
            small_cuda_graph=args.small_cuda_graph,
            extra_server_args=args.extra_server_args,
        )
        a_name = "default"
        b_name = "triton"

    print(f"[load] {a_name}: {trace_a_path}")
    print(f"[load] {b_name}: {trace_b_path}")
    a_traces = _load_trace(trace_a_path)
    b_traces = _load_trace(trace_b_path)

    print(_summarize(a_name, a_traces))
    print(_summarize(b_name, b_traces))

    _compare_traces(
        a_name=a_name,
        b_name=b_name,
        a_traces=a_traces,
        b_traces=b_traces,
        float_tol=args.float_tol,
    )

    print(
        "PASS: strict parity holds for request-level and cycle-level acceptance fields "
        f"between {a_name} and {b_name}."
    )


if __name__ == "__main__":
    main()
