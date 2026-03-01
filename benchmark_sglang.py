from __future__ import annotations

import argparse
from collections import defaultdict
import json
import shlex
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, TextIO

import requests
import torch
from transformers import AutoTokenizer
from model import load_and_process_dataset

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)

def _is_blackwell() -> bool:
    if envs.IS_BLACKWELL.get():
        return True
    return get_device_sm() >= 100


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


def _send_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> list[dict]:
    if not prompts:
        return []
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, list):
        raise RuntimeError(
            "Expected a list response for batched /generate, but got "
            f"type={type(out).__name__}."
        )
    return out


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    request_count: int
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int
    spec_accept_rate: Optional[float]
    spec_accept_token_sum: int
    spec_draft_token_sum: int
    e2e_latency_avg_s: Optional[float]
    e2e_latency_p50_s: Optional[float]
    e2e_latency_p95_s: Optional[float]
    inference_time_avg_s: Optional[float]
    decode_throughput_avg_tok_s: Optional[float]
    extra_timing_avgs_s: dict[str, Optional[float]]


def _mean_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def _percentile_or_none(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    p = min(max(float(pct), 0.0), 1.0)
    xs = sorted(float(x) for x in values)
    pos = p * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _extract_float(meta: dict, keys: list[str]) -> Optional[float]:
    for key in keys:
        if key not in meta:
            continue
        try:
            return float(meta[key])
        except (TypeError, ValueError):
            continue
    return None


def _fmt_opt(v: Optional[float], fmt: str) -> str:
    return "N/A" if v is None else format(v, fmt)


def _run_bench_requests(
    base_url: str,
    *,
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    batch_requests: bool,
    stop: list[str],
    timeout_s: int,
    expect_dflash: bool,
    trace_fp: Optional[TextIO] = None,
    trace_common: Optional[dict] = None,
    trace_include_prompt: bool = False,
    trace_include_raw_meta: bool = False,
) -> BenchMetrics:
    # Drop the first batch from metrics to exclude one-time JIT/cuda-graph overhead
    bs = max(int(concurrency), 1)
    if len(prompts) > bs:
        warmup_prompts = prompts[:bs]
        if batch_requests:
            _send_generate_batch(
                base_url,
                warmup_prompts,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
                futures = [
                    pool.submit(
                        _send_generate,
                        base_url,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        stop=stop,
                        timeout_s=timeout_s,
                    )
                    for prompt in warmup_prompts
                ]
                for fut in as_completed(futures):
                    fut.result()

        prompts = prompts[bs:]

    start = time.perf_counter()
    request_count = 0
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_token_sum = 0
    spec_draft_token_sum = 0
    spec_accept_lengths: list[float] = []
    spec_accept_rates: list[float] = []
    e2e_latencies: list[float] = []
    inference_times: list[float] = []
    decode_throughputs: list[float] = []
    # Capture optional fields if server starts exposing them in meta_info.
    extra_timing_fields = {
        "draft_time_s": [
            "spec_draft_time_s",
            "spec_draft_time",
            "draft_time_s",
            "draft_time",
        ],
        "verify_time_s": [
            "spec_verify_time_s",
            "spec_verify_time",
            "verify_time_s",
            "verify_time",
            "target_verify_time_s",
            "target_verify_time",
        ],
    }
    extra_timing_samples: dict[str, list[float]] = defaultdict(list)

    def _consume_meta(
        meta: dict,
        *,
        prompt_text: Optional[str] = None,
        client_request_wall_s: Optional[float] = None,
        client_batch_wall_s: Optional[float] = None,
        client_batch_size: Optional[int] = None,
    ) -> None:
        nonlocal request_count
        nonlocal total_tokens
        nonlocal spec_verify_ct_sum
        nonlocal spec_accept_token_sum
        nonlocal spec_draft_token_sum
        request_local_idx = request_count
        request_count += 1
        total_tokens += int(meta.get("completion_tokens", 0))
        spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
        spec_accept_token_sum += int(meta.get("spec_accept_token_num", 0))
        spec_draft_token_sum += int(meta.get("spec_draft_token_num", 0))

        spec_accept_length = _extract_float(meta, ["spec_accept_length"])
        if spec_accept_length is not None:
            spec_accept_lengths.append(spec_accept_length)

        spec_accept_rate = _extract_float(meta, ["spec_accept_rate"])
        if spec_accept_rate is not None:
            spec_accept_rates.append(spec_accept_rate)

        e2e_latency = _extract_float(meta, ["e2e_latency"])
        if e2e_latency is not None:
            e2e_latencies.append(e2e_latency)

        inference_time = _extract_float(meta, ["inference_time"])
        if inference_time is not None:
            inference_times.append(inference_time)

        decode_throughput = _extract_float(meta, ["decode_throughput"])
        if decode_throughput is not None:
            decode_throughputs.append(decode_throughput)

        for field_name, keys in extra_timing_fields.items():
            v = _extract_float(meta, keys)
            if v is not None:
                extra_timing_samples[field_name].append(v)

        if trace_fp is not None:
            draft_time_s = _extract_float(meta, extra_timing_fields["draft_time_s"])
            verify_time_s = _extract_float(meta, extra_timing_fields["verify_time_s"])
            row = dict(trace_common or {})
            row.update(
                {
                    "request_local_idx": int(request_local_idx),
                    "timestamp_unix_s": float(time.time()),
                    "completion_tokens": int(meta.get("completion_tokens", 0)),
                    "e2e_latency_s": _extract_float(meta, ["e2e_latency"]),
                    "spec_accept_length": spec_accept_length,
                    "spec_accept_rate": spec_accept_rate,
                    "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
                    "spec_accept_token_num": int(meta.get("spec_accept_token_num", 0)),
                    "spec_draft_token_num": int(meta.get("spec_draft_token_num", 0)),
                    "inference_time_s": _extract_float(meta, ["inference_time"]),
                    "decode_throughput_tok_s": _extract_float(meta, ["decode_throughput"]),
                    "draft_time_s": draft_time_s,
                    "verify_time_s": verify_time_s,
                    "client_request_wall_s": client_request_wall_s,
                    "client_batch_wall_s": client_batch_wall_s,
                    "client_batch_size": client_batch_size,
                }
            )
            if trace_include_prompt:
                row["prompt"] = prompt_text
            if trace_include_raw_meta:
                row["meta_info_raw"] = meta
            trace_fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            trace_fp.flush()

    if batch_requests:
        bs = max(int(concurrency), 1)
        for start_idx in range(0, len(prompts), bs):
            chunk_prompts = prompts[start_idx : start_idx + bs]
            batch_t0 = time.perf_counter()
            outs = _send_generate_batch(
                base_url,
                chunk_prompts,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
            batch_wall_s = time.perf_counter() - batch_t0
            if len(outs) != len(chunk_prompts):
                raise RuntimeError(
                    "Batched /generate output length mismatch: "
                    f"got {len(outs)} outputs for {len(chunk_prompts)} prompts."
                )

            for idx, out in enumerate(outs):
                meta = out.get("meta_info", {}) or {}
                _consume_meta(
                    meta,
                    prompt_text=chunk_prompts[idx],
                    client_batch_wall_s=float(batch_wall_s),
                    client_batch_size=int(len(chunk_prompts)),
                )
    else:
        def _timed_send(prompt_idx: int, prompt: str):
            t0 = time.perf_counter()
            out = _send_generate(
                base_url,
                prompt,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
            wall_s = time.perf_counter() - t0
            return prompt_idx, prompt, out, wall_s

        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = {
                pool.submit(_timed_send, i, prompt): i
                for i, prompt in enumerate(prompts)
            }
            for fut in as_completed(futures):
                _prompt_idx, prompt_text, out, req_wall_s = fut.result()
                meta = out.get("meta_info", {}) or {}
                _consume_meta(
                    meta,
                    prompt_text=prompt_text,
                    client_request_wall_s=float(req_wall_s),
                )

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_dflash and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "DFLASH sanity check failed: did not observe any `spec_verify_ct` in responses "
            "(DFLASH may not have been enabled)."
        )

    spec_accept_length = _mean_or_none(spec_accept_lengths)
    spec_accept_rate = _mean_or_none(spec_accept_rates)
    e2e_latency_avg_s = _mean_or_none(e2e_latencies)
    e2e_latency_p50_s = _percentile_or_none(e2e_latencies, 0.50)
    e2e_latency_p95_s = _percentile_or_none(e2e_latencies, 0.95)
    inference_time_avg_s = _mean_or_none(inference_times)
    decode_throughput_avg_tok_s = _mean_or_none(decode_throughputs)
    extra_timing_avgs_s = {
        k: _mean_or_none(vs) for k, vs in extra_timing_samples.items()
    }

    return BenchMetrics(
        latency_s=float(latency),
        request_count=int(request_count),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
        spec_accept_rate=spec_accept_rate,
        spec_accept_token_sum=int(spec_accept_token_sum),
        spec_draft_token_sum=int(spec_draft_token_sum),
        e2e_latency_avg_s=e2e_latency_avg_s,
        e2e_latency_p50_s=e2e_latency_p50_s,
        e2e_latency_p95_s=e2e_latency_p95_s,
        inference_time_avg_s=inference_time_avg_s,
        decode_throughput_avg_tok_s=decode_throughput_avg_tok_s,
        extra_timing_avgs_s=extra_timing_avgs_s,
    )


def _format_table(
    *,
    concurrencies: list[int],
    values: dict[int, Optional[float]],
    float_fmt: str,
) -> str:
    header = ["conc"] + [str(c) for c in concurrencies]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    row = ["value"]
    for c in concurrencies:
        v = values.get(c, None)
        row.append("N/A" if v is None else format(v, float_fmt))
    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write a markdown report to this file (disabled by default).",
    )
    parser.add_argument(
        "--save-call-trace-path",
        type=str,
        default=None,
        help="Optional JSONL path. Writes one row per request response with per-call meta (tau/verify/draft timings when available).",
    )
    parser.add_argument(
        "--save-call-trace-prompt",
        action="store_true",
        help="Include full prompt text in each call-trace row.",
    )
    parser.add_argument(
        "--save-call-trace-raw-meta",
        action="store_true",
        help="Include raw response meta_info dict in each call-trace row.",
    )
    parser.add_argument("--dataset-name", type=str, default="gsm8k")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline (target-only) sweep; only run DFLASH and report N/A for baseline/speedup.",
    )
    parser.add_argument(
        "--batch-requests",
        action="store_true",
        help="Send prompts as server-side batched /generate requests (batch size = concurrency) instead of client-side concurrent requests.",
    )
    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        default="DFLASH",
        help="Speculative algorithm for the speculative run (e.g., DFLASH/EAGLE/EAGLE3/STANDALONE/NGRAM/NEXTN).",
    )
    parser.add_argument(
        "--speculative-dflash-block-size",
        type=int,
        default=None,
        help="DFLASH only. Sets --speculative-dflash-block-size on server.",
    )
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=None,
        help="Optional override for --speculative-num-draft-tokens on server.",
    )
    parser.add_argument(
        "--speculative-num-steps",
        type=int,
        default=None,
        help="Optional override for --speculative-num-steps on server.",
    )
    parser.add_argument(
        "--speculative-eagle-topk",
        type=int,
        default=None,
        help="Optional override for --speculative-eagle-topk on server.",
    )
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="Force --disable-overlap-schedule on server.",
    )
    parser.add_argument(
        "--server-extra-args",
        type=str,
        default="",
        help="Raw extra args appended to launch_server (parsed with shlex.split).",
    )
    parser.add_argument(
        "--enable-server-metrics",
        action="store_true",
        help="Pass --enable-metrics to SGLang server so response meta_info includes extra timing fields like inference_time/decode_throughput.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (single value, no sweep).",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of client concurrency levels.",
    )
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=128,
        help="num_questions = base * concurrency (default matches the sweep plan).",
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=1024,
        help="Cap num_questions per (tp, concurrency) run (default: 1024).",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer,fa3,fa4",
        help="Comma-separated list. Will auto-skip fa3 unless SM90 (Hopper), and fa4 unless SM100+ (Blackwell).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_questions_by_conc = {
        c: min(int(args.questions_per_concurrency_base) * int(c), int(args.max_questions_per_config))
        for c in concurrencies
    }
    max_questions = max(num_questions_by_conc.values())
    max_concurrency = max(concurrencies)

    attention_backends = [s.strip() for s in args.attention_backends.split(",") if s.strip()]
    is_blackwell = _is_blackwell()
    device_sm = get_device_sm()
    if device_sm != 90:
        attention_backends = [b for b in attention_backends if b != "fa3"]
    if device_sm < 100:
        attention_backends = [b for b in attention_backends if b != "fa4"]
    attention_backends = attention_backends or ["flashinfer"]

    # --- Load Data using the new function ---
    print(f"Loading dataset: {args.dataset_name}...")
    dataset = load_and_process_dataset(args.dataset_name)
    required_questions = max_questions + max_concurrency
    
    if len(dataset) < required_questions:
         print(f"Warning: Dataset has {len(dataset)} items, but need up to {required_questions}. Reusing items.")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    prompts: list[str] = []
    # Build prompts list
    for i in range(max(len(dataset), required_questions)):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0] # Extract the formatted turn
        
        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)
        if len(prompts) >= required_questions:
            break

    # Results indexed by (backend, concurrency) for baseline + dflash.
    # Removed TP dimension from keys since we aren't sweeping it.
    baseline_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, int], Optional[float]] = {}
    baseline_metrics: dict[tuple[str, int], BenchMetrics] = {}
    dflash_metrics: dict[tuple[str, int], BenchMetrics] = {}
    
    tp = args.tp_size  # Fixed TP size

    call_trace_fp: Optional[TextIO] = None
    if args.save_call_trace_path:
        call_trace_fp = open(args.save_call_trace_path, "w", encoding="utf-8")

    try:
        for backend in attention_backends:
            port_base = find_available_port(20000)

            common_server_args: list[str] = [
                "--trust-remote-code",
                "--attention-backend",
                backend,
                "--tp-size",
                str(tp),
                "--dtype",
                str(args.dtype),
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--max-running-requests",
                str(args.max_running_requests),
            ]
            common_server_args.extend(
                ["--cuda-graph-bs", *[str(i) for i in range(1, 33)], "--cuda-graph-max-bs", "32"]
            )
            if args.disable_radix_cache:
                common_server_args.append("--disable-radix-cache")
            if args.enable_server_metrics:
                common_server_args.append("--enable-metrics")
            if args.disable_overlap_schedule:
                common_server_args.append("--disable-overlap-schedule")
            if args.server_extra_args.strip():
                common_server_args.extend(shlex.split(args.server_extra_args))

            if not args.skip_baseline:
                print(f"\n=== backend={backend} tp={tp} (baseline) ===")
                baseline_port = port_base
                baseline_url = f"http://127.0.0.1:{baseline_port}"
                baseline_proc = popen_launch_server(
                    args.target_model,
                    baseline_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=common_server_args,
                )
                try:
                    # Warm up.
                    _send_generate(
                        baseline_url,
                        "Hello",
                        max_new_tokens=8,
                        stop=[],
                        timeout_s=min(int(args.timeout_s), 300),
                    )

                    for conc in concurrencies:
                        n = num_questions_by_conc[conc]
                        _flush_cache(baseline_url)
                        print(
                            f"[warmup] run 1 warmup batch (size={conc}) after /flush_cache; excluded from metrics."
                        )
                        metrics = _run_bench_requests(
                            baseline_url,
                            prompts=prompts[: n + conc],
                            max_new_tokens=int(args.max_new_tokens),
                            concurrency=int(conc),
                            batch_requests=bool(args.batch_requests),
                            stop=[],
                            timeout_s=int(args.timeout_s),
                            expect_dflash=False,
                            trace_fp=call_trace_fp,
                            trace_common={
                                "mode": "baseline",
                                "speculative_algorithm": "NONE",
                                "backend": backend,
                                "tp_size": int(tp),
                                "concurrency": int(conc),
                                "question_count": int(n),
                                "batch_requests": bool(args.batch_requests),
                            },
                            trace_include_prompt=bool(args.save_call_trace_prompt),
                            trace_include_raw_meta=bool(args.save_call_trace_raw_meta),
                        )
                        baseline_toks[(backend, conc)] = metrics.output_toks_per_s
                        baseline_metrics[(backend, conc)] = metrics
                        print(
                            f"[baseline] conc={conc:>2} n={n:<4} "
                            f"toks/s={metrics.output_toks_per_s:,.2f} "
                            f"latency={metrics.latency_s:.1f}s "
                            f"e2e_avg={_fmt_opt(metrics.e2e_latency_avg_s, '.3f')}s "
                        )
                finally:
                    kill_process_tree(baseline_proc.pid)
                    try:
                        baseline_proc.wait(timeout=30)
                    except Exception:
                        pass

            spec_algo = args.speculative_algorithm.upper()
            print(f"\n=== backend={backend} tp={tp} ({spec_algo}) ===")
            spec_server_args = [
                *common_server_args,
                "--speculative-algorithm",
                spec_algo,
            ]
            if args.draft_model:
                spec_server_args.extend(
                    ["--speculative-draft-model-path", args.draft_model]
                )
            if args.speculative_dflash_block_size is not None:
                spec_server_args.extend(
                    [
                        "--speculative-dflash-block-size",
                        str(int(args.speculative_dflash_block_size)),
                    ]
                )
            if args.speculative_num_draft_tokens is not None:
                spec_server_args.extend(
                    [
                        "--speculative-num-draft-tokens",
                        str(int(args.speculative_num_draft_tokens)),
                    ]
                )
            if args.speculative_num_steps is not None:
                spec_server_args.extend(
                    ["--speculative-num-steps", str(int(args.speculative_num_steps))]
                )
            if args.speculative_eagle_topk is not None:
                spec_server_args.extend(
                    ["--speculative-eagle-topk", str(int(args.speculative_eagle_topk))]
                )
            dflash_port = find_available_port(port_base + 1)
            dflash_url = f"http://127.0.0.1:{dflash_port}"
            dflash_proc = popen_launch_server(
                args.target_model,
                dflash_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=spec_server_args,
            )
            try:
                _send_generate(
                    dflash_url,
                    "Hello",
                    max_new_tokens=8,
                    stop=[],
                    timeout_s=min(int(args.timeout_s), 300),
                )

                for conc in concurrencies:
                    n = num_questions_by_conc[conc]
                    _flush_cache(dflash_url)
                    print(
                        f"[warmup] run 1 warmup batch (size={conc}) after /flush_cache; excluded from metrics."
                    )
                    metrics = _run_bench_requests(
                        dflash_url,
                        prompts=prompts[: n + conc],
                        max_new_tokens=int(args.max_new_tokens),
                        concurrency=int(conc),
                        batch_requests=bool(args.batch_requests),
                        stop=[],
                        timeout_s=int(args.timeout_s),
                        expect_dflash=True,
                        trace_fp=call_trace_fp,
                        trace_common={
                            "mode": "speculative",
                            "speculative_algorithm": spec_algo,
                            "backend": backend,
                            "tp_size": int(tp),
                            "concurrency": int(conc),
                            "question_count": int(n),
                            "batch_requests": bool(args.batch_requests),
                        },
                        trace_include_prompt=bool(args.save_call_trace_prompt),
                        trace_include_raw_meta=bool(args.save_call_trace_raw_meta),
                    )
                    dflash_toks[(backend, conc)] = metrics.output_toks_per_s
                    dflash_accept_len[(backend, conc)] = metrics.spec_accept_length
                    dflash_metrics[(backend, conc)] = metrics
                    verify_calls_per_s = (
                        metrics.spec_verify_ct_sum / max(metrics.latency_s, 1e-6)
                        if metrics.spec_verify_ct_sum > 0
                        else None
                    )
                    draft_tokens_per_s = (
                        metrics.spec_draft_token_sum / max(metrics.latency_s, 1e-6)
                        if metrics.spec_draft_token_sum > 0
                        else None
                    )
                    print(
                        f"[{spec_algo}]   conc={conc:>2} n={n:<4} "
                        f"toks/s={metrics.output_toks_per_s:,.2f} "
                        f"latency={metrics.latency_s:.1f}s "
                        f"tau={_fmt_opt(metrics.spec_accept_length, '.3f')} "
                        f"accept_rate={_fmt_opt(metrics.spec_accept_rate, '.3f')} "
                        f"verify/s={_fmt_opt(verify_calls_per_s, ',.2f')} "
                        f"draft_tok/s={_fmt_opt(draft_tokens_per_s, ',.2f')} "
                        f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
                    )
            finally:
                kill_process_tree(dflash_proc.pid)
                try:
                    dflash_proc.wait(timeout=30)
                except Exception:
                    pass
    finally:
        if call_trace_fp is not None:
            call_trace_fp.close()

    # Render markdown.
    md_lines: list[str] = []
    md_lines.append("# DFLASH Bench Report")
    md_lines.append("")
    md_lines.append("## Settings")
    md_lines.append(f"- dataset: `{args.dataset_name}`")
    md_lines.append(f"- target_model: `{args.target_model}`")
    md_lines.append(f"- draft_model: `{args.draft_model}`")
    md_lines.append(f"- speculative_algorithm: `{args.speculative_algorithm.upper()}`")
    md_lines.append(
        f"- speculative_dflash_block_size: `{args.speculative_dflash_block_size}`"
    )
    md_lines.append(
        f"- speculative_num_draft_tokens: `{args.speculative_num_draft_tokens}`"
    )
    md_lines.append(f"- speculative_num_steps: `{args.speculative_num_steps}`")
    md_lines.append(f"- speculative_eagle_topk: `{args.speculative_eagle_topk}`")
    md_lines.append(f"- disable_overlap_schedule: `{bool(args.disable_overlap_schedule)}`")
    md_lines.append(f"- enable_server_metrics: `{bool(args.enable_server_metrics)}`")
    md_lines.append(f"- server_extra_args: `{args.server_extra_args}`")
    md_lines.append(f"- save_call_trace_path: `{args.save_call_trace_path}`")
    md_lines.append(f"- save_call_trace_prompt: `{bool(args.save_call_trace_prompt)}`")
    md_lines.append(f"- save_call_trace_raw_meta: `{bool(args.save_call_trace_raw_meta)}`")
    md_lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md_lines.append(f"- attention_backends: `{', '.join(attention_backends)}`")
    md_lines.append(f"- tp_size: `{tp}`")
    md_lines.append(f"- concurrencies: `{', '.join(str(x) for x in concurrencies)}`")
    md_lines.append(f"- questions_per_concurrency: `base={args.questions_per_concurrency_base}`")
    md_lines.append(f"- device_sm: `{device_sm}`")
    md_lines.append(f"- is_blackwell: `{is_blackwell}`")
    md_lines.append(f"- skip_baseline: `{bool(args.skip_baseline)}`")
    md_lines.append("- drop_first_batch: `true`")
    md_lines.append("")

    for backend in attention_backends:
        md_lines.append(f"## Backend: `{backend}`")
        md_lines.append("")

        baseline_values = {
            c: baseline_toks.get((backend, c), None) for c in concurrencies
        }
        dflash_values = {
            c: dflash_toks.get((backend, c), None) for c in concurrencies
        }
        speedup_values: dict[int, Optional[float]] = {}
        for c in concurrencies:
            b = baseline_values.get(c, None)
            d = dflash_values.get(c, None)
            speedup_values[c] = None if (b is None or d is None or b <= 0) else (d / b)

        md_lines.append("### Baseline output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=baseline_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")
        
        md_lines.append("### DFLASH output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=dflash_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

        md_lines.append("### Speedup (DFLASH / baseline)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=speedup_values,
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH tau (accept length)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: dflash_accept_len.get((backend, c), None)
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH acceptance rate")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].spec_accept_rate
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH verify calls total")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        float(dflash_metrics[(backend, c)].spec_verify_ct_sum)
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=",.0f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH verify calls per second")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].spec_verify_ct_sum
                        / max(dflash_metrics[(backend, c)].latency_s, 1e-6)
                        if (backend, c) in dflash_metrics
                        and dflash_metrics[(backend, c)].spec_verify_ct_sum > 0
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH drafted tokens per second")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].spec_draft_token_sum
                        / max(dflash_metrics[(backend, c)].latency_s, 1e-6)
                        if (backend, c) in dflash_metrics
                        and dflash_metrics[(backend, c)].spec_draft_token_sum > 0
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH accepted draft tokens per second")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].spec_accept_token_sum
                        / max(dflash_metrics[(backend, c)].latency_s, 1e-6)
                        if (backend, c) in dflash_metrics
                        and dflash_metrics[(backend, c)].spec_accept_token_sum > 0
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH wall time per verify call (s)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].latency_s
                        / max(float(dflash_metrics[(backend, c)].spec_verify_ct_sum), 1.0)
                        if (backend, c) in dflash_metrics
                        and dflash_metrics[(backend, c)].spec_verify_ct_sum > 0
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".6f",
            )
        )
        md_lines.append("")

        md_lines.append("### Request E2E latency avg (s)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].e2e_latency_avg_s
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### Request E2E latency p95 (s)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].e2e_latency_p95_s
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### Baseline request E2E latency avg (s)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        baseline_metrics[(backend, c)].e2e_latency_avg_s
                        if (backend, c) in baseline_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH inference_time avg (s, if server metrics enabled)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].inference_time_avg_s
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH decode_throughput avg (tok/s, if server metrics enabled)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].decode_throughput_avg_tok_s
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH reported draft time avg (s, if exposed by server)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].extra_timing_avgs_s.get("draft_time_s")
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".6f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH reported verify time avg (s, if exposed by server)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].extra_timing_avgs_s.get("verify_time_s")
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".6f",
            )
        )
        md_lines.append("")

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            f.write("\n")
        print(f"\nWrote markdown report to: {args.output_md}")
    else:
        print("\nMarkdown report disabled (pass --output-md to write one).")

    if args.save_call_trace_path:
        print(f"Wrote per-call JSONL trace to: {args.save_call_trace_path}")


if __name__ == "__main__":
    main()
