from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
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
    sampling_custom_params: Optional[dict] = None,
) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    if sampling_custom_params:
        sampling_params["custom_params"] = dict(sampling_custom_params)
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
    sampling_custom_params: Optional[dict] = None,
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
    if sampling_custom_params:
        sampling_params["custom_params"] = dict(sampling_custom_params)
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


@dataclass
class DynamicChunkRecord:
    chunk_idx: int
    block_size: int
    chunk_size: int
    request_count: int
    output_tokens: int
    latency_s: float
    output_toks_per_s: float
    tau: Optional[float]
    accept_rate: Optional[float]
    verify_calls: int
    drafted_tokens: int
    accepted_tokens: int


@dataclass(frozen=True)
class DynamicChoice:
    block_size: int
    batch_size: int


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
    sampling_custom_params: Optional[dict] = None,
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
                sampling_custom_params=sampling_custom_params,
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
                        sampling_custom_params=sampling_custom_params,
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
        "draft_time_per_cycle_s": [
            "spec_draft_time_per_cycle_s",
            "draft_time_per_cycle_s",
        ],
        "verify_time_per_cycle_s": [
            "spec_verify_time_per_cycle_s",
            "verify_time_per_cycle_s",
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
            draft_time_per_cycle_s = _extract_float(
                meta, extra_timing_fields["draft_time_per_cycle_s"]
            )
            verify_time_per_cycle_s = _extract_float(
                meta, extra_timing_fields["verify_time_per_cycle_s"]
            )
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
                    "draft_time_per_cycle_s": draft_time_per_cycle_s,
                    "verify_time_per_cycle_s": verify_time_per_cycle_s,
                    "spec_runtime_bs_hist": meta.get("spec_runtime_bs_hist"),
                    "spec_runtime_bs_mode": _extract_float(
                        meta, ["spec_runtime_bs_mode"]
                    ),
                    "spec_runtime_bs_avg": _extract_float(
                        meta, ["spec_runtime_bs_avg"]
                    ),
                    "spec_cycle_trace": meta.get("spec_cycle_trace"),
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
                sampling_custom_params=sampling_custom_params,
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
                sampling_custom_params=sampling_custom_params,
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


def _parse_int_csv(s: str) -> list[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    return vals


def _parse_dynamic_gpu_map(s: str) -> dict[int, str]:
    out: dict[int, str] = {}
    raw = str(s).strip()
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --dynamic-gpu-map entry '{item}'. Expected bs:gpu_id.")
        bs_str, gpu_str = item.split(":", 1)
        bs = int(bs_str.strip())
        gpu = gpu_str.strip()
        if not gpu:
            raise ValueError(f"Invalid --dynamic-gpu-map entry '{item}'. GPU id cannot be empty.")
        out[bs] = gpu
    return out


class DynamicAdaptiveController:
    def __init__(
        self,
        *,
        block_sizes: list[int],
        batch_sizes: list[int],
        initial_block_size: Optional[int],
        policy: str,
        ewma_alpha: float,
        exploration_c: float,
        switch_margin: float,
        required_streak: int,
        warmup_chunks: int,
        probe_interval: int,
    ) -> None:
        self.block_sizes = sorted({int(x) for x in block_sizes if int(x) >= 1})
        self.batch_sizes = sorted({int(x) for x in batch_sizes if int(x) >= 1})
        if not self.block_sizes:
            raise ValueError("No dynamic block-size candidates provided.")
        if not self.batch_sizes:
            raise ValueError("No dynamic batch-size candidates provided.")
        if policy not in {"ewma", "ucb"}:
            raise ValueError("dynamic-policy must be one of: ewma, ucb.")
        if not (0.0 < float(ewma_alpha) <= 1.0):
            raise ValueError("dynamic-ewma-alpha must be in (0, 1].")

        self.policy = str(policy)
        self.ewma_alpha = float(ewma_alpha)
        self.exploration_c = float(max(0.0, exploration_c))
        self.switch_margin = float(max(0.0, switch_margin))
        self.required_streak = int(max(1, required_streak))
        self.warmup_chunks = int(max(0, warmup_chunks))
        self.probe_interval = int(max(0, probe_interval))

        self.arms: list[DynamicChoice] = [
            DynamicChoice(block_size=bs, batch_size=bz)
            for bs in self.block_sizes
            for bz in self.batch_sizes
        ]
        if not self.arms:
            raise ValueError("No dynamic controller arms could be constructed.")

        if initial_block_size is None:
            init_block = max(self.block_sizes)
        else:
            init_block = int(max(1, initial_block_size))
        init_block = min(
            self.block_sizes, key=lambda b: (abs(int(b) - init_block), int(b))
        )
        init_batch = max(self.batch_sizes)

        self.score_hat: dict[tuple[int, int], Optional[float]] = {
            (a.block_size, a.batch_size): None for a in self.arms
        }
        self.obs_count: dict[tuple[int, int], int] = {
            (a.block_size, a.batch_size): 0 for a in self.arms
        }
        self.total_obs = 0
        self.current = DynamicChoice(block_size=int(init_block), batch_size=int(init_batch))
        self.pending_target = self.current
        self.pending_streak = 0
        self.probe_cursor = 0
        self.warmup_cursor = 0

    @staticmethod
    def _key(choice: DynamicChoice) -> tuple[int, int]:
        return (int(choice.block_size), int(choice.batch_size))

    def _ewma(self, old: Optional[float], new: float) -> float:
        if old is None:
            return float(new)
        return float((1.0 - self.ewma_alpha) * old + self.ewma_alpha * float(new))

    def _feasible(self, remaining_prompts: int) -> list[DynamicChoice]:
        rem = int(max(1, remaining_prompts))
        feasible = [a for a in self.arms if int(a.batch_size) <= rem]
        return feasible if feasible else [min(self.arms, key=lambda a: a.batch_size)]

    def _next_probe(self, feasible: list[DynamicChoice]) -> DynamicChoice:
        if not feasible:
            return self.current
        for _ in range(max(1, len(self.arms))):
            cand = self.arms[self.probe_cursor % len(self.arms)]
            self.probe_cursor += 1
            if cand in feasible and cand != self.current:
                return cand
        return self.current if self.current in feasible else feasible[0]

    def _ucb_value(self, arm: DynamicChoice) -> float:
        k = self._key(arm)
        base = self.score_hat.get(k)
        if base is None:
            base = 0.0
        n = int(self.obs_count.get(k, 0))
        bonus = self.exploration_c * math.sqrt(
            math.log(float(self.total_obs) + 2.0) / (float(n) + 1.0)
        )
        return float(base + bonus)

    def select(self, *, chunk_idx: int, remaining_prompts: int) -> DynamicChoice:
        feasible = self._feasible(int(remaining_prompts))
        if chunk_idx < self.warmup_chunks:
            choice = feasible[self.warmup_cursor % len(feasible)]
            self.warmup_cursor += 1
            return choice
        if self.probe_interval > 0:
            since_warmup = int(chunk_idx) - self.warmup_chunks
            if since_warmup >= 0 and since_warmup % self.probe_interval == 0:
                return self._next_probe(feasible)

        if self.policy == "ucb":
            return max(feasible, key=self._ucb_value)

        if self.current in feasible:
            return self.current
        scored = [(arm, self.score_hat.get(self._key(arm))) for arm in feasible]
        scored = [(a, s) for a, s in scored if s is not None]
        if scored:
            return max(scored, key=lambda x: x[1])[0]
        return max(feasible, key=lambda a: (a.batch_size, a.block_size))

    def update(self, *, choice: DynamicChoice, score: float) -> None:
        k = self._key(choice)
        if k not in self.score_hat:
            return
        self.score_hat[k] = self._ewma(self.score_hat[k], float(score))
        self.obs_count[k] = int(self.obs_count[k]) + 1
        self.total_obs += 1

        if self.policy != "ewma":
            self.current = choice
            return

        scored = [
            (DynamicChoice(block_size=b, batch_size=z), s)
            for (b, z), s in self.score_hat.items()
            if s is not None
        ]
        if not scored:
            return
        best_choice, best_score = max(scored, key=lambda x: x[1])
        cur_score = self.score_hat.get(self._key(self.current))
        if cur_score is None:
            self.current = best_choice
            self.pending_target = self.current
            self.pending_streak = 0
            return

        rel_improvement = (best_score - cur_score) / max(abs(cur_score), 1e-12)
        eligible = best_choice != self.current and rel_improvement > self.switch_margin
        if not eligible:
            self.pending_target = self.current
            self.pending_streak = 0
            return

        if self.pending_target == best_choice:
            self.pending_streak += 1
        else:
            self.pending_target = best_choice
            self.pending_streak = 1

        if self.pending_streak >= self.required_streak:
            self.current = best_choice
            self.pending_target = self.current
            self.pending_streak = 0


def _aggregate_bench_metrics(metrics_list: list[BenchMetrics]) -> BenchMetrics:
    if not metrics_list:
        raise ValueError("Cannot aggregate empty metrics list.")

    total_latency = float(sum(m.latency_s for m in metrics_list))
    total_requests = int(sum(m.request_count for m in metrics_list))
    total_tokens = int(sum(m.output_tokens for m in metrics_list))
    total_verify_ct = int(sum(m.spec_verify_ct_sum for m in metrics_list))
    total_accept_tokens = int(sum(m.spec_accept_token_sum for m in metrics_list))
    total_draft_tokens = int(sum(m.spec_draft_token_sum for m in metrics_list))

    def _wavg(attr: str) -> Optional[float]:
        num = 0.0
        den = 0.0
        for m in metrics_list:
            v = getattr(m, attr)
            if v is None:
                continue
            w = float(max(1, m.request_count))
            num += float(v) * w
            den += w
        return None if den <= 0.0 else float(num / den)

    all_extra_keys: set[str] = set()
    for m in metrics_list:
        all_extra_keys.update(m.extra_timing_avgs_s.keys())
    extra_timing_avgs: dict[str, Optional[float]] = {}
    for key in sorted(all_extra_keys):
        num = 0.0
        den = 0.0
        for m in metrics_list:
            v = m.extra_timing_avgs_s.get(key)
            if v is None:
                continue
            w = float(max(1, m.request_count))
            num += float(v) * w
            den += w
        extra_timing_avgs[key] = None if den <= 0.0 else float(num / den)

    return BenchMetrics(
        latency_s=total_latency,
        request_count=total_requests,
        output_tokens=total_tokens,
        output_toks_per_s=(float(total_tokens) / max(total_latency, 1e-6)),
        spec_accept_length=_wavg("spec_accept_length"),
        spec_verify_ct_sum=total_verify_ct,
        spec_accept_rate=_wavg("spec_accept_rate"),
        spec_accept_token_sum=total_accept_tokens,
        spec_draft_token_sum=total_draft_tokens,
        e2e_latency_avg_s=_wavg("e2e_latency_avg_s"),
        e2e_latency_p50_s=_wavg("e2e_latency_p50_s"),
        e2e_latency_p95_s=_wavg("e2e_latency_p95_s"),
        inference_time_avg_s=_wavg("inference_time_avg_s"),
        decode_throughput_avg_tok_s=_wavg("decode_throughput_avg_tok_s"),
        extra_timing_avgs_s=extra_timing_avgs,
    )


def _run_dynamic_spec(
    *,
    urls_by_bs: dict[int, str],
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    batch_requests: bool,
    stop: list[str],
    timeout_s: int,
    controller: DynamicAdaptiveController,
    score_metric: str,
    send_runtime_block_size_param: bool,
    trace_fp: Optional[TextIO],
    trace_common: Optional[dict],
    trace_include_prompt: bool,
    trace_include_raw_meta: bool,
) -> tuple[
    BenchMetrics,
    list[DynamicChunkRecord],
    dict[int, int],
    dict[int, int],
    dict[tuple[int, int], int],
]:
    if not urls_by_bs:
        raise ValueError("Dynamic spec requires at least one block-size server URL.")

    max_chunk_size = max(1, int(concurrency))
    chunk_records: list[DynamicChunkRecord] = []
    metrics_accum: list[BenchMetrics] = []
    usage_counts: dict[int, int] = {int(k): 0 for k in sorted(urls_by_bs.keys())}
    batch_usage_counts: dict[int, int] = {}
    arm_usage_counts: dict[tuple[int, int], int] = {}
    max_block_size = max(int(k) for k in urls_by_bs.keys())

    def _score_chunk(m: BenchMetrics) -> float:
        if score_metric == "accepted_toks_per_s":
            return float(m.spec_accept_token_sum) / max(float(m.latency_s), 1e-6)
        if score_metric == "tau_over_verify":
            return (
                float(m.spec_accept_length) / max(float(m.spec_verify_ct_sum), 1.0)
                if m.spec_accept_length is not None
                else 0.0
            )
        if score_metric == "multi_objective":
            throughput = float(m.output_toks_per_s)
            accept_rate = float(m.spec_accept_rate) if m.spec_accept_rate is not None else 0.0
            tau_norm = (
                float(m.spec_accept_length) / max(float(max_block_size), 1.0)
                if m.spec_accept_length is not None
                else 0.0
            )
            verify_per_out_tok = float(m.spec_verify_ct_sum) / max(float(m.output_tokens), 1.0)
            draft_per_out_tok = float(m.spec_draft_token_sum) / max(float(m.output_tokens), 1.0)
            return (
                throughput
                * (1.0 + 0.35 * accept_rate + 0.25 * tau_norm)
                / (1.0 + 0.75 * verify_per_out_tok + 0.15 * draft_per_out_tok)
            )
        return float(m.output_toks_per_s)

    start_idx = 0
    chunk_idx = 0
    while start_idx < len(prompts):
        remaining = len(prompts) - start_idx
        choice = controller.select(chunk_idx=chunk_idx, remaining_prompts=remaining)
        chosen_bs = int(choice.block_size)
        chosen_chunk_size = min(max(1, int(choice.batch_size)), max_chunk_size, remaining)
        chunk_prompts = prompts[start_idx : start_idx + chosen_chunk_size]
        if not chunk_prompts:
            break
        start_idx += chosen_chunk_size
        if chosen_bs not in urls_by_bs:
            # Safety fallback if controller selects a candidate not provisioned.
            chosen_bs = max(urls_by_bs.keys())
        usage_counts[chosen_bs] = usage_counts.get(chosen_bs, 0) + 1
        batch_usage_counts[chosen_chunk_size] = batch_usage_counts.get(chosen_chunk_size, 0) + 1
        arm_key = (int(chosen_bs), int(chosen_chunk_size))
        arm_usage_counts[arm_key] = arm_usage_counts.get(arm_key, 0) + 1

        m = _run_bench_requests(
            urls_by_bs[chosen_bs],
            prompts=chunk_prompts,
            max_new_tokens=max_new_tokens,
            concurrency=chosen_chunk_size,
            batch_requests=batch_requests,
            stop=stop,
            timeout_s=timeout_s,
            expect_dflash=True,
            sampling_custom_params=(
                {"dflash_block_size": int(chosen_bs)}
                if send_runtime_block_size_param
                else None
            ),
            trace_fp=trace_fp,
            trace_common={
                **(trace_common or {}),
                "dynamic_mode": True,
                "dynamic_block_size": int(chosen_bs),
                "dynamic_chunk_idx": int(chunk_idx),
                "dynamic_batch_size": int(chosen_chunk_size),
                "dynamic_chunk_size": int(len(chunk_prompts)),
            },
            trace_include_prompt=trace_include_prompt,
            trace_include_raw_meta=trace_include_raw_meta,
        )
        metrics_accum.append(m)
        chunk_records.append(
            DynamicChunkRecord(
                chunk_idx=int(chunk_idx),
                block_size=int(chosen_bs),
                chunk_size=int(chosen_chunk_size),
                request_count=int(m.request_count),
                output_tokens=int(m.output_tokens),
                latency_s=float(m.latency_s),
                output_toks_per_s=float(m.output_toks_per_s),
                tau=m.spec_accept_length,
                accept_rate=m.spec_accept_rate,
                verify_calls=int(m.spec_verify_ct_sum),
                drafted_tokens=int(m.spec_draft_token_sum),
                accepted_tokens=int(m.spec_accept_token_sum),
            )
        )
        score = _score_chunk(m)
        controller.update(
            choice=DynamicChoice(block_size=int(chosen_bs), batch_size=int(chosen_chunk_size)),
            score=float(score),
        )
        chunk_idx += 1

    return (
        _aggregate_bench_metrics(metrics_accum),
        chunk_records,
        usage_counts,
        batch_usage_counts,
        arm_usage_counts,
    )


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
        "--request-dflash-block-size",
        type=int,
        default=None,
        help="Optional per-request runtime DFLASH block size sent via sampling_params.custom_params['dflash_block_size'] (DFLASH only).",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-block-size",
        action="store_true",
        help="Enable true server-side adaptive DFLASH block size (per-request state updated each verify cycle).",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-rho",
        type=float,
        default=0.30,
        help="Server-side DFLASH adaptive EWMA rho.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-delta",
        type=float,
        default=1.0,
        help="Server-side DFLASH adaptive growth delta.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-k-min",
        type=int,
        default=None,
        help="Server-side DFLASH adaptive minimum block size.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-k-max",
        type=int,
        default=None,
        help="Server-side DFLASH adaptive maximum block size.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-k-start",
        type=int,
        default=None,
        help="Server-side DFLASH adaptive initial block size per request.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-low-accept-threshold",
        type=float,
        default=0.35,
        help="Server-side DFLASH adaptive immediate fallback threshold on acceptance ratio.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-low-accept-streak",
        type=int,
        default=2,
        help="Server-side DFLASH adaptive consecutive low-accept cycles before one-step fallback.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-high-accept-threshold",
        type=float,
        default=0.90,
        help="Server-side DFLASH adaptive threshold for allowing one-step block-size increase.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-high-accept-streak",
        type=int,
        default=2,
        help="Server-side DFLASH adaptive consecutive high-accept cycles before one-step increase.",
    )
    parser.add_argument(
        "--speculative-dflash-adaptive-cooldown-cycles",
        type=int,
        default=1,
        help="Server-side DFLASH adaptive hold cycles after each block-size change.",
    )
    parser.add_argument(
        "--dynamic-block-sizes",
        type=str,
        default="",
        help="Deprecated. Legacy benchmark-side dynamic routing removed; use server-side DFLASH adaptive flags instead.",
    )
    parser.add_argument(
        "--dynamic-batch-sizes",
        type=str,
        default="",
        help="Deprecated. Legacy benchmark-side dynamic routing removed; use server-side DFLASH adaptive flags instead.",
    )
    parser.add_argument(
        "--dynamic-policy",
        type=str,
        default="ewma",
        choices=["ewma", "ucb"],
        help="Dynamic controller policy. ewma = hysteresis switching; ucb = exploration-aware bandit.",
    )
    parser.add_argument(
        "--dynamic-gpu-map",
        type=str,
        default="",
        help="Deprecated. Legacy benchmark-side dynamic routing removed; use server-side DFLASH adaptive flags instead.",
    )
    parser.add_argument(
        "--dynamic-single-server",
        action="store_true",
        help="Deprecated. Legacy benchmark-side dynamic routing removed; use server-side DFLASH adaptive flags instead.",
    )
    parser.add_argument(
        "--dynamic-ewma-alpha",
        type=float,
        default=0.20,
        help="EWMA alpha for dynamic controller score smoothing.",
    )
    parser.add_argument(
        "--dynamic-exploration-c",
        type=float,
        default=0.15,
        help="Exploration coefficient for --dynamic-policy ucb.",
    )
    parser.add_argument(
        "--dynamic-switch-margin",
        type=float,
        default=0.02,
        help="Minimum relative score gain required to switch dynamic block size.",
    )
    parser.add_argument(
        "--dynamic-required-streak",
        type=int,
        default=2,
        help="Consecutive decisions needed before switching dynamic block size.",
    )
    parser.add_argument(
        "--dynamic-warmup-chunks",
        type=int,
        default=4,
        help="Number of initial chunks used for round-robin probing across candidate block sizes.",
    )
    parser.add_argument(
        "--dynamic-probe-interval",
        type=int,
        default=8,
        help="Periodic probe interval (in chunks) for non-current block sizes; 0 disables probing.",
    )
    parser.add_argument(
        "--dynamic-score-metric",
        type=str,
        default="output_toks_per_s",
        choices=[
            "output_toks_per_s",
            "accepted_toks_per_s",
            "tau_over_verify",
            "multi_objective",
        ],
        help="Objective metric used by dynamic block-size controller.",
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
    parser.add_argument(
        "--enable-dflash-stage-timing",
        action="store_true",
        help="Set SGLANG_DFLASH_REPORT_TIMING=1 for launched servers so DFLASH reports attributed draft/verify timing fields in meta_info.",
    )
    parser.add_argument(
        "--enable-dflash-cycle-trace",
        action="store_true",
        help="Pass --speculative-dflash-cycle-trace to SGLang so DFLASH emits per-request per-cycle trace in meta_info.",
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
        "--fixed-question-count",
        type=int,
        default=0,
        help="If > 0, use the same fixed number of prompts for every concurrency/config.",
    )
    parser.add_argument(
        "--fixed-question-offset",
        type=int,
        default=0,
        help="Start index for fixed-question mode (only used when --fixed-question-count > 0).",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer,fa3,fa4",
        help="Comma-separated list. Will auto-skip fa3 unless SM90 (Hopper), and fa4 unless SM100+ (Blackwell).",
    )
    args = parser.parse_args()

    if args.enable_dflash_stage_timing:
        os.environ["SGLANG_DFLASH_REPORT_TIMING"] = "1"
        print("[setup] enabled SGLANG_DFLASH_REPORT_TIMING=1 for launched SGLang servers")

    dynamic_block_sizes: list[int] = []
    dynamic_batch_sizes: list[int] = []
    dynamic_mode = False
    dynamic_gpu_map: dict[int, str] = {}
    request_dflash_block_size: Optional[int] = None

    # Legacy client-side dynamic routing is intentionally retired in favor of
    # true server-side adaptive DFLASH block size.
    if (
        args.dynamic_block_sizes.strip()
        or args.dynamic_batch_sizes.strip()
        or args.dynamic_gpu_map.strip()
        or args.dynamic_single_server
    ):
        raise RuntimeError(
            "Legacy benchmark-side dynamic block-size routing was removed. "
            "Use server-side adaptive mode with "
            "--speculative-dflash-adaptive-block-size and related flags."
        )

    if args.request_dflash_block_size is not None:
        req_bs = int(args.request_dflash_block_size)
        if req_bs <= 0:
            raise RuntimeError(
                f"--request-dflash-block-size must be > 0, got {req_bs}."
            )
        if args.speculative_algorithm.upper() != "DFLASH":
            print(
                "[warn] --request-dflash-block-size is ignored because speculative algorithm is not DFLASH.",
                flush=True,
            )
        else:
            request_dflash_block_size = req_bs

    if args.speculative_dflash_adaptive_block_size and args.speculative_algorithm.upper() != "DFLASH":
        raise RuntimeError(
            "--speculative-dflash-adaptive-block-size is only valid with --speculative-algorithm DFLASH."
        )
    if args.enable_dflash_cycle_trace and args.speculative_algorithm.upper() != "DFLASH":
        raise RuntimeError(
            "--enable-dflash-cycle-trace is only valid with --speculative-algorithm DFLASH."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    fixed_question_count = int(args.fixed_question_count)
    fixed_question_offset = int(args.fixed_question_offset)
    if fixed_question_count < 0:
        raise RuntimeError(
            f"--fixed-question-count must be >= 0, got {fixed_question_count}."
        )
    if fixed_question_offset < 0:
        raise RuntimeError(
            f"--fixed-question-offset must be >= 0, got {fixed_question_offset}."
        )
    if fixed_question_count == 0 and fixed_question_offset != 0:
        raise RuntimeError(
            "--fixed-question-offset requires --fixed-question-count > 0."
        )
    fixed_subset_mode = fixed_question_count > 0

    if fixed_subset_mode:
        num_questions_by_conc = {c: fixed_question_count for c in concurrencies}
    else:
        num_questions_by_conc = {
            c: min(
                int(args.questions_per_concurrency_base) * int(c),
                int(args.max_questions_per_config),
            )
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
    if not dataset:
        raise RuntimeError(f"Dataset {args.dataset_name} is empty after preprocessing.")
    if fixed_subset_mode:
        required_questions = (
            fixed_question_offset + fixed_question_count + max_concurrency
        )
    else:
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

    fixed_eval_prompts: list[str] = []
    fixed_warmup_prompts: list[str] = []
    if fixed_subset_mode:
        eval_start = fixed_question_offset
        eval_end = eval_start + fixed_question_count
        warmup_start = eval_end
        warmup_end = warmup_start + max_concurrency
        fixed_eval_prompts = prompts[eval_start:eval_end]
        fixed_warmup_prompts = prompts[warmup_start:warmup_end]
        if len(fixed_eval_prompts) != fixed_question_count:
            raise RuntimeError(
                "Failed to build fixed evaluation prompt window. "
                f"wanted={fixed_question_count}, got={len(fixed_eval_prompts)}"
            )
        if len(fixed_warmup_prompts) != max_concurrency:
            raise RuntimeError(
                "Failed to build fixed warmup prompt pool. "
                f"wanted={max_concurrency}, got={len(fixed_warmup_prompts)}"
            )

    def _prompts_for_run(n: int, conc: int) -> list[str]:
        if not fixed_subset_mode:
            return prompts[: n + conc]
        return fixed_warmup_prompts[:conc] + fixed_eval_prompts[:n]

    # Results indexed by (backend, concurrency) for baseline + dflash.
    # Removed TP dimension from keys since we aren't sweeping it.
    baseline_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, int], Optional[float]] = {}
    baseline_metrics: dict[tuple[str, int], BenchMetrics] = {}
    dflash_metrics: dict[tuple[str, int], BenchMetrics] = {}
    dynamic_usage_counts: dict[tuple[str, int], dict[int, int]] = {}
    dynamic_batch_usage_counts: dict[tuple[str, int], dict[int, int]] = {}
    dynamic_arm_usage_counts: dict[tuple[str, int], dict[tuple[int, int], int]] = {}
    dynamic_chunk_logs: dict[tuple[str, int], list[DynamicChunkRecord]] = {}
    
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
                            prompts=_prompts_for_run(n, int(conc)),
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
            request_custom_params = (
                {"dflash_block_size": int(request_dflash_block_size)}
                if (spec_algo == "DFLASH" and request_dflash_block_size is not None)
                else None
            )
            print(f"\n=== backend={backend} tp={tp} ({spec_algo}) ===")
            def _build_spec_server_args(block_size_override: Optional[int]) -> list[str]:
                spec_server_args = [
                    *common_server_args,
                    "--speculative-algorithm",
                    spec_algo,
                ]
                if args.draft_model:
                    spec_server_args.extend(
                        ["--speculative-draft-model-path", args.draft_model]
                    )
                if block_size_override is not None:
                    spec_server_args.extend(
                        [
                            "--speculative-dflash-block-size",
                            str(int(block_size_override)),
                        ]
                    )
                elif args.speculative_dflash_block_size is not None:
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
                if (
                    spec_algo == "DFLASH"
                    and bool(args.speculative_dflash_adaptive_block_size)
                ):
                    spec_server_args.extend(
                        ["--speculative-dflash-adaptive-block-size"]
                    )
                    spec_server_args.extend(
                        [
                            "--speculative-dflash-adaptive-rho",
                            str(float(args.speculative_dflash_adaptive_rho)),
                            "--speculative-dflash-adaptive-delta",
                            str(float(args.speculative_dflash_adaptive_delta)),
                            "--speculative-dflash-adaptive-low-accept-threshold",
                            str(float(args.speculative_dflash_adaptive_low_accept_threshold)),
                            "--speculative-dflash-adaptive-low-accept-streak",
                            str(int(args.speculative_dflash_adaptive_low_accept_streak)),
                            "--speculative-dflash-adaptive-high-accept-threshold",
                            str(float(args.speculative_dflash_adaptive_high_accept_threshold)),
                            "--speculative-dflash-adaptive-high-accept-streak",
                            str(int(args.speculative_dflash_adaptive_high_accept_streak)),
                            "--speculative-dflash-adaptive-cooldown-cycles",
                            str(int(args.speculative_dflash_adaptive_cooldown_cycles)),
                        ]
                    )
                    if args.speculative_dflash_adaptive_k_min is not None:
                        spec_server_args.extend(
                            [
                                "--speculative-dflash-adaptive-k-min",
                                str(int(args.speculative_dflash_adaptive_k_min)),
                            ]
                        )
                    if args.speculative_dflash_adaptive_k_max is not None:
                        spec_server_args.extend(
                            [
                                "--speculative-dflash-adaptive-k-max",
                                str(int(args.speculative_dflash_adaptive_k_max)),
                            ]
                        )
                    if args.speculative_dflash_adaptive_k_start is not None:
                        spec_server_args.extend(
                            [
                                "--speculative-dflash-adaptive-k-start",
                                str(int(args.speculative_dflash_adaptive_k_start)),
                            ]
                        )
                if spec_algo == "DFLASH" and bool(args.enable_dflash_cycle_trace):
                    spec_server_args.extend(["--speculative-dflash-cycle-trace"])
                return spec_server_args

            if dynamic_mode:
                urls_by_bs: dict[int, str] = {}
                procs_by_bs: dict[int, object] = {}
                try:
                    max_dynamic_bs = max(dynamic_block_sizes)
                    dflash_port = find_available_port(port_base + 1)
                    dflash_url = f"http://127.0.0.1:{dflash_port}"
                    launch_env = None
                    if max_dynamic_bs in dynamic_gpu_map:
                        launch_env = {
                            "CUDA_VISIBLE_DEVICES": str(dynamic_gpu_map[max_dynamic_bs])
                        }
                    proc = popen_launch_server(
                        args.target_model,
                        dflash_url,
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                        other_args=_build_spec_server_args(max_dynamic_bs),
                        env=launch_env,
                    )
                    for bs in dynamic_block_sizes:
                        urls_by_bs[int(bs)] = dflash_url
                        procs_by_bs[int(bs)] = proc
                    print(
                        f"[dynamic] launched single server at {dflash_url} "
                        f"(max_bs={max_dynamic_bs}, gpu={dynamic_gpu_map.get(max_dynamic_bs, 'inherit')})"
                    )

                    _send_generate(
                        dflash_url,
                        "Hello",
                        max_new_tokens=8,
                        stop=[],
                        timeout_s=min(int(args.timeout_s), 300),
                    )
                    print(f"[dynamic] warmed single server (max_bs={max_dynamic_bs})")

                    for conc in concurrencies:
                        n = num_questions_by_conc[conc]
                        _flush_cache(dflash_url)
                        print(f"[dynamic] flushed single server cache before conc={conc}")

                        batch_sizes_for_conc = (
                            sorted(
                                {
                                    int(x)
                                    for x in (
                                        dynamic_batch_sizes if dynamic_batch_sizes else [int(conc)]
                                    )
                                    if 1 <= int(x) <= int(conc)
                                }
                            )
                            or [int(conc)]
                        )

                        controller = DynamicAdaptiveController(
                            block_sizes=dynamic_block_sizes,
                            batch_sizes=batch_sizes_for_conc,
                            initial_block_size=min(
                                int(conc), int(max(dynamic_block_sizes))
                            ),
                            policy=str(args.dynamic_policy),
                            ewma_alpha=float(args.dynamic_ewma_alpha),
                            exploration_c=float(args.dynamic_exploration_c),
                            switch_margin=float(args.dynamic_switch_margin),
                            required_streak=int(args.dynamic_required_streak),
                            warmup_chunks=int(args.dynamic_warmup_chunks),
                            probe_interval=int(args.dynamic_probe_interval),
                        )

                        (
                            metrics,
                            chunk_records,
                            usage,
                            batch_usage,
                            arm_usage,
                        ) = _run_dynamic_spec(
                            urls_by_bs=urls_by_bs,
                            prompts=(
                                fixed_eval_prompts[:n]
                                if fixed_subset_mode
                                else prompts[:n]
                            ),
                            max_new_tokens=int(args.max_new_tokens),
                            concurrency=int(conc),
                            batch_requests=bool(args.batch_requests),
                            stop=[],
                            timeout_s=int(args.timeout_s),
                            controller=controller,
                            score_metric=str(args.dynamic_score_metric),
                            send_runtime_block_size_param=bool(
                                args.dynamic_single_server
                            ),
                            trace_fp=call_trace_fp,
                            trace_common={
                                "mode": "speculative_dynamic",
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
                        dynamic_usage_counts[(backend, conc)] = usage
                        dynamic_batch_usage_counts[(backend, conc)] = batch_usage
                        dynamic_arm_usage_counts[(backend, conc)] = arm_usage
                        dynamic_chunk_logs[(backend, conc)] = chunk_records

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
                        usage_str = ", ".join(
                            [f"bs{b}:{usage.get(b, 0)}" for b in sorted(dynamic_block_sizes)]
                        )
                        batch_usage_str = ", ".join(
                            [f"b{b}:{batch_usage.get(b, 0)}" for b in sorted(batch_sizes_for_conc)]
                        )
                        print(
                            f"[{spec_algo}-dynamic] conc={conc:>2} n={n:<4} "
                            f"toks/s={metrics.output_toks_per_s:,.2f} "
                            f"latency={metrics.latency_s:.1f}s "
                            f"tau={_fmt_opt(metrics.spec_accept_length, '.3f')} "
                            f"accept_rate={_fmt_opt(metrics.spec_accept_rate, '.3f')} "
                            f"verify/s={_fmt_opt(verify_calls_per_s, ',.2f')} "
                            f"draft_tok/s={_fmt_opt(draft_tokens_per_s, ',.2f')} "
                            f"bs_usage=[{usage_str}] "
                            f"batch_usage=[{batch_usage_str}]"
                        )
                finally:
                    stopped_pids: set[int] = set()
                    for bs, proc in procs_by_bs.items():
                        if proc.pid in stopped_pids:
                            continue
                        stopped_pids.add(proc.pid)
                        try:
                            kill_process_tree(proc.pid)
                        except Exception:
                            pass
                        try:
                            proc.wait(timeout=30)
                        except Exception:
                            pass
            else:
                dflash_port = find_available_port(port_base + 1)
                dflash_url = f"http://127.0.0.1:{dflash_port}"
                dflash_proc = popen_launch_server(
                    args.target_model,
                    dflash_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=_build_spec_server_args(None),
                )
                try:
                    _send_generate(
                        dflash_url,
                        "Hello",
                        max_new_tokens=8,
                        stop=[],
                        timeout_s=min(int(args.timeout_s), 300),
                        sampling_custom_params=request_custom_params,
                    )

                    for conc in concurrencies:
                        n = num_questions_by_conc[conc]
                        _flush_cache(dflash_url)
                        print(
                            f"[warmup] run 1 warmup batch (size={conc}) after /flush_cache; excluded from metrics."
                        )
                        metrics = _run_bench_requests(
                            dflash_url,
                            prompts=_prompts_for_run(n, int(conc)),
                            max_new_tokens=int(args.max_new_tokens),
                            concurrency=int(conc),
                            batch_requests=bool(args.batch_requests),
                            stop=[],
                            timeout_s=int(args.timeout_s),
                            expect_dflash=True,
                            sampling_custom_params=request_custom_params,
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
        f"- request_dflash_block_size: `{args.request_dflash_block_size}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_block_size: `{bool(args.speculative_dflash_adaptive_block_size)}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_rho: `{args.speculative_dflash_adaptive_rho}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_delta: `{args.speculative_dflash_adaptive_delta}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_k_min: `{args.speculative_dflash_adaptive_k_min}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_k_max: `{args.speculative_dflash_adaptive_k_max}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_k_start: `{args.speculative_dflash_adaptive_k_start}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_low_accept_threshold: `{args.speculative_dflash_adaptive_low_accept_threshold}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_low_accept_streak: `{args.speculative_dflash_adaptive_low_accept_streak}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_high_accept_threshold: `{args.speculative_dflash_adaptive_high_accept_threshold}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_high_accept_streak: `{args.speculative_dflash_adaptive_high_accept_streak}`"
    )
    md_lines.append(
        f"- speculative_dflash_adaptive_cooldown_cycles: `{args.speculative_dflash_adaptive_cooldown_cycles}`"
    )
    md_lines.append(f"- dynamic_mode: `{bool(dynamic_mode)}`")
    md_lines.append(
        f"- dynamic_block_sizes: `{', '.join(str(x) for x in dynamic_block_sizes) if dynamic_block_sizes else ''}`"
    )
    md_lines.append(
        f"- dynamic_batch_sizes: `{', '.join(str(x) for x in dynamic_batch_sizes) if dynamic_batch_sizes else ''}`"
    )
    md_lines.append(f"- dynamic_policy: `{args.dynamic_policy}`")
    md_lines.append(f"- dynamic_single_server: `{bool(args.dynamic_single_server)}`")
    md_lines.append(f"- dynamic_gpu_map: `{args.dynamic_gpu_map}`")
    md_lines.append(f"- dynamic_ewma_alpha: `{args.dynamic_ewma_alpha}`")
    md_lines.append(f"- dynamic_exploration_c: `{args.dynamic_exploration_c}`")
    md_lines.append(f"- dynamic_switch_margin: `{args.dynamic_switch_margin}`")
    md_lines.append(f"- dynamic_required_streak: `{args.dynamic_required_streak}`")
    md_lines.append(f"- dynamic_warmup_chunks: `{args.dynamic_warmup_chunks}`")
    md_lines.append(f"- dynamic_probe_interval: `{args.dynamic_probe_interval}`")
    md_lines.append(f"- dynamic_score_metric: `{args.dynamic_score_metric}`")
    md_lines.append(
        f"- speculative_num_draft_tokens: `{args.speculative_num_draft_tokens}`"
    )
    md_lines.append(f"- speculative_num_steps: `{args.speculative_num_steps}`")
    md_lines.append(f"- speculative_eagle_topk: `{args.speculative_eagle_topk}`")
    md_lines.append(f"- disable_overlap_schedule: `{bool(args.disable_overlap_schedule)}`")
    md_lines.append(
        f"- enable_dflash_stage_timing: `{bool(args.enable_dflash_stage_timing)}`"
    )
    md_lines.append(
        f"- enable_dflash_cycle_trace: `{bool(args.enable_dflash_cycle_trace)}`"
    )
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
    md_lines.append(f"- max_questions_per_config: `{args.max_questions_per_config}`")
    md_lines.append(f"- fixed_question_count: `{fixed_question_count}`")
    md_lines.append(f"- fixed_question_offset: `{fixed_question_offset}`")
    md_lines.append(f"- fixed_subset_mode: `{fixed_subset_mode}`")
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

        if dynamic_mode:
            md_lines.append("### DFLASH dynamic block usage (chunk counts)")
            md_lines.append("| conc | usage |")
            md_lines.append("| --- | --- |")
            for c in concurrencies:
                usage = dynamic_usage_counts.get((backend, c))
                if not usage:
                    md_lines.append(f"| {c} | N/A |")
                    continue
                usage_str = ", ".join(
                    [f"bs{b}:{usage.get(b, 0)}" for b in sorted(dynamic_block_sizes)]
                )
                md_lines.append(f"| {c} | {usage_str} |")
            md_lines.append("")

            md_lines.append("### DFLASH dynamic batch usage (chunk counts)")
            md_lines.append("| conc | usage |")
            md_lines.append("| --- | --- |")
            for c in concurrencies:
                usage = dynamic_batch_usage_counts.get((backend, c))
                if not usage:
                    md_lines.append(f"| {c} | N/A |")
                    continue
                usage_str = ", ".join([f"b{k}:{v}" for k, v in sorted(usage.items())])
                md_lines.append(f"| {c} | {usage_str} |")
            md_lines.append("")

            md_lines.append("### DFLASH dynamic arm usage (block,batch chunk counts)")
            md_lines.append("| conc | usage |")
            md_lines.append("| --- | --- |")
            for c in concurrencies:
                usage = dynamic_arm_usage_counts.get((backend, c))
                if not usage:
                    md_lines.append(f"| {c} | N/A |")
                    continue
                usage_str = ", ".join(
                    [f"(bs{bs},b{bz}):{ct}" for (bs, bz), ct in sorted(usage.items())]
                )
                md_lines.append(f"| {c} | {usage_str} |")
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

        md_lines.append(
            "### DFLASH reported draft time per cycle avg (s, if exposed by server)"
        )
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].extra_timing_avgs_s.get(
                            "draft_time_per_cycle_s"
                        )
                        if (backend, c) in dflash_metrics
                        else None
                    )
                    for c in concurrencies
                },
                float_fmt=".6f",
            )
        )
        md_lines.append("")

        md_lines.append(
            "### DFLASH reported verify time per cycle avg (s, if exposed by server)"
        )
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: (
                        dflash_metrics[(backend, c)].extra_timing_avgs_s.get(
                            "verify_time_per_cycle_s"
                        )
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
