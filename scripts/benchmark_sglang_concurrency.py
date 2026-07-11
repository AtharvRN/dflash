from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from dflash.benchmark import _apply_chat_template, load_and_process_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an SGLang /generate endpoint with concurrency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def _send_one(
    base_url: str,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout_s,
    )
    latency_s = time.perf_counter() - start
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, dict):
        out = out[0]
    return {"latency_s": latency_s, "response": out}


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[idx])


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = load_and_process_dataset(args.dataset)
    total_needed = args.num_prompts + args.concurrency
    prompts: list[str] = []
    for i in range(total_needed):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0]
        prompts.append(
            _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": user_content}],
                args.enable_thinking,
            )
        )

    def send(prompt: str) -> dict[str, Any]:
        return _send_one(
            args.base_url,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
        )

    if args.concurrency > 0:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(send, prompts[: args.concurrency]))
        prompts = prompts[args.concurrency :]

    start = time.perf_counter()
    request_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send, prompt) for prompt in prompts]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.label or "benchmark"):
            request_results.append(fut.result())
    wall_time_s = time.perf_counter() - start

    completion_tokens: list[int] = []
    spec_verify_ct: list[int] = []
    spec_accept_lengths: list[float] = []
    latencies = [float(item["latency_s"]) for item in request_results]
    for item in request_results:
        meta = item["response"].get("meta_info", {}) or {}
        completion_tokens.append(int(meta.get("completion_tokens", 0)))
        spec_verify_ct.append(int(meta.get("spec_verify_ct", 0)))
        if "spec_accept_length" in meta:
            try:
                spec_accept_lengths.append(float(meta["spec_accept_length"]))
            except (TypeError, ValueError):
                pass

    total_tokens = int(sum(completion_tokens))
    payload = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "dataset": args.dataset,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "wall_time_s": wall_time_s,
        "total_completion_tokens": total_tokens,
        "throughput_tok_s": total_tokens / max(wall_time_s, 1e-6),
        "mean_completion_tokens": _mean([float(x) for x in completion_tokens]),
        "mean_latency_s": _mean(latencies),
        "p50_latency_s": _percentile(latencies, 0.50),
        "p95_latency_s": _percentile(latencies, 0.95),
        "mean_spec_verify_ct": _mean([float(x) for x in spec_verify_ct]),
        "mean_spec_accept_length": _mean(spec_accept_lengths),
        "requests": request_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()
