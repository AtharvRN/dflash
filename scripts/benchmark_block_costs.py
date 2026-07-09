from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.model import DFlashDraftModel, dflash_generate


GSM8K_FORMAT = (
    "{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def _parse_arms(value: str) -> list[int]:
    arms = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not arms:
        raise ValueError("at least one block size is required")
    if arms != sorted(set(arms)):
        raise ValueError(f"block sizes must be sorted and unique, got {arms}")
    return arms


def _load_gsm8k_prompts(split: str, max_samples: int, seed: int) -> list[str]:
    rows = list(load_dataset("openai/gsm8k", "main", split=split))
    prompts = [GSM8K_FORMAT.format(**row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:max_samples]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _render_prompt(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure single-request DFlash cycle costs by block size.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dataset", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--arms", type=_parse_arms, default="4,8,12,16")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    prompts = _load_gsm8k_prompts(args.split, args.max_samples + args.warmup_samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    ).to("cuda").eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    ).to("cuda").eval()

    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    results: dict[str, Any] = {
        "model": args.model,
        "draft_model": args.draft_model,
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "warmup_samples": args.warmup_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "blocks": {},
    }

    for block_size in args.arms:
        cycle_ms_values: list[float] = []
        tpot_ms_values: list[float] = []
        tps_values: list[float] = []
        committed_values: list[float] = []
        output_token_values: list[float] = []
        cycle_count_values: list[float] = []

        for idx, prompt in enumerate(tqdm(prompts, desc=f"B={block_size}")):
            text = _render_prompt(tokenizer, prompt, args.enable_thinking)
            input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
            stats = dflash_generate(
                draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                block_size=block_size,
                return_stats=True,
            )
            if idx < args.warmup_samples:
                continue
            total_decode_s = float(stats.time_per_output_token) * float(stats.num_output_tokens)
            num_cycles = max(1, len(stats.acceptance_lengths))
            cycle_ms_values.append(1000.0 * total_decode_s / num_cycles)
            tpot_ms_values.append(1000.0 * float(stats.time_per_output_token))
            tps_values.append(1.0 / float(stats.time_per_output_token))
            committed_values.extend(float(x) for x in stats.acceptance_lengths)
            output_token_values.append(float(stats.num_output_tokens))
            cycle_count_values.append(float(num_cycles))

        results["blocks"][str(block_size)] = {
            "cycle_ms": _summary(cycle_ms_values)["mean"],
            "mean_cycle_ms": _summary(cycle_ms_values)["mean"],
            "cycle_ms_summary": _summary(cycle_ms_values),
            "tpot_ms_summary": _summary(tpot_ms_values),
            "tokens_per_s_summary": _summary(tps_values),
            "committed_per_cycle_summary": _summary(committed_values),
            "output_tokens_summary": _summary(output_token_values),
            "cycles_per_prompt_summary": _summary(cycle_count_values),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
