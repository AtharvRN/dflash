from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import chain
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.benchmark import _apply_chat_template, _get_transformers_attn_impl, load_and_process_dataset
from dflash.dynamic import dflash_generate_dynamic
from dflash.model import DFlashDraftModel, dflash_generate
from dflash.policy import DFlashSurvivalBlockPolicy, DFlashV2HorizonBlockPolicy


def _parse_ints(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if out != tuple(sorted(set(out))):
        raise ValueError(f"values must be sorted and unique, got {out}")
    return out


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _limit_dataset(dataset: list[dict[str, Any]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    dataset = list(dataset)
    rng.shuffle(dataset)
    return dataset[:max_samples]


def _mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _fixed_metrics(responses: list[Any], block_size: int) -> dict[str, float]:
    committed = list(chain.from_iterable(r.acceptance_lengths for r in responses))
    accepted = [max(0, x - 1) for x in committed]
    ratios = [x / max(block_size - 1, 1) for x in accepted]
    return {
        "mean_accepted_draft_len": _mean_or_zero([float(x) for x in accepted]),
        "mean_acceptance_ratio": _mean_or_zero([float(x) for x in ratios]),
        "mean_block_size": float(block_size),
        "mean_draft_budget": float(block_size - 1),
        "mean_time_per_output_token": _mean_or_zero([float(r.time_per_output_token) for r in responses]),
        "total_output_tokens": float(sum(int(r.num_output_tokens) for r in responses)),
    }


def _dynamic_metrics(responses: list[Any]) -> dict[str, Any]:
    accepted = list(chain.from_iterable(r.accepted_draft_lengths for r in responses))
    ratios = list(chain.from_iterable(r.acceptance_ratios for r in responses))
    block_sizes = list(chain.from_iterable(r.block_sizes for r in responses))
    draft_budgets = list(chain.from_iterable(r.draft_budgets for r in responses))
    hist = {str(block): block_sizes.count(block) for block in sorted(set(block_sizes))}
    return {
        "mean_accepted_draft_len": _mean_or_zero([float(x) for x in accepted]),
        "mean_acceptance_ratio": _mean_or_zero([float(x) for x in ratios]),
        "mean_block_size": _mean_or_zero([float(x) for x in block_sizes]),
        "mean_draft_budget": _mean_or_zero([float(x) for x in draft_budgets]),
        "mean_time_per_output_token": _mean_or_zero([float(r.time_per_output_token) for r in responses]),
        "total_output_tokens": float(sum(int(r.num_output_tokens) for r in responses)),
        "block_hist": hist,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep runtime DFlash survival-policy alpha values.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=["survival", "horizon_v2"], default="survival")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--fixed-block-size", type=int, default=16)
    parser.add_argument("--arms", type=_parse_ints, default=(4, 8, 12, 16))
    parser.add_argument("--alphas", type=_parse_floats, default=(0.80, 0.85, 0.90, 0.95))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    attn_impl = args.attn_implementation or _get_transformers_attn_impl()
    local_files_only = bool(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))

    print(f"loading target model: {args.model}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        dtype=dtype,
        local_files_only=local_files_only,
    ).to("cuda").eval()
    print("target model loaded", flush=True)
    print(f"loading draft model: {args.draft_model}", flush=True)
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model,
        attn_implementation=attn_impl,
        dtype=dtype,
        local_files_only=local_files_only,
    ).to("cuda").eval()
    print("draft model loaded", flush=True)
    print("loading tokenizer and dataset", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)

    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples, args.seed)
    print(f"loaded dataset rows: {len(dataset)}", flush=True)
    fixed_responses: list[Any] = []
    dynamic_responses: dict[str, list[Any]] = {str(alpha): [] for alpha in args.alphas}

    policy_cls = DFlashV2HorizonBlockPolicy if args.policy_kind == "horizon_v2" else DFlashSurvivalBlockPolicy
    policies = {}
    for alpha in args.alphas:
        kwargs: dict[str, Any] = {
            "checkpoint_path": args.policy_checkpoint,
            "arms": args.arms,
            "alpha": alpha,
            "monotonicize_probs": True,
        }
        if args.policy_kind == "survival":
            kwargs.update(
                {
                    "internal_feature_dim": 128,
                    "internal_feature_window": 16,
                    "internal_feature_seed": 0,
                }
            )
        policies[str(alpha)] = policy_cls(**kwargs)

    for instance in tqdm(dataset, desc="runtime alpha sweep"):
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            fixed = dflash_generate(
                draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=args.fixed_block_size,
                return_stats=True,
            )
            fixed_responses.append(fixed)

            generated_ids = fixed.output_ids[0, fixed.num_input_tokens :]
            messages.append({"role": "assistant", "content": tokenizer.decode(generated_ids, skip_special_tokens=True)})

            for alpha_key, policy in policies.items():
                dynamic = dflash_generate_dynamic(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    policy=policy,
                    return_stats=True,
                )
                dynamic_responses[alpha_key].append(dynamic)

    payload = {
        "model": args.model,
        "draft_model": args.draft_model,
        "policy_checkpoint": str(args.policy_checkpoint),
        "policy_kind": args.policy_kind,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "fixed_block_size": args.fixed_block_size,
        "arms": list(args.arms),
        "alphas": list(args.alphas),
        "num_responses": len(fixed_responses),
        "fixed": _fixed_metrics(fixed_responses, args.fixed_block_size),
        "dynamic_by_alpha": {
            alpha_key: _dynamic_metrics(responses)
            for alpha_key, responses in dynamic_responses.items()
        },
    }
    print(json.dumps(payload, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
