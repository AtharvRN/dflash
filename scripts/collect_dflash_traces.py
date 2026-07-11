from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dflash.model import DFlashDraftModel, extract_context_feature, sample


GSM8K_FORMAT = (
    "{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def _safe_float(x: float | None) -> float | None:
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return float(x)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "last": None,
            "slope": None,
        }
    slope = values[-1] - values[0] if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": _safe_float(mean(values)),
        "std": _safe_float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": _safe_float(min(values)),
        "max": _safe_float(max(values)),
        "last": _safe_float(values[-1]),
        "slope": _safe_float(slope),
    }


def _tail_padded(values: deque[float | None], length: int) -> list[float | None]:
    tail = list(values)[-length:]
    return [None] * (length - len(tail)) + tail


def _tail_mask(values: deque[float | None], length: int) -> list[int]:
    return [0 if x is None else 1 for x in _tail_padded(values, length)]


def _finite_values(values: deque[float | None]) -> list[float]:
    return [float(x) for x in values if x is not None]


def _rounded_vector(values: torch.Tensor, ndigits: int = 6) -> list[float]:
    return [round(float(x), ndigits) for x in values.detach().float().cpu().tolist()]


def _project_internal_context(
    *,
    draft_model: DFlashDraftModel,
    target_hidden: torch.Tensor,
    projector: torch.Tensor | None,
) -> torch.Tensor:
    fused = draft_model.hidden_norm(draft_model.fc(target_hidden))
    if projector is not None:
        fused = fused.float() @ projector
    return fused[0].detach().float()


def _append_internal_context(
    history: deque[list[float]],
    *,
    draft_model: DFlashDraftModel,
    target_hidden: torch.Tensor,
    projector: torch.Tensor | None,
) -> None:
    projected = _project_internal_context(
        draft_model=draft_model,
        target_hidden=target_hidden,
        projector=projector,
    )
    for row in projected:
        history.append(_rounded_vector(row))


def _padded_internal_context_window(
    history: deque[list[float]],
    *,
    window: int,
    dim: int,
) -> tuple[list[list[float]], list[int]]:
    tail = list(history)[-window:]
    pad = window - len(tail)
    return ([[0.0] * dim for _ in range(pad)] + tail, [0] * pad + [1] * len(tail))


def _logit_stats(logits: torch.Tensor, token_ids: torch.Tensor | None = None) -> dict[str, list[float]]:
    logits_f = logits.float()
    log_probs = torch.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top_probs, top_ids = torch.topk(probs, k=2, dim=-1)
    out: dict[str, list[float]] = {
        "entropy": entropy.detach().cpu().tolist(),
        "pmax": top_probs[..., 0].detach().cpu().tolist(),
        "margin": (top_probs[..., 0] - top_probs[..., 1]).detach().cpu().tolist(),
        "top1_id": top_ids[..., 0].detach().cpu().tolist(),
        "top2_id": top_ids[..., 1].detach().cpu().tolist(),
    }
    if token_ids is not None:
        gathered = torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        out["token_logprob"] = gathered.detach().cpu().tolist()
        out["token_prob"] = gathered.exp().detach().cpu().tolist()
    return out


def _load_gsm8k_prompts(
    split: str,
    max_samples: int | None,
    seed: int,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    rows = list(load_dataset("openai/gsm8k", "main", split=split))
    indexed = [{"dataset_index": i, "turns": [GSM8K_FORMAT.format(**row)]} for i, row in enumerate(rows)]
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    indexed = [item for pos, item in enumerate(indexed) if pos % num_shards == shard_index]
    if max_samples is not None and len(indexed) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(indexed)
        indexed = indexed[:max_samples]
    return indexed


def _load_jsonl_prompts(
    path: Path,
    max_samples: int | None,
    seed: int,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if "conversations" in row:
                prompt = row["conversations"][0]["value"]
            elif "turns" in row:
                prompt = row["turns"][0]
            elif "prompt" in row:
                prompt = row["prompt"]
            else:
                raise ValueError(
                    f"{path} row {i} must contain conversations[0].value, turns[0], or prompt"
                )
            indexed.append(
                {
                    "dataset_index": i,
                    "turns": [prompt],
                    "source_id": row.get("id", str(i)),
                    "metadata": row.get("metadata", {}),
                }
            )
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    indexed = [item for pos, item in enumerate(indexed) if pos % num_shards == shard_index]
    if max_samples is not None and len(indexed) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(indexed)
        indexed = indexed[:max_samples]
    return indexed


def _make_input_text(tokenizer: Any, prompt: str, *, enable_thinking: bool, input_is_chat_template: bool) -> str:
    if input_is_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _make_pre_features(
    *,
    latest_target_stats: dict[str, float | int | None],
    prev_cycle: dict[str, Any],
    accept_history: deque[int],
    target_entropy_history: deque[float | None],
    draft_entropy_history: deque[float | None],
    target_logprob_history: deque[float | None],
    window: int,
) -> dict[str, Any]:
    return {
        "latest_target": latest_target_stats,
        "prev_cycle": prev_cycle,
        "rolling_accept_len": _summary([float(x) for x in accept_history]),
        "last_target_entropy": _tail_padded(target_entropy_history, window),
        "last_target_entropy_summary": _summary(_finite_values(target_entropy_history)),
        "last_target_logprob": _tail_padded(target_logprob_history, window),
        "last_target_logprob_summary": _summary(_finite_values(target_logprob_history)),
        "last_draft_entropy": _tail_padded(draft_entropy_history, window),
        "last_draft_entropy_mask": _tail_mask(draft_entropy_history, window),
        "last_draft_entropy_summary": _summary(_finite_values(draft_entropy_history)),
    }


@torch.inference_mode()
def collect_one_prompt(
    *,
    draft_model: DFlashDraftModel,
    target: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    prompt_id: str,
    dataset_name: str,
    split: str,
    dataset_index: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    history_window: int,
    enable_thinking: bool,
    input_is_chat_template: bool,
    log_internal_features: bool,
    internal_projector: torch.Tensor | None,
    internal_feature_window: int,
    internal_feature_dim: int,
) -> list[dict[str, Any]]:
    input_text = _make_input_text(
        tokenizer,
        prompt,
        enable_thinking=enable_thinking,
        input_is_chat_template=input_is_chat_template,
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    mask_token_id = draft_model.mask_token_id
    num_draft_slots = block_size - 1

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    first_token = sample(output.logits, temperature)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token
    target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids)
    internal_context_history: deque[list[float]] = deque(maxlen=internal_feature_window)
    if log_internal_features:
        _append_internal_context(
            internal_context_history,
            draft_model=draft_model,
            target_hidden=target_hidden,
            projector=internal_projector,
        )

    prefill_stats = _logit_stats(output.logits[:, -1, :], first_token[:, -1])
    latest_target_stats: dict[str, float | int | None] = {
        "entropy": prefill_stats["entropy"][0],
        "pmax": prefill_stats["pmax"][0],
        "margin": prefill_stats["margin"][0],
        "token_logprob": prefill_stats["token_logprob"][0],
        "token_prob": prefill_stats["token_prob"][0],
        "token_id": int(first_token[0, -1].item()),
    }

    rows: list[dict[str, Any]] = []
    accept_history: deque[int] = deque(maxlen=8)
    target_entropy_history: deque[float | None] = deque(maxlen=history_window)
    draft_entropy_history: deque[float | None] = deque(maxlen=history_window)
    target_logprob_history: deque[float | None] = deque(maxlen=history_window)
    target_entropy_history.append(latest_target_stats["entropy"])
    target_logprob_history.append(latest_target_stats["token_logprob"])
    draft_entropy_history.append(None)

    prev_cycle: dict[str, Any] = {
        "exists": False,
        "accepted_draft_len": 0,
        "committed_len": 1,
        "first_reject_pos": None,
        "accepted_draft_entropy": _summary([]),
    }

    start = num_input_tokens
    cycle_id = 0
    eos_id = tokenizer.eos_token_id

    while start < max_length:
        pre_features = _make_pre_features(
            latest_target_stats=latest_target_stats,
            prev_cycle=prev_cycle,
            accept_history=accept_history,
            target_entropy_history=target_entropy_history,
            draft_entropy_history=draft_entropy_history,
            target_logprob_history=target_logprob_history,
            window=history_window,
        )
        if log_internal_features:
            context_window, context_mask = _padded_internal_context_window(
                internal_context_history,
                window=internal_feature_window,
                dim=internal_feature_dim,
            )
            pre_features["dflash_context"] = context_window[-1]
            pre_features["dflash_context_window"] = context_window
            pre_features["dflash_context_window_mask"] = context_mask

        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_hidden = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size :, :]
        draft_logits = target.lm_head(draft_hidden)
        past_key_values_draft.crop(start)
        draft_tokens = sample(draft_logits, temperature)
        block_output_ids[:, 1:] = draft_tokens

        draft_stats = _logit_stats(draft_logits[0], draft_tokens[0])

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(output.logits, temperature)
        accepted_draft_len = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        committed_len = accepted_draft_len + 1
        first_reject_pos = accepted_draft_len + 1 if accepted_draft_len < num_draft_slots else None

        target_for_draft_tokens = block_output_ids[:, 1:]
        target_stats_for_drafts = _logit_stats(output.logits[0, :num_draft_slots, :], target_for_draft_tokens[0])
        correction_token = posterior[:, accepted_draft_len]
        correction_stats = _logit_stats(
            output.logits[0, accepted_draft_len, :].unsqueeze(0),
            correction_token[0].unsqueeze(0),
        )

        accepted_draft_entropies = draft_stats["entropy"][:accepted_draft_len]
        accepted_target_entropies = target_stats_for_drafts["entropy"][:accepted_draft_len]
        accepted_target_logprobs = target_stats_for_drafts["token_logprob"][:accepted_draft_len]

        row = {
            "dataset": dataset_name,
            "split": split,
            "prompt_id": prompt_id,
            "dataset_index": dataset_index,
            "cycle_id": cycle_id,
            "prompt_tokens": num_input_tokens,
            "prefix_len_before_cycle": start,
            "generated_tokens_before_cycle": start - num_input_tokens,
            "remaining_budget": max_length - start,
            "block_size": block_size,
            "num_draft_slots": num_draft_slots,
            "temperature": temperature,
            "inputs": pre_features,
            "labels": {
                "accepted_draft_len": int(accepted_draft_len),
                "committed_len": int(committed_len),
                "first_reject_pos": first_reject_pos,
                "draft_survival": [1 if accepted_draft_len >= i else 0 for i in range(1, num_draft_slots + 1)],
                "committed_survival": [1 if committed_len >= i else 0 for i in range(1, block_size + 1)],
            },
            "current_cycle_debug": {
                "draft_entropy_by_pos": draft_stats["entropy"],
                "draft_pmax_by_pos": draft_stats["pmax"],
                "draft_margin_by_pos": draft_stats["margin"],
                "target_entropy_for_draft_by_pos": target_stats_for_drafts["entropy"],
                "target_prob_of_draft_by_pos": target_stats_for_drafts["token_prob"],
                "target_logprob_of_draft_by_pos": target_stats_for_drafts["token_logprob"],
                "draft_target_match_by_pos": [
                    int(block_output_ids[0, i + 1].item() == posterior[0, i].item())
                    for i in range(num_draft_slots)
                ],
                "accepted_draft_entropy": _summary(accepted_draft_entropies),
                "accepted_target_entropy": _summary(accepted_target_entropies),
                "correction_target": {
                    "entropy": correction_stats["entropy"][0],
                    "pmax": correction_stats["pmax"][0],
                    "margin": correction_stats["margin"][0],
                    "token_logprob": correction_stats["token_logprob"][0],
                    "token_prob": correction_stats["token_prob"][0],
                    "token_id": int(correction_token[0].item()),
                },
            },
        }
        rows.append(row)

        output_ids[:, start : start + accepted_draft_len + 1] = block_output_ids[:, : accepted_draft_len + 1]
        output_ids[:, start + accepted_draft_len + 1] = correction_token
        start += committed_len
        past_key_values_target.crop(start)

        for i in range(accepted_draft_len):
            target_entropy_history.append(target_stats_for_drafts["entropy"][i])
            target_logprob_history.append(target_stats_for_drafts["token_logprob"][i])
            draft_entropy_history.append(draft_stats["entropy"][i])
        target_entropy_history.append(correction_stats["entropy"][0])
        target_logprob_history.append(correction_stats["token_logprob"][0])
        draft_entropy_history.append(None)

        latest_target_stats = row["current_cycle_debug"]["correction_target"]
        accept_history.append(int(accepted_draft_len))
        prev_cycle = {
            "exists": True,
            "accepted_draft_len": int(accepted_draft_len),
            "committed_len": int(committed_len),
            "first_reject_pos": first_reject_pos,
            "accepted_draft_entropy": _summary(accepted_draft_entropies),
            "accepted_target_entropy": _summary(accepted_target_entropies),
            "accepted_target_logprob": _summary(accepted_target_logprobs),
        }

        target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids)[
            :, : accepted_draft_len + 1, :
        ]
        if log_internal_features:
            _append_internal_context(
                internal_context_history,
                draft_model=draft_model,
                target_hidden=target_hidden,
                projector=internal_projector,
            )
        cycle_id += 1

        if eos_id is not None and eos_id in output_ids[:, num_input_tokens : start + 1]:
            break

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DFlash pre-draft trace rows.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dataset", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--input-is-chat-template", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--history-window", type=int, default=50)
    parser.add_argument("--log-internal-features", action="store_true")
    parser.add_argument(
        "--internal-feature-dim",
        type=int,
        default=256,
        help="Random projection dimension for pre-draft DFlash fused context. Use 0 to store the full vector.",
    )
    parser.add_argument("--internal-feature-window", type=int, default=16)
    parser.add_argument("--internal-feature-seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    dataset_name = args.dataset_name or args.dataset
    if args.dataset_jsonl is not None:
        prompts = _load_jsonl_prompts(
            args.dataset_jsonl,
            args.max_samples,
            args.seed,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        dataset_name = args.dataset_name or args.dataset_jsonl.stem
    else:
        prompts = _load_gsm8k_prompts(
            args.split,
            args.max_samples,
            args.seed,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    ).to("cuda").eval()
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    ).to("cuda").eval()
    internal_projector = None
    if args.log_internal_features and args.internal_feature_dim > 0:
        generator = torch.Generator(device="cpu").manual_seed(args.internal_feature_seed)
        internal_projector = torch.randn(
            draft_model.config.hidden_size,
            args.internal_feature_dim,
            generator=generator,
            dtype=torch.float32,
        )
        internal_projector = (internal_projector / math.sqrt(draft_model.config.hidden_size)).to(target.device)

    num_rows = 0
    with args.output.open("w") as f:
        for local_idx, item in enumerate(tqdm(prompts, desc=f"{dataset_name}:{args.split}")):
            rows = collect_one_prompt(
                draft_model=draft_model,
                target=target,
                tokenizer=tokenizer,
                prompt=item["turns"][0],
                prompt_id=f"{dataset_name}-{args.split}-{item['dataset_index']}",
                dataset_name=dataset_name,
                split=args.split,
                dataset_index=item["dataset_index"],
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                temperature=args.temperature,
                history_window=args.history_window,
                enable_thinking=args.enable_thinking,
                input_is_chat_template=args.input_is_chat_template,
                log_internal_features=args.log_internal_features,
                internal_projector=internal_projector,
                internal_feature_window=args.internal_feature_window,
                internal_feature_dim=args.internal_feature_dim if args.internal_feature_dim > 0 else draft_model.config.hidden_size,
            )
            for row in rows:
                row["local_sample_index"] = local_idx
                f.write(json.dumps(row) + "\n")
            f.flush()
            num_rows += len(rows)

    print(f"wrote {num_rows} rows to {args.output}")


if __name__ == "__main__":
    main()
