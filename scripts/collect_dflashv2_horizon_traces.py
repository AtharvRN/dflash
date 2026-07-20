from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import DFlashDraftModel, extract_context_feature, sample


def _load_manifest(
    path: Path,
    *,
    max_prompts: int | None,
    seed: int,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]
    if max_prompts is not None and len(rows) > max_prompts:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_prompts]
    return rows


def _input_ids_from_messages(tokenizer: Any, row: dict[str, Any], *, enable_thinking: bool) -> torch.Tensor:
    if "messages" in row:
        messages = row["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    elif "prompt" in row:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        raise ValueError("manifest row must contain messages or prompt")
    return tokenizer.encode(text, return_tensors="pt")


def _fused_context(draft_model: DFlashDraftModel, target_hidden: torch.Tensor) -> torch.Tensor:
    return draft_model.hidden_norm(draft_model.fc(target_hidden))[0].detach()


def _padded_context_window(
    history: deque[torch.Tensor],
    *,
    window: int,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tail = list(history)[-window:]
    pad = window - len(tail)
    features = np.zeros((window, hidden_size), dtype=np.float16)
    mask = np.zeros((window,), dtype=np.uint8)
    for out_idx, row in enumerate(tail, start=pad):
        features[out_idx] = row.float().cpu().numpy().astype(np.float16)
        mask[out_idx] = 1
    return features, mask


def _padded_token_window(
    output_ids: torch.Tensor,
    *,
    end_exclusive: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = output_ids[0, :end_exclusive].detach().cpu().numpy().astype(np.int64)
    tail = ids[-window:]
    pad = window - int(tail.shape[0])
    token_ids = np.zeros((window,), dtype=np.int64)
    token_mask = np.zeros((window,), dtype=np.uint8)
    token_ids[pad:] = tail
    token_mask[pad:] = 1
    return token_ids, token_mask


def _padded_float_window(
    history: deque[float],
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    tail = np.asarray(list(history)[-window:], dtype=np.float16)
    pad = window - int(tail.shape[0])
    values = np.zeros((window,), dtype=np.float16)
    mask = np.zeros((window,), dtype=np.uint8)
    values[pad:] = tail
    mask[pad:] = 1
    return values, mask


def _draft_confidence_features(logits: torch.Tensor, token_ids: torch.Tensor) -> np.ndarray:
    return _token_confidence_features(logits, token_ids)


def _token_confidence_features(logits: torch.Tensor, token_ids: torch.Tensor) -> np.ndarray:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    token_logprob = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    token_prob = token_logprob.exp()
    features = torch.stack(
        [
            entropy,
            token_prob,
            top2[:, 0] - top2[:, 1],
            token_logprob,
        ],
        dim=-1,
    )
    return features.detach().cpu().numpy().astype(np.float16)


class HorizonShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        rows_per_shard: int,
        window: int,
        hidden_size: int,
        num_slots: int,
        log_postdraft_confidence: bool,
        log_postdraft_hidden: bool,
        log_predraft_token_ids: bool,
        token_window: int,
        log_verifier_entropy_window: bool,
        verifier_entropy_window: int,
    ) -> None:
        self.output_dir = output_dir
        self.rows_per_shard = rows_per_shard
        self.window = window
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.log_postdraft_confidence = log_postdraft_confidence
        self.log_postdraft_hidden = log_postdraft_hidden
        self.log_predraft_token_ids = log_predraft_token_ids
        self.token_window = token_window
        self.log_verifier_entropy_window = log_verifier_entropy_window
        self.verifier_entropy_window = verifier_entropy_window
        self.shard_idx = -1
        self.row_idx = 0
        self.total_rows = 0
        self.features: np.memmap | None = None
        self.mask: np.memmap | None = None
        self.survival: np.memmap | None = None
        self.accepted_len: np.memmap | None = None
        self.prompt_index: np.memmap | None = None
        self.cycle_id: np.memmap | None = None
        self.predraft_stats: np.memmap | None = None
        self.predraft_token_ids: np.memmap | None = None
        self.predraft_token_mask: np.memmap | None = None
        self.predraft_verifier_entropy: np.memmap | None = None
        self.predraft_verifier_entropy_mask: np.memmap | None = None
        self.postdraft_confidence: np.memmap | None = None
        self.postdraft_hidden: np.memmap | None = None
        self.postdraft_token_ids: np.memmap | None = None
        self.meta_file = None
        self.shards: list[dict[str, Any]] = []

    def _open_next_shard(self) -> None:
        self.close_current()
        self.shard_idx += 1
        self.row_idx = 0
        shard_dir = self.output_dir / f"shard_{self.shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        self.features = np.lib.format.open_memmap(
            shard_dir / "features.npy",
            mode="w+",
            dtype=np.float16,
            shape=(self.rows_per_shard, self.window, self.hidden_size),
        )
        self.mask = np.lib.format.open_memmap(
            shard_dir / "mask.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(self.rows_per_shard, self.window),
        )
        self.survival = np.lib.format.open_memmap(
            shard_dir / "survival.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(self.rows_per_shard, self.num_slots),
        )
        self.accepted_len = np.lib.format.open_memmap(
            shard_dir / "accepted_len.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(self.rows_per_shard,),
        )
        self.prompt_index = np.lib.format.open_memmap(
            shard_dir / "prompt_index.npy",
            mode="w+",
            dtype=np.int64,
            shape=(self.rows_per_shard,),
        )
        self.cycle_id = np.lib.format.open_memmap(
            shard_dir / "cycle_id.npy",
            mode="w+",
            dtype=np.int32,
            shape=(self.rows_per_shard,),
        )
        self.predraft_stats = np.lib.format.open_memmap(
            shard_dir / "predraft_stats.npy",
            mode="w+",
            dtype=np.float16,
            shape=(self.rows_per_shard, 8),
        )
        if self.log_predraft_token_ids:
            self.predraft_token_ids = np.lib.format.open_memmap(
                shard_dir / "predraft_token_ids.npy",
                mode="w+",
                dtype=np.int64,
                shape=(self.rows_per_shard, self.token_window),
            )
            self.predraft_token_mask = np.lib.format.open_memmap(
                shard_dir / "predraft_token_mask.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(self.rows_per_shard, self.token_window),
            )
        if self.log_verifier_entropy_window:
            self.predraft_verifier_entropy = np.lib.format.open_memmap(
                shard_dir / "predraft_verifier_entropy.npy",
                mode="w+",
                dtype=np.float16,
                shape=(self.rows_per_shard, self.verifier_entropy_window),
            )
            self.predraft_verifier_entropy_mask = np.lib.format.open_memmap(
                shard_dir / "predraft_verifier_entropy_mask.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(self.rows_per_shard, self.verifier_entropy_window),
            )
        if self.log_postdraft_confidence:
            self.postdraft_confidence = np.lib.format.open_memmap(
                shard_dir / "postdraft_confidence.npy",
                mode="w+",
                dtype=np.float16,
                shape=(self.rows_per_shard, self.num_slots, 4),
            )
        if self.log_postdraft_hidden:
            self.postdraft_hidden = np.lib.format.open_memmap(
                shard_dir / "postdraft_hidden.npy",
                mode="w+",
                dtype=np.float16,
                shape=(self.rows_per_shard, self.num_slots, self.hidden_size),
            )
            self.postdraft_token_ids = np.lib.format.open_memmap(
                shard_dir / "postdraft_token_ids.npy",
                mode="w+",
                dtype=np.int64,
                shape=(self.rows_per_shard, self.num_slots + 1),
            )
        self.meta_file = (shard_dir / "metadata.jsonl").open("w")
        self.shards.append({"path": shard_dir.name, "rows": 0})

    def write(
        self,
        *,
        features: np.ndarray,
        mask: np.ndarray,
        accepted_len: int,
        prompt_index: int,
        cycle_id: int,
        predraft_stats: np.ndarray,
        metadata: dict[str, Any],
        predraft_token_ids: np.ndarray | None = None,
        predraft_token_mask: np.ndarray | None = None,
        predraft_verifier_entropy: np.ndarray | None = None,
        predraft_verifier_entropy_mask: np.ndarray | None = None,
        postdraft_confidence: np.ndarray | None = None,
        postdraft_hidden: np.ndarray | None = None,
        postdraft_token_ids: np.ndarray | None = None,
    ) -> None:
        if self.features is None or self.row_idx >= self.rows_per_shard:
            self._open_next_shard()
        assert self.features is not None
        assert self.mask is not None
        assert self.survival is not None
        assert self.accepted_len is not None
        assert self.prompt_index is not None
        assert self.cycle_id is not None
        assert self.predraft_stats is not None
        assert self.meta_file is not None
        idx = self.row_idx
        self.features[idx] = features
        self.mask[idx] = mask
        self.survival[idx] = np.asarray(
            [1 if accepted_len >= k else 0 for k in range(1, self.num_slots + 1)],
            dtype=np.uint8,
        )
        self.accepted_len[idx] = np.uint8(accepted_len)
        self.prompt_index[idx] = int(prompt_index)
        self.cycle_id[idx] = int(cycle_id)
        self.predraft_stats[idx] = predraft_stats
        if self.log_predraft_token_ids:
            assert self.predraft_token_ids is not None
            assert self.predraft_token_mask is not None
            if predraft_token_ids is None or predraft_token_mask is None:
                raise ValueError("predraft_token_ids/mask are required when log_predraft_token_ids=True")
            self.predraft_token_ids[idx] = predraft_token_ids
            self.predraft_token_mask[idx] = predraft_token_mask
        if self.log_verifier_entropy_window:
            assert self.predraft_verifier_entropy is not None
            assert self.predraft_verifier_entropy_mask is not None
            if predraft_verifier_entropy is None or predraft_verifier_entropy_mask is None:
                raise ValueError(
                    "predraft_verifier_entropy/mask are required when "
                    "log_verifier_entropy_window=True"
                )
            self.predraft_verifier_entropy[idx] = predraft_verifier_entropy
            self.predraft_verifier_entropy_mask[idx] = predraft_verifier_entropy_mask
        if self.log_postdraft_confidence:
            assert self.postdraft_confidence is not None
            if postdraft_confidence is None:
                raise ValueError("postdraft_confidence is required when log_postdraft_confidence=True")
            self.postdraft_confidence[idx] = postdraft_confidence
        if self.log_postdraft_hidden:
            assert self.postdraft_hidden is not None
            assert self.postdraft_token_ids is not None
            if postdraft_hidden is None:
                raise ValueError("postdraft_hidden is required when log_postdraft_hidden=True")
            if postdraft_token_ids is None:
                raise ValueError("postdraft_token_ids is required when log_postdraft_hidden=True")
            self.postdraft_hidden[idx] = postdraft_hidden
            self.postdraft_token_ids[idx] = postdraft_token_ids
        self.meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        self.row_idx += 1
        self.total_rows += 1
        self.shards[-1]["rows"] = self.row_idx

    def close_current(self) -> None:
        if self.meta_file is not None:
            self.meta_file.flush()
            self.meta_file.close()
            self.meta_file = None
        for arr in (
            self.features,
            self.mask,
            self.survival,
            self.accepted_len,
            self.prompt_index,
            self.cycle_id,
            self.predraft_stats,
            self.predraft_token_ids,
            self.predraft_token_mask,
            self.predraft_verifier_entropy,
            self.predraft_verifier_entropy_mask,
            self.postdraft_confidence,
            self.postdraft_hidden,
            self.postdraft_token_ids,
        ):
            if arr is not None:
                arr.flush()
        self.features = None
        self.mask = None
        self.survival = None
        self.accepted_len = None
        self.prompt_index = None
        self.cycle_id = None
        self.predraft_stats = None
        self.predraft_token_ids = None
        self.predraft_token_mask = None
        self.predraft_verifier_entropy = None
        self.predraft_verifier_entropy_mask = None
        self.postdraft_confidence = None
        self.postdraft_hidden = None
        self.postdraft_token_ids = None

    def close(self) -> None:
        self.close_current()


@torch.inference_mode()
def collect_one_prompt(
    *,
    row: dict[str, Any],
    prompt_index: int,
    draft_model: DFlashDraftModel,
    target: torch.nn.Module,
    tokenizer: Any,
    writer: HorizonShardWriter,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    context_window: int,
    enable_thinking: bool,
    max_cycles_total: int | None,
) -> bool:
    input_ids = _input_ids_from_messages(
        tokenizer,
        row,
        enable_thinking=enable_thinking,
    ).to(target.device)
    if input_ids.shape[1] == 0:
        return False

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    mask_token_id = draft_model.mask_token_id
    num_draft_slots = block_size - 1
    hidden_size = int(draft_model.config.hidden_size)

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
    latest_verifier_confidence = _token_confidence_features(
        output.logits[0, -1:], first_token[0].reshape(1)
    )[0]
    verifier_entropy_history: deque[float] = deque(maxlen=writer.verifier_entropy_window)
    verifier_entropy_history.append(float(latest_verifier_confidence[0]))
    prev_cycle_summary = np.zeros((4,), dtype=np.float16)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token
    target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids)

    context_history: deque[torch.Tensor] = deque(maxlen=context_window)
    for item in _fused_context(draft_model, target_hidden):
        context_history.append(item.to("cpu", dtype=torch.float16))

    start = num_input_tokens
    cycle_id = 0
    eos_id = tokenizer.eos_token_id
    while start < max_length:
        if max_cycles_total is not None and writer.total_rows >= max_cycles_total:
            return True

        context_features, context_mask = _padded_context_window(
            context_history,
            window=context_window,
            hidden_size=hidden_size,
        )
        predraft_token_ids, predraft_token_mask = _padded_token_window(
            output_ids,
            end_exclusive=start + 1,
            window=writer.token_window,
        )
        predraft_verifier_entropy, predraft_verifier_entropy_mask = _padded_float_window(
            verifier_entropy_history,
            window=writer.verifier_entropy_window,
        )

        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_hidden = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size :, :]
        draft_logits = target.lm_head(draft_hidden)
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits, temperature)
        postdraft_confidence = (
            _draft_confidence_features(draft_logits[0], block_output_ids[0, 1:])
            if writer.log_postdraft_confidence
            else None
        )
        postdraft_hidden = (
            draft_hidden[0].detach().float().cpu().numpy().astype(np.float16)
            if writer.log_postdraft_hidden
            else None
        )
        postdraft_token_ids = (
            block_output_ids[0, :block_size].detach().cpu().numpy().astype(np.int64)
            if writer.log_postdraft_hidden
            else None
        )

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(output.logits, temperature)
        verifier_confidence_for_block = _token_confidence_features(
            output.logits[0, :block_size],
            posterior[0, :block_size],
        )
        accepted_draft_len = int(
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[
                0
            ].item()
        )
        committed_len = accepted_draft_len + 1
        correction_token = posterior[:, accepted_draft_len]

        writer.write(
            features=context_features,
            mask=context_mask,
            accepted_len=accepted_draft_len,
            prompt_index=prompt_index,
            cycle_id=cycle_id,
            predraft_stats=np.concatenate(
                [latest_verifier_confidence, prev_cycle_summary], axis=0
            ).astype(np.float16),
            metadata={
                "prompt_index": prompt_index,
                "manifest_index": row.get("manifest_index"),
                "source": row.get("source"),
                "source_id": row.get("source_id"),
                "category": row.get("category"),
                "reasoning": row.get("reasoning"),
                "cycle_id": cycle_id,
                "prompt_tokens": int(num_input_tokens),
                "prefix_len_before_cycle": int(start),
                "generated_tokens_before_cycle": int(start - num_input_tokens),
                "accepted_draft_len": accepted_draft_len,
                "committed_len": committed_len,
                "predraft_stats_columns": [
                    "verifier_entropy_latest",
                    "verifier_token_prob_latest",
                    "verifier_top1_top2_margin_latest",
                    "verifier_token_logprob_latest",
                    "prev_accepted_len_norm",
                    "prev_accept_ratio",
                    "prev_budget_norm",
                    "has_prev_cycle",
                ],
                "postdraft_confidence_columns": [
                    "draft_entropy",
                    "draft_token_prob",
                    "draft_top1_top2_margin",
                    "draft_token_logprob",
                ]
                if writer.log_postdraft_confidence
                else None,
                "postdraft_hidden": writer.log_postdraft_hidden,
                "predraft_token_ids": writer.log_predraft_token_ids,
                "predraft_token_window": writer.token_window if writer.log_predraft_token_ids else None,
                "predraft_verifier_entropy_window": writer.log_verifier_entropy_window,
                "predraft_verifier_entropy_window_size": (
                    writer.verifier_entropy_window if writer.log_verifier_entropy_window else None
                ),
            },
            predraft_token_ids=predraft_token_ids if writer.log_predraft_token_ids else None,
            predraft_token_mask=predraft_token_mask if writer.log_predraft_token_ids else None,
            predraft_verifier_entropy=(
                predraft_verifier_entropy if writer.log_verifier_entropy_window else None
            ),
            predraft_verifier_entropy_mask=(
                predraft_verifier_entropy_mask if writer.log_verifier_entropy_window else None
            ),
            postdraft_confidence=postdraft_confidence,
            postdraft_hidden=postdraft_hidden,
            postdraft_token_ids=postdraft_token_ids,
        )

        output_ids[:, start : start + accepted_draft_len + 1] = block_output_ids[
            :, : accepted_draft_len + 1
        ]
        output_ids[:, start + accepted_draft_len + 1] = correction_token
        start += committed_len
        past_key_values_target.crop(start)

        for entropy in verifier_confidence_for_block[:committed_len, 0]:
            verifier_entropy_history.append(float(entropy))
        latest_verifier_confidence = verifier_confidence_for_block[accepted_draft_len]
        prev_cycle_summary = np.asarray(
            [
                accepted_draft_len / max(num_draft_slots, 1),
                accepted_draft_len / max(num_draft_slots, 1),
                num_draft_slots / max(num_draft_slots, 1),
                1.0,
            ],
            dtype=np.float16,
        )

        target_hidden = extract_context_feature(output.hidden_states, draft_model.target_layer_ids)[
            :, :committed_len, :
        ]
        for item in _fused_context(draft_model, target_hidden):
            context_history.append(item.to("cpu", dtype=torch.float16))
        cycle_id += 1

        if eos_id is not None and eos_id in output_ids[:, num_input_tokens : start + 1]:
            break
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect full-fused-context DFlashv2 horizon traces as binary shards."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--context-window", type=int, default=16)
    parser.add_argument("--rows-per-shard", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--log-postdraft-confidence",
        action="store_true",
        help="Store per-drafted-position drafter confidence stats for post-draft verification-length policies.",
    )
    parser.add_argument(
        "--log-postdraft-hidden",
        action="store_true",
        help=(
            "Store per-drafted-position DFlash hidden states and token ids for a "
            "DSPARK-style confidence head. This is storage-heavy."
        ),
    )
    parser.add_argument(
        "--log-predraft-token-ids",
        action="store_true",
        help=(
            "Store the last prefix token ids before each draft cycle. This supports "
            "token-prefix horizon heads that use actual token identity, not entropy history."
        ),
    )
    parser.add_argument(
        "--token-window",
        type=int,
        default=32,
        help="Number of latest prefix token ids to store when --log-predraft-token-ids is enabled.",
    )
    parser.add_argument(
        "--log-verifier-entropy-window",
        action="store_true",
        help=(
            "Store a causal window of verifier entropies for the last committed prefix tokens "
            "before each draft cycle."
        ),
    )
    parser.add_argument(
        "--verifier-entropy-window",
        type=int,
        default=64,
        help="Number of latest committed-token verifier entropies to store.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    prompts = _load_manifest(
        args.manifest,
        max_prompts=args.max_prompts,
        seed=args.seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    hidden_size = int(draft_model.config.hidden_size)
    writer = HorizonShardWriter(
        args.output_dir,
        rows_per_shard=args.rows_per_shard,
        window=args.context_window,
        hidden_size=hidden_size,
        num_slots=args.block_size - 1,
        log_postdraft_confidence=args.log_postdraft_confidence,
        log_postdraft_hidden=args.log_postdraft_hidden,
        log_predraft_token_ids=args.log_predraft_token_ids,
        token_window=args.token_window,
        log_verifier_entropy_window=args.log_verifier_entropy_window,
        verifier_entropy_window=args.verifier_entropy_window,
    )
    stopped_by_cycle_budget = False
    try:
        for prompt_index, row in enumerate(tqdm(prompts, desc="collect")):
            stopped_by_cycle_budget = collect_one_prompt(
                row=row,
                prompt_index=prompt_index,
                draft_model=draft_model,
                target=target,
                tokenizer=tokenizer,
                writer=writer,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                temperature=args.temperature,
                context_window=args.context_window,
                enable_thinking=args.enable_thinking,
                max_cycles_total=args.max_cycles,
            )
            if stopped_by_cycle_budget:
                break
    finally:
        writer.close()

    manifest = {
        "format": "dflashv2_horizon_shards_v1",
        "source_manifest": str(args.manifest),
        "model": args.model,
        "draft_model": args.draft_model,
        "num_prompts_loaded": len(prompts),
        "num_rows": writer.total_rows,
        "hidden_size": hidden_size,
        "context_window": args.context_window,
        "block_size": args.block_size,
        "num_slots": args.block_size - 1,
        "postdraft_confidence": args.log_postdraft_confidence,
        "predraft_stats": True,
        "predraft_stats_columns": [
            "verifier_entropy_latest",
            "verifier_token_prob_latest",
            "verifier_top1_top2_margin_latest",
            "verifier_token_logprob_latest",
            "prev_accepted_len_norm",
            "prev_accept_ratio",
            "prev_budget_norm",
            "has_prev_cycle",
        ],
        "postdraft_confidence_columns": [
            "draft_entropy",
            "draft_token_prob",
            "draft_top1_top2_margin",
            "draft_token_logprob",
        ]
        if args.log_postdraft_confidence
        else None,
        "postdraft_hidden": args.log_postdraft_hidden,
        "postdraft_hidden_shape": [args.block_size - 1, hidden_size]
        if args.log_postdraft_hidden
        else None,
        "postdraft_token_ids_shape": [args.block_size] if args.log_postdraft_hidden else None,
        "predraft_token_ids": args.log_predraft_token_ids,
        "predraft_token_ids_shape": [args.token_window] if args.log_predraft_token_ids else None,
        "predraft_verifier_entropy_window": args.log_verifier_entropy_window,
        "predraft_verifier_entropy_window_shape": [args.verifier_entropy_window]
        if args.log_verifier_entropy_window
        else None,
        "rows_per_shard": args.rows_per_shard,
        "shards": writer.shards,
        "stopped_by_cycle_budget": stopped_by_cycle_budget,
        "args": vars(args),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
