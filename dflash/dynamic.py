from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache

from .model import DFlashDraftModel, _cuda_time, extract_context_feature, sample
from .policy import DFlashSurvivalBlockPolicy


@torch.inference_mode()
def dflash_generate_dynamic(
    model: DFlashDraftModel,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    policy: DFlashSurvivalBlockPolicy,
    mask_token_id: Optional[int] = None,
    return_stats: bool = False,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    max_block_size = max(policy.arms)
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + max_block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    policy.reset(model, target.device)

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    policy.observe_context(model, target_hidden)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths: list[int] = []
    accepted_draft_lengths: list[int] = []
    block_sizes: list[int] = []
    expected_draft_lengths: list[list[float]] = []
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        block_size = policy.select_block_size()
        block_sizes.append(block_size)
        if policy.last_expected_by_arm is not None:
            expected_draft_lengths.append([float(x) for x in policy.last_expected_by_arm.tolist()])

        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - block_size :, :]
        )
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits, temperature)
        if draft_prefill and return_stats:
            draft_prefill = False
            decode_start = _cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample(output.logits, temperature)
        accepted_draft_len = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        committed_len = accepted_draft_len + 1
        output_ids[:, start : start + committed_len] = block_output_ids[:, :committed_len]
        output_ids[:, start + committed_len] = posterior[:, accepted_draft_len]
        start += committed_len
        past_key_values_target.crop(start)
        acceptance_lengths.append(committed_len)
        accepted_draft_lengths.append(accepted_draft_len)

        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :committed_len, :]
        policy.observe_context(model, target_hidden)

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens : start + 1] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    draft_budgets = [block_size - 1 for block_size in block_sizes]
    acceptance_ratios = [
        accepted / max(budget, 1)
        for accepted, budget in zip(accepted_draft_lengths, draft_budgets)
    ]
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        accepted_draft_lengths=accepted_draft_lengths,
        block_sizes=block_sizes,
        draft_budgets=draft_budgets,
        acceptance_ratios=acceptance_ratios,
        expected_draft_lengths=expected_draft_lengths,
    )
