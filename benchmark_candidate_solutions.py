import argparse
import atexit
import builtins
import json
import os
import random
import time
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import distributed as dist
from model import DFlashDraftModel, extract_context_feature, load_and_process_dataset, sample


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def summarize_mode(samples: list[SimpleNamespace]) -> dict[str, float]:
    total_wall_s = float(np.sum([getattr(s, "wall_time_s") for s in samples]))
    total_tokens = int(np.sum([s.num_output_tokens for s in samples]))
    avg_ttft_s = float(np.mean([s.time_to_first_token for s in samples]))
    avg_tpot_s = float(np.mean([s.time_per_output_token for s in samples]))
    avg_wall_s = float(np.mean([getattr(s, "wall_time_s") for s in samples]))
    tokens_per_sec = float(total_tokens / max(total_wall_s, 1e-8))
    return {
        "total_wall_s": total_wall_s,
        "avg_wall_s": avg_wall_s,
        "avg_ttft_s": avg_ttft_s,
        "avg_tpot_s": avg_tpot_s,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": float(total_tokens),
    }


def summarize_profile(samples: list[SimpleNamespace]) -> dict[str, float] | None:
    profiles = [getattr(s, "profile_summary", None) for s in samples]
    profiles = [p for p in profiles if p is not None]
    if not profiles:
        return None

    total_target_prefill_s = float(np.sum([p["target_prefill_s"] for p in profiles]))
    total_target_decode_s = float(np.sum([p["target_decode_s"] for p in profiles]))
    total_draft_decode_s = float(np.sum([p["draft_decode_s"] for p in profiles]))
    total_cycle_decode_s = float(np.sum([p["cycle_decode_s_sum"] for p in profiles]))
    total_decode_wall_s = float(np.sum([p["decode_wall_s"] for p in profiles]))
    total_cycles = int(np.sum([p["profiled_cycles"] for p in profiles]))
    denom = max(1e-12, total_draft_decode_s + total_target_decode_s)

    return {
        "total_target_prefill_s": total_target_prefill_s,
        "total_target_decode_s": total_target_decode_s,
        "total_draft_decode_s": total_draft_decode_s,
        "total_cycle_decode_s": total_cycle_decode_s,
        "total_decode_wall_s": total_decode_wall_s,
        "total_profiled_cycles": float(total_cycles),
        "draft_share_decode": float(total_draft_decode_s / denom),
        "target_share_decode": float(total_target_decode_s / denom),
        "avg_target_prefill_s": float(total_target_prefill_s / len(profiles)),
        "avg_target_decode_s": float(total_target_decode_s / len(profiles)),
        "avg_draft_decode_s": float(total_draft_decode_s / len(profiles)),
        "avg_decode_wall_s": float(total_decode_wall_s / len(profiles)),
    }


def clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    # Clone tensors to avoid mutating the live prefix cache when expanding batch
    # for candidate verification.
    legacy = cache.to_legacy_cache()
    cloned_legacy = tuple((k.clone(), v.clone()) for k, v in legacy)
    return DynamicCache.from_legacy_cache(cloned_legacy)


def select_branch_positions(
    draft_logits: torch.Tensor,
    effective_block_size: int,
    branch_depth: int,
    margin_threshold: float,
) -> list[int]:
    max_pos = min(effective_block_size - 1, branch_depth)
    if max_pos <= 0:
        return []

    positions = list(range(1, max_pos + 1))
    if margin_threshold < 0:
        return positions

    probs = torch.softmax(draft_logits.float(), dim=-1)
    selected = []
    for pos in positions:
        top2 = torch.topk(probs[0, pos - 1], k=2).values
        margin = float(top2[0].item() - top2[1].item())
        if margin <= margin_threshold:
            selected.append(pos)
    return selected


def build_candidate_blocks(
    base_block_output_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    selected_positions: list[int],
    branch_top_k: int,
    max_candidates: int,
) -> tuple[list[torch.Tensor], list[dict]]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    if not selected_positions:
        return [base_block_output_ids.clone()], [{"candidate_idx": 0, "draft_score": 0.0, "replaced_positions": []}]

    log_probs = torch.log_softmax(draft_logits.float(), dim=-1)

    options_per_pos = []
    for pos in selected_positions:
        vocab_k = min(branch_top_k, int(log_probs.shape[-1]))
        values, indices = torch.topk(log_probs[0, pos - 1], k=vocab_k, dim=-1)
        options_per_pos.append((pos, indices.tolist(), values.tolist()))

    base_score = 0.0
    for pos in selected_positions:
        token_id = int(base_block_output_ids[0, pos].item())
        base_score += float(log_probs[0, pos - 1, token_id].item())

    beams: list[tuple[dict[int, int], float]] = [({}, 0.0)]
    for pos, token_ids, token_scores in options_per_pos:
        expanded: list[tuple[dict[int, int], float]] = []
        for assignment, score in beams:
            for token_id, token_score in zip(token_ids, token_scores):
                updated = dict(assignment)
                updated[pos] = int(token_id)
                expanded.append((updated, score + float(token_score)))
        expanded.sort(key=lambda x: x[1], reverse=True)
        beams = expanded[:max_candidates]

    candidates: list[torch.Tensor] = []
    metadata: list[dict] = []
    seen = set()

    # Always include base candidate.
    base_key = tuple(int(x) for x in base_block_output_ids[0, 1:].tolist())
    seen.add(base_key)
    candidates.append(base_block_output_ids.clone())
    metadata.append({"candidate_idx": 0, "draft_score": base_score, "replaced_positions": []})

    for assignment, score in beams:
        candidate = base_block_output_ids.clone()
        replaced_positions = []
        for pos, token_id in assignment.items():
            old_id = int(candidate[0, pos].item())
            if old_id != token_id:
                replaced_positions.append(int(pos))
            candidate[0, pos] = token_id
        key = tuple(int(x) for x in candidate[0, 1:].tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        metadata.append(
            {
                "candidate_idx": len(candidates) - 1,
                "draft_score": float(score),
                "replaced_positions": replaced_positions,
            }
        )
        if len(candidates) >= max_candidates:
            break

    return candidates, metadata


def build_fixed_prefix_rank_candidates(
    base_block_output_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    fixed_prefix_len: int,
    rank_top_k: int,
    max_candidates: int,
) -> tuple[list[torch.Tensor], list[dict], list[int]]:
    """Build candidates with shared prefix and global rank-suffix variants.

    Candidate 0: greedy base.
    Candidate r>0: keep prefix fixed, set every suffix position to rank-(r+1)
    token under draft logits at that position.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if rank_top_k < 1:
        raise ValueError("rank_top_k must be >= 1")

    effective_block_size = int(base_block_output_ids.shape[1])
    # block position 0 has no draft logits (context token for this cycle).
    suffix_start = max(1, min(fixed_prefix_len, effective_block_size))
    suffix_positions = list(range(suffix_start, effective_block_size))

    if not suffix_positions:
        return (
            [base_block_output_ids.clone()],
            [{"candidate_idx": 0, "draft_score": 0.0, "replaced_positions": [], "rank_variant": 1}],
            [],
        )

    # Number of total candidates to emit (including greedy base).
    # Example: top-4 => greedy + (rank-2, rank-3, rank-4) = 4 total.
    candidate_total = min(max_candidates, rank_top_k)
    if candidate_total <= 1:
        return (
            [base_block_output_ids.clone()],
            [{"candidate_idx": 0, "draft_score": 0.0, "replaced_positions": [], "rank_variant": 1}],
            suffix_positions,
        )

    log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    rank_table: dict[int, tuple[list[int], list[float]]] = {}
    for pos in suffix_positions:
        values, indices = torch.topk(log_probs[0, pos - 1], k=candidate_total, dim=-1)
        rank_table[pos] = (
            [int(x) for x in indices.tolist()],
            [float(x) for x in values.tolist()],
        )

    candidates: list[torch.Tensor] = []
    metadata: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    # Always include greedy base candidate first.
    base_score = 0.0
    for pos in suffix_positions:
        token_id = int(base_block_output_ids[0, pos].item())
        base_score += float(log_probs[0, pos - 1, token_id].item())
    base_key = tuple(int(x) for x in base_block_output_ids[0, 1:].tolist())
    seen.add(base_key)
    candidates.append(base_block_output_ids.clone())
    metadata.append(
        {
            "candidate_idx": 0,
            "draft_score": float(base_score),
            "replaced_positions": [],
            "rank_variant": 1,
        }
    )

    # Add rank-2, rank-3, ... variants on the suffix.
    for rank_idx in range(1, candidate_total):
        candidate = base_block_output_ids.clone()
        replaced_positions: list[int] = []
        score = 0.0
        for pos in suffix_positions:
            token_ids, token_scores = rank_table[pos]
            token_id = int(token_ids[rank_idx])
            old_id = int(candidate[0, pos].item())
            if old_id != token_id:
                replaced_positions.append(pos)
            candidate[0, pos] = token_id
            score += float(token_scores[rank_idx])

        key = tuple(int(x) for x in candidate[0, 1:].tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        metadata.append(
            {
                "candidate_idx": len(candidates) - 1,
                "draft_score": float(score),
                "replaced_positions": replaced_positions,
                "rank_variant": int(rank_idx + 1),
            }
        )
        if len(candidates) >= max_candidates:
            break

    return candidates, metadata, suffix_positions


@torch.inference_mode()
def dflash_generate_candidate_solutions(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    branch_depth: int,
    branch_top_k: int,
    max_candidates: int,
    margin_threshold: float,
    candidate_mode: str,
    fixed_prefix_len: int,
    temperature: float = 0.0,
    collect_profile: bool = False,
) -> SimpleNamespace:
    if temperature >= 1e-5:
        raise ValueError("benchmark_candidate_solutions.py currently supports only temperature=0.0")

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    else:
        target_hidden = None
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths: list[int] = []
    cycle_trace: list[dict] = []

    candidate_count_sum = 0
    candidate_verify_calls = 0
    first_prompt_cycle_done = False

    while start < max_length:
        cycle_start = None
        cycle_end = None
        draft_events = None
        target_events = None
        if collect_profile:
            cycle_start = torch.cuda.Event(enable_timing=True)
            cycle_end = torch.cuda.Event(enable_timing=True)
            cycle_start.record()

        remaining = max_length - start
        effective_block_size = min(block_size, remaining)
        block_output_ids = output_ids[:, start : start + effective_block_size].clone()
        block_position_ids = position_ids[:, start : start + effective_block_size]

        selected_positions = []
        candidate_blocks = [block_output_ids.clone()]
        candidate_meta = [{"candidate_idx": 0, "draft_score": 0.0, "replaced_positions": []}]

        if effective_block_size > 1:
            if collect_profile:
                draft_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                draft_events[0].record()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + effective_block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -effective_block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits, temperature)
            if candidate_mode == "fixed_prefix_rank":
                candidate_blocks, candidate_meta, selected_positions = build_fixed_prefix_rank_candidates(
                    base_block_output_ids=block_output_ids,
                    draft_logits=draft_logits,
                    fixed_prefix_len=fixed_prefix_len,
                    rank_top_k=branch_top_k,
                    max_candidates=max_candidates,
                )
            else:
                selected_positions = select_branch_positions(
                    draft_logits=draft_logits,
                    effective_block_size=effective_block_size,
                    branch_depth=branch_depth,
                    margin_threshold=margin_threshold,
                )
                candidate_blocks, candidate_meta = build_candidate_blocks(
                    base_block_output_ids=block_output_ids,
                    draft_logits=draft_logits,
                    selected_positions=selected_positions,
                    branch_top_k=branch_top_k,
                    max_candidates=max_candidates,
                )
            if collect_profile:
                draft_events[1].record()

        candidate_count_sum += len(candidate_blocks)

        # Verify all candidates in one target call by expanding batch.
        num_candidates = len(candidate_blocks)
        stacked_candidates = torch.cat(candidate_blocks, dim=0)
        stacked_position_ids = block_position_ids.repeat(num_candidates, 1)
        verify_cache = clone_dynamic_cache(past_key_values_target)
        if num_candidates > 1:
            verify_cache.batch_repeat_interleave(num_candidates)

        if collect_profile:
            target_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            target_events[0].record()
        verify_output = target(
            stacked_candidates,
            position_ids=stacked_position_ids,
            past_key_values=verify_cache,
            use_cache=True,
            output_hidden_states=True if effective_block_size > 1 else False,
        )
        if collect_profile:
            target_events[1].record()
        candidate_verify_calls += 1

        posterior_all = sample(verify_output.logits, temperature)
        acceptance_lengths_all = (
            (stacked_candidates[:, 1:] == posterior_all[:, :-1]).cumprod(dim=1).sum(dim=1)
        )

        candidate_evals = []
        for cand_idx, meta in enumerate(candidate_meta):
            acceptance_length = int(acceptance_lengths_all[cand_idx].item())
            tau = acceptance_length + 1
            candidate_evals.append(
                {
                    "candidate_idx": int(meta["candidate_idx"]),
                    "tau": int(tau),
                    "draft_score": float(meta["draft_score"]),
                    "acceptance_length": int(acceptance_length),
                }
            )

        best = max(candidate_evals, key=lambda x: (x["tau"], x["draft_score"], -x["candidate_idx"]))
        chosen_candidate_idx = int(best["candidate_idx"])
        chosen_candidate_block = candidate_blocks[chosen_candidate_idx]
        posterior = posterior_all[chosen_candidate_idx : chosen_candidate_idx + 1]
        acceptance_length = int(acceptance_lengths_all[chosen_candidate_idx].item())
        tau = acceptance_length + 1

        # Keep only chosen branch cache for next cycle.
        if num_candidates > 1:
            verify_cache.batch_select_indices(
                torch.tensor([chosen_candidate_idx], dtype=torch.long, device=stacked_candidates.device)
            )
        past_key_values_target = verify_cache

        output_ids[:, start : start + tau] = chosen_candidate_block[:, :tau]
        output_ids[:, start + tau] = posterior[:, acceptance_length]
        acceptance_lengths.append(tau)

        cycle_row = {
            "cycle_idx": int(len(acceptance_lengths) - 1),
            "generated_tokens_before": int(start - num_input_tokens),
            "effective_block_size": int(effective_block_size),
            "tau": int(tau),
            "acceptance_ratio": float(tau / max(1, effective_block_size)),
            "num_candidates": int(len(candidate_blocks)),
            "selected_positions": [int(x) for x in selected_positions],
            "chosen_candidate_idx": int(chosen_candidate_idx),
            "candidate_taus": [int(x["tau"]) for x in candidate_evals],
            "candidate_draft_scores": [float(x["draft_score"]) for x in candidate_evals],
            "candidate_rank_variants": [int(x.get("rank_variant", 1)) for x in candidate_meta],
        }
        if collect_profile:
            cycle_end.record()
            cycle_row["_events"] = {
                "draft": draft_events,
                "target": target_events,
                "cycle": (cycle_start, cycle_end),
            }
        cycle_trace.append(cycle_row)

        start += tau
        past_key_values_target.crop(start)
        if effective_block_size > 1:
            chosen_hidden_states = [h[chosen_candidate_idx : chosen_candidate_idx + 1] for h in verify_output.hidden_states]
            target_hidden = extract_context_feature(chosen_hidden_states, model.target_layer_ids)[:, :tau, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

        if not first_prompt_cycle_done:
            decode_start = cuda_time()
            first_prompt_cycle_done = True

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(1, num_output_tokens)
    avg_candidates_per_cycle = float(candidate_count_sum / max(1, len(acceptance_lengths)))

    candidate_summary = {
        "candidate_mode": str(candidate_mode),
        "fixed_prefix_len": int(fixed_prefix_len),
        "avg_candidates_per_cycle": float(avg_candidates_per_cycle),
        "candidate_verify_calls": int(candidate_verify_calls),
        "candidate_count_sum": int(candidate_count_sum),
    }

    profile_summary = None
    if collect_profile:
        torch.cuda.synchronize()
        target_prefill_s = float(time_to_first_token)
        total_target_decode_s = 0.0
        total_draft_decode_s = 0.0
        total_cycle_s = 0.0
        for row in cycle_trace:
            draft_pair = row["_events"]["draft"]
            target_pair = row["_events"]["target"]
            cycle_pair = row["_events"]["cycle"]
            draft_s = 0.0
            if draft_pair is not None:
                draft_s = draft_pair[0].elapsed_time(draft_pair[1]) / 1000.0
            target_s = 0.0
            if target_pair is not None:
                target_s = target_pair[0].elapsed_time(target_pair[1]) / 1000.0
            cycle_s = cycle_pair[0].elapsed_time(cycle_pair[1]) / 1000.0
            row["draft_s"] = float(draft_s)
            row["target_s"] = float(target_s)
            row["cycle_s"] = float(cycle_s)
            row.pop("_events", None)
            total_draft_decode_s += draft_s
            total_target_decode_s += target_s
            total_cycle_s += cycle_s
        profile_summary = {
            "target_prefill_s": float(target_prefill_s),
            "target_decode_s": float(total_target_decode_s),
            "draft_decode_s": float(total_draft_decode_s),
            "cycle_decode_s_sum": float(total_cycle_s),
            "decode_wall_s": float(total_decode_time),
            "profiled_cycles": int(len(cycle_trace)),
            "draft_share_decode": float(
                total_draft_decode_s / max(1e-12, total_draft_decode_s + total_target_decode_s)
            ),
            "target_share_decode": float(
                total_target_decode_s / max(1e-12, total_draft_decode_s + total_target_decode_s)
            ),
        }

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        cycle_trace=cycle_trace,
        candidate_summary=candidate_summary,
        profile_summary=profile_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash benchmark with candidate-solution generation per cycle.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--branch-depth", type=int, default=6, help="Maximum early positions to branch on.")
    parser.add_argument("--branch-top-k", type=int, default=2, help="Top-k tokens per selected position.")
    parser.add_argument("--max-candidates", type=int, default=4, help="Maximum candidate blocks per cycle.")
    parser.add_argument(
        "--candidate-mode",
        type=str,
        choices=["branch_beam", "fixed_prefix_rank"],
        default="branch_beam",
        help="Candidate generation mode. fixed_prefix_rank = greedy + rank-2/3/4 suffix variants.",
    )
    parser.add_argument(
        "--fixed-prefix-len",
        type=int,
        default=5,
        help="For fixed_prefix_rank mode: keep first N block positions fixed, vary suffix by rank.",
    )
    parser.add_argument(
        "--branch-margin-threshold",
        type=float,
        default=-1.0,
        help="If >=0, only branch where (p1 - p2) <= threshold in selected early positions.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-all-ranks", action="store_true")
    parser.add_argument("--save-outputs-path", type=str, default=None)
    parser.add_argument("--save-cycle-trace-path", type=str, default=None)
    parser.add_argument(
        "--collect-profile",
        action="store_true",
        help="Collect per-cycle profiling stats (target/draft/cycle timings).",
    )
    args = parser.parse_args()

    if args.branch_depth < 0:
        raise ValueError("--branch-depth must be >= 0")
    if args.branch_top_k < 1:
        raise ValueError("--branch-top-k must be >= 1")
    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be >= 1")
    if args.fixed_prefix_len < 0:
        raise ValueError("--fixed-prefix-len must be >= 0")
    if args.temperature >= 1e-5:
        raise ValueError("This research script currently supports only --temperature 0.0")
    collect_profile = args.collect_profile or (args.save_cycle_trace_path is not None)

    setup_t0 = time.perf_counter()

    def setup_log(msg: str, *, pre_dist: bool = False) -> None:
        if pre_dist:
            builtins.print(f"[setup][pre-dist] +{time.perf_counter() - setup_t0:.2f}s {msg}", flush=True)
            return
        if args.log_all_ranks or dist.is_main():
            builtins.print(
                f"[setup][rank{dist.rank()}] +{time.perf_counter() - setup_t0:.2f}s {msg}",
                flush=True,
            )

    setup_log(
        (
            f"pid={os.getpid()} starting; "
            f"candidate_mode={args.candidate_mode}, fixed_prefix_len={args.fixed_prefix_len}, "
            f"branch_depth={args.branch_depth}, top_k={args.branch_top_k}, max_candidates={args.max_candidates}, "
            f"margin_threshold={args.branch_margin_threshold}"
        ),
        pre_dist=True,
    )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    setup_log("random seeds configured", pre_dist=True)

    setup_log("initializing distributed process group...", pre_dist=True)
    dist.init()
    atexit.register(dist.destroy)
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    setup_log(f"distributed ready (world_size={dist.size()}, local_rank={dist.local_rank()}, device={device})")

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()
    setup_log(f"attention backend={'flash_attention_2' if installed_flash_attn else 'sdpa'}")

    setup_log("loading target model...")
    t_load_target = time.perf_counter()
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    setup_log(f"target model loaded in {time.perf_counter() - t_load_target:.2f}s")

    setup_log("loading draft model...")
    t_load_draft = time.perf_counter()
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    setup_log(f"draft model loaded in {time.perf_counter() - t_load_draft:.2f}s")

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    setup_log(f"effective block_size={block_size}")

    setup_log("loading tokenizer...")
    t_tokenizer = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    setup_log(f"tokenizer loaded in {time.perf_counter() - t_tokenizer:.2f}s")

    setup_log(f"loading dataset `{args.dataset}`...")
    t_dataset = time.perf_counter()
    dataset = load_and_process_dataset(args.dataset)
    setup_log(f"dataset loaded with {len(dataset)} rows in {time.perf_counter() - t_dataset:.2f}s")

    if args.max_samples is not None and len(dataset) > args.max_samples:
        t_slice = time.perf_counter()
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
        setup_log(f"dataset reduced to {len(dataset)} rows in {time.perf_counter() - t_slice:.2f}s")

    responses = []
    output_records = []
    cycle_trace_records = []
    first_prompt_logged = False
    baseline_enabled = not args.skip_baseline

    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        first_prompt_t0 = time.perf_counter()
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            bs_candidates = [block_size] if args.skip_baseline else [1, block_size]
            for bs in dict.fromkeys(bs_candidates):
                t_call = cuda_time()
                response[bs] = dflash_generate_candidate_solutions(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    branch_depth=args.branch_depth,
                    branch_top_k=args.branch_top_k,
                    max_candidates=args.max_candidates,
                    margin_threshold=args.branch_margin_threshold,
                    candidate_mode=args.candidate_mode,
                    fixed_prefix_len=args.fixed_prefix_len,
                    temperature=args.temperature,
                    collect_profile=collect_profile,
                )
                response[bs].wall_time_s = cuda_time() - t_call

            baseline_response = response.get(1)
            baseline_output_text = None
            if baseline_response is not None:
                baseline_generated_ids = baseline_response.output_ids[0, baseline_response.num_input_tokens :]
                baseline_output_text = tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

            if args.save_cycle_trace_path:
                for mode_name, mode_bs in [("baseline", 1), ("speculative", block_size)]:
                    mode_resp = response.get(mode_bs)
                    if mode_resp is None:
                        continue
                    for row in getattr(mode_resp, "cycle_trace", []):
                        cycle_trace_records.append(
                            {
                                "rank": dist.rank(),
                                "dataset": args.dataset,
                                "dataset_row_idx": idx,
                                "turn_index": turn_index,
                                "mode": mode_name,
                                "block_size": int(mode_bs),
                                **row,
                            }
                        )

            output_records.append(
                {
                    "rank": dist.rank(),
                    "dataset_row_idx": idx,
                    "turn_index": turn_index,
                    "dataset": args.dataset,
                    "prompt_text": user_content,
                    "input_text": input_text,
                    "block_size": block_size,
                    "branch_depth": args.branch_depth,
                    "branch_top_k": args.branch_top_k,
                    "max_candidates": args.max_candidates,
                    "branch_margin_threshold": args.branch_margin_threshold,
                    "candidate_mode": args.candidate_mode,
                    "fixed_prefix_len": args.fixed_prefix_len,
                    "baseline": None
                    if baseline_response is None
                    else {
                        "output_text": baseline_output_text,
                        "num_input_tokens": baseline_response.num_input_tokens,
                        "num_output_tokens": baseline_response.num_output_tokens,
                        "wall_time_s": baseline_response.wall_time_s,
                        "ttft_s": baseline_response.time_to_first_token,
                        "tpot_s": baseline_response.time_per_output_token,
                        "acceptance_lengths": baseline_response.acceptance_lengths,
                        "candidate_summary": baseline_response.candidate_summary,
                        "profile_summary": baseline_response.profile_summary,
                    },
                    "speculative": {
                        "output_text": output_text,
                        "num_input_tokens": spec_response.num_input_tokens,
                        "num_output_tokens": spec_response.num_output_tokens,
                        "wall_time_s": spec_response.wall_time_s,
                        "ttft_s": spec_response.time_to_first_token,
                        "tpot_s": spec_response.time_per_output_token,
                        "acceptance_lengths": spec_response.acceptance_lengths,
                        "candidate_summary": spec_response.candidate_summary,
                        "profile_summary": spec_response.profile_summary,
                    },
                }
            )

        if not first_prompt_logged:
            setup_log(f"first local prompt finished in {time.perf_counter() - first_prompt_t0:.2f}s")
            first_prompt_logged = True

    if dist.size() > 1:
        setup_log("gathering responses from all ranks...")
        responses = dist.gather(responses, dst=0)
        output_records = dist.gather(output_records, dst=0)
        if args.save_cycle_trace_path:
            cycle_trace_records = dist.gather(cycle_trace_records, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        output_records = list(chain(*output_records))
        if args.save_cycle_trace_path:
            cycle_trace_records = list(chain(*cycle_trace_records))
        setup_log("gather complete")

    spec_samples = [r[block_size] for r in responses]
    spec_metrics = summarize_mode(spec_samples)

    if baseline_enabled:
        baseline_samples = [r[1] for r in responses]
        baseline_metrics = summarize_mode(baseline_samples)
        print(f"Baseline total_wall_s: {baseline_metrics['total_wall_s']:.6f}")
        print(f"Baseline avg_wall_s: {baseline_metrics['avg_wall_s']:.6f}")
        print(f"Baseline TTFT: {baseline_metrics['avg_ttft_s']:.6f}")
        print(f"Baseline TPOT: {baseline_metrics['avg_tpot_s']:.6f}")
        print(f"Baseline tokens_per_sec: {baseline_metrics['tokens_per_sec']:.6f}")

    print(f"Speculative total_wall_s: {spec_metrics['total_wall_s']:.6f}")
    print(f"Speculative avg_wall_s: {spec_metrics['avg_wall_s']:.6f}")
    print(f"Speculative TTFT: {spec_metrics['avg_ttft_s']:.6f}")
    print(f"Speculative TPOT: {spec_metrics['avg_tpot_s']:.6f}")
    print(f"Speculative tokens_per_sec: {spec_metrics['tokens_per_sec']:.6f}")

    if baseline_enabled:
        print(f"Decoding speedup: {baseline_metrics['avg_tpot_s'] / spec_metrics['avg_tpot_s']:.2f}")
    else:
        print("Decoding speedup: N/A (baseline skipped)")

    if collect_profile:
        spec_profile = summarize_profile(spec_samples)
        if spec_profile is not None:
            print(f"Speculative profile avg_target_prefill_s: {spec_profile['avg_target_prefill_s']:.6f}")
            print(f"Speculative profile avg_target_decode_s: {spec_profile['avg_target_decode_s']:.6f}")
            print(f"Speculative profile avg_draft_decode_s: {spec_profile['avg_draft_decode_s']:.6f}")
            print(f"Speculative profile target_share_decode: {spec_profile['target_share_decode']:.4f}")
            print(f"Speculative profile draft_share_decode: {spec_profile['draft_share_decode']:.4f}")
            print(f"Speculative profile total_profiled_cycles: {int(spec_profile['total_profiled_cycles'])}")
        if baseline_enabled:
            baseline_profile = summarize_profile(baseline_samples)
            if baseline_profile is not None:
                print(f"Baseline profile avg_target_prefill_s: {baseline_profile['avg_target_prefill_s']:.6f}")
                print(f"Baseline profile avg_target_decode_s: {baseline_profile['avg_target_decode_s']:.6f}")
                print(f"Baseline profile avg_draft_decode_s: {baseline_profile['avg_draft_decode_s']:.6f}")
                print(f"Baseline profile target_share_decode: {baseline_profile['target_share_decode']:.4f}")
                print(f"Baseline profile draft_share_decode: {baseline_profile['draft_share_decode']:.4f}")
                print(f"Baseline profile total_profiled_cycles: {int(baseline_profile['total_profiled_cycles'])}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")
    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    avg_candidates_per_cycle = float(np.mean([r[block_size].candidate_summary["avg_candidates_per_cycle"] for r in responses]))
    avg_verify_calls = float(np.mean([r[block_size].candidate_summary["candidate_verify_calls"] for r in responses]))
    print(f"Candidate avg_candidates_per_cycle: {avg_candidates_per_cycle:.3f}")
    print(f"Candidate avg_verify_calls_per_sample: {avg_verify_calls:.1f}")
    print(f"Candidate mode: {args.candidate_mode}")
    print(f"Candidate fixed_prefix_len: {args.fixed_prefix_len}")
    print(f"Candidate branch_depth: {args.branch_depth}")
    print(f"Candidate branch_top_k: {args.branch_top_k}")
    print(f"Candidate max_candidates: {args.max_candidates}")
    print(f"Candidate margin_threshold: {args.branch_margin_threshold}")

    print(f"Hardware GPU: {torch.cuda.get_device_name(device)}")
    print(f"Hardware CUDA: {torch.version.cuda}")
    print(f"Hardware Torch: {torch.__version__}")
    print(f"Hardware World Size: {dist.size()}")

    if args.save_outputs_path:
        out_path = Path(args.save_outputs_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in output_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved per-sample outputs to: {out_path}")

    if args.save_cycle_trace_path:
        trace_path = Path(args.save_cycle_trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as f:
            for row in cycle_trace_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved per-cycle trace to: {trace_path}")


if __name__ == "__main__":
    main()
