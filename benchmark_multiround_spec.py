import argparse
import atexit
import builtins
import json
import os
import random
import time
from collections import Counter
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
from model import DFlashDraftModel, extract_context_feature, load_and_process_dataset


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def parse_round_blocks(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("No round block sizes provided.")
    if any(v < 1 for v in vals):
        raise ValueError("Round block sizes must be >= 1.")
    return vals


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


def softmax_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return probs


def normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.clamp(probs, min=0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    return probs / torch.clamp(denom, min=1e-12)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = normalize_probs(probs)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def proposal_distribution_from_round(
    *,
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    target_hidden: torch.Tensor,
    last_token: torch.Tensor,
    current_pos: int,
    block_size: int,
    mask_token_id: int,
    temperature: float,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    # Round block size 1 means "proposal = current target distribution".
    if block_size == 1:
        return target_probs

    block_output_ids = torch.full(
        (1, block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    block_output_ids[:, 0] = last_token[:, 0]
    ctx_len = int(target_hidden.shape[1])
    # DFlash attention builds K by concatenating context + noise tokens.
    # Rotary embeddings therefore need positions covering both segments.
    pos_start = max(0, current_pos - ctx_len)
    block_position_ids = torch.arange(
        pos_start,
        current_pos + block_size,
        device=model.device,
    ).unsqueeze(0)

    noise_embedding = target.model.embed_tokens(block_output_ids)
    draft_hidden = model(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=block_position_ids,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )
    draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
    next_token_logits = draft_logits[:, 0, :]
    return softmax_probs(next_token_logits, temperature)


@torch.inference_mode()
def dflash_generate_multiround_exact(
    *,
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    round_block_sizes: list[int],
    stop_token_ids: list[int],
    temperature: float,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.full(
        (1, max_length + 1),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()

    output_ids[:, :num_input_tokens] = input_ids
    stop_ids = set(stop_token_ids or [])

    prefill_start = cuda_time()
    prefill_output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    first_probs = softmax_probs(prefill_output.logits[:, -1, :], temperature)
    first_token = sample_from_probs(first_probs)
    output_ids[:, num_input_tokens] = first_token[:, 0]
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    accepted_rounds: list[int] = []
    round_trace: list[dict] = []

    current_pos = num_input_tokens
    stop_hit = int(first_token.item()) in stop_ids

    while (current_pos + 1) < max_length and not stop_hit:
        last_token = output_ids[:, current_pos : current_pos + 1]
        step_output = target(
            last_token,
            position_ids=position_ids[:, current_pos : current_pos + 1],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        target_probs = softmax_probs(step_output.logits[:, -1, :], temperature)
        target_hidden = extract_context_feature(step_output.hidden_states, model.target_layer_ids)
        residual = target_probs.clone()

        step_idx = current_pos - num_input_tokens + 1
        chosen_token = None
        accepted_round_idx = len(round_block_sizes) + 1

        for round_idx, configured_bs in enumerate(round_block_sizes, start=1):
            proposal_probs = proposal_distribution_from_round(
                model=model,
                target=target,
                target_hidden=target_hidden,
                last_token=last_token,
                current_pos=current_pos,
                block_size=configured_bs,
                mask_token_id=mask_token_id,
                temperature=temperature,
                target_probs=target_probs,
            )

            sampled_token = sample_from_probs(proposal_probs)
            sampled_token_id = int(sampled_token.item())
            q_t = float(proposal_probs[0, sampled_token_id].item())
            p_t = float(residual[0, sampled_token_id].item())
            accept_prob = min(1.0, p_t / max(q_t, 1e-12))
            u = float(torch.rand((), device=model.device).item())
            accepted = u < accept_prob

            round_trace.append(
                {
                    "step_idx": int(step_idx),
                    "round_idx": int(round_idx),
                    "configured_block_size": int(configured_bs),
                    "token_id": sampled_token_id,
                    "proposal_prob": q_t,
                    "residual_prob": p_t,
                    "accept_prob": accept_prob,
                    "u": u,
                    "accepted": bool(accepted),
                }
            )

            if accepted:
                chosen_token = sampled_token
                accepted_round_idx = round_idx
                break

            residual = torch.clamp(residual - proposal_probs, min=0.0)
            residual_mass = float(residual.sum().item())
            if residual_mass <= 1e-12:
                residual = target_probs.clone()
            else:
                residual = residual / residual_mass

        if chosen_token is None:
            chosen_token = sample_from_probs(residual)

        current_pos += 1
        output_ids[:, current_pos] = chosen_token[:, 0]
        accepted_rounds.append(accepted_round_idx)

        if int(chosen_token.item()) in stop_ids:
            stop_hit = True

    output_ids = output_ids[:, : current_pos + 1]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(1, num_output_tokens)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        accepted_rounds=accepted_rounds,
        round_trace=round_trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact multi-round speculative sampling benchmark for DFlash.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for both target and proposal distributions. Must be > 0 for exact sampling.",
    )
    parser.add_argument("--round-block-sizes", type=str, default="16,12")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-all-ranks", action="store_true")
    parser.add_argument("--save-outputs-path", type=str, default=None)
    args = parser.parse_args()

    if args.temperature <= 0.0:
        raise ValueError("Exact multi-round speculative sampling requires --temperature > 0.0.")

    round_blocks = parse_round_blocks(args.round_block_sizes)
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

    setup_log(f"pid={os.getpid()} starting; round_blocks={round_blocks}", pre_dist=True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    setup_log("random seeds configured", pre_dist=True)

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
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa.")
            return False

    installed_flash_attn = has_flash_attn()
    setup_log(f"attention backend={'flash_attention_2' if installed_flash_attn else 'sdpa'}")

    setup_log("loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(device).eval()

    setup_log("loading draft model...")
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    if draft_model.mask_token_id is None:
        raise ValueError("Draft model has no mask_token_id in config; cannot run DFlash proposal rounds.")

    setup_log("loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
    setup_log(f"dataset ready with {len(dataset)} rows")

    responses_baseline = []
    responses_multiround = []
    output_records = []
    first_prompt_logged = False
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

            baseline_response = None
            if not args.skip_baseline:
                t_base = cuda_time()
                baseline_response = dflash_generate_multiround_exact(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    round_block_sizes=[],
                    stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
                    temperature=args.temperature,
                )
                baseline_response.wall_time_s = cuda_time() - t_base
                responses_baseline.append(baseline_response)

            t_multi = cuda_time()
            multi_response = dflash_generate_multiround_exact(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                round_block_sizes=round_blocks,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
                temperature=args.temperature,
            )
            multi_response.wall_time_s = cuda_time() - t_multi
            responses_multiround.append(multi_response)

            generated_ids = multi_response.output_ids[0, multi_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})

            baseline_text = None
            if baseline_response is not None:
                baseline_ids = baseline_response.output_ids[0, baseline_response.num_input_tokens :]
                baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)

            output_records.append(
                {
                    "rank": dist.rank(),
                    "dataset_row_idx": idx,
                    "turn_index": turn_index,
                    "dataset": args.dataset,
                    "prompt_text": user_content,
                    "input_text": input_text,
                    "baseline": None
                    if baseline_response is None
                    else {
                        "output_text": baseline_text,
                        "num_input_tokens": baseline_response.num_input_tokens,
                        "num_output_tokens": baseline_response.num_output_tokens,
                        "wall_time_s": baseline_response.wall_time_s,
                        "ttft_s": baseline_response.time_to_first_token,
                        "tpot_s": baseline_response.time_per_output_token,
                    },
                    "multiround": {
                        "round_block_sizes": round_blocks,
                        "output_text": output_text,
                        "num_input_tokens": multi_response.num_input_tokens,
                        "num_output_tokens": multi_response.num_output_tokens,
                        "wall_time_s": multi_response.wall_time_s,
                        "ttft_s": multi_response.time_to_first_token,
                        "tpot_s": multi_response.time_per_output_token,
                        "accepted_rounds": multi_response.accepted_rounds,
                        "round_trace": multi_response.round_trace,
                    },
                }
            )
        if not first_prompt_logged:
            setup_log(f"first local prompt finished in {time.perf_counter() - first_prompt_t0:.2f}s")
            first_prompt_logged = True

    if dist.size() > 1:
        responses_multiround = dist.gather(responses_multiround, dst=0)
        output_records = dist.gather(output_records, dst=0)
        if not args.skip_baseline:
            responses_baseline = dist.gather(responses_baseline, dst=0)
        if not dist.is_main():
            return
        responses_multiround = list(chain(*responses_multiround))
        output_records = list(chain(*output_records))
        if not args.skip_baseline:
            responses_baseline = list(chain(*responses_baseline))

    multi_metrics = summarize_mode(responses_multiround)
    print(f"Multiround total_wall_s: {multi_metrics['total_wall_s']:.6f}")
    print(f"Multiround avg_wall_s: {multi_metrics['avg_wall_s']:.6f}")
    print(f"Multiround TTFT: {multi_metrics['avg_ttft_s']:.6f}")
    print(f"Multiround TPOT: {multi_metrics['avg_tpot_s']:.6f}")
    print(f"Multiround tokens_per_sec: {multi_metrics['tokens_per_sec']:.6f}")

    if not args.skip_baseline:
        baseline_metrics = summarize_mode(responses_baseline)
        print(f"Baseline total_wall_s: {baseline_metrics['total_wall_s']:.6f}")
        print(f"Baseline avg_wall_s: {baseline_metrics['avg_wall_s']:.6f}")
        print(f"Baseline TTFT: {baseline_metrics['avg_ttft_s']:.6f}")
        print(f"Baseline TPOT: {baseline_metrics['avg_tpot_s']:.6f}")
        print(f"Baseline tokens_per_sec: {baseline_metrics['tokens_per_sec']:.6f}")
        print(f"Decoding speedup: {baseline_metrics['avg_tpot_s'] / multi_metrics['avg_tpot_s']:.2f}")
    else:
        print("Decoding speedup: N/A (baseline skipped)")

    accepted_rounds = list(chain(*[r.accepted_rounds for r in responses_multiround]))
    if accepted_rounds:
        avg_round = float(np.mean(accepted_rounds))
        max_round = len(round_blocks) + 1
        round_hist = [accepted_rounds.count(i) / len(accepted_rounds) for i in range(1, max_round + 1)]
        round_labels = [f"round_{i}" for i in range(1, len(round_blocks) + 1)] + ["residual"]
        print(f"Average accepted round index: {avg_round:.3f}")
        print(f"Accepted round histogram ({round_labels}): {[f'{x * 100:.1f}%' for x in round_hist]}")

        usage = Counter(accepted_rounds)
        usage_sorted = {k: usage[k] for k in sorted(usage)}
        usage_pct = {k: (v / len(accepted_rounds)) for k, v in usage_sorted.items()}
        print(f"Accepted round counts: {usage_sorted}")
        print(f"Accepted round pct: {usage_pct}")
    else:
        print("Accepted round histogram: N/A (no decode steps after first token)")

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


if __name__ == "__main__":
    main()
