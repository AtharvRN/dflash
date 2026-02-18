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


def pick_seed_positions(bs: int, mode: str) -> list[int]:
    # Index 0 is always the corrected/start token. We can seed only indices [1, bs-1].
    if bs <= 1 or mode == "none":
        return []
    if mode == "dense":
        return list(range(1, bs))
    # sparse: [corr, M, old, M, old, ...]
    return list(range(2, bs, 2))


@torch.inference_mode()
def dflash_generate_with_suffix_seed(
    *,
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    suffix_seed_mode: str = "none",
    suffix_seed_max_tokens: int = -1,
) -> SimpleNamespace:
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
    target_hidden = None
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths: list[int] = []
    cycle_trace: list[dict] = []
    draft_prefill = True

    recycled_suffix: list[int] = []
    seed_attempt_cycles = 0
    seeded_cycles = 0
    seeded_token_total = 0
    recycled_generated_total = 0

    while start < max_length:
        remaining = max_length - start
        bs = min(block_size, remaining)

        block_output_ids = output_ids[:, start : start + bs].clone()
        block_position_ids = position_ids[:, start : start + bs]

        recycled_available = len(recycled_suffix)
        seeded_count = 0
        if bs > 1 and suffix_seed_mode != "none" and recycled_available > 0:
            seed_attempt_cycles += 1
            seed_positions = pick_seed_positions(bs, suffix_seed_mode)
            if suffix_seed_max_tokens >= 0:
                seed_positions = seed_positions[:suffix_seed_max_tokens]
            n = min(len(seed_positions), recycled_available)
            if n > 0:
                token_tensor = torch.tensor(recycled_suffix[:n], device=model.device, dtype=torch.long)
                pos_tensor = torch.tensor(seed_positions[:n], device=model.device, dtype=torch.long)
                block_output_ids[0, pos_tensor] = token_tensor
                seeded_count = n
                seeded_cycles += 1
                seeded_token_total += n

        if bs > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + bs],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -bs + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        tau = acceptance_length + 1

        output_ids[:, start : start + tau] = block_output_ids[:, :tau]
        output_ids[:, start + tau] = posterior[:, acceptance_length]

        # Recycle discarded suffix as hints for next cycle.
        if tau < bs:
            recycled_suffix = block_output_ids[0, tau:bs].tolist()
            recycled_generated_total += len(recycled_suffix)
        else:
            recycled_suffix = []

        acceptance_lengths.append(tau)
        cycle_trace.append(
            {
                "cycle_idx": int(len(acceptance_lengths) - 1),
                "start_idx": int(start),
                "block_size": int(bs),
                "tau": int(tau),
                "acceptance_ratio": float(tau / max(1, bs)),
                "seed_mode": suffix_seed_mode,
                "recycled_available": int(recycled_available),
                "seeded_count": int(seeded_count),
                "recycled_next": int(len(recycled_suffix)),
            }
        )

        start += tau
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :tau, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

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
    total_cycles = len(acceptance_lengths)

    seed_summary = {
        "mode": suffix_seed_mode,
        "seed_attempt_cycles": int(seed_attempt_cycles),
        "seeded_cycles": int(seeded_cycles),
        "seeded_token_total": int(seeded_token_total),
        "recycled_generated_total": int(recycled_generated_total),
        "seeded_cycle_rate": float(seeded_cycles / max(1, total_cycles)),
        "seeded_tokens_per_cycle": float(seeded_token_total / max(1, total_cycles)),
    }

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        cycle_trace=cycle_trace,
        seed_summary=seed_summary,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash benchmark with recycled rejected-suffix seeding.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--suffix-seed-mode", type=str, choices=["none", "dense", "sparse"], default="sparse")
    parser.add_argument(
        "--suffix-seed-max-tokens",
        type=int,
        default=-1,
        help="Maximum seeded positions per cycle; -1 means unlimited.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-all-ranks", action="store_true")
    parser.add_argument("--save-outputs-path", type=str, default=None)
    parser.add_argument("--save-cycle-trace-path", type=str, default=None)
    args = parser.parse_args()

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
            f"pid={os.getpid()} starting; seed_mode={args.suffix_seed_mode}, "
            f"seed_max={args.suffix_seed_max_tokens}"
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

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    setup_log(f"effective block_size={block_size}")

    setup_log("loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
    setup_log(f"dataset ready with {len(dataset)} rows")

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
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            bs_candidates = [block_size] if args.skip_baseline else [1, block_size]
            for bs in dict.fromkeys(bs_candidates):
                t_call = cuda_time()
                response[bs] = dflash_generate_with_suffix_seed(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    suffix_seed_mode=args.suffix_seed_mode if bs != 1 else "none",
                    suffix_seed_max_tokens=args.suffix_seed_max_tokens,
                )
                response[bs].wall_time_s = cuda_time() - t_call

            baseline_response = response.get(1)
            baseline_text = None
            if baseline_response is not None:
                baseline_ids = baseline_response.output_ids[0, baseline_response.num_input_tokens:]
                baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)

            spec_response = response[block_size]
            spec_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(spec_ids, skip_special_tokens=True)
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
                    "baseline": None
                    if baseline_response is None
                    else {
                        "output_text": baseline_text,
                        "num_input_tokens": baseline_response.num_input_tokens,
                        "num_output_tokens": baseline_response.num_output_tokens,
                        "wall_time_s": baseline_response.wall_time_s,
                        "ttft_s": baseline_response.time_to_first_token,
                        "tpot_s": baseline_response.time_per_output_token,
                        "acceptance_lengths": baseline_response.acceptance_lengths,
                        "seed_summary": baseline_response.seed_summary,
                    },
                    "speculative": {
                        "seed_mode": args.suffix_seed_mode,
                        "seed_max_tokens": args.suffix_seed_max_tokens,
                        "output_text": output_text,
                        "num_input_tokens": spec_response.num_input_tokens,
                        "num_output_tokens": spec_response.num_output_tokens,
                        "wall_time_s": spec_response.wall_time_s,
                        "ttft_s": spec_response.time_to_first_token,
                        "tpot_s": spec_response.time_per_output_token,
                        "acceptance_lengths": spec_response.acceptance_lengths,
                        "seed_summary": spec_response.seed_summary,
                    },
                }
            )
        if not first_prompt_logged:
            setup_log(f"first local prompt finished in {time.perf_counter() - first_prompt_t0:.2f}s")
            first_prompt_logged = True

    if dist.size() > 1:
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

    tau = np.mean([np.mean(r.acceptance_lengths) for r in spec_samples])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r.acceptance_lengths for r in spec_samples]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    seed_attempt_cycles = int(np.sum([r.seed_summary["seed_attempt_cycles"] for r in spec_samples]))
    seeded_cycles = int(np.sum([r.seed_summary["seeded_cycles"] for r in spec_samples]))
    seeded_token_total = int(np.sum([r.seed_summary["seeded_token_total"] for r in spec_samples]))
    total_cycles = int(np.sum([len(r.acceptance_lengths) for r in spec_samples]))
    print(f"Suffix seed mode: {args.suffix_seed_mode}")
    print(f"Suffix seed attempt cycles: {seed_attempt_cycles}")
    print(f"Suffix seeded cycles: {seeded_cycles} ({seeded_cycles / max(1, total_cycles):.2%})")
    print(f"Suffix seeded tokens total: {seeded_token_total}")
    print(f"Suffix seeded tokens per cycle: {seeded_token_total / max(1, total_cycles):.3f}")

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
