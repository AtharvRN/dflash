import argparse
import atexit
import builtins
import json
import os
import time
import random
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    collect_profile: bool = False,
    draft_steps: int = 1,
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

    # Prefill stage
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
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True
    cycle_trace = []

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
        if effective_block_size > 1:
            if collect_profile:
                draft_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                draft_events[0].record()
            # Multi-step draft refinement: repeatedly denoise/update the draft block
            # before target verification.
            use_draft_cache = draft_steps == 1
            for _ in range(draft_steps):
                noise_embedding = target.model.embed_tokens(block_output_ids)
                if use_draft_cache:
                    draft_position_ids = position_ids[
                        :, past_key_values_draft.get_seq_length() : start + effective_block_size
                    ]
                    draft_past = past_key_values_draft
                else:
                    # Cache reuse across refinement steps is not valid here because each step
                    # rewrites the same token range. Recompute on the local block positions.
                    draft_position_ids = block_position_ids
                    draft_past = None
                draft_logits = target.lm_head(
                    model(
                        target_hidden=target_hidden,
                        noise_embedding=noise_embedding,
                        position_ids=draft_position_ids,
                        past_key_values=draft_past,
                        use_cache=use_draft_cache,
                        is_causal=False,
                    )[:, -effective_block_size + 1 :, :]
                )
                block_output_ids[:, 1:] = sample(draft_logits)
            if use_draft_cache:
                past_key_values_draft.crop(start)
            if collect_profile:
                draft_events[1].record()
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        if collect_profile:
            target_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            target_events[0].record()
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if effective_block_size > 1 else False,
        )
        if collect_profile:
            target_events[1].record()

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        tau = acceptance_length + 1
        acceptance_lengths.append(tau)
        generated_tokens_before = start - num_input_tokens
        if collect_profile:
            cycle_end.record()
            cycle_trace.append(
                {
                    "cycle_idx": int(len(acceptance_lengths) - 1),
                    "generated_tokens_before": int(generated_tokens_before),
                    "effective_block_size": int(effective_block_size),
                    "tau": int(tau),
                    "acceptance_ratio": float(tau / max(1, effective_block_size)),
                    "_events": {
                        "draft": draft_events,
                        "target": target_events,
                        "cycle": (cycle_start, cycle_end),
                    },
                }
            )
        start += tau
        past_key_values_target.crop(start)
        if effective_block_size > 1:
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
    time_per_output_token = total_decode_time / num_output_tokens

    profile_summary = None
    if collect_profile:
        torch.cuda.synchronize()
        target_prefill_s = float(time_to_first_token)
        total_target_decode_s = 0.0
        total_draft_decode_s = 0.0
        total_cycle_s = 0.0
        for row in cycle_trace:
            draft_events = row["_events"]["draft"]
            target_events = row["_events"]["target"]
            cycle_events = row["_events"]["cycle"]
            draft_s = 0.0
            if draft_events is not None:
                draft_s = draft_events[0].elapsed_time(draft_events[1]) / 1000.0
            target_s = target_events[0].elapsed_time(target_events[1]) / 1000.0
            cycle_s = cycle_events[0].elapsed_time(cycle_events[1]) / 1000.0
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
            "draft_share_decode": float(total_draft_decode_s / max(1e-12, total_draft_decode_s + total_target_decode_s)),
            "target_share_decode": float(total_target_decode_s / max(1e-12, total_draft_decode_s + total_target_decode_s)),
        }

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        cycle_trace=cycle_trace,
        profile_summary=profile_summary,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip per-sample baseline (bs=1) generation and run only speculative decode.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local cache; fail fast if missing.",
    )
    parser.add_argument(
        "--log-all-ranks",
        action="store_true",
        help="Print setup timing logs from every rank (default: rank0 only).",
    )
    parser.add_argument(
        "--save-outputs-path",
        type=str,
        default=None,
        help="Optional JSONL path to save per-sample outputs and timing metrics.",
    )
    parser.add_argument(
        "--save-cycle-trace-path",
        type=str,
        default=None,
        help="Optional JSONL path to save per-cycle trace rows (tau, draft/target/cycle timing, token index).",
    )
    parser.add_argument(
        "--collect-profile",
        action="store_true",
        help="Collect per-cycle profiling stats (target/draft/cycle timings).",
    )
    parser.add_argument(
        "--draft-steps",
        type=int,
        default=1,
        help="Number of draft refinement passes per speculative cycle (default: 1).",
    )
    args = parser.parse_args()
    if args.draft_steps < 1:
        raise ValueError("--draft-steps must be >= 1")
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
            f"pid={os.getpid()} starting; model={args.model_name_or_path}, "
            f"draft={args.draft_name_or_path}, draft_steps={args.draft_steps}"
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

    def has_flash_attn():
        try:
            import flash_attn
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
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            bs_candidates = [block_size] if args.skip_baseline else [1, block_size]
            for bs in dict.fromkeys(bs_candidates):
                t_call = cuda_time()
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    collect_profile=collect_profile,
                    draft_steps=args.draft_steps,
                )
                response[bs].wall_time_s = cuda_time() - t_call

            baseline_response = response.get(1)
            baseline_output_text = None
            if baseline_response is not None:
                baseline_generated_ids = baseline_response.output_ids[0, baseline_response.num_input_tokens:]
                baseline_output_text = tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
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
                    "draft_steps": args.draft_steps,
                    "baseline": None if baseline_response is None else {
                        "output_text": baseline_output_text,
                        "num_input_tokens": baseline_response.num_input_tokens,
                        "num_output_tokens": baseline_response.num_output_tokens,
                        "wall_time_s": baseline_response.wall_time_s,
                        "ttft_s": baseline_response.time_to_first_token,
                        "tpot_s": baseline_response.time_per_output_token,
                        "acceptance_lengths": baseline_response.acceptance_lengths,
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
    print(f"Draft steps per cycle: {args.draft_steps}")
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
