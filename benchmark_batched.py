import argparse
import atexit
import builtins
import json
import os
import random
import time
from itertools import chain
from pathlib import Path

_BOOT_T0 = time.perf_counter()
builtins.print(f"[boot] +0.00s entering benchmark_batched.py pid={os.getpid()}", flush=True)
_IMPORT_DEBUG = os.environ.get("DFLASH_IMPORT_DEBUG", "0") == "1"


def _boot_import_log(msg: str) -> None:
    if _IMPORT_DEBUG:
        builtins.print(f"[boot][imports] +{time.perf_counter() - _BOOT_T0:.2f}s {msg}", flush=True)


_boot_import_log("importing numpy")
import numpy as np
_boot_import_log("importing torch")
import torch
_boot_import_log("importing loguru")
from loguru import logger
_boot_import_log("importing rich/tqdm")
from rich import print
from tqdm import tqdm
_boot_import_log("importing transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer
_boot_import_log("importing distributed utils")
import distributed as dist
_boot_import_log("importing benchmark helpers")
from benchmark import cuda_time, dflash_generate, summarize_mode, summarize_profile
_boot_import_log("importing local model modules")
from model import DFlashDraftModel, load_and_process_dataset
_boot_import_log("all imports finished")


def chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batched benchmark harness for DFlash. "
            "Processes prompts in user-defined batches while preserving benchmark.py metrics/profile outputs."
        )
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=4,
        help="Number of prompts to process per local batch loop.",
    )
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

    if args.prompt_batch_size < 1:
        raise ValueError("--prompt-batch-size must be >= 1")
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
            f"draft={args.draft_name_or_path}, prompt_batch_size={args.prompt_batch_size}, "
            f"draft_steps={args.draft_steps}"
        ),
        pre_dist=True,
    )
    setup_log(
        "batch mode=prompt-harness (per-sample decode path; grouped dataset processing for batch-size sweeps)",
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
    first_batch_logged = False
    baseline_enabled = not args.skip_baseline

    indices = list(range(dist.rank(), len(dataset), dist.size()))
    local_batches = chunked(indices, args.prompt_batch_size)

    for batch_idx, idx_batch in enumerate(tqdm(local_batches, disable=not dist.is_main())):
        first_batch_t0 = time.perf_counter()
        for idx in idx_batch:
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
                                    "local_batch_idx": batch_idx,
                                    "prompt_batch_size": args.prompt_batch_size,
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
                        "local_batch_idx": batch_idx,
                        "prompt_batch_size": args.prompt_batch_size,
                        "dataset": args.dataset,
                        "prompt_text": user_content,
                        "input_text": input_text,
                        "block_size": block_size,
                        "draft_steps": args.draft_steps,
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
        if not first_batch_logged:
            setup_log(f"first local prompt batch finished in {time.perf_counter() - first_batch_t0:.2f}s")
            first_batch_logged = True

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
    print(f"Prompt batch size: {args.prompt_batch_size}")
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
