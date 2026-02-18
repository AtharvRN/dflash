import argparse
import atexit
import builtins
import json
import os
import random
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from benchmark import dflash_generate
from model import DFlashDraftModel, load_and_process_dataset


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def summarize_mode(samples: list[dict]) -> dict[str, float]:
    total_wall_s = float(np.sum([s["wall_time_s"] for s in samples]))
    total_tokens = int(np.sum([s["num_output_tokens"] for s in samples]))
    avg_ttft_s = float(np.mean([s["ttft_s"] for s in samples]))
    avg_tpot_s = float(np.mean([s["tpot_s"] for s in samples]))
    avg_wall_s = float(np.mean([s["wall_time_s"] for s in samples]))
    tokens_per_sec = float(total_tokens / max(total_wall_s, 1e-8))
    return {
        "total_wall_s": total_wall_s,
        "avg_wall_s": avg_wall_s,
        "avg_ttft_s": avg_ttft_s,
        "avg_tpot_s": avg_tpot_s,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": float(total_tokens),
    }


def parse_block_sizes(raw: str) -> list[int]:
    values = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        val = int(token)
        if val < 1:
            raise ValueError(f"Invalid block size: {val}")
        values.append(val)
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError("No block sizes provided.")
    return values


def maybe_float_str(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple speculative block sizes in one process (shared imports/model load)."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-sizes", type=str, required=True, help="Comma-separated list, e.g. 8,12,16")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip shared baseline generation (bs=1).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-all-ranks", action="store_true")
    parser.add_argument(
        "--save-outputs-path",
        type=str,
        default=None,
        help="Optional JSONL path for per-sample outputs/metrics.",
    )
    parser.add_argument(
        "--save-summary-path",
        type=str,
        default=None,
        help="Optional CSV path for per-block aggregate metrics.",
    )
    args = parser.parse_args()

    block_sizes = parse_block_sizes(args.block_sizes)
    if (not args.skip_baseline) and 1 in block_sizes:
        logger.warning("Removing block size 1 from speculative list; shared baseline already covers bs=1.")
        block_sizes = [b for b in block_sizes if b != 1]
    if not block_sizes:
        raise ValueError("After filtering, no speculative block sizes remain.")

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
            f"draft={args.draft_name_or_path}, block_sizes={block_sizes}"
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
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa.")
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

    setup_log("loading tokenizer...")
    t_tok = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    setup_log(f"tokenizer loaded in {time.perf_counter() - t_tok:.2f}s")

    setup_log(f"loading dataset `{args.dataset}`...")
    t_dataset = time.perf_counter()
    dataset = load_and_process_dataset(args.dataset)
    setup_log(f"dataset loaded with {len(dataset)} rows in {time.perf_counter() - t_dataset:.2f}s")

    if args.max_samples is not None and len(dataset) > args.max_samples:
        t_slice = time.perf_counter()
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
        setup_log(f"dataset reduced to {len(dataset)} rows in {time.perf_counter() - t_slice:.2f}s")

    baseline_enabled = not args.skip_baseline
    per_block_samples: dict[int, list[dict]] = {b: [] for b in block_sizes}
    baseline_samples: list[dict] = []
    output_records: list[dict] = []
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

            sample_record = {
                "rank": dist.rank(),
                "dataset_row_idx": idx,
                "turn_index": turn_index,
                "dataset": args.dataset,
                "prompt_text": user_content,
                "input_text": input_text,
                "baseline": None,
                "speculative": {},
            }

            if baseline_enabled:
                t_call = cuda_time()
                baseline_resp = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=1,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    collect_profile=False,
                )
                baseline_wall = cuda_time() - t_call
                baseline_ids = baseline_resp.output_ids[0, baseline_resp.num_input_tokens:]
                baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)
                baseline_entry = {
                    "num_input_tokens": baseline_resp.num_input_tokens,
                    "num_output_tokens": baseline_resp.num_output_tokens,
                    "wall_time_s": baseline_wall,
                    "ttft_s": baseline_resp.time_to_first_token,
                    "tpot_s": baseline_resp.time_per_output_token,
                    "acceptance_lengths": baseline_resp.acceptance_lengths,
                }
                baseline_samples.append(baseline_entry)
                sample_record["baseline"] = {"output_text": baseline_text, **baseline_entry}

            for bs in block_sizes:
                t_call = cuda_time()
                spec_resp = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    collect_profile=False,
                )
                spec_wall = cuda_time() - t_call
                spec_ids = spec_resp.output_ids[0, spec_resp.num_input_tokens:]
                spec_text = tokenizer.decode(spec_ids, skip_special_tokens=True)
                entry = {
                    "num_input_tokens": spec_resp.num_input_tokens,
                    "num_output_tokens": spec_resp.num_output_tokens,
                    "wall_time_s": spec_wall,
                    "ttft_s": spec_resp.time_to_first_token,
                    "tpot_s": spec_resp.time_per_output_token,
                    "acceptance_lengths": spec_resp.acceptance_lengths,
                }
                per_block_samples[bs].append(entry)
                sample_record["speculative"][str(bs)] = {"output_text": spec_text, **entry}

            # Keep the final assistant message from the largest block size for multi-turn datasets.
            final_bs = block_sizes[-1]
            messages.append(
                {
                    "role": "assistant",
                    "content": sample_record["speculative"][str(final_bs)]["output_text"],
                }
            )
            output_records.append(sample_record)
        if not first_prompt_logged:
            setup_log(f"first local prompt finished in {time.perf_counter() - first_prompt_t0:.2f}s")
            first_prompt_logged = True

    if dist.size() > 1:
        setup_log("gathering results from all ranks...")
        gathered_per_block = {}
        for bs in block_sizes:
            chunks = dist.gather(per_block_samples[bs], dst=0)
            if dist.is_main():
                gathered_per_block[bs] = list(chain(*chunks))
        output_chunks = dist.gather(output_records, dst=0)
        baseline_chunks = dist.gather(baseline_samples, dst=0)
        if not dist.is_main():
            return
        per_block_samples = gathered_per_block
        output_records = list(chain(*output_chunks))
        baseline_samples = list(chain(*baseline_chunks))
        setup_log("gather complete")

    baseline_metrics = None
    if baseline_enabled:
        baseline_metrics = summarize_mode(baseline_samples)
        print(f"Baseline total_wall_s: {baseline_metrics['total_wall_s']:.6f}")
        print(f"Baseline avg_wall_s: {baseline_metrics['avg_wall_s']:.6f}")
        print(f"Baseline TTFT: {baseline_metrics['avg_ttft_s']:.6f}")
        print(f"Baseline TPOT: {baseline_metrics['avg_tpot_s']:.6f}")
        print(f"Baseline tokens_per_sec: {baseline_metrics['tokens_per_sec']:.6f}")

    summary_rows = []
    for bs in block_sizes:
        samples = per_block_samples[bs]
        metrics = summarize_mode(samples)
        tau = float(np.mean([np.mean(s["acceptance_lengths"]) for s in samples]))
        acceptance_lengths = list(chain(*[s["acceptance_lengths"] for s in samples]))
        histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(bs + 1)]
        histogram_str = [f"{x * 100:.1f}%" for x in histogram]

        print(f"[bs={bs}] Speculative total_wall_s: {metrics['total_wall_s']:.6f}")
        print(f"[bs={bs}] Speculative avg_wall_s: {metrics['avg_wall_s']:.6f}")
        print(f"[bs={bs}] Speculative TTFT: {metrics['avg_ttft_s']:.6f}")
        print(f"[bs={bs}] Speculative TPOT: {metrics['avg_tpot_s']:.6f}")
        print(f"[bs={bs}] Speculative tokens_per_sec: {metrics['tokens_per_sec']:.6f}")
        if baseline_metrics is not None:
            speedup = baseline_metrics["avg_tpot_s"] / metrics["avg_tpot_s"]
            print(f"[bs={bs}] Decoding speedup: {speedup:.2f}")
        else:
            speedup = None
            print(f"[bs={bs}] Decoding speedup: N/A (baseline skipped)")
        print(f"[bs={bs}] Average Acceptance length: {tau:.2f}")
        print(f"[bs={bs}] Acceptance length histogram: {histogram_str}")

        summary_rows.append(
            {
                "dataset": args.dataset,
                "max_samples": len(dataset),
                "block_size": bs,
                "speedup": speedup,
                "tau": tau,
                "acceptance_histogram": histogram_str,
                "baseline_total_wall_s": None if baseline_metrics is None else baseline_metrics["total_wall_s"],
                "speculative_total_wall_s": metrics["total_wall_s"],
                "baseline_tokens_per_sec": None if baseline_metrics is None else baseline_metrics["tokens_per_sec"],
                "speculative_tokens_per_sec": metrics["tokens_per_sec"],
                "baseline_tpot": None if baseline_metrics is None else baseline_metrics["avg_tpot_s"],
                "speculative_tpot": metrics["avg_tpot_s"],
                "baseline_ttft": None if baseline_metrics is None else baseline_metrics["avg_ttft_s"],
                "speculative_ttft": metrics["avg_ttft_s"],
                "gpu_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "world_size": dist.size(),
            }
        )

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

    if args.save_summary_path:
        summary_path = Path(args.save_summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "dataset,max_samples,block_size,speedup,tau,gpu_name,cuda_version,torch_version,"
            "baseline_total_wall_s,speculative_total_wall_s,baseline_tokens_per_sec,speculative_tokens_per_sec,"
            "baseline_tpot,speculative_tpot,baseline_ttft,speculative_ttft,acceptance_histogram,world_size"
        )
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for row in summary_rows:
                f.write(
                    ",".join(
                        [
                            str(row["dataset"]),
                            str(row["max_samples"]),
                            str(row["block_size"]),
                            maybe_float_str(row["speedup"]),
                            maybe_float_str(row["tau"]),
                            str(row["gpu_name"]),
                            str(row["cuda_version"]),
                            str(row["torch_version"]),
                            maybe_float_str(row["baseline_total_wall_s"]),
                            maybe_float_str(row["speculative_total_wall_s"]),
                            maybe_float_str(row["baseline_tokens_per_sec"]),
                            maybe_float_str(row["speculative_tokens_per_sec"]),
                            maybe_float_str(row["baseline_tpot"]),
                            maybe_float_str(row["speculative_tpot"]),
                            maybe_float_str(row["baseline_ttft"]),
                            maybe_float_str(row["speculative_ttft"]),
                            json.dumps(row["acceptance_histogram"]),
                            str(row["world_size"]),
                        ]
                    )
                    + "\n"
                )
        print(f"Saved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
