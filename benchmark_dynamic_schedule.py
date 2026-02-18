import argparse
import atexit
import builtins
import json
import os
import random
import time
from collections import Counter, deque
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


def parse_block_sizes(s: str) -> list[int]:
    vals = sorted({int(x) for x in s.split(",") if x.strip()})
    if not vals:
        raise ValueError("No block sizes provided.")
    if any(v < 2 for v in vals):
        raise ValueError("Dynamic candidate block sizes must be >= 2.")
    return vals


class RollingTauScheduler:
    def __init__(
        self,
        *,
        candidates: list[int],
        warmup_cycles: int,
        decision_window: int,
        tau_high: float,
        tau_mid: float,
        required_streak: int,
        cooldown_cycles: int,
    ) -> None:
        self.candidates = sorted(candidates)
        self.low = self.candidates[0]
        self.mid = self.candidates[len(self.candidates) // 2]
        self.high = self.candidates[-1]
        self.current = self.high
        self.warmup_cycles = int(max(0, warmup_cycles))
        self.tau_hist: deque[float] = deque(maxlen=max(1, decision_window))
        self.tau_high = float(tau_high)
        self.tau_mid = float(tau_mid)
        self.required_streak = int(max(1, required_streak))
        self.cooldown_cycles = int(max(0, cooldown_cycles))
        self.cooldown_left = 0
        self.pending_target = self.current
        self.pending_streak = 0
        self.last_rolling_tau = None

    def select(self, cycle_idx: int) -> int:
        if cycle_idx < self.warmup_cycles:
            return self.high
        return self.current

    def update(self, tau: float, cycle_idx: int) -> None:
        self.tau_hist.append(float(tau))
        if cycle_idx < self.warmup_cycles:
            return
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return
        if len(self.tau_hist) < self.tau_hist.maxlen:
            return

        rolling_tau = float(sum(self.tau_hist) / len(self.tau_hist))
        self.last_rolling_tau = rolling_tau

        if rolling_tau >= self.tau_high:
            target = self.high
        elif rolling_tau >= self.tau_mid:
            target = self.mid
        else:
            target = self.low

        if target == self.current:
            self.pending_target = self.current
            self.pending_streak = 0
            return

        if target == self.pending_target:
            self.pending_streak += 1
        else:
            self.pending_target = target
            self.pending_streak = 1

        if self.pending_streak >= self.required_streak:
            self.current = target
            self.pending_streak = 0
            self.cooldown_left = self.cooldown_cycles


@torch.inference_mode()
def dflash_generate_policy(
    *,
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float,
    fixed_block_size: int | None = None,
    scheduler: RollingTauScheduler | None = None,
) -> SimpleNamespace:
    if fixed_block_size is None and scheduler is None:
        raise ValueError("Either fixed_block_size or scheduler must be provided.")

    max_block_size = fixed_block_size if fixed_block_size is not None else max(scheduler.candidates)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + max_block_size),
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
        output_hidden_states=True if max_block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = None
    if max_block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths: list[int] = []
    used_block_sizes: list[int] = []
    cycle_trace: list[dict] = []
    draft_prefill = True
    cycle_idx = 0

    while start < max_length:
        chosen_bs = fixed_block_size if fixed_block_size is not None else scheduler.select(cycle_idx)
        remaining = max_length - start
        bs = max(1, min(chosen_bs, remaining))

        block_output_ids = output_ids[:, start : start + bs].clone()
        block_position_ids = position_ids[:, start : start + bs]

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
            block_output_ids[:, 1:] = sample(draft_logits, temperature)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if max_block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        tau = acceptance_length + 1

        output_ids[:, start : start + tau] = block_output_ids[:, :tau]
        output_ids[:, start + tau] = posterior[:, acceptance_length]

        acceptance_lengths.append(tau)
        used_block_sizes.append(bs)
        cycle_trace.append(
            {
                "cycle_idx": cycle_idx,
                "start_idx": int(start),
                "block_size": int(bs),
                "tau": int(tau),
                "rolling_tau": None if scheduler is None else scheduler.last_rolling_tau,
            }
        )

        if scheduler is not None:
            scheduler.update(tau, cycle_idx)

        start += tau
        past_key_values_target.crop(start)
        if max_block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :tau, :]

        cycle_idx += 1

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_idx = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_idx[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(1, num_output_tokens)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        used_block_sizes=used_block_sizes,
        cycle_trace=cycle_trace,
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
    parser = argparse.ArgumentParser(description="Dynamic block-size scheduling benchmark for DFlash.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--candidate-block-sizes", type=str, default="8,12,16")
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--decision-window", type=int, default=4)
    parser.add_argument("--tau-high", type=float, default=7.0)
    parser.add_argument("--tau-mid", type=float, default=6.0)
    parser.add_argument("--required-streak", type=int, default=2)
    parser.add_argument("--cooldown-cycles", type=int, default=2)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-all-ranks", action="store_true")
    parser.add_argument("--save-outputs-path", type=str, default=None)
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

    candidates = parse_block_sizes(args.candidate_block_sizes)
    setup_log(f"pid={os.getpid()} starting; candidates={candidates}", pre_dist=True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    setup_log("loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
    setup_log(f"dataset ready with {len(dataset)} rows")

    responses_baseline = []
    responses_dynamic = []
    output_records = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            baseline_response = None
            if not args.skip_baseline:
                t_base = cuda_time()
                baseline_response = dflash_generate_policy(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    fixed_block_size=1,
                    scheduler=None,
                )
                baseline_response.wall_time_s = cuda_time() - t_base
                responses_baseline.append(baseline_response)

            scheduler = RollingTauScheduler(
                candidates=candidates,
                warmup_cycles=args.warmup_cycles,
                decision_window=args.decision_window,
                tau_high=args.tau_high,
                tau_mid=args.tau_mid,
                required_streak=args.required_streak,
                cooldown_cycles=args.cooldown_cycles,
            )
            t_dyn = cuda_time()
            dynamic_response = dflash_generate_policy(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                fixed_block_size=None,
                scheduler=scheduler,
            )
            dynamic_response.wall_time_s = cuda_time() - t_dyn
            responses_dynamic.append(dynamic_response)

            dyn_generated_ids = dynamic_response.output_ids[0, dynamic_response.num_input_tokens:]
            dyn_output_text = tokenizer.decode(dyn_generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": dyn_output_text})

            baseline_text = None
            if baseline_response is not None:
                baseline_ids = baseline_response.output_ids[0, baseline_response.num_input_tokens:]
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
                    "dynamic": {
                        "candidate_block_sizes": candidates,
                        "output_text": dyn_output_text,
                        "num_input_tokens": dynamic_response.num_input_tokens,
                        "num_output_tokens": dynamic_response.num_output_tokens,
                        "wall_time_s": dynamic_response.wall_time_s,
                        "ttft_s": dynamic_response.time_to_first_token,
                        "tpot_s": dynamic_response.time_per_output_token,
                        "acceptance_lengths": dynamic_response.acceptance_lengths,
                        "used_block_sizes": dynamic_response.used_block_sizes,
                        "cycle_trace": dynamic_response.cycle_trace,
                    },
                }
            )

    if dist.size() > 1:
        responses_dynamic = dist.gather(responses_dynamic, dst=0)
        output_records = dist.gather(output_records, dst=0)
        if not args.skip_baseline:
            responses_baseline = dist.gather(responses_baseline, dst=0)
        if not dist.is_main():
            return
        responses_dynamic = list(chain(*responses_dynamic))
        output_records = list(chain(*output_records))
        if not args.skip_baseline:
            responses_baseline = list(chain(*responses_baseline))

    dynamic_metrics = summarize_mode(responses_dynamic)
    print(f"Dynamic total_wall_s: {dynamic_metrics['total_wall_s']:.6f}")
    print(f"Dynamic avg_wall_s: {dynamic_metrics['avg_wall_s']:.6f}")
    print(f"Dynamic TTFT: {dynamic_metrics['avg_ttft_s']:.6f}")
    print(f"Dynamic TPOT: {dynamic_metrics['avg_tpot_s']:.6f}")
    print(f"Dynamic tokens_per_sec: {dynamic_metrics['tokens_per_sec']:.6f}")

    if not args.skip_baseline:
        baseline_metrics = summarize_mode(responses_baseline)
        print(f"Baseline total_wall_s: {baseline_metrics['total_wall_s']:.6f}")
        print(f"Baseline avg_wall_s: {baseline_metrics['avg_wall_s']:.6f}")
        print(f"Baseline TTFT: {baseline_metrics['avg_ttft_s']:.6f}")
        print(f"Baseline TPOT: {baseline_metrics['avg_tpot_s']:.6f}")
        print(f"Baseline tokens_per_sec: {baseline_metrics['tokens_per_sec']:.6f}")
        print(f"Decoding speedup: {baseline_metrics['avg_tpot_s'] / dynamic_metrics['avg_tpot_s']:.2f}")
    else:
        print("Decoding speedup: N/A (baseline skipped)")

    tau = np.mean([np.mean(r.acceptance_lengths) for r in responses_dynamic])
    print(f"Average Acceptance length: {tau:.2f}")

    all_acceptance = list(chain(*[r.acceptance_lengths for r in responses_dynamic]))
    max_bs = max(candidates)
    histogram = [all_acceptance.count(b) / len(all_acceptance) for b in range(max_bs + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    block_usage = Counter(chain(*[r.used_block_sizes for r in responses_dynamic]))
    block_usage_sorted = {k: block_usage[k] for k in sorted(block_usage)}
    total_cycles = float(sum(block_usage_sorted.values()))
    block_usage_pct = {k: (v / total_cycles) for k, v in block_usage_sorted.items()} if total_cycles > 0 else {}
    print(f"Dynamic block usage counts: {block_usage_sorted}")
    print(f"Dynamic block usage pct: {block_usage_pct}")

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
