# Quick Experiments for Dynamic Scheduling

These commands are designed for rapid iteration before full benchmark runs.

## 1) Very fast smoke test (Transformers backend)

```bash
FAST=1 NPROC=1 ./run_benchmark.sh
```

Override task/samples:

```bash
FAST=1 NPROC=1 TASKS="aime25:4" MAX_NEW_TOKENS=128 ./run_benchmark.sh
```

## 2) Block-size sweep on one dataset

Runs one dataset across multiple block sizes and writes a CSV summary.

```bash
DATASET=aime25 MAX_SAMPLES=8 MAX_NEW_TOKENS=256 BLOCK_SIZES="4 8 12 16" NPROC=1 ./run_block_sweep.sh
```

Output:
- `logs/sweep_*/summary.csv`
- per-block logs in the same folder
- `summary.csv` now includes hardware + absolute timing fields:
  - `gpu_name`, `cuda_version`, `torch_version`
  - `baseline_total_wall_s`, `speculative_total_wall_s`
  - `baseline_tokens_per_sec`, `speculative_tokens_per_sec`
  - `baseline_tpot`, `speculative_tpot`
  - `baseline_ttft`, `speculative_ttft`

To avoid duplicate baseline compute in every block-size run:

```bash
DATASET=aime25 MAX_SAMPLES=8 MAX_NEW_TOKENS=256 BLOCK_SIZES="4 8 12 16" NPROC=1 SKIP_BASELINE=1 ./run_block_sweep.sh
```

This runs one shared baseline pass (`bs=1`) and computes speedup for each block size using shared baseline TPOT.

## 3) Single-run explicit command (Transformers)

```bash
torchrun --nproc_per_node=1 --master_port=29600 benchmark.py \
  --dataset aime25 \
  --max-samples 8 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 256 \
  --temperature 0.0 \
  --block-size 16
```

## 4) Dynamic block-size scheduler run (inference-time only)

Uses candidates `{8,12,16}` by default and compares dynamic policy against baseline.
Scheduler is EWMA throughput-based (`score = tau_hat / cycle_hat`) with hysteresis,
cooldown, periodic probes, and low-accept fallback.

```bash
python benchmark_dynamic_schedule.py \
  --dataset aime25 \
  --max-samples 30 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 2048 \
  --candidate-block-sizes 8,12,16 \
  --warmup-cycles 2 \
  --ewma-alpha 0.10 \
  --switch-margin 0.05 \
  --required-streak 2 \
  --cooldown-cycles 2 \
  --probe-interval 12 \
  --low-accept-threshold 0.35 \
  --low-accept-streak 2 \
  --save-outputs-path outputs/dynamic_aime25.jsonl
```

## 5) SGLang quick throughput run

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python benchmark_sglang.py \
  --dataset-name aime25 \
  --target-model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --concurrencies 1,2,4 \
  --questions-per-concurrency-base 8 \
  --max-questions-per-config 64 \
  --max-new-tokens 256 \
  --output-md sglang_quick.md
```

## 6) Multi-round speculative sampling (exact accept/reject)

This implements the exact multi-round acceptance-residual algorithm:
- Sample proposal token `t_i` from round proposal distribution `q_i`
- Accept with probability `min(1, p(t_i) / q_i(t_i))`
- If rejected, update residual `p <- norm(max(0, p - q_i))`
- If all rounds reject, sample from final residual

```bash
python benchmark_multiround_spec.py \
  --dataset aime25 \
  --max-samples 8 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --round-block-sizes 16,12 \
  --save-outputs-path outputs/multiround_aime25.jsonl
```

## Suggested first experiment matrix (for scheduling signal)

Run block sweep on 3 categories:
- reasoning: `aime25`
- code: `humaneval`
- chat: `mt-bench`

Then correlate:
- `tau` vs speedup
- `tau / block_size` vs speedup

Use this to define a first scheduler threshold policy.
