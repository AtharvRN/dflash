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

## 4) SGLang quick throughput run

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

## Suggested first experiment matrix (for scheduling signal)

Run block sweep on 3 categories:
- reasoning: `aime25`
- code: `humaneval`
- chat: `mt-bench`

Then correlate:
- `tau` vs speedup
- `tau / block_size` vs speedup

Use this to define a first scheduler threshold policy.
