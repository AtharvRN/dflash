# SGLang C=16 Policy Matrix Protocol (Static + EWMA/UCB/LinUCB)

Date: 2026-03-03  
Scope: TP=1, FlashInfer, A100, fixed GSM8K subset for apples-to-apples comparisons.

## Objective

Run one consistent matrix on the same fixed subset to compare:

- baseline
- static DFLASH block sizes: `8, 10, 12, 14, 16, 18`
- adaptive DFLASH:
  - `ewma` with rewards `{accept_length, throughput}`
  - `ucb` with rewards `{accept_length, throughput}`
  - `linucb` with rewards `{accept_length, throughput}`

## Why this protocol

- Fixing the GSM8K subset removes prompt-mix variance from throughput/tau comparisons.
- Keeping concurrency fixed (`c=16`) isolates policy effects from queueing effects.
- Capturing per-cycle trace and GPU metrics supports root-cause analysis (not just headline tok/s).

## One-command matrix run

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_TAG_PREFIX=sglang_c16_policy_matrix_$(date +%Y%m%d_%H%M%S) \
CONCURRENCY=16 \
FIXED_QUESTION_COUNT=256 \
FIXED_QUESTION_OFFSET=0 \
QUESTIONS_PER_CONCURRENCY_BASE=8 \
MAX_QUESTIONS_PER_CONFIG=256 \
MAX_RUNNING_REQUESTS=128 \
ENABLE_DFLASH_CYCLE_TRACE=1 \
ENABLE_GPU_MONITOR=1 \
ADAPTIVE_K_MIN=1 \
ADAPTIVE_K_MAX=18 \
ADAPTIVE_K_START=12 \
ADAPTIVE_BLOCK_BUCKETS=8,10,12,14,16,18 \
ADAPTIVE_RHO=0.30 \
ADAPTIVE_DELTA=1.0 \
ADAPTIVE_UCB_C=1.0 \
ADAPTIVE_UCB_DELTA=0.05 \
ADAPTIVE_LINUCB_ALPHA=1.0 \
ADAPTIVE_LINUCB_LAMBDA=1.0 \
bash run_sglang_c16_policy_matrix.sh
```

## Outputs

For each run `logs/<run_tag>/`:

- benchmark markdown: `<run_tag>.md`
- per-call trace: `<run_tag>_calls.jsonl`
- call trace summary: `<run_tag>_calls_summary.md`
- GPU metrics: `<run_tag>_gpu_metrics.csv`
- GPU metrics summary: `<run_tag>_gpu_metrics_summary.md`

Matrix-level aggregates (auto-generated):

- `logs/<RUN_TAG_PREFIX>_matrix_summary.md`
- `logs/<RUN_TAG_PREFIX>_matrix_summary.csv`

## Metrics to report in advisor update

- Throughput: `DFLASH output tok/s`, plus speedup vs baseline
- Quality/efficiency: `tau`, `accept_rate`
- Work decomposition: `verify/s`, `draft_tok/s`
- Per-cycle timing: `draft ms/cycle`, `verify ms/cycle`
- Runtime policy behavior: block-size histogram and mode
- Resource behavior: GPU mem and utilization summaries

## Story template

1. Fixed-subset setup and why it matters.
2. Best static block size and margin vs neighboring sizes.
3. EWMA/UCB/LinUCB results with both reward modes.
4. Whether adaptive closes or exceeds best static throughput.
5. Whether adaptive improves tau/accept_rate without inflating per-cycle verify cost.
6. GPU utilization/memory observations that explain wins or regressions.
