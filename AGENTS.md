# DFlash Agent Handoff (2026-03-02)

## Current Working Branches
- Superproject branch: `dflash-flex-experiments`
- Submodule path: `third_party/sglang`
- Submodule URL: `https://github.com/AtharvRN/sglang`
- Submodule branch used for DFLASH changes: `pr16818-flex-experiments`

## Pull + Setup (fresh session)
```bash
cd /workspace/DSC291/dflash
git pull origin dflash-flex-experiments
git submodule sync --recursive
git submodule update --init --recursive

# Ensure CLI uses local SGLang with latest DFLASH flags.
pip install -e ./third_party/sglang/python --no-deps
```

Sanity check:
```bash
sglang serve --help | grep speculative-dflash-adaptive-k-start
```

## DFLASH Adaptive Controls (important)
- `--speculative-dflash-block-size`:
  - Sets DFLASH configured max block size.
  - Also drives `speculative_num_draft_tokens` and draft/verify memory/capture sizing.
- `--speculative-dflash-adaptive-k-min`, `--speculative-dflash-adaptive-k-max`:
  - Runtime adaptive range.
  - Must satisfy `1 <= k_min <= k_max <= speculative_dflash_block_size`.
- `--speculative-dflash-adaptive-k-start`:
  - Initial runtime block size per request.
  - Must satisfy `k_min <= k_start <= k_max`.
  - Lets us run `k_max=16` but start each request at `k_start=8`.
- `--speculative-dflash-adaptive-block-buckets`:
  - Optional runtime bucket set for adaptive mode (e.g. `8 12 16`).
  - For FlashInfer + CUDA graph, use explicit buckets to keep replay on captured shapes.
  - If unset, server auto-picks `{8,12,16}` intersected with `[k_min, k_max]`.
- Less brittle adaptive knobs (hysteresis + cooldown):
  - `--speculative-dflash-adaptive-low-accept-threshold`, `--speculative-dflash-adaptive-low-accept-streak`:
    - Downshift gate on EWMA acceptance ratio.
  - `--speculative-dflash-adaptive-high-accept-threshold`, `--speculative-dflash-adaptive-high-accept-streak`:
    - Upshift gate on EWMA acceptance ratio.
  - `--speculative-dflash-adaptive-cooldown-cycles`:
    - Hold cycles after block-size changes to reduce oscillation.

## Fixed Subset Mode (apples-to-apples)
- Benchmark now supports:
  - `--fixed-question-count N`
  - `--fixed-question-offset O`
- When `N > 0`, every config/concurrency uses the exact same eval window
  `dataset[O : O+N]`.
- Warmup prompts are sourced from a separate prompt pool so drop-first-batch
  no longer shifts which eval prompts are measured across concurrencies.
- Runner env wiring:
  - `run_sglang_tp1_sweep.sh`: `FIXED_QUESTION_COUNT`, `FIXED_QUESTION_OFFSET`
  - `run_sglang_dynamic_c16.sh`: `FIXED_QUESTION_COUNT`, `FIXED_QUESTION_OFFSET`

## Key Scripts
- Benchmark: `benchmark_sglang.py`
- C=16 adaptive launcher: `run_sglang_dynamic_c16.sh`
- Call-trace summarizer: `scripts/summarize_sglang_calls.py`

## Recommended C=16 Run (start at 8, allow up to 16)
```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_BASELINE=0 \
RUN_TAG=sg_adaptive_c16_kstart8_$(date +%Y%m%d_%H%M%S) \
CONCURRENCY=16 \
QUESTIONS_PER_CONCURRENCY_BASE=8 \
MAX_QUESTIONS_PER_CONFIG=256 \
DFLASH_BLOCK_SIZE=16 \
ADAPTIVE_ENABLED=1 \
ADAPTIVE_RHO=0.30 \
ADAPTIVE_DELTA=1.0 \
ADAPTIVE_K_MIN=1 \
ADAPTIVE_K_MAX=16 \
ADAPTIVE_K_START=8 \
ADAPTIVE_BLOCK_BUCKETS=8,12,16 \
ADAPTIVE_LOW_ACCEPT_THRESHOLD=0.35 \
ADAPTIVE_LOW_ACCEPT_STREAK=2 \
ADAPTIVE_HIGH_ACCEPT_THRESHOLD=0.90 \
ADAPTIVE_HIGH_ACCEPT_STREAK=2 \
ADAPTIVE_COOLDOWN_CYCLES=1 \
FIXED_QUESTION_COUNT=256 \
FIXED_QUESTION_OFFSET=0 \
bash run_sglang_dynamic_c16.sh
```

## Where Outputs Go
- Markdown report: `logs/<run_tag>/<run_tag>.md`
- Per-call JSONL: `logs/<run_tag>/<run_tag>_calls.jsonl`
- Derived summary: `logs/<run_tag>/<run_tag>_calls_summary.md`
- Raw log: `logs/<run_tag>/<run_tag>.log`

## Call-Trace Fields To Use
- `spec_accept_rate`
- `spec_accept_length`
- `spec_verify_ct`
- `spec_draft_token_num`
- `draft_time_s`, `verify_time_s`
- `draft_time_per_cycle_s`, `verify_time_per_cycle_s`
- `spec_runtime_bs_hist` (exact runtime block-size cycle histogram per request)
- `spec_runtime_bs_mode`
- `spec_runtime_bs_avg`
- `spec_cycle_trace[*].adaptive_decision`:
  - includes per-cycle `prev_bs`, `next_bs`, `action`, `reason`,
    `accept_ratio`, `accept_ratio_ewma`, and streak/cooldown counters.

## Known Failure Modes
- Unknown adaptive CLI args:
  - Reinstall editable SGLang (`pip install -e ./third_party/sglang/python --no-deps`).
- `ModuleNotFoundError: imageio` after editable install:
  - `pip install imageio`
- `libnuma.so.1` missing:
  - install `libnuma1` in pod image/env.
- FlashInfer CUDA-graph mismatch (e.g. `qo_indptr ... cannot exceed ... set during initialization`):
  - Ensure bucketed adaptive flags are wired through benchmark/server.
  - Use `--speculative-dflash-adaptive-block-buckets` (or `ADAPTIVE_BLOCK_BUCKETS` in runner).
