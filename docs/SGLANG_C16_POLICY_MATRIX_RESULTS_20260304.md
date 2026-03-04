# SGLang C=16 Policy Matrix Results (Static + EWMA/UCB/LinUCB)

Date: 2026-03-04  
Run prefix: `sglang_c16_policy_matrix_20260304_080903`  
Setup: TP=1, FlashInfer, fixed GSM8K subset (`256` examples, offset `0`), A100.

## Artifacts

- Driver log: `logs/sglang_c16_policy_matrix_20260304_080903_driver.log`
- Matrix summary: `logs/sglang_c16_policy_matrix_20260304_080903_matrix_summary.md`
- Matrix CSV: `logs/sglang_c16_policy_matrix_20260304_080903_matrix_summary.csv`
- Per-run artifacts: `logs/sglang_c16_policy_matrix_20260304_080903_*`

## Core Comparison

Baseline reference throughput (from `baseline_and_static_bs8` run):
- baseline tok/s: `1086.34`

| Policy/Setting | DFLASH tok/s | Speedup vs baseline | tau | accept_rate | verify/s | draft_tok/s | draft ms/cycle | verify ms/cycle | Runtime block usage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Static `bs=8` | 889.63 | 0.819x | 4.938 | 0.560 | 183.36 | 1283.50 | 0.197 | 1.026 | `8:100%` |
| Static `bs=10` | 2091.74 | 1.925x | 5.468 | 0.494 | 391.40 | 3522.57 | 0.220 | 1.200 | `10:100%` |
| Static `bs=12` | 2036.99 | 1.875x | 5.790 | 0.434 | 362.59 | 3988.52 | 0.230 | 1.275 | `12:100%` |
| Static `bs=14` | 1967.30 | 1.811x | 6.099 | 0.390 | 329.99 | 4289.87 | 0.247 | 1.376 | `14:100%` |
| Static `bs=16` | **2245.87** | **2.067x** | **6.376** | 0.357 | 362.85 | 5442.70 | 0.249 | 1.387 | `16:100%` |
| Static `bs=18` | 1913.78 | 1.762x | 6.149 | 0.302 | 322.37 | 5480.30 | 0.260 | 1.519 | `18:100%` |
| EWMA (`accept_length`) | 2098.46 | 1.932x | 4.982 | 0.551 | 428.56 | 3093.03 | 0.200 | 1.039 | `8:94.57%, 12:5.43%` |
| EWMA (`throughput`) | 2105.01 | 1.938x | 4.987 | 0.552 | 429.60 | 3095.96 | 0.208 | 1.075 | `8:94.84%, 12:5.16%` |
| UCB (`accept_length`) | **2114.82** | **1.947x** | 5.859 | 0.387 | 369.48 | 4644.46 | 0.234 | 1.307 | `8:22.79%, 12:26.10%, 16:29.26%, 18:21.85%` |
| UCB (`throughput`) | 1937.41 | 1.783x | 5.782 | 0.406 | 342.59 | 4036.17 | 0.237 | 1.291 | `8:29.96%, 12:28.51%, 16:25.63%, 18:15.90%` |
| LinUCB (`accept_length`) | 1800.85 | 1.658x | 5.007 | 0.530 | 365.48 | 2782.66 | 0.199 | 1.042 | `8:91.38%, 12:3.18%, 16:2.85%, 18:2.58%` |
| LinUCB (`throughput`) | 1943.07 | 1.789x | 5.528 | 0.453 | 359.39 | 3613.36 | 0.216 | 1.170 | `8:50.01%, 12:26.67%, 16:17.21%, 18:6.11%` |

## Ranking

Top throughput:
1. Static `bs=16`: `2245.87 tok/s`
2. UCB (`accept_length`): `2114.82 tok/s`
3. EWMA (`throughput`): `2105.01 tok/s`
4. EWMA (`accept_length`): `2098.46 tok/s`
5. Static `bs=10`: `2091.74 tok/s`

Adaptive best (`UCB accept_length`) vs static best (`bs=16`):
- absolute gap: `131.05 tok/s`
- relative gap: `5.84%` below static best

## Interpretation

- For this fixed subset and hardware, the best point is still a static block size (`bs=16`).
- EWMA is stable but mostly collapses to low block sizes (`8/12`), yielding good verify/s but lower tau.
- UCB with `accept_length` reward explores all block sizes and is the strongest adaptive policy in this run.
- LinUCB underperforms here, mainly because it over-selects lower block sizes in this configuration.

## Notes

- `bs=8` is weak in this matrix because the draft model is `b16`; forcing small runtime block size can reduce speculative efficiency.
- Per-cycle draft/verify time increases with larger static block sizes, but throughput still depends on how much tau improves.
- GPU monitor scripts were missing from git during this matrix run, so per-run GPU CSV/summary files were not generated for `sglang_c16_policy_matrix_20260304_080903`.
  - Fixed in commit `b18738b` by adding:
    - `scripts/record_gpu_metrics.sh`
    - `scripts/summarize_gpu_metrics.py`
  - Verified via smoke run:
    - `logs/sg_gpu_metrics_smoke_20260304_084123/sg_gpu_metrics_smoke_20260304_084123_gpu_metrics.csv`
    - `logs/sg_gpu_metrics_smoke_20260304_084123/sg_gpu_metrics_smoke_20260304_084123_gpu_metrics_summary.md`
