# SGLang GSM8K TP=1 Block-Size x Concurrency Sweep (2026-03-02)

## Run Setup
- Run tags: `sg_tp1_bsweep_g0_20260302_024551`, `sg_tp1_bsweep_g1_20260302_024701`
- Source summaries: `pulled/sglang_sweeps_20260302_full/sg_tp1_bsweep_g0_20260302_024551/summary.csv`, `pulled/sglang_sweeps_20260302_full/sg_tp1_bsweep_g1_20260302_024701/summary.csv`
- Combined summary: `pulled/sglang_sweeps_20260302_full/combined_summary.csv`
- Dataset: `gsm8k`
- Attention backend: `flashinfer`
- Tensor parallel size: `1`
- Speculative algorithm: `DFLASH`
- Block sizes swept: `8,12,16,20`
- Concurrency swept: `1,2,4,8,16,32`
- Questions per config: `n=min(32*conc,256)`
- Status: `24/24 configs completed (all OK)`

## Dataset Coverage Per Config
| Concurrency | Measured questions `n` | Call rows in trace |
|---:|---:|---:|
| 1 | 32 | 64 |
| 2 | 64 | 128 |
| 4 | 128 | 256 |
| 8 | 256 | 512 |
| 16 | 256 | 512 |
| 32 | 256 | 512 |

## Headline Results
- Best speedup: `x3.519` at `bs=16, c=1`
- Best speculative throughput: `3826.57 tok/s` at `bs=8, c=32`
- Best tau: `6.507` at `bs=16, c=32`

## Best Config Per Concurrency

| Concurrency | Best by speedup (bs, speedup, spec tok/s, tau) | Best by spec tok/s (bs, spec tok/s, speedup, tau) |
|---:|---|---|
| 1 | `bs=16`, `x3.519`, `490.25`, `tau=6.225` | `bs=16`, `490.25`, `x3.519`, `tau=6.225` |
| 2 | `bs=16`, `x3.455`, `775.92`, `tau=6.323` | `bs=16`, `775.92`, `x3.455`, `tau=6.323` |
| 4 | `bs=16`, `x3.105`, `1280.82`, `tau=6.415` | `bs=16`, `1280.82`, `x3.105`, `tau=6.415` |
| 8 | `bs=16`, `x2.860`, `1878.51`, `tau=6.421` | `bs=16`, `1878.51`, `x2.860`, `tau=6.421` |
| 16 | `bs=8`, `x2.535`, `2895.37`, `tau=4.979` | `bs=8`, `2895.37`, `x2.535`, `tau=4.979` |
| 32 | `bs=8`, `x2.195`, `3826.57`, `tau=4.982` | `bs=8`, `3826.57`, `x2.195`, `tau=4.982` |

## Block-Size Averages Across Concurrency

| Block size | Avg speedup | Avg spec tok/s | Avg tau | Avg accept rate | Avg spec e2e (s) | Avg wall/verify (s) | Avg draft time (s) | Avg verify time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2.603 | 1772.84 | 4.935 | 0.559 | 0.956 | 0.004915 | 0.036060 | 0.193929 |
| 12 | 2.733 | 1781.05 | 5.798 | 0.434 | 0.947 | 0.005195 | 0.032426 | 0.175688 |
| 16 | 2.910 | 1834.81 | 6.391 | 0.358 | 0.912 | 0.005329 | 0.030254 | 0.165771 |
| 20 | 2.283 | 1388.54 | 5.427 | 0.232 | 1.220 | 0.005641 | 0.037702 | 0.210079 |

## Full Per-Config Results

| bs | conc | n | baseline tok/s | spec tok/s | speedup | tau | accept rate | verify/s | draft tok/s | accepted tok/s | wall/verify (s) | spec e2e avg (s) | draft time avg (s) | verify time avg (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 32 | 139.50 | 400.49 | 2.871 | 4.834 | 0.545 | 84.45 | 591.15 | 314.72 | 0.011841 | 0.757 | 0.089590 | 0.482386 |
| 8 | 2 | 64 | 240.94 | 639.57 | 2.655 | 4.895 | 0.554 | 133.83 | 936.84 | 503.68 | 0.007472 | 0.823 | 0.055985 | 0.296894 |
| 8 | 4 | 128 | 411.29 | 1086.62 | 2.642 | 4.953 | 0.562 | 224.22 | 1569.57 | 858.79 | 0.004460 | 0.822 | 0.032019 | 0.172048 |
| 8 | 8 | 256 | 657.99 | 1788.39 | 2.718 | 4.969 | 0.564 | 364.82 | 2553.71 | 1417.56 | 0.002741 | 0.867 | 0.019039 | 0.102435 |
| 8 | 16 | 256 | 1142.06 | 2895.37 | 2.535 | 4.979 | 0.566 | 590.14 | 4130.98 | 2295.33 | 0.001695 | 1.017 | 0.011328 | 0.061847 |
| 8 | 32 | 256 | 1743.22 | 3826.57 | 2.195 | 4.982 | 0.566 | 781.24 | 5468.67 | 3032.31 | 0.001280 | 1.448 | 0.008400 | 0.047961 |
| 12 | 1 | 32 | 139.31 | 457.34 | 3.283 | 5.701 | 0.426 | 82.96 | 912.53 | 372.88 | 0.012054 | 0.663 | 0.078170 | 0.419649 |
| 12 | 2 | 64 | 240.87 | 713.96 | 2.964 | 5.727 | 0.428 | 129.00 | 1418.98 | 582.63 | 0.007752 | 0.736 | 0.049186 | 0.262268 |
| 12 | 4 | 128 | 410.01 | 1198.82 | 2.924 | 5.828 | 0.437 | 212.54 | 2337.92 | 982.26 | 0.004705 | 0.728 | 0.028321 | 0.152948 |
| 12 | 8 | 256 | 663.04 | 1850.03 | 2.790 | 5.825 | 0.437 | 327.92 | 3607.10 | 1515.94 | 0.003050 | 0.822 | 0.018435 | 0.100440 |
| 12 | 16 | 256 | 1213.19 | 2737.26 | 2.256 | 5.848 | 0.439 | 480.83 | 5289.13 | 2247.11 | 0.002080 | 1.099 | 0.011907 | 0.068064 |
| 12 | 32 | 256 | 1708.89 | 3728.86 | 2.182 | 5.860 | 0.440 | 653.41 | 7187.49 | 3062.78 | 0.001530 | 1.632 | 0.008534 | 0.050759 |
| 16 | 1 | 32 | 139.33 | 490.25 | 3.519 | 6.225 | 0.347 | 82.13 | 1231.92 | 406.51 | 0.012176 | 0.618 | 0.072517 | 0.391509 |
| 16 | 2 | 64 | 224.60 | 775.92 | 3.455 | 6.323 | 0.353 | 127.72 | 1915.75 | 645.67 | 0.007830 | 0.670 | 0.044808 | 0.241502 |
| 16 | 4 | 128 | 412.54 | 1280.82 | 3.105 | 6.415 | 0.359 | 207.32 | 3109.82 | 1069.18 | 0.004823 | 0.675 | 0.026240 | 0.142737 |
| 16 | 8 | 256 | 656.73 | 1878.51 | 2.860 | 6.421 | 0.360 | 306.16 | 4592.34 | 1566.13 | 0.003266 | 0.789 | 0.018282 | 0.100867 |
| 16 | 16 | 256 | 1199.45 | 2876.60 | 2.398 | 6.458 | 0.362 | 460.23 | 6903.39 | 2406.54 | 0.002173 | 1.076 | 0.011199 | 0.065820 |
| 16 | 32 | 256 | 1745.54 | 3706.76 | 2.124 | 6.507 | 0.366 | 585.49 | 8782.30 | 3108.71 | 0.001708 | 1.642 | 0.008477 | 0.052190 |
| 20 | 1 | 32 | 139.51 | 405.27 | 2.905 | 5.331 | 0.227 | 78.81 | 1497.32 | 325.12 | 0.012689 | 0.743 | 0.087545 | 0.472721 |
| 20 | 2 | 64 | 241.31 | 643.47 | 2.667 | 5.382 | 0.230 | 124.34 | 2362.43 | 517.07 | 0.008043 | 0.805 | 0.055308 | 0.301516 |
| 20 | 4 | 128 | 411.26 | 1036.22 | 2.520 | 5.442 | 0.233 | 198.14 | 3764.71 | 834.64 | 0.005047 | 0.839 | 0.033121 | 0.183913 |
| 20 | 8 | 256 | 658.06 | 1520.33 | 2.310 | 5.438 | 0.233 | 289.14 | 5493.61 | 1226.07 | 0.003459 | 1.070 | 0.022051 | 0.128233 |
| 20 | 16 | 256 | 1161.61 | 2036.82 | 1.753 | 5.478 | 0.235 | 381.30 | 7244.66 | 1648.69 | 0.002623 | 1.558 | 0.016216 | 0.098410 |
| 20 | 32 | 256 | 1740.67 | 2689.14 | 1.545 | 5.490 | 0.235 | 503.69 | 9570.15 | 2176.43 | 0.001985 | 2.308 | 0.011971 | 0.075683 |

## Notes
- `draft_time_avg_s` and `verify_time_avg_s` are populated for all DFLASH rows in this run.
- Lower block sizes (8/12) give better acceptance rate; block size 16 gives the strongest speedup at low concurrency.
- Block size 20 underperforms in speedup despite decent tau, consistent with higher verification burden per accepted token.

## Per-Cycle Timing vs Block Size
- Plot: `outputs/analysis_sglang_tp1_bsz_sweep/per_cycle_timing_vs_block_by_concurrency.png`
- Table: `outputs/analysis_sglang_tp1_bsz_sweep/per_cycle_timing_vs_block_by_concurrency.csv`
- Generation command:
  - `python scripts/plot_sglang_per_cycle_timing.py --input-root pulled/sglang_sweeps_20260302_full --output-dir outputs/analysis_sglang_tp1_bsz_sweep`

![Per-cycle draft/verify timing vs block size](../outputs/analysis_sglang_tp1_bsz_sweep/per_cycle_timing_vs_block_by_concurrency.png)

### Quick Takeaway
- This plot is **per-request attributed** cycle timing (shared batch time divided across active requests).
- At fixed concurrency, increasing block size (`8 -> 12 -> 16 -> 20`) increases both `draft_time_per_cycle` and `verify_time_per_cycle`.
- The verify-time slope is steeper than draft-time, especially at higher concurrency.

## Batch-Cycle Timing (Estimated)
- Plot: `outputs/analysis_sglang_tp1_bsz_sweep/batch_cycle_timing_est_vs_block_by_concurrency.png`
- Interpretation:
  - This is the estimated **batch** time per cycle in ms (`per_request_cycle_ms * concurrency`).
  - Use this when comparing how long one draft/verify cycle takes for the full active batch.

![Estimated batch-cycle draft/verify timing vs block size](../outputs/analysis_sglang_tp1_bsz_sweep/batch_cycle_timing_est_vs_block_by_concurrency.png)

## Tau And Speedup vs Block Size
- Tau plot: `outputs/analysis_sglang_tp1_bsz_sweep/tau_vs_block_by_concurrency.png`
- Speedup plot: `outputs/analysis_sglang_tp1_bsz_sweep/speedup_vs_block_by_concurrency.png`

![Tau vs block size by concurrency](../outputs/analysis_sglang_tp1_bsz_sweep/tau_vs_block_by_concurrency.png)

![Speedup vs block size by concurrency](../outputs/analysis_sglang_tp1_bsz_sweep/speedup_vs_block_by_concurrency.png)

### Dynamic Block Size Hypothesis
- Hypothesis: dynamic block sizing helps at high concurrency.
- From this sweep, that is directionally supported, but likely a moderate gain, not a dramatic one.
- At low-mid concurrency (`c <= 8`), static `bs=16` is already best for speedup.
- At high concurrency (`c=16,32`), best static shifts to `bs=8`.
  - `c=16`: best speedup `2.535` at `bs=8` vs `2.398` at `bs=16` (`+5.7%`)
  - `c=32`: best speedup `2.195` at `bs=8` vs `2.124` at `bs=16` (`+3.3%`)
- Important nuance:
  - Tau alone increases with larger block sizes even at high concurrency, but speedup does not.
  - The batch-cycle draft/verify costs rise enough that they can outweigh tau gains.

## Total Batch Timing (Not Per-Prompt Averaged)
- Plot: `outputs/analysis_sglang_tp1_bsz_sweep/total_batch_timing_vs_block_by_concurrency.png`
- Interpretation:
  - This is the run-level total draft/verify compute time per config (seconds), reconstructed by summing request-side attributed times.
  - Use this view when you want "actual time spent drafting/verifying" for the whole run, not per-request/per-cycle averages.

![Total batch draft/verify time vs block size](../outputs/analysis_sglang_tp1_bsz_sweep/total_batch_timing_vs_block_by_concurrency.png)

## Derived Timing Diagnostics
- Definitions:
  - `verify/draft ratio = verify_time_avg_s / draft_time_avg_s`
  - `accounted_frac = (draft_time_avg_s + verify_time_avg_s) / spec_e2e_avg_s`
  - `missing_time_avg_s = spec_e2e_avg_s - (draft_time_avg_s + verify_time_avg_s)`

### By Concurrency (averaged over block sizes)
| Concurrency | verify/draft ratio | accounted frac | missing time (s) | e2e avg (s) |
|---:|---:|---:|---:|---:|
| 1 | 5.388 | 0.753 | 0.172 | 0.695 |
| 2 | 5.369 | 0.431 | 0.432 | 0.758 |
| 4 | 5.442 | 0.252 | 0.573 | 0.766 |
| 8 | 5.540 | 0.144 | 0.760 | 0.887 |
| 16 | 5.780 | 0.072 | 1.101 | 1.188 |
| 32 | 6.034 | 0.038 | 1.692 | 1.757 |

### By Block Size (averaged over concurrencies)
| Block size | verify/draft ratio | accounted frac | missing time (s) | e2e avg (s) |
|---:|---:|---:|---:|---:|
| 8 | 5.435 | 0.281 | 0.726 | 0.956 |
| 12 | 5.536 | 0.279 | 0.739 | 0.947 |
| 16 | 5.630 | 0.281 | 0.716 | 0.912 |
| 20 | 5.768 | 0.285 | 0.973 | 1.221 |

### Does `verify + draft` equal E2E?
- No. `verify_time_avg_s + draft_time_avg_s` does **not** fully explain request E2E latency.
- Missing components include: prefill, scheduling/queueing overlap effects, request orchestration, non-draft target work, and transport/serving overhead.
- In this sweep, the accounted fraction drops strongly with concurrency, so most E2E time at high concurrency is outside request-attributed draft/verify compute.

## Adaptive Follow-Up (Same Day, c=16)

### Run A: adaptive with `k_max=16` (prior)
- Run tag: `sg_adaptive_c16_20260302_203536`
- Baseline output tok/s: `983.18`
- DFLASH output tok/s: `1346.38`
- Speedup: `1.369x`
- Tau: `6.280`
- Acceptance rate: `0.372`
- Verify/s: `223.66`
- E2E avg latency: `1.441s`

### Run B: adaptive with start-at-8 behavior (effective cap at 8 in that run)
- Run tag: `sg_adaptive_c16_bs8_20260302_214001`
- DFLASH output tok/s: `1125.08` (baseline skipped)
- Tau: `4.732`
- Acceptance rate: `0.598`
- Verify/s: `240.89`
- E2E avg latency: `1.845s`
- Exact runtime block-size cycle histogram:
  - `7`: `6064` cycles (`76.07%`)
  - `8`: `1908` cycles (`23.93%`)
- Request-level behavior:
  - `128/128` requests had mixed runtime block sizes (`7` and `8`)

### Interpretation
- Lower start/effective block sizes increased acceptance rate but reduced end-to-end throughput in this setup.
- Higher tau in the `k_max=16` run still translated to better output tok/s and latency.
- This motivated adding a **separate adaptive `k_start` control**, so we can test:
  - `block_size=16`, `k_max=16`, `k_start=8`
  - instead of forcing `k_max <= 8` when starting at 8.

## New Adaptive Knob Added
- New server flag: `--speculative-dflash-adaptive-k-start`
- Purpose: separate initial block size from max block size.
- Constraints:
  - `k_min <= k_start <= k_max`
  - `k_max <= speculative_dflash_block_size`

Example:
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
ADAPTIVE_LOW_ACCEPT_THRESHOLD=0.35 \
ADAPTIVE_LOW_ACCEPT_STREAK=2 \
bash run_sglang_dynamic_c16.sh
```
