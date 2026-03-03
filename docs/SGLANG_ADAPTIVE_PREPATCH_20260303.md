# SGLang DFLASH Adaptive: Pre-Patch Baseline Notes (2026-03-03)

This note records the previously observed behavior before removing hardcoded adaptive block buckets from `dflash_worker.py`.

## Scope
- Backend: `flashinfer`
- TP: `1`
- Concurrency: `16`
- Dataset: `gsm8k`
- Source runs: `pulled/cycle_trace_runs_20260303/*`

## Key Observation
- Adaptive runtime block size was effectively constrained to coarse values (not natural integer transitions), e.g. `{15,16}` or `{7,8}` depending on run setup.
- This came from hardcoded bucketization in `dflash_worker.py` (now removed).

## Representative Runs

| Run tag | Mode | Tok/s | Tau | Accept rate | Verify/s | Draft tok/s |
|---|---|---:|---:|---:|---:|---:|
| `sg_hp_sweep_20260303_070749_static_bs8` | static `bs=8` | `1545.15` | `4.975` | `0.565` | `315.78` | `2210.49` |
| `sg_hp_sweep_20260303_070749_static_bs16` | static `bs=16` | `1510.81` | `6.444` | `0.361` | `244.43` | `3666.44` |
| `sg_rho_sweep_kstart12_20260303_072551_rho030` | adaptive (`k_start=12`) | `1076.92` | `5.696` | `0.461` | `194.66` | `1979.27` |
| `sglang_dynamic_c16_20260303_062130` | adaptive (older setup) | `1326.01` | `6.363` | `0.378` | `216.13` | `3057.99` |
| `sglang_dynamic_c16_20260303_070008` | adaptive (`k_start=8`) | `981.03` | `4.960` | `0.574` | `200.24` | `1375.75` |

## Runtime Block-Size Histograms (Cycle-Weighted)

| Run tag | Histogram |
|---|---|
| `sglang_dynamic_c16_20260303_062130` | `15: 85.11%`, `16: 14.89%` |
| `sglang_dynamic_c16_20260303_070008` | `7: 12.96%`, `8: 87.04%` |
| `sg_rho_sweep_kstart12_20260303_072551_rho030` | `11: 83.21%`, `12: 16.79%` |
| `sg_hp_sweep_20260303_070749_static_bs8` | `8: 100.00%` |
| `sg_hp_sweep_20260303_070749_static_bs16` | `16: 100.00%` |

## Per-Cycle Timing (from Call-Trace Summaries)

| Run tag | Draft ms/cycle | Verify ms/cycle |
|---|---:|---:|
| `sg_hp_sweep_20260303_070749_static_bs8` | `0.198` | `1.027` |
| `sg_hp_sweep_20260303_070749_static_bs16` | `0.281` | `1.476` |
| `sg_rho_sweep_kstart12_20260303_072551_rho030` | `0.400` | `2.650` |
| `sglang_dynamic_c16_20260303_062130` | `0.329` | `2.048` |
| `sglang_dynamic_c16_20260303_070008` | `0.415` | `2.700` |

## Raw Inputs
- `pulled/cycle_trace_runs_20260303/sg_hp_sweep_20260303_070749_static_bs8/sg_hp_sweep_20260303_070749_static_bs8.md`
- `pulled/cycle_trace_runs_20260303/sg_hp_sweep_20260303_070749_static_bs8/sg_hp_sweep_20260303_070749_static_bs8_calls_summary.md`
- `pulled/cycle_trace_runs_20260303/sg_hp_sweep_20260303_070749_static_bs16/sg_hp_sweep_20260303_070749_static_bs16.md`
- `pulled/cycle_trace_runs_20260303/sg_hp_sweep_20260303_070749_static_bs16/sg_hp_sweep_20260303_070749_static_bs16_calls_summary.md`
- `pulled/cycle_trace_runs_20260303/sg_rho_sweep_kstart12_20260303_072551_rho030/sg_rho_sweep_kstart12_20260303_072551_rho030.md`
- `pulled/cycle_trace_runs_20260303/sg_rho_sweep_kstart12_20260303_072551_rho030/sg_rho_sweep_kstart12_20260303_072551_rho030_calls_summary.md`
- `pulled/cycle_trace_runs_20260303/sglang_dynamic_c16_20260303_062130/sglang_dynamic_c16_20260303_062130.md`
- `pulled/cycle_trace_runs_20260303/sglang_dynamic_c16_20260303_062130/sglang_dynamic_c16_20260303_062130_calls_summary.md`
- `pulled/cycle_trace_runs_20260303/sglang_dynamic_c16_20260303_070008/sglang_dynamic_c16_20260303_070008.md`
- `pulled/cycle_trace_runs_20260303/sglang_dynamic_c16_20260303_070008/sglang_dynamic_c16_20260303_070008_calls_summary.md`
