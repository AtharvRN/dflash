# Results Summary (Downloaded `logs/` + `outputs/`)

Generated on: 2026-02-18  
Scope: artifacts currently present in local `logs/` and `outputs/`.

## 1) Completed AIME25 Block-Size Sweeps (`max_new_tokens=256`)

### AIME25, `max_samples=8` (aggregated across completed sweeps)
| Block Size | Runs | Speedup Mean | Speedup Range | Tau Mean |
|---|---:|---:|---:|---:|
| 4  | 4 | 2.89 | 2.82 to 3.05 | 3.46 |
| 8  | 3 | 4.46 | 4.35 to 4.57 | 5.46 |
| 12 | 3 | 5.60 | 5.53 to 5.73 | 6.80 |
| 16 | 3 | 6.24 | 6.18 to 6.32 | 7.67 |
| 20 | 1 | 5.07 | 5.07 to 5.07 | 6.40 |
| 24 | 1 | 4.01 | 4.01 to 4.01 | 5.04 |

### AIME25, `max_samples=10` (one completed sweep)
| Block Size | Speedup | Tau |
|---|---:|---:|
| 4  | 2.94 | 3.40 |
| 8  | 4.32 | 5.32 |
| 12 | 5.14 | 6.46 |
| 16 | 5.63 | 7.31 |

### Interpretation
- Best fixed block size in all completed sweeps: `bs=16`.
- `bs=20` and `bs=24` are worse than `bs=16` on available runs.

## 2) Completed Full-Length AIME25 Runs on A100 (`max_samples=30`, `max_new_tokens=2048`)

| Mode | TPOT (s) | TTFT (s) | Tokens/s | Tau | Avg Output Tokens | Hit 2048 Cap |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (`bs=1`) | 0.046162 | 0.110155 | 21.654 | 1.00 | 1844.6 | 18/30 |
| Static (`bs=8`) | 0.010251 | 0.060146 | 97.113 | 5.40 | 1833.7 | 20/30 |
| Static (`bs=12`) | 0.009111 | 0.081716 | 110.458 | 6.61 | 1785.3 | 18/30 |
| Static (`bs=16`) | 0.007873 | 0.225684 | 127.653 | 7.47 | 1768.6 | 16/30 |

Derived TPOT speedup vs baseline:
- `bs=8`: `4.50x`
- `bs=12`: `5.07x`
- `bs=16`: `5.86x` (best)

## 3) Completed Dynamic-Scheduling Runs

| Run | Samples | Baseline TPOT | Dynamic TPOT | Speedup | Baseline Tokens/s | Dynamic Tokens/s | Dynamic Tau |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dynamic_aime25_smoke.jsonl` | 4  | 0.053484 | 0.013311 | 4.02x | 11.326 | 73.766 | 6.80 |
| `dynamic_aime25_smok_2048e.jsonl` | 10 | 0.047953 | 0.008854 | 5.42x | 20.792 | 111.354 | 6.75 |

Dynamic block usage (top):
- 4-sample run: `16` (54.9%), `8` (32.0%), `12` (8.5%)
- 10-sample run: `8` (54.0%), `16` (38.2%), `12` (7.4%)

## 4) Completed vs Incomplete Artifacts

### Completed (non-empty summaries or full logs)
- `logs/sweep_20260217_120717/summary.csv`
- `logs/sweep_20260217_221618/summary.csv`
- `logs/sweep_20260217_225314/summary.csv`
- `logs/sweep_20260217_235048/summary.csv` (only `bs=4`)
- `logs/sweep_20260217_235553/summary.csv`
- `logs/aime25_bs12_only.log`
- `logs/aime25_bs16_only.log`
- `logs/sweep_20260218_003245/aime25_baseline_bs1.log`
- `logs/sweep_20260218_003245/aime25_bs8.log`

### Incomplete / partial (not used for conclusions)
- Empty summary files:
  - `logs/sweep_20260217_215840/summary.csv`
  - `logs/sweep_20260217_220930/summary.csv`
  - `logs/sweep_20260218_001458/summary.csv`
  - `logs/sweep_20260218_003245/summary.csv`
- Incomplete GSM8K launch-only logs:
  - `logs/20260217_215956_gsm8k_bsauto.log`
  - `logs/20260217_220819_gsm8k_bsauto.log`
  - `logs/20260217_235013_gsm8k_bsauto.log`
  - `logs/20260217_235017_gsm8k_bsauto.log`

## 5) Current Takeaway
- For your current setup and AIME25 runs, `bs=16` is the strongest fixed choice.
- Going beyond `16` (`20`, `24`) is currently not useful.
- Dynamic scheduling is strong vs baseline, but on present runs it has not clearly beaten the best fixed `bs=16`.

## 6) EWMA Scheduler Run (A100 80GB PCIe, `max_samples=10`, `max_new_tokens=2048`)

Run:
- `outputs/dynamic_aime25_ewma.jsonl` (pulled and re-analyzed as `outputs/remote_dynamic_aime25_ewma.jsonl`)
- Flags: `candidate-block-sizes=8,12,16`, `ewma-alpha=0.10`, `switch-margin=0.05`, `required-streak=2`, `cooldown-cycles=2`, `probe-interval=12`, `low-accept-threshold=0.35`, `low-accept-streak=2`

Primary outcomes:
- Baseline total wall time: `798.54s`
- Dynamic total wall time: `143.20s`
- Wall-time speedup: `5.58x`
- Baseline output tokens: `19565`
- Dynamic output tokens: `18505` (`0.946x` baseline tokens)
- Token-normalized throughput speedup: `5.27x` (fairer than raw wall-time ratio)
- Dynamic mean acceptance length (sample-mean): `6.55`
- Cycle-weighted mean tau: `6.28`
- Mean cycle time: `0.0481s`

Dynamic block usage (cycle level):
- `bs=12`: `1219` cycles (`41.34%`)
- `bs=16`: `894` cycles (`30.32%`)
- `bs=8`: `826` cycles (`28.01%`)
- Others are negligible probe/edge cycles.

Important diagnostic:
- On cycles where `bs=16` was used, `tau < 8` occurred in `60.18%` of cycles.
- Despite this, realized cycle throughput remained strongest for `bs=16` overall in this run:
  - `bs=8`: `~104.25` tokens/s
  - `bs=12`: `~133.63` tokens/s
  - `bs=16`: `~150.81` tokens/s

Switching behavior:
- Block-switch rate: `32.15%` (`945` switches over `2939` transitions), indicating high scheduler churn.

Saved local analysis artifacts:
- `outputs/analysis_remote_dynamic_aime25_ewma/summary.md`
- `outputs/analysis_remote_dynamic_aime25_ewma/summary.json`
- `outputs/analysis_remote_dynamic_aime25_ewma/per_sample_summary.csv`
- `outputs/analysis_remote_dynamic_aime25_ewma/cycle_trace_flat.csv`
- `outputs/analysis_remote_dynamic_aime25_ewma/per_block_realized_throughput.csv`
- `outputs/analysis_remote_dynamic_aime25_ewma/tau_vs_cycle_colored_block.png`
- `outputs/analysis_remote_dynamic_aime25_ewma/rolling_tau_cycle_time.png`

## 7) Suffix-Seed Experiments (Latest Pulled from Pod)

Source files pulled from pod:
- `outputs/remote_suffix_none_aime25.jsonl`
- `outputs/remote_suffix_sparse_aime25.jsonl`
- `outputs/remote_suffix_dense_aime25.jsonl`

Important run-context note:
- Latest `sparse` and `dense` files were generated with `--skip-baseline`, so those files do not contain their own baseline rows.
- To compare all modes consistently, derived speedups for `sparse` and `dense` are normalized against the `none` baseline from the same pulled batch.

| Mode | `seed_max_tokens` | Baseline in file | Spec TPOT (s) | Spec Tokens/s | Tau | Seeded tokens/cycle | Reported Speedup | Derived Speedup vs `none` baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | -1 | yes | 0.006258 | 158.176 | 7.92 | 0.000 | 6.44x | 6.44x |
| sparse | 2 | no | 0.011713 | 84.678 | 4.27 | 1.914 | N/A | 3.44x |
| dense | -1 | no | 0.019733 | 50.720 | 2.52 | 13.442 | N/A | 2.04x |

Takeaways from pulled suffix-seed runs:
- Any suffix seeding currently degrades performance vs `none`.
- Degradation tracks acceptance collapse:
  - `none`: `tau=7.92`
  - `sparse`: `tau=4.27`
  - `dense`: `tau=2.52`
- Dense seeding is the most harmful in these runs, with very high seed intensity (`13.442` seeded tokens/cycle).

Saved local analysis artifacts:
- `outputs/analysis_remote_suffix_seed/summary.md`
- `outputs/analysis_remote_suffix_seed/summary.json`
- `outputs/analysis_remote_suffix_seed/summary_table.csv`

## 8) Latest Implementation Update (Candidate-Generation Testbed)

Added a new research benchmark script:
- `benchmark_candidate_solutions.py`

Purpose:
- test whether per-cycle draft candidate diversity can improve accepted length (`tau`) for DFlash.

Current behavior:
- run normal DFlash draft pass for a block,
- branch early uncertain positions (`--branch-depth`, optional `--branch-margin-threshold`),
- build bounded candidate set (`--branch-top-k`, `--max-candidates`),
- verify candidates with target and commit the best-by-`tau` candidate.

Status:
- implementation complete; no production speed claims yet.
- this is intended for controlled ablations against fixed-block baseline.
