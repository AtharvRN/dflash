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

## 9) Candidate-On vs Vanilla (Pulled from Pod, AIME25 `max_samples=30`, `max_new_tokens=512`, `bs=16`)

Source files pulled from pod:
- `outputs/remote_aime25_bs16_vanilla_profiled.jsonl`
- `outputs/remote_aime25_bs16_vanilla_cycle.jsonl`
- `outputs/remote_aime25_bs16_candidate_on.jsonl`
- `outputs/remote_aime25_bs16_candidate_on_cycle.jsonl`

Aggregate comparison:

| Mode | Total Wall (s) | Avg Wall (s) | TTFT (s) | TPOT (s) | Tokens/s | Tau (sample-mean) |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla `benchmark.py` | 128.027446 | 4.267582 | 0.138115 | 0.008027 | 119.974275 | 6.84 |
| Candidate-on `benchmark_candidate_solutions.py` | 130.420388 | 4.347346 | 0.054765 | 0.008261 | 117.772997 | 7.20 |

Cycle-level comparison:
- Vanilla cycles: `2387` (cycle-mean tau `6.43`, median tau `5`)
- Candidate-on cycles: `2284` (cycle-mean tau `6.73`, median tau `5`)
- Candidate-on mean candidates/cycle from trace: `2.187`
- Candidate-on reported avg candidates/cycle: `2.078`
- Candidate-on reported avg verify-calls/sample: `76.1`

Delta (Candidate-on minus Vanilla):
- Tau (sample-mean): `+0.355` (`+5.18%`)
- TPOT: `+0.000234s` (`+2.91%`, slower)
- Tokens/s: `-2.201` (`-1.83%`)
- Total wall time: `+2.393s` (`+1.87%`)

Interpretation:
- Candidate generation improves acceptance (`tau`) as intended.
- At current settings (`branch_depth=6`, `top_k=2`, `max_candidates=4`), the extra verify/branch overhead still outweighs the tau gain, so end-to-end throughput is slightly worse than vanilla.
- Next tuning target is to reduce average candidates per cycle (more selective branching) while preserving most of the tau gain.

Saved local analysis artifacts:
- `outputs/analysis_remote_aime25_bs16_candidate_vs_vanilla/summary.json`
- `outputs/analysis_remote_aime25_bs16_candidate_vs_vanilla/summary.md`

## 10) Plot Pack: Vanilla vs Candidate-Off vs Candidate-On (`max_new_tokens=512`, `bs=16`)

Matched-file set used for plotting:
- Vanilla sample/profile: `outputs/remote_aime25_bs16_vanilla_profiled.jsonl`
- Vanilla cycle trace: `outputs/remote_aime25_bs16_vanilla_cycle.jsonl`
- Candidate-off sample/profile: `outputs/remote_full_aime25_bs16_candidate_off.jsonl`
- Candidate-off cycle trace: `outputs/remote_full_aime25_bs16_candidate_off_cycle.jsonl`
- Candidate-on sample/profile: `outputs/remote_full_aime25_bs16_candidate_on.jsonl`
- Candidate-on cycle trace: `outputs/remote_full_aime25_bs16_candidate_on_cycle.jsonl`

Summary metrics (30 samples each):

| Mode | Avg Tokens | Cycles | Avg Wall (s) | Avg TPOT (s) | Tokens/s | Sample-mean Tau | Cycle-mean Tau |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 512.00 | 2387 | 4.267582 | 0.008027 | 119.974275 | 6.843090 | 6.434855 |
| Candidate-off | 512.00 | 2387 | 4.510838 | 0.008420 | 113.504399 | 6.843090 | 6.434855 |
| Candidate-on | 512.00 | 2284 | 4.463596 | 0.008068 | 114.705726 | 7.197762 | 6.725044 |

Interpretation:
- Candidate-on increases acceptance (`tau`) relative to vanilla.
- Throughput is still lower than vanilla at current settings.
- Candidate-off isolates overhead: turning on candidate machinery without branching is significantly slower than vanilla.

Generated plots:
- `outputs/analysis_remote_aime25_candidate_plots_512/throughput_tpot_bars.png`
- `outputs/analysis_remote_aime25_candidate_plots_512/sample_tau_boxplot.png`
- `outputs/analysis_remote_aime25_candidate_plots_512/cycle_tau_hist_overlay.png`
- `outputs/analysis_remote_aime25_candidate_plots_512/cycle_tau_cdf.png`
- `outputs/analysis_remote_aime25_candidate_plots_512/rolling_cycle_tau.png`
- `outputs/analysis_remote_aime25_candidate_plots_512/candidate_count_distribution.png`
- `outputs/analysis_remote_aime25_candidate_plots_512/sample_walltime_vs_tau.png`
- `outputs/analysis_remote_aime25_candidate_plots_512/vanilla_profile_decode_breakdown_boxplot.png`

Saved analysis files:
- `outputs/analysis_remote_aime25_candidate_plots_512/summary_metrics.csv`
- `outputs/analysis_remote_aime25_candidate_plots_512/README.md`

## 11) Latest Matched AA10 Runs (10 samples, `bs=16`, `max_new_tokens=512`)

Source logs:
- `logs/aa10_vanilla_bs16.log`
- `logs/aa10_candidate_off_bs16.log`
- `logs/aa10_candidate_on_bs16.log`

Source outputs:
- `outputs/aa10_vanilla_bs16_profiled.jsonl`
- `outputs/aa10_candidate_off_bs16_profiled.jsonl`
- `outputs/aa10_candidate_on_bs16_profiled.jsonl`
- cycle traces: corresponding `*_cycle.jsonl`

Fresh local analysis bundle:
- `outputs/analysis_aa10_candidate_compare/summary.md`
- `outputs/analysis_aa10_candidate_compare/summary_metrics.csv`
- `outputs/analysis_aa10_candidate_compare/speedup_vs_vanilla.csv`
- `outputs/analysis_aa10_candidate_compare/steady_state_drop_first_sample.csv`
- `outputs/analysis_aa10_candidate_compare/cycle_summary.csv`

Full-run aggregate (includes startup effects):

| Mode | Total Wall (s) | Avg Wall (s) | TTFT (s) | TPOT (s) | Tokens/s | Mean Tau | Avg Draft Decode (s/sample) | Avg Target Decode (s/sample) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 78.8840 | 7.8884 | 3.1446 | 0.008660 | 64.9054 | 7.0816 | 0.8699 | 3.2219 |
| Candidate-Off | 65.3856 | 6.5386 | 1.6966 | 0.008225 | 78.3047 | 7.0816 | 0.8115 | 3.2488 |
| Candidate-On | 71.4976 | 7.1498 | 1.6426 | 0.008815 | 71.6108 | 7.1210 | 1.3445 | 3.2743 |

Steady-state (drop first sample to reduce startup distortion):

| Mode | Avg TTFT (s) | Avg TPOT (s) | Decode Tokens/s |
|---|---:|---:|---:|
| Vanilla | 0.1117 | 0.007285 | 137.2659 |
| Candidate-Off | 0.1132 | 0.007481 | 133.6688 |
| Candidate-On | 0.0447 | 0.007570 | 132.1054 |

Cycle-level view:
- Vanilla mean tau: `7.0816`, mean draft/cycle: `0.01203s`, mean verify/cycle: `0.04456s`.
- Candidate-On mean tau: `7.1210` (`+0.0394`), mean draft/cycle: `0.01870s`, mean verify/cycle: `0.04554s`.
- Candidate-On mean candidates/cycle: `~2.09`.

Interpretation:
- Candidate-On improves tau slightly.
- The added branching/candidate logic increases draft-side cycle cost materially.
- On steady-state metrics, Candidate-On remains slower than Vanilla at current settings.

## 12) Dense Fixed Block Sweep (`bs=9..16`, latest run)

Run:
- `logs/quick_bs_compare_20260224_220740/summary.csv`
- per-block logs in `logs/quick_bs_compare_20260224_220740/`
- per-sample outputs in `outputs/quick_bs_compare_20260224_220740_aime25_bs*.jsonl`

Config:
- Dataset: `aime25`
- `max_samples=5`
- `max_new_tokens=2048`
- `temperature=0.0`
- `skip_baseline=1`

Decode metrics:

| Block Size | Tau | TPOT (s) | Tokens/s | Total Wall (s) | TTFT (s) |
|---|---:|---:|---:|---:|---:|
| 9  | 5.86 | 0.008819 | 111.043 | 84.643 | 0.272130 |
| 10 | 6.16 | 0.008246 | 118.937 | 79.235 | 0.232203 |
| 11 | 6.45 | 0.007926 | 123.932 | 75.840 | 0.161983 |
| 12 | 6.71 | 0.007804 | 126.307 | 74.414 | 0.167105 |
| 13 | 6.97 | 0.007321 | 134.419 | 69.923 | 0.160631 |
| 14 | 7.38 | 0.006980 | 141.636 | 66.960 | 0.166662 |
| 15 | 7.45 | 0.006947 | 142.692 | 66.416 | 0.153220 |
| 16 | 7.28 | 0.007153 | 138.724 | 68.438 | 0.115137 |

Takeaways:
- Best throughput in this run is `bs=15` (142.692 tokens/s).
- `bs=14` and `bs=15` both outperform `bs=16` on TPOT/tokens-per-second.
- Tau increases from `bs=9` to `bs=15`, then dips at `bs=16`.

Startup overhead note (same run):
- Import and dataset load were heavily cold-start sensitive on early jobs in the sweep.
- Import time to `all imports finished` dropped from ~`387s` (`bs=9`) to ~`4.4s` (`bs=16`) as cache warmed.
- This affects end-to-end wall-clock for scripted sweeps, but does not change steady-state decode comparisons above.

Profiling caveat:
- This sweep did not use `--collect-profile`, so target vs draft decode breakdown is not available for these exact runs.
- For target/draft timing by block size, rerun the same sweep with `--collect-profile`.

Local analysis artifacts for this run:
- `logs/quick_bs_compare_20260224_220740/analysis.md`
- `logs/quick_bs_compare_20260224_220740/metrics_vs_bs.png`
- `logs/quick_bs_compare_20260224_220740/startup_overhead.png`

## 13) AIME25 Profiled Sweep (`bs=12..19`, `max_samples=30`, `max_new_tokens=2048`)

Run sources:
- `logs/aime25_bs12.log` ... `logs/aime25_bs16.log`
- `logs/aime25_bs17_19_profiled_20260224_233943/aime25_bs17.log`
- `logs/aime25_bs17_19_profiled_20260224_233943/aime25_bs18.log`
- `logs/aime25_bs17_19_profiled_20260224_233943/aime25_bs19.log`
- Baseline reference: `logs/aa30_bs12_16_profiled_20260224_224339/aime25_bs1_baseline.log`

Baseline (`bs=1`) reference:
- TPOT: `0.043140`
- Tokens/s: `22.999258`
- TTFT: `0.699609`

Profiled speculative comparison:

| bs | Tau | TPOT (s) | Tokens/s | TPOT Speedup vs bs1 | Throughput Gain vs bs1 | Avg Target Decode (s/sample) | Avg Draft Decode (s/sample) | Target Share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 6.60 | 0.008311 | 115.952 | 5.19x | 5.04x | 12.297534 | 2.108032 | 0.8537 |
| 13 | 6.73 | 0.008053 | 125.009 | 5.36x | 5.44x | 11.923771 | 2.007209 | 0.8559 |
| 14 | 7.02 | 0.007554 | 132.146 | 5.71x | 5.75x | 11.601284 | 1.941967 | 0.8566 |
| 15 | 7.37 | 0.007210 | 138.306 | 5.98x | 6.01x | 11.148856 | 1.866264 | 0.8566 |
| 16 | 7.46 | 0.007273 | 138.694 | 5.93x | 6.03x | 10.769086 | 1.800859 | 0.8567 |
| 17 | 7.32 | 0.007619 | 123.806 | 5.66x | 5.38x | 11.204767 | 2.036019 | 0.8462 |
| 18 | 7.20 | 0.007815 | 128.969 | 5.52x | 5.61x | 11.558331 | 1.933780 | 0.8567 |
| 19 | 6.86 | 0.008072 | 125.805 | 5.34x | 5.47x | 11.984842 | 2.010128 | 0.8564 |

Takeaways:
- Best TPOT: `bs=15` (`0.007210`).
- Best tokens/s and best tau in this sweep: `bs=16` (`138.694 tokens/s`, `tau=7.46`).
- `bs>=17` is consistently worse than the `bs=15/16` band on throughput and TPOT.
- Decode remains verification-dominated at all sizes (target share ~`85.6%`), with verify/draft ratio around `~5.8x-6.0x`.
- `bs=17` has an outlier-high TTFT/prefill (`1.038781s`) due cold-start effects, but decode-side metrics also do not beat `15/16`.

Saved local analysis artifacts:
- `outputs/analysis_aime25_bs12_19_profiled/summary.md`
- `outputs/analysis_aime25_bs12_19_profiled/summary.csv`
- `outputs/analysis_aime25_bs12_19_profiled/tau_throughput_vs_bs.png`
- `outputs/analysis_aime25_bs12_19_profiled/target_draft_decode_vs_bs.png`

## 14) Per-Cycle Decode Cost Check (`bs=12..19`)

Using profiled totals plus cycle counts from the same AIME25 run (`max_samples=30`):

- `verify_s_per_cycle = target_decode_s / (cycles / 30)`
- `draft_s_per_cycle = draft_decode_s / (cycles / 30)`

| bs | Verify (s/cycle) | Draft (s/cycle) | Total (s/cycle) |
|---:|---:|---:|---:|
| 12 | 0.043915 | 0.007528 | 0.051442 |
| 13 | 0.043618 | 0.007343 | 0.050961 |
| 14 | 0.042894 | 0.007180 | 0.050074 |
| 15 | 0.043079 | 0.007211 | 0.050290 |
| 16 | 0.043985 | 0.007355 | 0.051341 |
| 17 | 0.044027 | 0.008000 | 0.052027 |
| 18 | 0.045150 | 0.007554 | 0.052704 |
| 19 | 0.044274 | 0.007426 | 0.051699 |

Conclusions:
- In `bs=12..19`, per-cycle decode cost is nearly flat.
- Verification remains the dominant term at about `~5.5x` to `~6.0x` draft per cycle.
- Therefore, for this range, dynamic scheduling should focus on improving `tau` (accepted tokens per cycle), not expecting large cycle-time reductions from smaller block sizes alone.

## 15) Fixed-Prefix Rank-Suffix (`p4_k4_c4`) on AIME25 (`max_samples=30`, `max_new_tokens=2048`)

Run:
- command used through `run_fixed_prefix_sweep.sh` with:
  - `fixed_prefix_len=4`
  - `branch_top_k=4`
  - `max_candidates=4`
  - `candidate_mode=fixed_prefix_rank`
- log/output:
  - `logs/fprefix_full_nobase/aime25_p4_k4_c4.log`
  - `outputs/fprefix_full_nobase_aime25_p4_k4_c4.jsonl`
  - `outputs/fprefix_full_nobase_aime25_p4_k4_c4_cycle.jsonl`
  - summary: `logs/fprefix_full_nobase/summary.csv`

Raw speculative metrics:
- `Speculative total_wall_s`: `395.160765`
- `Speculative TPOT`: `0.007011`
- `Speculative tokens_per_sec`: `140.727534`
- `Average Acceptance length (tau)`: `7.66`
- `avg_target_decode_s`: `9.973618`
- `avg_draft_decode_s`: `1.847648`
- `draft_share_decode`: `0.1563`

Reference baselines:
- `bs=1` matched baseline:
  - total wall `2406.121140`, TPOT `0.043140`, tokens/s `22.999258`
- vanilla `bs=16` reference:
  - total wall `382.560704`, TPOT `0.007273`, tokens/s `138.694329`, tau `7.46`

Comparison:

| Metric | vs `bs=1` | vs vanilla `bs=16` |
|---|---:|---:|
| E2E wall speedup | `6.09x` | `0.968x` |
| TPOT speedup | `6.15x` | `1.037x` |
| Throughput gain | `6.12x` | `1.015x` |
| Tau delta | `N/A` | `+0.20` (`7.66` vs `7.46`) |

Interpretation:
- This config improves decode efficiency (`TPOT`, tokens/s, and tau) over vanilla `bs=16`.
- E2E wall time is still slightly worse than vanilla in this run.
- Cause: this run produced more output tokens overall (about `+4.8%` vs vanilla), so total elapsed time increased despite better per-token speed.

Per-cycle timing (derived):
- fixed-prefix `p4_k4_c4`:
  - draft: `~7.33 ms/cycle`
  - target verify: `~39.59 ms/cycle`
  - total decode: `~46.92 ms/cycle`
- vanilla `bs=16`:
  - draft: `~7.36 ms/cycle`
  - target verify: `~43.99 ms/cycle`
  - total decode: `~51.34 ms/cycle`

Takeaway (from this specific single-config run):
- `p4_k4_c4` improved decode-side metrics versus vanilla `bs=16`.
- Full prefix tuning is summarized in Section 16; that later sweep shows `prefix_len=2` is stronger overall.

## 16) Fixed-Prefix Sweep Update (`k=4`, `c=4`, AIME25 full 30; pulled from pod)

Pulled artifacts:
- `pulled/2026-02-25/fprefix_full_nobase/logs/aime25_p1_k4_c4.log`
- `pulled/2026-02-25/fprefix_full_nobase/logs/aime25_p2_k4_c4.log`
- `pulled/2026-02-25/fprefix_full_nobase/logs/aime25_p3_k4_c4.log`
- `pulled/2026-02-25/fprefix_full_nobase/logs/aime25_p4_k4_c4.log`
- `pulled/2026-02-25/fprefix_full_nobase/logs/aime25_p6_k4_c4.log`
- `pulled/2026-02-25/fprefix_full_nobase/logs/aime25_p7_k4_c4.log`
- `pulled/2026-02-25/fprefix_full_nobase/logs/aime25_p5_k4_c4.log` (incomplete run; excluded)

Baseline references used:
- `bs=1` baseline: total wall `2406.121140`, TPOT `0.043140`, tokens/s `22.999258`
- vanilla `bs=16`: total wall `382.560704`, TPOT `0.007273`, tokens/s `138.694329`, tau `7.46`

### Raw metrics

| Fixed Prefix Len | Total Wall (s) | TPOT (s) | Tokens/s | Tau | Avg Target Decode (s/sample) | Avg Draft Decode (s/sample) | Avg Verify Calls/sample |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 393.331782 | 0.007303 | 137.146304 | 7.67 | 9.962570 | 1.839576 | 242.5 |
| 2 | 377.158110 | 0.006818 | 145.973263 | 7.86 | 9.532146 | 1.757153 | 240.1 |
| 3 | 411.910601 | 0.007452 | 132.698697 | 7.59 | 10.431105 | 1.926425 | 251.3 |
| 4 | 395.160765 | 0.007011 | 140.727534 | 7.66 | 9.973618 | 1.847648 | 251.9 |
| 6 | 481.025378 | 0.008837 | 112.931672 | 7.56 | 10.672296 | 2.381083 | 247.8 |
| 7 | 486.731377 | 0.008773 | 112.657623 | 7.63 | 10.826066 | 2.392658 | 251.4 |

### Derived vs baseline (`bs=1`)

| Fixed Prefix Len | E2E Wall Speedup | TPOT Speedup | Throughput Gain |
|---:|---:|---:|---:|
| 1 | 6.12x | 5.91x | 5.96x |
| 2 | 6.38x | 6.33x | 6.35x |
| 3 | 5.84x | 5.79x | 5.77x |
| 4 | 6.09x | 6.15x | 6.12x |
| 6 | 5.00x | 4.88x | 4.91x |
| 7 | 4.94x | 4.92x | 4.90x |

### Delta vs vanilla `bs=16`

| Fixed Prefix Len | E2E Wall Ratio | TPOT Ratio | Throughput Ratio | Tau Delta |
|---:|---:|---:|---:|---:|
| 1 | 0.973x | 0.996x | 0.989x | +0.21 |
| 2 | 1.014x | 1.067x | 1.052x | +0.40 |
| 3 | 0.929x | 0.976x | 0.957x | +0.13 |
| 4 | 0.968x | 1.037x | 1.015x | +0.20 |
| 6 | 0.795x | 0.823x | 0.814x | +0.10 |
| 7 | 0.786x | 0.829x | 0.812x | +0.17 |

Interpretation:
- `fixed_prefix_len=2` is best among completed runs:
  - highest tau (`7.86`)
  - best TPOT (`0.006818`)
  - best throughput (`145.97` tok/s)
  - best E2E wall speedup vs baseline (`6.38x`)
  - slight E2E win over vanilla `bs=16` (`1.014x` wall ratio).
- `fixed_prefix_len=6/7` over-branches and hurts throughput hard:
  - draft decode cost jumps to ~`2.39s/sample`
  - throughput falls to ~`112.8` tok/s despite tau staying above vanilla.
- Practical rule from this sweep:
  - keep fixed-prefix branching shallow (`1-4`),
  - centered near `prefix_len=2`.

Saved local analysis artifacts:
- `outputs/analysis_fixed_prefix_tuning_20260225/summary.csv`
- `outputs/analysis_fixed_prefix_tuning_20260225/summary.md`

## 17) Rerun of Old Fixed-Prefix Settings (`p6`, `p7`) on A100

Run tag:
- `logs/fixed_prefix_sweep_20260225_033258/`

Config:
- Dataset: `aime25`, `max_samples=30`, `max_new_tokens=2048`
- Method: `candidate_mode=fixed_prefix_rank`, `branch_top_k=4`, `max_candidates=4`
- Tested: `fixed_prefix_len in {6,7}`

Matched baseline from same run:
- `bs=1` total wall: `2437.238799`
- `bs=1` TPOT: `0.043839`
- `bs=1` tokens/s: `22.705613`

Raw speculative metrics:

| Prefix | Total Wall (s) | TPOT (s) | Tokens/s | Tau | Avg Target Decode (s/sample) | Avg Draft Decode (s/sample) |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 416.679801 | 0.007598 | 130.371090 | 7.56 | 10.491714 | 1.999206 |
| 7 | 411.389279 | 0.007456 | 133.289813 | 7.63 | 10.433355 | 1.924455 |

Derived vs same-run baseline:

| Prefix | E2E Wall Speedup | TPOT Speedup | Throughput Gain |
|---:|---:|---:|---:|
| 6 | 5.85x | 5.77x | 5.74x |
| 7 | 5.92x | 5.88x | 5.87x |

Interpretation:
- This rerun confirms the earlier trend: long fixed prefixes (`6`, `7`) remain clearly worse than shallow settings.
- `prefix=7` is slightly better than `prefix=6`, but both are still well below the best shallow run (`prefix=2` in Section 16: TPOT `0.006818`, tokens/s `145.973`, tau `7.86`).

## 18) Adaptive Candidate Budget (`fixed_prefix_rank`, `p2`, AIME25 full 30)

Run A (adaptive 1/4/8):
- Command core:
  - `--candidate-mode fixed_prefix_rank --fixed-prefix-len 2 --branch-top-k 4 --max-candidates 8`
  - `--adaptive-candidates --adaptive-budgets 1,4,8 --adaptive-accept-thresholds 0.85,0.65 --adaptive-warmup-cycles 4 --adaptive-probe-interval 16`
- Results:
  - `Speculative total_wall_s`: `435.822242`
  - `Speculative TPOT`: `0.007584`
  - `Speculative tokens_per_sec`: `125.291907`
  - `Average Acceptance length (tau)`: `7.57`
  - `avg_target_decode_s`: `10.552797`
  - `avg_draft_decode_s`: `2.084895`
  - `avg_candidates_per_cycle`: `3.407`
  - adaptive usage: `{1: 1337, 4: 567, 8: 5620}` (`17.8%`, `7.5%`, `74.7%`)

Run B (adaptive 2/3/4):
- Command core:
  - `--candidate-mode fixed_prefix_rank --fixed-prefix-len 2 --branch-top-k 4 --max-candidates 4`
  - `--adaptive-candidates --adaptive-budgets 2,3,4 --adaptive-accept-thresholds 0.90,0.75 --adaptive-warmup-cycles 8 --adaptive-probe-interval 0`
- Results:
  - `Speculative total_wall_s`: `449.192554`
  - `Speculative TPOT`: `0.007887`
  - `Speculative tokens_per_sec`: `122.007810`
  - `Average Acceptance length (tau)`: `7.49`
  - `avg_target_decode_s`: `10.803554`
  - `avg_draft_decode_s`: `2.146270`
  - `avg_candidates_per_cycle`: `3.573`
  - adaptive usage: `{2: 1244, 3: 500, 4: 5817}` (`16.5%`, `6.6%`, `76.9%`)

Reference static run (Section 16 best):
- `fixed_prefix_len=2, branch_top_k=4, max_candidates=4`
- `total_wall_s=377.158110`, `TPOT=0.006818`, `tokens/s=145.973263`, `tau=7.86`
- `avg_target_decode_s=9.532146`, `avg_draft_decode_s=1.757153`

Delta vs static `p2_k4_c4`:

| Config | Wall | TPOT | Tokens/s | Tau | Avg Target Decode | Avg Draft Decode |
|---|---:|---:|---:|---:|---:|---:|
| adaptive `1/4/8` | `0.865x` | `0.899x` | `0.858x` | `-0.29` | `+10.7%` | `+18.7%` |
| adaptive `2/3/4` | `0.840x` | `0.864x` | `0.836x` | `-0.37` | `+13.3%` | `+22.1%` |

Notes:
- Ratios above are relative to static best (`1.0` = parity with static).
- In both adaptive runs, reduced candidate budgets on some cycles lowered acceptance enough to hurt throughput.
- For this method on AIME25, static `p2,k4,c4` remains stronger than these adaptive policies.
