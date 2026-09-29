# Intermediate-target acceptance probe: bounded feasibility pilot

Authorized 2026-09-29 after reviewing related mechanisms in FASER,
Speculative Streaming and Lever. This is **not a claim of a novel mechanism**.
No SGLang serving code is modified by this pilot.

## Question and fixed data

Does intermediate target computation give a useful retention/work frontier
over replayed drafter-confidence trimming on unchanged B16 candidates?

- Target Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- DFlash B16 revision `b74e3a329c4d963783143b1e970d95b002be72bd`.
- First 2,000 eligible rows in the audited, immutable 10k training selection.
- All existing 342 calibration and 1,416 assessment states, prompt groups fixed.
- Input sources: `runs/actual_block_predictor_10k_finish_20260928/cache`
  and `runs/policy_granularity_20260927/cache` beneath the workstation data root
  `/data/scratch/zekaili/atharv/dflash`.

Saved committed prefixes, anchors and 15 B16 candidate IDs are immutable.
Recompute target labels with intermediate features in the SAME target forward;
old labels are an audit comparison, never supervision for new features.
The saved states are conditioned states, not fresh closed-loop trajectories.

## Collection and correctness scope

BF16 SDPA, TF32 off, Transformers 4.57.1. Length-sorted batches of eight with
explicit left-padding masks and per-request logical positions. Final arrays
retain original row ordering. Independent single-state replays check numerical
drift on spread-out batches. Greedy-only; no stochastic exactness claim.

Capture after 6, 9, 12, 18 and 24 completed target decoder layers, before
final RMSNorm. Candidate j is predicted by query row j-1. Query row zero is
the known anchor. Retaining K query rows permits K-1 proposals and one bonus;
the fixed-candidate acceptance is min(A_B16, K-1), and a hindsight keep-all-
accepted oracle needs A_B16+1 rows.

Capture the candidate's frozen LM-head weight vector. An untuned logit lens
(target final RMSNorm + original LM head) supplies rank and margin baselines;
this diagnostic full-vocabulary projection is NOT assumed free at deployment.

Draft baseline: recompute the B16 draft distribution using full-prefix target
features and score the **saved** candidate IDs. This is not identical to the
original incremental cached draft computation; report candidate-argmax drift.
All policies use exactly the same candidates and fresh target labels. Report
anchor and acceptance replay differences rather than silently dropping them.
Fresh accepted-EOS cases halt automatic training pending a censoring review.

## Probe and evaluation

Candidate-conditioned linear and 128-wide GELU MLP heads. Inputs are separately
RMS-normalized target h and candidate vector e, concatenated with h*e (7,680
dimensions). MLP dropout .05. Binary match loss covers only the at-risk prefix
through the first rejection; no fictional rejection beyond position 15.

Eight epochs, AdamW 3e-4, weight decay .01, clip1, 128 states/batch,
FP32/TF32-off training, seeds913/914/915. Select each checkpoint on calibration
kept rows at >=96% retention only. Exact score thresholds calibrated at
90/95/96/98/99%; apply frozen settings to assessment. Do not choose a winner
from assessment outcomes. Report all seeds and whole-prompt bootstrap intervals.

Baseline families: replayed draft candidate logprob and entropy; intermediate
untuned Top-K/rank and margin; fixed truncation of unchanged B16 candidates;
candidate-preserving 100%-retention oracle. These are mechanism controls,
**not full reproductions** of FASER or Lever systems.

Primary work proxy: [L*16 + (36-L)*K]/36, together with retained accepted tokens
and committed tokens A_trim+1. Neither this proxy nor acceptance ratio is a
throughput result. Probe, compaction, graph dispatch, cache bookkeeping,
nonlinear attention and full B16 drafting costs still need measurement.

## Execution and next gate

Run a 24-state collection/training smoke before the full 3,758-state replay.
Use only idle authorized GPU4; verify again immediately before launch.
Temporary collection uses a fresh `/tmp/dflash-midverify.*` directory and
checksummed per-batch durable backups. Never overwrite source evidence.
Both collection and training have explicit time bounds. No recurring monitor.

Only if the diagnostic frontier is promising should we implement and measure
segmented verification. No 10k expansion, pre-draft hybrid, or serving-engine
integration is automatically launched by this wrapper.

Related mechanisms reviewed:

- https://arxiv.org/html/2604.20503v1 (FASER, token-wise early exiting).
- https://arxiv.org/html/2402.11131 (Speculative Streaming, parallel tree pruning).
- https://arxiv.org/html/2605.16786v1 (Lever, predictor-based verification pruning).

## Verified smoke and launch

Implementation commit: `e6d7a01`. The 28 focused unit/regression tests passed
on the workstation's existing environment (4.45 seconds).

`runs/midverify_probe_smoke_20260929` completed collection and ten one-epoch
smoke trainings. All 24 fresh acceptance labels and 24 anchors matched the
saved originals. Six batched-versus-single numerical controls had zero target
argmax differences. Hidden relative L2 differences were approximately
0.6–1.7%, so this is **not** bitwise parity. Replayed draft argmax agreed with
98.33% of saved candidate IDs; this validates why the recomputed confidence
baseline must be explicitly labelled as replayed.

The full `runs/midverify_probe_2k_20260929` pipeline was launched on the idle
authorized GPU4, UUID `GPU-2b489243-113f-1e33-ee0b-e4d28423e006`, parent PID
3070962. Early progress: 456/3,758 states, 37.2 states/s, GPU utilization 86%
and 11,455 MiB device memory. These are collection performance observations,
not speculative-decoding throughput. Final audit/results are pending at this
entry; a running process is not a completed experiment.

## Completed results (2026-09-29)

The full pipeline completed with exit code zero. Collection/audit took 186.6
seconds (capture itself 176.5 seconds); all 30 probes trained in 9.1 seconds.
GPU4 was subsequently verified idle. No segmented-target implementation,
additional GPU job, larger collection or recurring monitor was launched.

Final data: 2,000 train states / 263 prompts, 342 calibration / 47 prompts,
1,416 assessment / 187 prompts. All original fixed assessment rows retained.
Fresh same-forward labels were used throughout. Original labels agree in
3,722/3,758 states (20 train, 4 calibration, 12 assessment disagreements).
Anchors agree in 3,719 states. Recomputed draft argmax matches 97.784% of saved
candidate IDs. No fresh accepted-EOS state was found. All eight batch/single
controls agree on accepted length, although one had two posterior argmax
differences after the first rejection; this is not bitwise equality. Maximum
controlled hidden relative L2 difference was 2.58%.

### Corrected scoring is authoritative

A final audit found a NumPy weak-scalar promotion edge case: applying a Python
float `nextafter` threshold to float32 scores could round back to an exact tie.
Calibration already used float64; checkpoint and threshold selection were
unaffected. Commit `6548729` makes inference comparisons float64 too. The
original data, weights and reports were preserved; frozen scores were rescored
into **`assessment_fp64`**. Fourteen of 210 policy/retention points changed,
at most four assessment decisions each. The primary table below and the
confidence baseline did not change at the displayed precision. Thirty focused
tests passed after adding an exact float32-threshold regression test.

Use these authoritative artifacts under
`/data/scratch/zekaili/atharv/dflash/runs/midverify_probe_2k_20260929`:

- `cache/COMPLETE.json`, `cache/audit.json`: immutable collection audit.
- `training/`: immutable original training histories and all 30 checkpoints.
- `assessment_fp64/summary.json`, `assessment_fp64/report.md`: corrected
  assessment, unchanged calibration settings, all seeds and bootstrap intervals.
- `assessment_fp64/retention_vs_work.png` and `retention_vs_kept.png`: corrected plots.
- `matched_progress_and_sensitivity_fp64/`: matched-progress oracle and replay
  stability checks. `matched_progress_bound/` is the earlier bound-only output.

Small reports, frozen scores and plots are also saved beneath local
`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_probe_2k_20260929`.
Large captured vectors and checkpoints remain on the workstation.

### Calibration-96% operating points

MLP entries are arithmetic means across seeds913/914/915, not selected best
seeds. Assessment retentions differ; these are calibration-matched operating
points, not exactly matched assessment retention. All use the same B16
candidates and fresh labels. Assessment mean B16 accepted length is 6.5840.

| Policy | Assessment retention | Kept query rows | Equivalent full-depth rows |
| --- | ---: | ---: | ---: |
| Replayed draft candidate logprob | 95.25% | 9.049 | 9.049 |
| Replayed draft entropy | 95.32% | 9.507 | 9.507 |
| MLP after L6 | 95.86% | 11.283 | 12.069 |
| MLP after L9 | 95.15% | 10.794 | 12.095 |
| MLP after L12 | 95.14% | 10.540 | 12.360 |
| MLP after L18 | 95.98% | 10.554 | 13.277 |
| MLP after L24 | 95.21% | 9.655 | 13.885 |

Draft logprob's aggregate accept ratio is 0.7791 at 0.95248 assessment
retention. This is **post-draft fixed-candidate trimming**, not achievement of
the earlier pre-draft 70% target, and not directly comparable to historical
task tables. Its retention prompt-bootstrap 95% interval is [0.9420, 0.9626].

The learned probes do not justify segmented-engine integration on this pilot:
later layers reduce kept rows somewhat, but the early full-width layer cost
outweighs that benefit. This finding does not establish an information limit
or rule out different training, features, models or workloads.

### Matched-progress ideal bound

For unchanged candidates, `sum(K) >= N + sum(A_trim)`. Thus at a specified
accepted-token total a clairvoyant probe has minimum row-layer work
`[16*L + (36-L)*(1+mean(A_trim))]/36`. This is a work bound, **not a latency bound**.

At exactly the draft-logprob control's 95.248% assessment retention, a perfect
L12 decision costs at least 10.181 full-depth-equivalent rows, versus the
control's 9.049. The break-even depth is 7.33 layers: a decision after eight
or more completed layers cannot beat that control in this proxy, even with a
perfect probe and free compaction. At 98.809% retention the break-even depth
increases to 14.94 layers; the bound is operating-point dependent.

The comparison survives replay-stability checks without retraining or
recalibration. Among the 1,033 assessment states where all 15 saved candidates
also match the recomputed drafter argmax, the frozen confidence control has
96.47% retention / 10.486 work rows; the L12 MLP seed mean has 95.68% /
13.027. These subsets are secondary diagnostics, not replacements for the
full assessment set.

**Decision:** no large integration or scaling job on these results. Preserve
post-draft confidence as the strong matched control; any revisited mid-target
proposal must beat it after accounting for the early full-width computation.

Subsequent user-authorized follow-up: a controlled 2k/5k/9,999 data/update scaling
study is now complete; see [the scaling note](midverify_scaling_20260929.md).
It preserves this pilot's features/validation bytes and uses rerun 2k probes as
its equal-update controls. More data improves the early probe, but it still
does not beat the frozen confidence control after early-layer work is counted.
