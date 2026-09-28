# Frozen-predictor policy granularity test

Protocol fixed before collecting assessment outcomes. No training or architecture
search. Recover the exact 100k-trained last-fused MLP, epoch 4, SHA256
`84a37d00f61b1c8ae5ecd76e1ad49b4cd00850613e783edf72ecdf875663af35`.
The temporary CPU recovery pod mounts the original PVC read-only and is removed
after a hash-verified copy. GPU inference stays on the designated free workstation
GPU. Preserve original caches, splits, models, and other users' processes.

Use the original Qwen3-4B and DFlash B16 revisions, BF16/SDPA, greedy, thinking
off, TF32 off. Preserve all 47 calibration and 187 assessment prompts with rows
in the completed paired pilot. Verify content hashes, canonical validation
membership, and calibration membership embedded in the 100k checkpoint.
These are previously inspected development prompts, not a pristine final test.

Generate fresh B16 reference trajectories, at most 256 generated / 2048 prompt
tokens, eight generated-offset-spaced states per prompt including cycle zero.
Independently restore both caches and actually draft/verify every B2--B16 at
each selected state. Randomize alternative execution order. Reverse-order
replay checks cover four states; canonical greedy checks cover eight states.
Exclude a paired state from every policy if any block accepts EOS, and do not
sample insufficient-output-budget states. Do not clip old B16 labels or impose
monotonic acceptance across B. Record fresh causal fused vectors, prefixes,
token candidates, outcomes, hashes, and per-prompt backups.

The existing training-only B2--B20 matrix is not assessment data. We do not reuse
its labels with newly computed features or claim it validates generalization.

Policies:

1. Fixed: select one integer B2--B16 on calibration.
2. Request: use the MLP's cycle-zero survival curve for every sampled state of
   that request. Keep this initial feature even if its own outcome is excluded.
3. Cycle: use the MLP's current-state survival curve.

For adaptive policies, choose the smallest draft budget retaining alpha of the
predicted B16 expected accepted length. Calibrate alpha on actual outcomes over
0.5--1.0 in increments of .001; alpha=1 explicitly means budget 15. At each
retention target (90/95/96/98/100%), select minimum calibration mean budget,
breaking ties by higher retention then smaller setting. Fixed policies search
all 15 integer budgets. Primary target is 96%. Persist selections before
assessment evaluation. The request control is not a reproduction of BlockPilot.

Report actual mean accepted, mean proposed budget, aggregate accepted/proposed,
retention against matched B16, full curves, and paired whole-prompt bootstrap
intervals (2,000 draws, seed 928), with settings/weights frozen. Unequal actual
assessment retention and discrete fixed-size operating points must be explicit;
do not claim an exact matched-retention speedup from these points.

This first gate is a common-state policy evaluation, not closed-loop deployment,
an all-cycle population estimate, or throughput. The 1.48M MLP inference uses CPU
FP32 for offline scoring with dropout disabled. The collector saves a separate
causal `hidden_norm(fc(latest_target_features))` replay, not a same-forward hook.
If a material advantage is established, a separate bounded closed-loop test is
needed before any serving/runtime claim. No recurring monitor is created.

## Post-run calibration-resolution check

The primary .001-grid run completed first and is preserved unchanged. Its
request policy chose alpha=1: calibration retention at .999 was only 93.58%,
and alpha=1 forces every request to B16. This identifies a possible grid artifact,
not proof that request-level information is absent. A separately named exploratory
analysis enumerates every representable decision transition using calibration
survival curves only, with unchanged minimum-budget/retention selection. No
assessment outcomes select thresholds, and no weights or GPU outcomes change.
This is a post-hoc robustness check, not a replacement for the primary result.

## Completed result

Collection/inference and primary analysis exited 0. GPU 4 on
`zekaili@tianhaowang-gpu0.ucsd.edu` (RTX PRO 6000 Blackwell Server Edition) was
released after collection; all subsequent analysis was CPU-only. No model was
trained. Collection took 1,691 seconds (28.2 minutes, excluding model loading).
All 234 planned prompts completed: 1,764 paired states, six common accepted-EOS
exclusions, 1,758 eligible states. Calibration has 342 states / 47 prompts;
assessment has 1,416 states / 187 prompts. No prompt reassignment occurred.

The four reverse-order state checks passed exactly. All 120 canonical-greedy
comparisons across eight states agreed. Immutable per-prompt files, receipt
hashes, prefix hashes, B16 reference continuity, feature alignment/finiteness,
and exact initial-feature reuse passed the local artifact audit. Sixteen unit
tests passed after adding the calibration-resolution regression test.

### Primary preregistered .001-grid comparison

Settings below were selected using calibration outcomes only, targeting at
least 96% calibration retention. Mean budget excludes the known anchor.

| Policy | Selected setting | Calibration retention | Assessment accepted | Assessment budget | Aggregate ratio | Assessment retention |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | B15 | 97.785% | 6.3249 | 14.0000 | 0.45178 | 96.146% |
| Request | alpha=1, hence B16 | 100% | 6.5784 | 15.0000 | 0.43856 | 100% |
| Cycle | alpha=0.990 | 96.338% | 6.3157 | 11.7083 | 0.53942 | 96.006% |

Cycle selection uses 16.369% less proposed-token work than fixed B15, with
paired prompt-bootstrap 95% interval [13.744%, 19.014%]. Its aggregate ratio
improves by 0.08764 absolute; interval [0.07507, 0.09945]. Observed retention
differs by -0.140 percentage points, but the paired interval is [-1.387, +1.024]
points: close point estimates are not proof of retention equivalence. The cycle
retention interval itself is [94.862%, 97.108%], not a guaranteed 96% floor.

The request control's 100% retention is not matched to the cycle point, and its
collapse to B16 should not be sold as absence of request-level information.

### Exploratory calibration-resolution sensitivity

Using all calibration-input decision breakpoints instead of the coarse alpha
grid changes neither model weights nor GPU outcomes. The exact thresholds are
request 0.9997000825731314 and cycle 0.9894556174155835. The original generated
sensitivity report rounds settings to three decimals; these values and its
`summary.json` are authoritative. The reporting script now preserves 12
significant digits in future reports. Original completed artifacts are unchanged.

| Policy | Assessment accepted | Assessment budget | Aggregate ratio | Assessment retention |
|---|---:|---:|---:|---:|
| Fixed B15 | 6.3249 | 14.0000 | 0.45178 | 96.146% |
| Request | 6.3072 | 13.7811 | 0.45767 | 95.878% |
| Cycle | 6.3051 | 11.6568 | 0.54089 | 95.845% |

Cycle selection saves 15.415% of proposed work versus request selection,
paired 95% interval [12.642%, 18.096%]. Retention differs by -0.032 percentage
points, with a wider paired interval [-1.885, +1.755] points. Both adaptive
policies slightly miss the 96% assessment target despite meeting it on
calibration. This is an exploratory robustness result on the same development
prompts, not an independent confirmation or a replacement of the primary test.

### Interpretation and next gate

The frozen small MLP already contains useful signal for refreshing block choices
within a request. The result supports a bounded closed-loop runtime test against
fixed B15 and the request-level control before training another predictor.
It does not demonstrate throughput, beat a trained BlockPilot implementation,
or establish a publishable end-to-end system. Runtime/trajectory changes and
predictor/bookkeeping overhead remain unmeasured in this experiment.

### Artifact locations

- Workstation run: `/data/scratch/zekaili/atharv/dflash/runs/policy_granularity_20260927`.
- Workstation checkpoint: `/data/scratch/zekaili/atharv/dflash/checkpoints/context_residual_100k_20260913/last_mlp_best_epoch_4.pt`.
- Local run: `/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/policy_granularity_20260927`.
- Primary: `analysis/report.md`, `analysis/summary.json`, `analysis/calibration.json`.
- Sensitivity: `analysis_exact_calibration/summary.json`, `analysis_exact_calibration/calibration.json`.
- Fresh paired records and completion receipts: `cache/`.
- Collection/primary source commit: `87985ea`; sensitivity source: `22f6c90`.
