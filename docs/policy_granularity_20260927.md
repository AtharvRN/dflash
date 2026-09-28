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
