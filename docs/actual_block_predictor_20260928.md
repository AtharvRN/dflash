# Actual-block supervision pilot: fixed protocol

User authorized proceeding toward 70% aggregate acceptance at approximately
96% retention relative to matched B16. This is a bounded predictor experiment,
not a throughput benchmark or a promise to reach the target.

## Data and collection

Collect 10,000 eligible training-cycle states on one idle workstation GPU,
with a four-hour collection bound (plus model loading and the last prompt).
Use a seed-929 uniform shuffle of canonical training prompts; deduplicate
exact message content and exclude cross-split duplicate content. Plan at most
2,000 prompts, skipping overlength prompts without replacements outside this
plan. Stop after reaching 10,000 eligible rows, finish the last prompt, and
retain its extra evidence; training uses the first 10,000 eligible rows in saved
receipt and within-prompt order. Incomplete collection must not trigger training.
Smoke output is separate and never used as a scientific result.

Original frozen Qwen3-4B and DFlash B16 revisions from models.json; BF16 SDPA,
greedy, thinking off, TF32 off. Maximum prompt 2,048 tokens, output 256 tokens,
eight progress-spaced states/prompt including cycle zero. At each common B16
reference state, restore caches independently and execute every integer B2--B16.
Randomize alternative order. Keep the existing reverse-order checks (four states),
canonical greedy checks (eight states), and same-candidate target-width controls.
Reject training if checked canonical comparisons disagree.

Input: latest 2,560-dimensional pre-draft fused target feature, generated in
the same collection as its labels by a separate causal fusion replay. No known
anchor embedding, current draft features, verification results, future tokens,
or assessment labels are predictor inputs. Exclude paired states if any action
accepts EOS; exclude insufficient output slots. Save prefix IDs/hashes and all
actual draft candidates. Back up each prompt from /tmp to the designated scratch
run directory with atomic checksum-verified copies. Do not overwrite prior runs.

Read-only evaluation: the completed policy_granularity_20260927/cache,
342 eligible calibration states/47 prompts and 1,416 assessment states/187 prompts.
Their membership and bytes are unchanged. They are inspected development data,
not an untouched final test. Audit canonical split membership, exact-content
disjointness, all receipt hashes, prefix continuity, complete action matrices,
model/collection configuration, feature shape/finiteness, and the selected row
index before training.

## Matched models

Three seed pairs: 913, 914, 915. Train from scratch with identical parameter
initialization, minibatch order, and dropout random stream within each pair.
No seed selection using assessment results.

Shared architecture: 2560 -> 512 -> 256 -> 128 -> 15, GELU after hidden linears,
LayerNorm after the first GELU, dropout 0.05 after each hidden stage.
1,478,415 trainable parameters. Output for proposed budget d is
mu[d] = d * sigmoid(logit[d]), d=1,...,15. These are independent expected actual
accepted-token counts; do not enforce monotonicity across different drafts.

- Actual arm: labels are the observed vector (A_B2,...,A_B16).
- Clipped arm: labels are min(A_B16,d) for each d on exactly the same states.

Both use unweighted MSE in accepted-token counts, averaged over all state/action
pairs. Six epochs, batch 128, AdamW 3e-4, weight decay 0.01, gradient norm clip 1,
no scheduler. FP32 training, TF32 off, deterministic algorithms. This compares
supervision, not another survival-loss variant or a larger context architecture.

The recovered 100k-trained small survival MLP remains a frozen control,
checkpoint SHA256 84a37d00f61b1c8ae5ecd76e1ad49b4cd00850613e783edf72ecdf875663af35.
It uses the original .001-grid alpha rule. It is NOT training-size/data-matched
to the new 10k models; only the actual/clipped pair isolates label supervision.

## Decisions, selection, and reporting

Choose d maximizing mu[d] - lambda*d, with smaller d breaking ties. This is
the Lagrangian of minimizing token budget under an accepted-token constraint,
not a hardware timing utility. Search nonnegative upper-envelope decision
breakpoints from calibration predictions, plus interval representatives. Include
fixed B16 as an explicit safe calibration fallback. No assessment predictions
or labels generate settings. No imposed monotonicity of actual A across B.

For each epoch, select minimum calibration mean budget with at least 96%
actual B16 retention; ties prefer higher retention. Select the checkpoint with
lowest such budget, then higher retention, then earlier epoch. Record the selected
checkpoint for every seed before computing any assessment metrics. Offline
scoring uses CPU FP32 with dropout disabled for both new and frozen models.

Report every seed, aggregate accepted/proposed (not mean per-cycle ratios),
retention, mean accepted, mean budget, per-block prediction errors, and operating
curves at calibration targets 90/95/96/98/100%. Calibration retention is not a
guarantee of assessment retention. Save both complete calibration/assessment
curves; assessment curves are descriptive and do not select deployed settings.
Use 2,000 paired whole-prompt bootstrap draws (seed 929) for fixed-checkpoint
uncertainty and actual-versus-clipped differences. Report seed mean/range/std;
do not select the best seed. Verify checkpoint reload and initialization/order
hashes. The 70%/96% target is a joint point-estimate check, not a confidence bound.

No throughput or closed-loop claim. All policies are evaluated at shared B16
states, not their own induced trajectories. At most eight sampled states/prompt
also does not represent the full cycle population. Any promising result needs
independent confirmation and later matched runtime evaluation.

## Paths

- Code: /home/zekaili/atharv/dflash
- Run: /data/scratch/zekaili/atharv/dflash/runs/actual_block_predictor_10k_20260928
- Temporary collection: /tmp/actual_block_predictor_10k_20260928_cache
- Durable paired data: run/cache
- Models, audit, selected settings, predictions and report: run/training
- Pipeline exit status: run/pipeline_exit.txt

The wrapper is one bounded collection-and-training batch. It creates no recurring
monitor and does not use the cluster/PVC or other users' GPU processes.
