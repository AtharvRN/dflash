# Raw Target Features Versus DFlash Fused Features

## Question

Does the frozen drafter's target-feature fusion discard information useful for
predicting its B16 accepted draft length? This changes the predictor input, not
the target model, draft model, decoding algorithm, or acceptance labels.

The checkpoint uses target layer IDs `[1,9,17,25,33]`, corresponding to hidden
state tuple indices `[2,10,18,26,34]`. Five 2560-dimensional vectors at the latest
committed token are concatenated into a 12800-dimensional RAW vector. DFlash's
existing `hidden_norm(fc(raw))` maps this to 2560 dimensions.

## Matched Models

| Variant | Input | MLP widths | Parameters |
| --- | --- | --- | --- |
| Raw | Latest pre-fusion target vector | 12800, 512, 256, 128, 15 | 6,721,295 |
| Fused baseline | Actual draft fusion output | 2560, 512, 256, 128, 15 | 1,478,415 |
| Fused capacity control | Actual draft fusion output | 2560, 2372, 256, 128, 15 | 6,721,755 |

All models use the existing last-vector MLP, GELU, projection LayerNorm, and
dropout 0.05. No anchor, confidence statistics, token IDs, or previous-token
features enter the predictor. Saved anchor IDs and trajectories are audit-only
metadata. No random projection is used.

## Collection

- Pod: `wenglab-interpretable-ai/dflash-a100-gpu-test`.
- Observed hardware: one NVIDIA A100-SXM4-80GB.
- Source prompts: existing `z-lab/qwen3-4b-instruct-100k` message manifest.
- Target: cached Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c`.
- Drafter: Qwen3-4B-DFlash-b16 snapshot `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Paired fresh collection is necessary: the historical fused cache cannot be
  inverted into raw target features. Old labels are not joined to new cycles.
- Planned pilot: 512 canonical training prompts and 256 canonical validation
  prompts, at most 32 cycles per prompt; fewer rows when EOS, prompt-length, or
  output-length exclusions apply. This is at most 16,384 training rows and
  8,192 validation rows, not the full historical 3.3M-row training set.
- Canonical split files remain unchanged. Existing calibration-prompt membership
  is inherited, with 51 calibration and 205 assessment prompts selected before
  collection exclusions. Exact duplicate message content crossing groups is
  excluded before sampling.
- Greedy B16 reference trajectories, thinking disabled, SDPA, BF16, TF32 disabled,
  max prompt 2048 tokens, max generated 512 tokens, seed 913.
- Raw feature captured before draft forward, at position `start-1`; fused feature
  captured by a transparent hook from that SAME forward. The anchor at `start`
  has not been target-processed and does not enter either predictor input.
- Label counts accepted draft tokens only, in `[0,15]`. Maximum label 15 has no
  invented rejection. Accepted-EOS cycles and states with fewer than 16 remaining
  output slots are excluded. Thus this pilot measures eligible nonterminal states.
- Writes use `/tmp`; immutable per-prompt shards and receipts are backed up to
  `/workspace` in the background. Completed caches and model checkpoints persist.

This is a fresh pilot subset preserving the old prompt groups, NOT the exact
historical validation states or their full distribution. Results must not be
compared directly to older full-validation/Math500 numbers.

## Training And Evaluation

Six epochs, batch 128, evaluation batch 256, AdamW learning rate 3e-4 and weight
decay 0.01, gradient clip 1.0, seed 913, BF16 model operations with FP32 likelihood.
All variants use identical rows and identically seeded training batch orders.

Loss: per-example first-rejection negative log likelihood, averaged over rows.
The 15 logits describe conditional token acceptance; cumulative products give
`P(A >= k)`. No extra distance, entropy, or length-reward loss.

Checkpoint and alpha selection share the calibration subset, as in the recent
100k-row residual experiment. At each epoch select the lowest-budget alpha-grid
point meeting 96% observed calibration retention; choose the checkpoint with
highest aggregate calibration acceptance ratio. Assessment prompts are excluded
from both choices. Calibration metrics are selection-biased and the previously
used validation suite is development evidence, not a pristine final test set.

Report expected/median MAE, per-length MAE, Brier, mean accepted length, mean draft
budget, aggregate accepted/budget ratio, and retention. Policy metrics clip the
observed B16 acceptance; they are NOT measured acceptance from shorter drafting
or online throughput. The paired-length diagnostic already demonstrated this
distinction matters.

## Verification And Execution

Entry point: `scripts/run_prefusion_pilot.sh`. It collects pairs, audits the
frozen fusion and provenance, then trains all three models. Code deployment is
through git push/pull, not file copying.

Initial real-model smoke: 26 eligible rows, all three models completed a one-epoch
train/save/reload/evaluation cycle. These tiny-run metrics are not research results.
Frozen-fusion replay relative RMS error was 0.334%, maximum row 0.413%, consistent
with comparing FP32 replay to the original BF16 forward.

Three subagents independently reviewed feature timing/caches, experiment config,
and implementation. Follow-up tests cover temporal alignment, rejected-state
isolation, split inheritance, feature/label alignment, hash tampering, and stale
audit rejection. Runtime checks reject nonfinite features and reused persistent
destinations. The target and drafter are frozen and never loaded by the trainer.

Pilot artifact root:
`/workspace/dflashv2_data/runs/prefusion_pilot_20260915`.

Status: implementation and smoke verification complete; full pilot results pending.
