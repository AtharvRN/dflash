# Pre-Draft Context Attention Pilot

## Question

Does reading recent fused context with learned attention queries improve acceptance prediction over a last-vector MLP? Does a distinct query for each draft position help beyond one pooled query?

## Models

- `last_mlp`: existing last-valid-fused-vector MLP, 2560 -> 512 -> 256 -> 128 -> 15 logits.
- `one_query`: one learned query reads 16 fused context vectors through two cross-attention/FFN blocks, then predicts 15 logits.
- `position_queries`: 15 learned queries read the same context through the same blocks; a shared scalar head produces one logit per query.

Attention uses width 2560, 16 heads, FFN width 1024, dropout 0.05, and learned relative context-age embeddings. No standalone input compression layer. The two attention variants have approximately 63.06M parameters each; they are parameter-matched to each other, not to the smaller MLP.

Each logit models conditional acceptance at position k, given that the preceding draft positions were accepted. Cumulative products of sigmoid outputs give P(A >= k). The queries do not predict token identities and do not read current-cycle draft outputs.

## Data And Leakage Controls

- Source: existing Qwen3-4B / DFlash B16 traces from `z-lab/qwen3-4b-instruct-100k` prompts.
- Canonical split: 3,308,764 training rows and 172,794 validation rows, separated by prompt.
- Pilot: 20,000 uniformly sampled training rows, seed 913; all 172,794 canonical validation rows retained.
- Checkpoint/threshold selection: deterministic 20% subset of validation prompts. Assessment: remaining disjoint 80% of validation prompts.
- Input: original saved 16-token fused-context window, 2560 dimensions per token. The pilot is not a full-prefix or 128-token-context experiment.
- Known anchor token embedding omitted from every model because these original traces do not save anchor IDs.
- Metadata determines real row counts; padded shard tails are excluded. Prompt/cycle IDs, feature masks, accepted-length labels, and stored cumulative labels are checked for alignment.

## Training And Evaluation

All three models use the same rows, shuffled training order, batch size 128, six epochs, AdamW learning rate 3e-4, weight decay 0.01, and gradient clipping at 1. A100 execution uses BF16 autocast and FP32 loss calculations.

The loss is first-rejection negative log likelihood: success terms through the observed accepted length, followed by one failure term when A < 15. A = 15 contributes 15 success terms without inventing a rejection at position 16. Terms are summed per example, then averaged over examples. No auxiliary distance loss in this controlled pilot.

Checkpoint selection uses expected-length MAE on calibration prompts. A separate alpha sweep on those same calibration prompts chooses the smallest average budget satisfying at least 96% observed retention. Alpha is frozen for assessment prompts. Full-validation metrics include calibration rows and are labeled accordingly.

Reported policy metrics are **clipped-B16 offline proxies**: retained tokens are min(A, budget), with draft budgets 1 through 15 (block size = budget + 1). They are not measurements of actual shorter-block drafting, online acceptance, or speedup. Aggregate accept ratio is sum(retained tokens) / sum(draft budgets).

## Correctness Tests

- Both query configurations: shape, gradients, checkpoint reload, tiny-set overfitting.
- Padding invariance, masked NaN isolation, context-order sensitivity, and per-request isolation.
- Every accepted length 0 through 15: likelihood values, first-failure masking, and censoring gradients.
- Extreme logits, monotone cumulative probabilities, budget bounds, and full-budget alpha=1 fallback.
- Canonical prompt separation, padded shard exclusion, cache partition alignment, invalid feature-kind/mask rejection.
- End-to-end train/save/reload/calibrate/evaluate/persistent-backup test for all three models.
- Independent read-only subagent review, including all 256 binary masks of width 8 and adversarial bulk-staging fixtures.

Fifteen repository tests pass locally and on the pod. Full-width forward/backward smoke tests pass on the A100.

## Artifacts

- Remote source branch: `codex/context-attention-predictor` (deployed using Git push/pull).
- Pod: `wenglab-interpretable-ai/dflash-a100-gpu-test`.
- Staged cache: `/tmp/context-attention-cache-20k-bulk-20260913`.
- Training output: `/tmp/context-attention-pilot-20k-20260913`.
- Persistent checkpoints/results: `/workspace/dflashv2_data/runs/context_attention_pilot_20k_20260913`.

## Results

All three six-epoch runs completed on one A100. Calibration used 34,293 rows from 837 validation prompts; assessment used 138,501 rows from the remaining 3,350 prompts. The 20,000 training rows represented 17,243 distinct training prompts.

| Model | Best epoch | Full-val expected MAE | Assessment expected MAE | Assessment accepted | Assessment budget | Accept ratio | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last-vector MLP | 2 | 2.643 | 2.646 | 4.727 | 9.746 | 0.4851 | 96.39% |
| One attention query | 4 | 2.764 | 2.769 | 4.725 | 10.038 | 0.4708 | 96.35% |
| 15 attention queries | 6 | 2.781 | 2.783 | 4.715 | 9.998 | 0.4716 | 96.13% |

The fixed-B16 reference on these assessment rows accepts 4.904 draft tokens on average, with budget 15, accept ratio 0.3270, and retention 100%.

**Conclusion: neither attention model beats the matched MLP control in this pilot.** The one-query model has 4.7% higher assessment MAE; 15 queries have 5.2% higher MAE. Both use more draft budget while retaining slightly fewer accepted tokens than the MLP. Distinct per-position queries do not establish a benefit over one pooled query here.

Training curves show overfitting in the MLP and one-query model. The 15-query model's best checkpoint is the last tested epoch, so this run does not establish its converged performance. Context length, optimization, and capacity remain confounded relative to the MLP. This single-seed 20k-row pilot is not evidence that context attention cannot help at larger scale, but it does not justify claiming an improvement or launching a full-scale run yet.

No Math500/HumanEval or online throughput run was performed in this pilot. These validation-domain figures are not directly comparable with earlier Math500 tables.

Local machine-readable results: `outputs/context_attention_pilot_20k_20260913/summary.json`; exact training config and epoch logs are alongside it. Training used source commit `9caaa75b836a249a54fe2f95115f1c08426365bd`. A subsequent change affects checkpoint copying only: buffered writes replace slow kernel sendfile copies to Ceph. Training/evaluation completed before the old backup queue was stopped; final selected checkpoints and results are synchronized separately with checksum verification.
