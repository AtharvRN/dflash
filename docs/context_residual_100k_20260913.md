# Frozen-MLP Residual Context Experiment

## Setup

The previous 20k-row pilot did not improve on the last-vector MLP. This follow-up preserves a trained MLP and learns only an additive correction to its 15 conditional-acceptance logits.

1. Train the matched last-vector MLP on 100k rows and select its checkpoint on calibration prompts.
2. Freeze that exact MLP, including evaluation-mode dropout behavior.
3. Add a one-query, two-layer cross-attention head with a zero-initialized output layer. Initial logits exactly equal the frozen MLP.
4. Compare a correction that sees only the latest valid fused vector against an otherwise identical correction that sees all 16 saved context vectors.

Both correction heads retain feature width 2560, use 16 attention heads and FFN width 1024, and have 63,059,983 trainable parameters. The frozen MLP has 1,478,415 parameters. Each combined model has 64,538,398 parameters. There are no current-cycle draft features, token IDs, anchor embeddings, or confidence statistics in these inputs.

## Data Integrity

- Existing Qwen3-4B / DFlash B16 traces from the `z-lab/qwen3-4b-instruct-100k` prompt dataset.
- Canonical pool: 3,308,764 training rows and 172,794 validation rows.
- The larger sample exposed a partial-shard inconsistency. A complete small-array audit found 101 rows with label/cumulative-label/cycle-ID inconsistencies in eight shards.
- No inconsistent row belonged to fixed validation, or to the prior 20k training sample.
- Conservatively exclude all 19,970 training rows in those eight shards. Do not repair source labels or change validation membership.
- Uniformly sample 100,000 rows from the remaining 3,288,794 training rows, seed 913. The sample contains 53,262 distinct training prompts.
- Reuse identical validation files and ordering from the previous cache: 172,794 rows / 4,187 prompts.
- Calibration: 34,293 rows / 837 prompts. Assessment: 138,501 rows / 3,350 disjoint prompts.

The audit is a consistency check, not a proof that every source feature is correct. Original feature arrays are not modified. The source-label audit and exact row indices are saved alongside the run.

## Training And Selection

- One A100-PCIE-40GB in `wenglab-interpretable-ai/dflash-a100-gpu-test`.
- Six epochs for each stage, batch 128, AdamW learning rate 3e-4, weight decay 0.01, dropout 0.05, gradient clipping 1.0, BF16 autocast with FP32 likelihood calculations.
- Same first-rejection NLL as the preceding pilot: sum successful conditional-acceptance terms plus the first rejection term; A=15 has no invented rejection term.
- Same training rows and deterministic shuffled row order for all models. Both correction models start from the same frozen MLP and identical correction initialization.
- At each checkpoint, alpha calibration chooses the smallest mean budget attaining at least 96% calibration retention over the existing alpha grid (0.5 to 1, step 0.005).
- Checkpoint selection maximizes the resulting aggregate calibration accept ratio. This differs from the earlier pilot's MAE-based checkpoint selection; the 100k MLP is the matched control for this experiment.
- Include epoch zero as a fallback for residual models. Calibration selection cannot guarantee an improvement on assessment prompts.
- Assessment prompts are not used for checkpoint or alpha selection. MAE and policy metrics are reported separately.

Policy results remain **clipped-B16 offline proxies**, not actual shorter-block drafting or online speedups. Draft budgets range from 1 to 15; block size equals draft budget plus one. Retained accepted tokens are min(observed B16 accepted length, chosen budget). Aggregate accept ratio is total retained accepted tokens divided by total draft budget.

## Verification

Twenty-two tests pass locally and on the pod. They cover exact baseline equality at initialization, frozen weights and dropout, correction gradients, last-only feature isolation, checkpoint replay, prompt separation, validation-cache identity, partial-label auditing, censoring boundaries, complete CLI training/backups, and paired prompt-bootstrap edge cases. An independent subagent reviewed correctness and adversarial cases.

Training verifies frozen baseline tensors remain bit-identical after optimization. Prompt-bootstrap comparisons use the same resampled assessment prompts for each model, with thresholds frozen from calibration. These confidence intervals do not quantify training-seed variability.

All three selected PVC checkpoints match the temporary copies by SHA-256. Reloading them reproduces the first 256 saved validation predictions with maximum absolute difference zero. Selected checkpoints, full validation predictions, histories, metrics, bootstrap results, and cache provenance are persisted under the run directory.

## Artifacts

- Source branch: `codex/context-attention-predictor`, deployed with Git push/pull.
- Training commit: `10fc9a873e72ea36e1782eb36a88a13265c02363`.
- Cache: `/tmp/context-attention-cache-100k-audited-20260913`.
- Local pod run: `/tmp/context-residual-100k-20260913`.
- Persistent run: `/workspace/dflashv2_data/runs/context_residual_100k_20260913`.
- Local retrieved results: `outputs/context_residual_100k_20260913/`.

## Results

All three runs completed, and all selected checkpoints were from epoch 4. Both residual models verified that the frozen MLP tensors remained unchanged. This is a single-seed controlled experiment, not a Math500/HumanEval or throughput benchmark.

Results below use only the 138,501 assessment rows from 3,350 prompts:

| Model | Expected MAE | Mean accepted | Mean draft budget | Aggregate accept ratio | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched MLP | 2.6023 | 4.7206 | 9.3263 | 0.50616 | 96.252% |
| Frozen MLP + last-vector-only correction | 2.5664 | 4.7213 | 9.2813 | 0.50869 | 96.266% |
| Frozen MLP + 16-token context correction | 2.5653 | 4.7277 | 9.3304 | 0.50670 | 96.395% |

Full-validation expected MAE, including calibration rows: 2.5988 / 2.5641 / 2.5627, respectively. The fixed-B16 assessment reference accepts 4.9044 tokens per cycle with budget 15, accept ratio 0.3270, and retention 100%.

### Paired Uncertainty

Prompt-bootstrap 95% intervals, 2,000 paired resamples, holding trained weights and calibration thresholds fixed:

- Last-vector-only correction versus MLP: accept-ratio difference +0.002525 [0.002350, 0.002700]; MAE difference -0.03583 [-0.03922, -0.03232].
- Full-context correction versus MLP: accept-ratio difference +0.000532 [0.000312, 0.000744]; MAE difference -0.03696 [-0.03984, -0.03405].
- Full-context versus last-vector-only correction: accept-ratio difference -0.001993 [-0.002164, -0.001825]; MAE difference -0.001129 [-0.002211, -0.000135]. Full context also retains approximately 0.130 percentage points more accepted tokens, so these are close but not exactly equal-retention operating points.

These small differences are measurable on this assessment sample; they are not evidence of a large practical gain or reproducibility across training seeds.

### Interpretation

Residual training avoided the regression of the original attention-from-scratch pilot. However, most of the MAE improvement is already present when the correction sees only the last vector. Adding earlier context improves MAE by only 0.0011 tokens versus that control and does not improve accept ratio at the selected operating points.

Relative to the matched MLP, the full-context model improves MAE by about 1.4% and accept ratio by about 0.1%, while adding approximately 0.007 accepted tokens per cycle. The last-only correction improves accept ratio by about 0.5% at essentially the same retention. These are not the substantial gains sought, particularly for an added 63M-parameter head. Do not scale this attention design based on this result alone.

The alpha grid is relatively coarse near one, only one training seed was tested, and the input window remains 16 tokens. The experiment therefore does not rule out every possible use of earlier context. It does establish that this specific residual-attention formulation provides little additional value over a last-vector correction in the current protocol.
