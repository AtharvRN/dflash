# Actual Draft Length Diagnostic: Qwen3-4B

## Conclusion

The clipped-label assumption is materially inaccurate for actual shorter drafting:

`A_b(x) != min(A_16(x), b - 1)`

The two quantities disagree on 20-23% of sampled states for B4/B8/B12. Existing frozen policies lose 1.61-2.34 percentage points of retention relative to the clipped estimate. This supports testing block-conditioned supervision before another architecture or loss sweep. It does not establish a new predictor improvement, explain the earlier approximately 3-token B16 prediction MAE, or demonstrate a throughput gain.

## Protocol

- Target: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Draft: `z-lab/Qwen3-4B-DFlash-b16`, revision `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Existing `z-lab/qwen3-4b-instruct-100k` prompt manifest, sampled only from current canonical validation prompt IDs; seed 913. No new external dataset.
- 2,000 matched states from 212 prompts; 1,991 nonterminal states used for primary metrics. Nine accepted-EOS states excluded. Thirteen additional prompts exceeded 2,048 tokens and were skipped, not truncated.
- Greedy, thinking disabled, maximum 256 new tokens; up to ten evenly spaced sampled positions per prompt. B16 determines the reference trajectory.
- Independently restored target and draft caches, identical prefix/anchor/context/positions, randomized alternative order. Actually draft and directly verify B4/B8/B12/B16 plus the exact lengths chosen by both frozen policies.
- B includes the known anchor. Draft budget is `b-1`, maximum 15; accepted length counts draft tokens only.
- One A100-PCIE-40GB, Transformers 4.57.1, torch 2.13.0+cu130, SDPA, BF16, TF32 disabled. Collection took 1,264 seconds (21.1 minutes), excluding setup/model loading.
- Source commit `170d38f`; no training or policy retuning during this diagnostic.

These are development diagnostics, not a fresh held-out ranking. Historical checkpoint training splits differ from the current canonical split, and their prompt-disjointness was not certified here. These new early-generation states are also not the old cached validation/Math500/HumanEval rows.

## Paired Labels

| Block | Mean clipped estimate | Mean actual accepted | Label MAE | Different labels | MAE, 95% prompt-bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- |
| B4 | 2.349 | 2.166 | 0.328 | 20.14% | 0.295-0.363 |
| B8 | 4.384 | 4.133 | 0.522 | 22.60% | 0.471-0.576 |
| B12 | 5.760 | 5.555 | 0.539 | 21.09% | 0.472-0.607 |
| B16 | 6.727 | 6.727 | 0 | 0% | Reference |

Here label MAE is disagreement between actual short-draft acceptance and clipped B16 acceptance, **not prediction MAE of an MLP**. Uncertainty uses 1,000 whole-prompt bootstrap resamples.

### Numerical Control

Verify the unchanged B16 candidate prefix at the same shorter width. This isolates target-width numerical effects from the changed draft sequence.

| Block | Target-width-only disagreement | Target-width-only MAE | Actual short draft vs same-width truncated B16 MAE |
| --- | ---: | ---: | ---: |
| B4 | 0.70% | 0.0105 | 0.3194 |
| B8 | 1.16% | 0.0236 | 0.5038 |
| B12 | 1.21% | 0.0281 | 0.5163 |

Thus changed drafts account for most observed differences; target-width numerics alone do not explain them. These MAEs are not an additive decomposition. A small FP32 smoke test also observed short-draft differences without target-width acceptance differences, but it is not a population estimate or a matched cross-precision comparison.

All five reverse-order cache-isolation checks passed. The full run's 42 sampled canonical greedy comparisons agreed; 6,720 same-width truncated verification controls were collected. Separate early BF16 smoke tests did encounter width-sensitive argmax ties, which motivated the control.

## Frozen Policies

Direct head: alpha 0.92, existing B4/B8/B12/B16 choices. Entropy head: threshold 2.6000006198883057, existing integer B2-B16 choices. No thresholds selected on these new states.

| Policy | Mean draft budget | Actual accepted | Clipped retention | Actual retention | Retention loss, percentage points (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct head | 11.094 | 6.489 | 98.79% | 96.45% | 2.34 (1.48-3.24) |
| Predicted entropy | 10.995 | 6.469 | 97.77% | 96.16% | 1.61 (0.99-2.30) |

Retention is total accepted draft tokens divided by B16 accepted draft tokens on these identical states. Both policies remain above 95% in this sample. The direct-head gap clearly exceeds the prior one-percentage-point screening threshold; the entropy gap's lower confidence bound is approximately that threshold.

| Policy | Actual aggregate accept ratio | Actual mean per-cycle accept ratio | Clipped aggregate ratio | Clipped mean per-cycle ratio |
| --- | ---: | ---: | ---: | ---: |
| Direct head | 0.5849 | 0.5175 | 0.5990 | 0.5404 |
| Predicted entropy | 0.5883 | 0.5430 | 0.5982 | 0.5671 |

Aggregate ratio is `sum(accepted)/sum(budget)`; mean cycle ratio is `mean(accepted/budget)`. They must not be interchanged. Neither is measured online throughput. This experiment does not establish a statistically significant ranking between the two policies.

## Source Breakdown

These counts describe this diagnostic sample, not the original dataset mixture.

| Manifest source | Prompts | Eligible states | B4 label MAE | B8 label MAE | B12 label MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| nemotron | 118 | 1,102 | 0.373 | 0.598 | 0.586 |
| evol_codealpaca | 27 | 263 | 0.350 | 0.548 | 0.574 |
| opencodeinstruct | 42 | 410 | 0.276 | 0.420 | 0.459 |
| openr1_math | 25 | 216 | 0.171 | 0.301 | 0.412 |

## Next Experiment

Use a separate subset of the canonical **training** prompts to collect actual per-length outcomes. Keep this diagnostic sample out of training. Begin with the same fused vector and a block-size-conditioned head:

`(fused_context, b) -> P(A_b >= k), 1 <= k < b`.

Compare it with the existing frozen heads and a block-only correction control. Fit calibration on a disjoint calibration subset, and compare actual draft budget at matched retention. This tests whether prefix-dependent conditioning helps beyond simply correcting each block's average bias. Expand beyond the pilot grid only after that comparison; the grid is not an eventual restriction to four arms.

Do not train on these validation states or launch a large collection solely on the strength of a claimed speedup: no online speedup has been measured here. Full B16 drafting followed by shorter verification of unchanged candidates remains a different case, where clipping is mathematically appropriate apart from the measured numerical effects.

## Artifacts And Checks

- Code: `scripts/diagnose_dflash_paired_lengths.py`; plotting: `scripts/plot_dflash_paired_lengths.py`; four unit tests in `tests/test_paired_lengths.py` pass.
- Git delivery branch: `codex/paired-length-diagnostic`.
- PVC: `/workspace/dflashv2_data/diagnostics/paired_lengths_2000_20260913`.
- Local: `outputs/paired_lengths_20260913/full`, including `summary.json`, `config.json`, prompt records, aligned fused vectors, and `plots/`.
- All 225 per-prompt JSON shards backed up; all 2,000 saved 2,560-dimensional fused vectors checked for alignment and finite values. Temporary collection IO was under `/tmp`, with per-prompt asynchronous atomic PVC backups.
- Summary SHA256 matches local and PVC: `39fa064890af0355139190f8ecec2ba018000dbb6766c19a75f59b6ede9bb5de`.
- Direct checkpoint SHA256: `7fc61bdcee28264cedbd5e8923d935bc423816a220aa9afe04e4eca71a33112c`.
- Entropy checkpoint SHA256: `62fe4cbca31c47521a49e95cde9fc077b77d4e25cb07cb696f118abf18e99217`.
- Complete paths and frozen settings are recorded in the run's `config.json`. Raw traces and model files are not committed to Git.
