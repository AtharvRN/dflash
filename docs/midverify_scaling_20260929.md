# Intermediate-target probe scaling: protocol and results

Authorized 2026-09-29 following the 2k feasibility pilot. This is a data and
optimization scaling study, NOT a new architecture or serving-engine change.

## Fixed data and collection

- Preserve every original 2k training state and all 342 calibration / 1,416
  assessment states from `midverify_probe_2k_20260929/cache`.
- Reuse their input features, candidate vectors, labels, confidence scores and
  metadata. Hash the array contents separately by group to establish bitwise
  preservation. Original files remain read-only.
- Replay only the additional 8,000 states in the immutable canonical 10k
  training selection from `actual_block_predictor_10k_finish_20260928/cache`.
  The 10k source has 1,317 training prompts; these are cycles, not 10k prompts.
- Same frozen Qwen3-4B and original DFlash B16 revisions, same saved B16
  candidates, BF16 SDPA / TF32 off, layers6/9/12/18/24, batches8. New features
  and labels come from the same verification forward; retain the replay audits.
- Fresh separate output, per-batch durable backups, time bound. Use only idle
  authorized GPU4. A 16-new-state append smoke precedes the full expansion.

## Controlled training

Nested training sizes 2,000 / 5,000 / 10,000, unchanged linear and 128-wide
GELU MLP heads, seeds913/914/915: **90 independently trained models**.
Inputs, masked binary loss, AdamW3e-4, weight decay.01, clip1 and FP32/TF32-off
are unchanged. Keep the same initialization for a given architecture/seed
across sizes, verified by weight hashes.

Use full 128-state minibatches drawn from a shuffled stream that carries the
tail across epoch boundaries. This makes exposures exactly matched at equal
update counts. It differs from the original pilot's partial final minibatch;
the rerun 2k models, not the old 2k weights, are the primary scaling control.

Evaluate calibration every 64 updates. Report best calibration-selected
checkpoints within each of these budgets:

1. **128 updates:** equal-compute short control (16,384 state exposures).
2. **1,024 updates:** equal-compute primary comparison (131,072 exposures).
3. **Extended:** at least 1,024 and at most 2,048 updates, stopping after eight
   calibration evaluations without improvement once the minimum is reached.

Selection minimizes calibration kept rows at >=96% accepted-token retention;
ties prefer higher retention and earlier update. Save calibration BCE and
learning curves too. Primary result reports all three seeds, not best-seed
assessment selection. All five retention thresholds (90/95/96/98/99%) are
chosen on calibration. Assessment is scored only after checkpoint selection.

Read the original 2k pilot's corrected `assessment_fp64` for fixed confidence
and lens baselines. Verify their calibration and assessment row identities and
scores against the preserved validation cache. All scores use FP64 threshold
comparisons, including NumPy exact-tie transitions.

## Evaluation / interpretation

Report retained accepted tokens, retained query rows, equivalent full-depth
rows, work per committed token, whole-prompt bootstrap intervals, and paired
10k-minus-2k intervals for the same depth/architecture/seed/update budget.
Show validation learning curves and size scaling. No throughput claims follow
from these work proxies. No actual target-layer pruning is implemented here.

No automatic escalation to larger targets, >10k data, wider probes, target
fine-tuning, hybrid confidence inputs or SGLang integration. A positive result
must first beat the matched post-draft confidence control with its early-layer
work accounted for. Prior oracle limits remain operating-point-specific work
bounds, not measured latency bounds.

## Collection-time amendment, before training

The completed 10k replay found one newly terminal training state: prompt54855,
cycle59, original row8273. Its same-forward accepted length changed from the
source's6 to9, crossing EOS at candidate8. The collector refused completion
and no training started on that cache. Both original 2k/5k subsets and all
validation states remain nonterminal.

An explicit CPU finalization rechecks every new shard against its receipt,
every materialized array against its shard or frozen seed, label/candidate
alignment, prefix hashes and preserved group bytes. It copies only nonterminal
rows into a separate `cache_nonterminal`, preserving the failed cache intact.
The largest training size is therefore **9,999**, not10,000. No replacement
state, label repair or validation change is introduced. All other preregistered
training and evaluation settings above are unchanged.

## Completed run and audit

Completed 2026-09-29 on workstation GPU4 (RTX PRO 6000 Blackwell Server Edition).
All 90 independent training trajectories completed, producing 270 selected
checkpoint policies (three update budgets per trajectory). Training/scoring/
reporting took 156.4 seconds, excluding collection and cache audits. Final GPU4
inspection showed 0 MiB and 0% utilization. No follow-on training, integration or
recurring monitor was launched.

| Nested training cycles | Distinct prompts | Supervised at-risk positions |
| ---: | ---: | ---: |
| 2,000 | 263 | 14,837 |
| 5,000 | 662 | 36,837 |
| 9,999 | 1,317 | 72,794 |

Validation remains 342 calibration cycles/47 prompts and 1,416 assessment
cycles/187 prompts. The three sizes are nested prefixes of the preselected
training order, apart from the single terminal exclusion beyond the 5k prefix.

Audits/checks:

- The 16-new-state append smoke and bounded CPU trainer smoke both passed.
- The finalizer rechecked all replay-chunk receipts and exact correspondence
  between materialized arrays and their seed/chunk evidence. Every original
  pilot array's training/calibration/assessment content hashes were preserved.
- Original replay labels agree in 11,616/11,757 retained states. The fresh labels
  remain authoritative and paired with same-forward features; this is not a
  bitwise reproduction of the original decoding trajectory.
- All eight new batch/single numerical controls agree on acceptance length;
  one differs at a posterior target argmax. Maximum hidden relative L2 is about
  1.63%. These controls do not establish bitwise equivalence.
- All frozen confidence/lens calibration and assessment metric dictionaries
  exactly reproduce the corrected pilot baseline.
- All 270 checkpoint reloads reproduce identical same-shape score arrays.
- Remote verification passed every completion-bound file for the nonterminal
  cache and all 279 training-result/checkpoint files. The 13 downloaded bound
  metadata/report/plot/score files were independently checksum-verified locally;
  large feature arrays and checkpoints were not downloaded.
- Focused final tests: 31 passed across `test_midverify.py`,
  `test_midverify_scaling.py`, `test_verifier_signal_replay.py` and
  `test_actual_block_predictor.py`.

Collection deployment: `78ff645`; scaling trainer: `592ce6e`; terminal finalization
and final training deployment: `3b0f012`. Run configs and hashes pin the actual
code and inputs. Source evidence and the failed raw-cache audit remain intact.

## Primary result: data helps, but does not beat confidence trimming

All rows below are at thresholds calibrated for >=96% retention. Assessment
retentions are the actual measured values, not constrained to 96%. Probe rows
are three-seed means at the equal 1,024-update budget. Costs are full-depth-
equivalent target query rows, NOT latency or throughput; lower is better.

| Policy | Training states | Assessment retention | Kept rows | Work proxy |
| --- | ---: | ---: | ---: | ---: |
| Replayed draft candidate logprob | None | 95.25% | 9.049 | 9.049 |
| L6 MLP | 2,000 | 95.72% | 11.246 | 12.038 |
| L6 MLP | 5,000 | 95.18% | 10.085 | 11.071 |
| L6 MLP | 9,999 | 95.48% | 9.739 | 10.782 |
| L9 MLP | 9,999 | 93.68% | 9.305 | 10.978 |
| L12 MLP | 9,999 | 94.20% | 9.363 | 11.575 |
| L18 MLP | 9,999 | 94.42% | 9.031 | 12.515 |
| L24 MLP | 9,999 | 95.77% | 9.063 | 13.688 |

L6's 2k-to-9,999 comparison reduces estimated work by 10.43%, while mean
assessment retention changes by -0.23 percentage points. Each seed's paired
whole-prompt bootstrap 95% interval for its work delta excludes zero; each
retention-delta interval includes zero. These are intervals conditional on
fitted policies, not proof of equal retention, noninferiority, or population-wide
robustness. The 9,999-state L6 retention range across seeds is 94.92–96.18%.

Despite the improvement, L6 still uses 19.15% more row-layer work than the
confidence control, at slightly higher retention (95.48% versus 95.25%). The
10k-scale linear L6 control has 95.67% retention and 11.234 work rows. Later
probes' smaller kept prefixes do not offset their full-width early-layer work.
Some miss the calibration retention target materially on assessment: do not
rank by work alone or call these exactly matched-retention comparisons.

The L6 MLP seed-mean aggregate acceptance ratio is 0.7195. This is **after
drafting AND six target layers**, not achievement of the original pre-draft
70% objective. The cheaper replayed draft-confidence control is 0.7791 on these
same states. Historical pre-draft/task tables are different comparisons.

### Optimization versus data

| L6 MLP size | Short 128-update work / retention | Primary 1,024-update work / retention |
| ---: | --- | --- |
| 2,000 | 12.096 / 96.39% | 12.038 / 95.72% |
| 5,000 | 11.291 / 95.61% | 11.071 / 95.18% |
| 9,999 | 11.501 / 96.46% | 10.782 / 95.48% |

These are calibration-matched, not identical assessment retention. At 128
updates, 9,999 does not uniformly dominate 5k. With the larger update budget the
larger dataset yields a better L6 work/retention tradeoff than the 2k control.

The extended budget selects the same L6 MLP checkpoints as the 1,024 budget at
all three sizes. The selected updates are 192/192/192 for 2k, 128/576/320 for 5k,
and 704/1024/512 for 9,999 (seeds 913/914/915). Thus simply training these models
longer did not give an additional L6 policy improvement. Training BCE keeps
falling while calibration BCE eventually rises, especially with 2k data; the
optimization plot shows overfitting, not merely a lack of optimization steps.
More data substantially reduces this generalization gap but does not eliminate
it. This is evidence against inferring an intrinsic information ceiling from
the small pilot; it is not evidence that an arbitrarily larger sweep will win.

### Higher-retention operating points

| Calibration target | Confidence assessment retention / work | 9,999-state L6 MLP assessment retention / work |
| --- | --- | --- |
| 90% | 89.16% / 7.695 | 90.67% / 9.947 |
| 95% | 93.76% / 8.638 | 94.43% / 10.555 |
| 96% | 95.25% / 9.049 | 95.48% / 10.782 |
| 98% | 97.81% / 10.226 | 97.87% / 11.329 |
| 99% | 98.81% / 11.030 | 99.13% / 11.844 |

The work gap narrows at higher retention, but these measured points still do
not establish a probe advantage over the matched confidence baseline. No
assessment-driven layer/seed selection or new threshold tuning was performed.

**Decision:** preserve the data-scaling improvement as a real finding; do not
claim the probe is useless or that intermediate-target signal is exhausted. This
test still does not justify segmented SGLang integration or a throughput claim.
Any further proposal needs a path below the strong post-draft confidence work
frontier, including early target computation and probe/compaction overhead.

## Full paths

Remote run root:
`/data/scratch/zekaili/atharv/dflash/runs/midverify_scaling_10k_20260929`

- `cache/`: unmodified 10k capture, including the failed terminal-state audit.
- `cache_nonterminal/`: completed 9,999-state training cache, frozen validation,
  `source_verification.json`, audited exclusion and completion hashes.
- `training/`: all 270 checkpoints, config, histories, scores, summary,
  completion hashes and plots. The wrapper exited 0.

Local copied artifacts:
`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_scaling_10k_20260929`

- [Data-scaling plot](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_scaling_10k_20260929/training/data_scaling.png)
- [Optimization/overfitting plot](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_scaling_10k_20260929/training/optimization_L6.png)
- [Machine-readable results and all paired intervals](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_scaling_10k_20260929/training/summary.json)
- [Generated compact report](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_scaling_10k_20260929/training/report.md)

Subsequent user-authorized follow-up: the bounded
[confidence-first L6 cascade test](midverify_cascade_20260929.md) is complete.
It reuses this cache without collection and evaluates target information beyond
confidence/candidate-only controls. Gains are modest and retention differs;
no online integration was launched.
