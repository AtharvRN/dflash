# Intermediate-target probe scaling: pre-run protocol

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
