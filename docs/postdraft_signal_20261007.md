# Matched post-draft signal audit

Prepared Oct 6 PDT / Oct 7 UTC 2026. This is a registered experiment, not a result.

## Scope and source

Frozen original Qwen3-4B / DFlash B16; unchanged greedy proposals; no editor,
redrafting, new target forwards, history feature collection, or cluster changes.
The completed cache at
`/workspace/dflashv2_data/runs/soft_supervision_2k_r2_20261005/cache` is reused.
Its 2,000 / 342 / 1,416 planned train/calibration/assessment cycles become
1,981 / 336 / 1,399 after its existing anchor/EOS exclusions. Prompt groups and
row ordering remain fixed. Assessment is repeatedly inspected development data.
This is not the historical 172,794-row validation dataset.

Existing hard / hard+TV / hard+margin candidate-vector heads are recovered for
all three seeds, not retrained or selected using assessment. Their cached raw
confidence control achieved 0.77346 ratio at 0.95404 assessment retention, versus
approximately 0.716–0.722 ratio at 0.942–0.944 retention for those learned heads.
These replay metrics are not closed-loop serving ratios or speedups.

## Registered new arms

1. Confidence MLP: log candidate probability, entropy, normalized position;
   3–128–1, GELU/dropout .05. The old third statistic, candidate logit minus max,
   is identically zero for greedy candidates and is not called a top-two margin.
2. DSpark-form hard: linear `[draft_hidden; learned previous-token embedding32]`
   predicts greedy match using at-risk BCE.
3. DSpark-form TV: identical initialization/architecture/order, with TV-overlap
   soft BCE instead. TV at temperature one is not greedy acceptance probability.
4. Frozen confidence + local residual: hidden/candidate vectors projected to 32
   each; concatenate h, e, h*e, duplicate this 96-vector and append confidence
   features; 195–128–1, zero final layer.
5. Frozen confidence + prefix residual: identical parameters, but second
   96-vector is the mean of projected features from positions 1 through j.

The residual comparison is matched in capacity and initialization. No invented
semantic states. Draft hidden, candidate identities/vectors, and draft statistics
are the only inference inputs. Target information is supervision only.

DSpark-form means a frozen-DFlash adaptation of Eq. 7/8 in
https://arxiv.org/html/2607.05147v1, NOT full DSpark: the embedding is separately
trained rather than shared with a trained Markov correction module; rank is 32,
not its default 256. We use greedy at-risk masking and our common offline policy
family, not its full joint training, STS calibration, or serving scheduler.

## Training and evaluation

Seeds 913/914/915; 1,024 updates; batch 128; AdamW 3e-4, weight decay .01,
gradient clipping 1; TF32 off. Same shuffled minibatch stream. No sweep. All
normalization and confidence-bin edges use eligible training at-risk positions.
No auxiliary TV/margin loss on the proposed residual arm in this first test.

Every 64 updates, calibration selects the checkpoint with minimum mean retained
query rows at >=96% retention (then higher retention, then earlier update).
Step zero is included so the residual can choose its unchanged frozen base.
Each model uses the same two policy candidates: first-low-local-score gate or
cumulative conditional log-survival gate. Calibration alone chooses rule and
threshold for each retention target: 90/95/96/98/99/100%. This common family is
an offline predictor comparison, not a claim to reproduce DSpark scheduling.
The raw confidence baseline is evaluated under both rules as well.

Report accepted tokens, aggregate accepted/proposed ratio, mean verification
length INCLUDING anchor, assessment retention, and paired prompt-bootstrap
differences versus raw. Different realized assessment retention is not matched
retention merely because calibration targets match. Inspect full operating
curves and uncertainty; do not tune on assessment results.

Diagnostic: greedy-match AUC within position and training-fitted confidence
quintile, at-risk positions only. It measures residual ranking signal; bins are
coarse and do not remove all confidence variation. No fundamental information
limit can be inferred from a negative result on this small dataset.

## Execution and limitations

Runner: `scripts/run_postdraft_signal_nrp.sh`. Tests and an eight-update smoke
precede full fitting; fresh output paths; GPU locks and occupancy checks;
source checksums; persisted checkpoint reload; per-model PVC backups; bounded
timeouts. Base Torch 2.13.0+cu130 is sufficient for cache-only training; target
Transformers is not invoked. Preserve all earlier artifacts.

Planned durable output:
`/workspace/dflashv2_data/runs/postdraft_signal_20261007/r1`.
Do not infer throughput, cross-drafter generalization, or stochastic correctness
from these offline greedy results. Serving profiling follows only if useful
signal survives the matched controls. This test does not establish novelty.
