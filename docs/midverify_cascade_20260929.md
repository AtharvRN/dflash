# Confidence-first / L6 correction: bounded preregistered test

Authorized 2026-09-29 after the controlled 2k/5k/9,999 scaling study. Question:
does intermediate target computation add useful information beyond already
available drafter confidence and candidate identity, after paying for that
computation? No new collection, model fine-tuning, SGLang integration or monitor.

## Frozen inputs and scope

Use `midverify_scaling_10k_20260929/cache_nonterminal`: 9,999 train cycles,
342 calibration and 1,416 assessment cycles. Preserve prompt membership, saved
B16 candidates, all features and same-forward greedy labels. Verify completion
hashes and the previous scaling result binding before any fitting.

Only L6 is used. Full B16 drafting cost is unchanged. First-stage thresholding
of saved replayed candidate logprob chooses K0 query rows including the anchor.
Second-stage scoring chooses K1 <= K0. Work proxy is
`[6*K0 + 30*K1]/36`, not `[6*16 + 30*K1]/36`. This is not measured latency;
predictor/packing/graph overhead and nonlinear attention costs are unmeasured.

Cached L6 vectors come from full-width verification. Causal prefix invariance
justifies their use for this offline diagnostic in exact arithmetic, but does
not establish numerical parity of actual shortened or segmented GPU forwards.
No deployed correctness or stochastic-sampling guarantee is claimed.

## Three learned feature ablations, three seeds each

All heads have one 128-wide GELU hidden layer, dropout .05, and one binary output
shared across candidate positions. Inputs:

1. `confidence_only`: three draft statistics: candidate logprob, entropy and
   candidate-minus-top logit. All available before target verification.
2. `candidate_confidence`: RMS-normalized frozen candidate LM-head vector
   (2,560 dimensions) plus the same three statistics; also target-free.
3. `target_candidate_confidence`: `[RMS(h_L6), RMS(e), RMS(h_L6)*RMS(e)]`
   (7,680 dimensions) plus the same statistics.

Confidence statistics use means/stds fitted only on training at-risk positions.
Candidate and hidden normalization have no learned parameters. Hidden width is
matched; input dimensionality and parameter counts are NOT matched. The
candidate-only control prevents attributing token-identity gains to L6 work.

Train nine independent trajectories: seeds913/914/915, all 9,999 states,
AdamW3e-4, weight decay.01, clip1, FP32/TF32 off, full128-state shuffled batches,
1,024 updates each. Same first-rejection masked BCE, no auxiliary objective.
Assess calibration every64 updates. No learning-rate or architecture search.

## Joint calibration and selection

Primary operating points: 96% and 99% calibration accepted-token retention.
For each checkpoint, search all distinct prefix-score threshold transitions
jointly, minimizing total row-layer work subject to the aggregate retention
constraint. First threshold uses draft candidate logprob; the second uses the
learned score. Include keep-all and explicit second-stage bypass. Integer
lengths 1–16 remain available, not a fixed four-arm grid.

For target-free controls, BOTH thresholds act before the target, so charge
`K1`, not six layers on K0. For target-feature models, charge the actual cascade
proxy above. Ties prefer greater accepted-token total, then bypass/fewer front
rows and fixed transition ordering. Select checkpoints independently per
operating point using calibration work, then retention, then earlier update.
Assessment never selects checkpoints, thresholds, seeds or feature variants.

Save both chosen checkpoints per trajectory (18 policies), calibration history,
normalization, score arrays, settings and exact-reload checks. Calibration and
assessment scoring use separately fixed batching, preventing a shape change
from altering the scores used for calibration versus assessment application.

## Controls and reporting

- Frozen raw draft-confidence policy from the preceding study, exactly reproduced.
- Frozen L6 target-only MLP from that study, with newly jointly calibrated cascade
  thresholds, no weight changes (three seeds). This isolates staging from fusion.
- Both newly learned target-free controls, allowed the same two-threshold search.
- A clairvoyant second-stage oracle following frozen confidence, preserving its
  exact accepted tokens; diagnostic bound only, never a trainable policy.

For each seed/operating point, choose the strongest target-free family using
calibration only (raw confidence, confidence-only head, candidate-confidence
head). Report every family separately as well as this reference choice.
Report actual assessment retention, K0/K1, work/committed token, held-out risk
BCE, and paired whole-prompt bootstrap intervals (2,000 resamples) versus raw
confidence and versus the calibration-chosen target-free reference.

These are calibration-matched comparisons; assessment retentions can differ.
Do not claim a win from smaller work alone when retention is also lower.
Bootstrap intervals condition on fitted policies and do not cover calibration
selection or training uncertainty. This repeatedly inspected assessment remains
development data, not an untouched paper test. No automatic integration follows.

## Execution

CPU unit tests and a bounded three-model CPU smoke precede the main GPU run.
Use only the idle authorized GPU4, checking immediately before allocation.
Fresh durable run destination, max runtime1,800 seconds, no overwrite or monitor.
Models/data remain under `/data/scratch/zekaili/atharv/dflash`; deploy via git.
