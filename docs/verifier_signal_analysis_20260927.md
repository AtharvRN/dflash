# Causal verifier-signal analysis

## Protocol fixed before replay

Question: does information already available before drafting add useful acceptance
signal beyond the collected fused vector? The completed paired cache has no saved
verifier probabilities; historical confidence experiments must not be represented
as newly verified results.

Use the same label-independent plot sample (2,766 states, at most four/prompt).
Replay each exact saved prefix, including its known anchor, with the original
frozen Qwen3-4B and DFlash B16 revisions, BF16/SDPA, TF32 off. Start with a
16-state smoke. Use only a GPU with less than 1 GiB reported usage; do not modify
other jobs. Bound collection at 1,800 seconds; preserve partial evidence.

This is full-prefix inference replay, not the original incremental GPU execution.
Capture fresh fused features and a fresh B16 label from the same draft forward.
Keep old labels solely as replay controls. Record anchor-argmax agreement,
original/fresh label agreement, fused-vector cosine, terminal status, prefix and
file hashes. Primary analyses exclude changed-anchor and accepted-EOS states.
Do not join freshly computed inputs to old acceptance labels. Preserve the
original cache and fixed train/calibration/assessment prompt membership.

## Availability boundary

For prefix `x[:start]` and known anchor `x[start]`:

- Pre-draft target distribution is `p(. | x[:start])`, at position `start-1`.
  It produced the known anchor; it is NOT the distribution of the first proposed
  draft token conditional on the anchor.
- Save the full 151,936-way distribution in FP32, final target hidden vector,
  anchor embedding, and same-forward fused vector.
- Save confidence history for the last 16 committed positions, masking prompt
  interior positions: original prefill computed only its last logit. Previous
  committed verification positions are causally available. Never treat logits
  after a previous rejection as distributions on the committed continuation.
- Save current draft-confidence and current verifier-confidence profiles ONLY
  as explicitly post-draft/post-verification diagnostics. Current verification
  plus candidate identities already determines acceptance; it is not a useful
  pre-draft predictor and would be leakage if supplied to one.
- Probabilities are softmax at temperature 1 for diagnostic confidence, while
  candidate generation and acceptance remain greedy.

## Planned matched analyses

Fit all transformations on replay training prompts; choose Ridge regularization
using calibration MSE, never assessment. Report MAE/R2 and paired prompt-bootstrap
intervals as signal probes, not policy/throughput results. This is a smaller
fresh-label dataset, so do not compare absolute MAEs with the previous 20,970-row
analysis or older MLP results.

Compare source/progress controls, fused PCA64, final-target-state PCA64,
anchor-embedding PCA32, current causal confidence, confidence history, and
full probability-vector PCA64. Add each causal feature group to fused controls
to measure incremental information. Use square-root probabilities before PCA
so Euclidean distance corresponds to Hellinger distance up to a constant. Also
inspect sorted top-probability shape to distinguish concentration from token
identity. Learned PCA features avoid manually labeled semantic states.

Plots: joint PCA/t-SNE with points colored by exact acceptance, 16-panel grouped
views on shared coordinates, and per-length probability/entropy distributions.
Labels only color/group points; they do not construct embeddings. Full target
probability vectors may cluster predominantly by predicted-token identity, so
visually cleaner clusters alone are not grounds for adding a signal.

## Research hypothesis, not an established result

The recent fused/raw pilot omitted the known anchor even though the drafter
conditions on it. That omission and the final target state are specific causal
inputs worth testing; history/entropy alone and heavier context architectures
have already been explored. Do not claim anchor conditioning is novel or untested
throughout all historical work: token-input variants were explored earlier.

For a final adaptive policy, the relevant estimand is the distribution of actual
block-conditioned acceptance given pre-draft information, not clipped B16 labels.
Target confidence describes only one side of draft–target agreement. A useful
signal must improve a matched held-out conditional predictor or policy frontier,
not just correlate with acceptance or improve a t-SNE plot.

## Completed results

Collection commit `279c815`; primary analysis `d1d869e`; nonlinear follow-up
`7ecae92`; distributional follow-up `ca78b2b`. Ten targeted unit tests passed.
One GPU was used for frozen-model replay (203.6 seconds after loading), then
released. All subsequent probes used four CPU threads. No target/draft weights
were updated, original caches/splits were not changed, and no recurring monitor
was created.

Workstation roots, all under `/data/scratch/zekaili/atharv/dflash/runs/`:

- `verifier_signal_smoke_20260927`: 16-state smoke, all anchor checks pass.
- `verifier_signal_replay_20260927`: full distributions and fresh paired data.
- `verifier_signal_analysis_20260927`: PCA/t-SNE, primary linear probes, figures.
- `verifier_signal_nonlinear_20260927`: exploratory boosted-tree check.
- `verifier_signal_survival_20260927`: exploratory proper-scoring check.

The last three output directories were copied to local `outputs/` with the same
names. Full 151,936-way probability arrays remain on the workstation. Collection
and primary-analysis completion files bind their evidence by SHA256.

### Replay audit and sample

Of 2,766 replayed states, 2,728 preserve the original greedy anchor; those are
the primary eligible states. There were no additional accepted-EOS exclusions.
Original/fresh acceptance agrees on 2,647/2,766 states. All 119 label changes
occur among anchor-matched states. Mean fused cosine is 0.9998406, median
0.9999147, minimum 0.9449522; four rows have cosine below 0.99. These four retain
A=15 and are not selectively removed. The replay changes hardware/execution
strategy from the old incremental run; the causes of all deviations were NOT
isolated. Do not attribute everything to hardware alone or mix these rates with
the separate actual-shorter-block disagreement rates.

Every new feature is paired with its freshly verified label. Counts:

| Partition | Cycle rows | Prompts |
| --- | ---: | ---: |
| Training | 1,812 | 464 |
| Calibration | 184 | 47 |
| Assessment | 732 | 187 |

The fixed prompt assignments are preserved. This is a small, previously inspected
development sample, not the full pilot assessment or a final test.

### Confidence is correlated, but incremental gains are weak

Assessment Spearman correlations with A: entropy **-0.45388**, known-anchor
probability **+0.44685**, top1/top2 probability margin **+0.44173**. The confidence
boxplots show a clear marginal trend: long accepted blocks often follow sharply
peaked target distributions. Many short blocks also follow peaked distributions;
target certainty is not sufficient for draft–target agreement.

Primary Ridge probes all include the same source/progress controls. PCA fits
only train; alpha is selected by calibration MSE. Negative deltas mean lower MAE.

| Probe | MAE | Delta vs fused | 95% paired prompt-bootstrap interval |
| --- | ---: | ---: | ---: |
| Fused PCA64 | 2.97954 | 0 | reference |
| Fused + latest verifier confidence | 2.99041 | +0.01087 | [-0.02501, +0.04350] |
| Fused + verifier history | 2.97155 | -0.00799 | [-0.05804, +0.04556] |
| Fused + probability PCA64 | 3.00205 | +0.02251 | [-0.02956, +0.07627] |
| Fused + sorted probability shape | 3.00859 | +0.02905 | [-0.02053, +0.07769] |
| Fused + anchor embedding | 2.97174 | -0.00780 | [-0.05155, +0.03432] |
| Fused + final target state | 3.01884 | +0.03930 | [-0.00844, +0.08937] |
| Fused + anchor + confidence | 2.97771 | -0.00183 | [-0.05092, +0.04409] |

Confidence + controls without fused gives MAE 3.47220; probability PCs + controls
give 3.56625; source/progress alone gives 3.61223. Thus probabilities carry signal,
but these probes do not establish additional predictive information beyond fused.

Current draft-confidence-only (+ controls) gives MAE **1.80444**. Current
verifier-confidence-only (+ controls) gives **1.71628**. Both are future
information relative to the initial block-size decision. The latter is a
label-related diagnostic, not a useful formal bound: full verification and draft
identities already determine A exactly.

Probability PC1/PC2 explain 4.92% of variance in square-root-probability space;
64 PCs explain 39.33%. A post-hoc descriptive 15-neighbor check in saved t-SNE
coordinates, excluding same-prompt neighbors, finds same-anchor fractions 31.42%
for probability geometry versus 6.84% for fused geometry (random-pair reference
0.446%). Mean neighbor A difference is 5.221 versus 4.275. This suggests many
probability neighborhoods reflect token identity, not acceptance grouping. It is
a 2-D descriptive check, not a held-out feature-selection or significance test.

### Nonlinear and distributional follow-ups

These follow-ups were added after inspecting primary results. They are
exploratory, not a preregistered confirmation. Hyperparameters still use only
calibration; assessment is not used to choose settings within a model family.

Boosted trees (200 iterations, learning rate .05, min leaf20; leaves7/15 and
L2=1/10 selected by calibration MSE):

| Input | MAE | Delta vs same-family fused | 95% paired interval |
| --- | ---: | ---: | ---: |
| Fused | 2.99851 | 0 | reference |
| + latest verifier confidence | 3.03828 | +0.03976 | [-0.05042, +0.13324] |
| + verifier history | 3.03671 | +0.03819 | [-0.06142, +0.13683] |
| + anchor + confidence | 3.02819 | +0.02968 | [-0.05920, +0.12167] |

Conditional-success logistic heads, correctly censored at A=15; regularization
C in .01/.1/1/10 selected by calibration first-rejection NLL:

| Input | Assessment NLL | Integrated survival Brier | Expected-A MAE |
| --- | ---: | ---: | ---: |
| Fused | 2.33510 | 0.14084 | 3.38250 |
| + latest verifier confidence | 2.33047 | 0.14031 | 3.34821 |
| + verifier history | 2.34612 | 0.13793 | 3.24834 |
| + anchor + confidence | 2.33071 | 0.13817 | 3.29763 |

Every added-feature NLL and Brier paired interval includes zero. History and
anchor+confidence DO improve MAE within this weaker survival-head family, with
paired intervals excluding zero, but not the likelihood and Brier criteria;
they do not establish a robust large gain over the stronger mean probes. Do not
report "no effect in every metric". No policy/throughput was measured.

### Decision after this analysis

Verifier uncertainty is a useful marginal signal, not an established missing
ingredient for a much stronger pre-draft predictor. This is evidence from limited
probes and sample size, not a proof of conditional independence or a Bayes limit.
The anchor omission was worth checking; this run does NOT justify presenting
anchor conditioning as a breakthrough or a new paper contribution.

The next discriminating development test should use the already collected
actual-integer-block data, rather than redoing its hindsight oracle analysis:
does a small shared head predict actual block-conditioned acceptance/relative
block benefit on held-out prompts better than the existing B16 proxy? Keep
feature/architecture comparisons matched. Learn survival over token horizon k
conditioned on proposed B; enforce horizon survival, not monotonic acceptance
across B. All-accepted observations are censored at that block's budget.
Evaluate calibration/proper scores and actual budget–retention curves, not MAE
alone. A small prompt-disjoint development test should precede larger collection
or training. No such policy training was launched in this analysis.

The large post-draft advantage is not a justification to repeat generic teacher
distillation: earlier matched controls already found it neutral/slightly worse.
If actual-B supervision also fails to improve policy decisions, a materially
stronger project may require changing how the drafter is trained or relaxing the
strict pre-draft constraint; neither change is assumed authorized here.
