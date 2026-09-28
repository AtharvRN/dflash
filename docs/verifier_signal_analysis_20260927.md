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
