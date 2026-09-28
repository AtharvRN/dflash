# Pre-draft acceptance geometry / recent-text analysis

Protocol fixed before running on the UCSD workstation. This is a CPU-only
analysis of existing data, not new drafter training, collection, or a deployment
benchmark. The completed paired pilot is used; the unfinished expansion is not.

## Questions

1. Do pre-draft hidden features form neighborhoods with similar B16 acceptance?
2. Is that organization predominantly dataset source/prompt identity, or does
   it also track acceptance variation within a request?
3. Does lexical content in recent committed text generalize across prompts,
   beyond original-prompt text, source, and decoding progress?
4. Is there useful temporal persistence in previously observed acceptance?

## Data and safeguards

- Completed `prefusion_pilot_20260915/cache`: 13,935 train, 1,373 calibration,
  5,662 assessment cycle rows. Preserve exact prompt membership.
- Check all original audit-bound file hashes, per-shard hashes, state-prefix
  hashes, anchor identities, and row/cycle/label alignment before analysis.
- Raw vectors: 12,800 dimensions; actual fused vectors: 2,560 dimensions.
  Both precede the current draft backbone. Labels count 0–15 proposed tokens.
- Text windows end strictly before the current anchor, matching feature timing.
  No rejected tokens, current candidates, verifier outputs, or future text are
  inputs. Prior-label features use only earlier observed cycles.
- Validation was inspected in earlier work: development assessment, not final
  untouched tests. No model/hyperparameter choice is made on assessment labels.

## Descriptive analyses

- Row-L2-normalize representations; fit centered randomized PCA64 on train only.
  Plot raw/fused PC1–PC2 and report explained variance.
- For t-SNE, at most four uniformly sampled cycles per prompt (seed 927), with
  no label balancing or label-based selection. Use first 50 fused PCs, p=15/50
  with seed 0 and p=50 with seed 1; show all maps, not a selected attractive one.
- Recolor the same coordinates by exact acceptance 0–15, upstream source, and
  A minus the full same-prompt mean. The latter is descriptive, never an input.
- Compute 15-neighbor label differences in PCA64, excluding all same-prompt
  neighbors. Compare a source/cycle-bin-stratified shuffled-label reference;
  this is descriptive, not a cluster-independent significance test.
- Decompose label sums of squares between/within prompts; examine lags 1/2/4/8/16
  before and after prompt demeaning, against within-prompt shuffles.

## Diagnostic predictive probes

These are inexpensive linear/retrieval tests of accessible signal, not a new
block-size policy. Labels remain B16 outcomes; no clipping-policy claims.

- Training-mean baseline; source/progress Ridge control.
- Control plus original-prompt or recent 16/64/256-token unigram/bigram TF-IDF.
  Token IDs preserve tokenizer units. This is lexical, not a semantic encoder.
- Control plus causal previous/running/EMA acceptance summaries.
- Control plus raw/fused PCA64; fused-plus-history and recent64-plus-history.
- Fixed k=50 retrieval from training prompts in PCA64 for both representations.
- Fit vocabulary, PCA, and scaling only on training. Select Ridge alpha from
  0.1/1/10/100 by calibration MSE. Clip predictions to the observed 0–15 range.
- Report assessment MAE, RMSE, R², Spearman, equal-prompt MAE, descriptive
  within-prompt R², and 1,000 prompt-bootstrap MAE/difference intervals.

No GPU is required; cap CPU math libraries at four threads. Install analysis
dependencies into a separate task-owned directory, leaving the model environment
unchanged. Deploy the analysis script through git push/pull. Save complete
configuration, software versions, inputs' hashes, coordinates, and predictions.

PCA maximizes feature variance, not label separation. t-SNE is an exploratory
local-neighborhood visualization; separation is not evidence of held-out
predictability, and overlap is not evidence of unpredictability. Even genuine
B16 acceptance signal does not establish predictability of actual best block
size, nor an online speedup.

## Completed run: 2026-09-27

Analysis commit: `d4232ffccc63aa22c9ab7191cc590832a54a2f0a`.
All original audit bindings, shard hashes, label/cycle alignment, prefix hashes,
and anchor identities passed. Five targeted tests passed locally and remotely.
No target or draft model was loaded, no GPU work was launched, and the partial
100k expansion was not changed. Runtime was 82.1 seconds after dependency setup.

Durable workstation output:
`/data/scratch/zekaili/atharv/dflash/runs/acceptance_geometry_pilot_20260927`.
Local copy: `outputs/acceptance_geometry_pilot_20260927`.
`COMPLETE.json` binds the final `summary.json` SHA256
`017523cd5fe6f8846dd8cf80ce5ca263603e1ec439a43eef53df34d178eea80f`.

Actual partitions contain 13,935 training rows / 464 prompts, 1,373 calibration
rows / 47 prompts, and 5,662 assessment rows / 187 prompts. Descriptive figures
use 2,766 rows across all 698 prompts, at most four rows per prompt. These are
the completed paired pilot, NOT the historical full validation dataset.

### Geometry and temporal findings

- Fused PC1/PC2 explain only **5.78%** of feature variance; 64 PCs explain
  **34.63%**. Raw equivalents are 7.79% and 39.52%. A mixed 2-D scatter therefore
  cannot rule out predictive information in the original vectors.
- PCA/t-SNE show overlapping acceptance gradients and neighborhoods rather
  than 16 separable classes. Similar broad organization appears under both
  tested perplexities; seed changes at perplexity 50 have little visible effect.
  Source coloring shows some source-specific structure, but not a complete
  explanation of acceptance coloring.
- With all same-prompt neighbors excluded, mean absolute label difference among
  15 fused-PCA neighbors is **4.183**, versus **5.789** under a source/cycle-bin
  stratified label shuffle (27.7% lower). Raw: 4.288 versus 5.755 (25.5% lower).
  This is a descriptive neighborhood statistic, not prediction MAE or a formal
  independent-sample significance test.
- **66.17%** of label sums of squares are within prompts; **33.83%** are between
  prompt means. This is a finite-sample descriptive decomposition, not a
  noise-corrected variance-components estimate or an adaptive-policy gain bound.
- Raw lag-1 acceptance correlation is **0.543**; after full-prompt demeaning,
  correlations at lags 1/2/4/8/16 are **0.305 / 0.165 / 0.069 / -0.038 / -0.143**.
  The within-prompt shuffled reference is approximately -0.033 (negative partly
  because finite sequences are demeaned). By lag 8, observed correlation is near
  this reference. Different lags have different eligible state pairs; do not
  interpret the lag-16 negative value as evidence for a semantic alternation.

### Held-out diagnostic results

Every nonconstant linear probe below includes the same source/progress control.
Its inputs are source one-hots, prompt/prefix lengths, cycle, and generated-token
offset. Text features are token unigram/bigram TF-IDF, not semantic embeddings.
PCA/vocabulary/scaling fit only training; Ridge alpha is selected by calibration
MSE. Assessment labels are not used for fitting or selection.

| Added information | Assessment MAE | R² | Within-prompt R², descriptive |
| --- | ---: | ---: | ---: |
| None: source/progress control | 3.7111 | 0.2599 | -0.0222 |
| Original-prompt lexical features | 3.7412 | 0.2600 | -0.0051 |
| Most recent 16 tokens | 3.5915 | 0.3284 | 0.0894 |
| Most recent 64 tokens | 3.6893 | 0.3021 | 0.0734 |
| Most recent 256 tokens | 3.7309 | 0.2654 | -0.0013 |
| Causal prior-acceptance summaries | 3.3287 | 0.3778 | 0.0945 |
| Raw PCA64 | 3.0693 | 0.4621 | 0.2352 |
| Fused PCA64 | 2.9641 | 0.4959 | 0.2815 |
| Fused PCA64 + prior acceptance | 2.9211 | 0.5056 | 0.2855 |

Feature-only k=50 retrieval from training prompts gives MAE **3.0716** with
fused PCA64 and **3.2022** with raw PCA64. It uses neither source/progress inputs
nor same-prompt neighbors. The constant training-mean MAE is 4.6092; the training
median constant gives 4.3207. The mean is not an MAE-optimal constant baseline.

Paired prompt-bootstrap differences, lower is better (1,000 resamples, seed927):

| Comparison | MAE difference | 95% paired interval |
| --- | ---: | ---: |
| Recent16 minus source/progress | -0.1195 | [-0.1803, -0.0564] |
| Fused PCs + history minus fused PCs | -0.0429 | [-0.0616, -0.0255] |
| Fused PCs minus raw PCs | -0.1053 | [-0.1489, -0.0562] |

The last two contrasts were calculated after the run from the saved predictions
with exactly the same paired prompt-resampling method; no probes were refit.
Intervals are conditional on these fitted probes, not multi-seed training
uncertainty. Within-prompt R² removes full-prompt means from both predictions
and labels only for evaluation; those means are unavailable at prediction time.

### Research interpretation

1. Request identity alone is insufficient to summarize B16 acceptance. There
   is substantial within-request variation, short-range temporal dependence,
   and cross-prompt predictive signal. This motivates evaluating cycle-level
   adaptation, but does not contradict request-level policies or establish
   that actual best block sizes vary to the same degree.
2. Recent lexical content adds a modest signal beyond the controls; longer
   lexical windows do not automatically help. This does not establish the
   optimal semantic context length or refute semantic text encoders.
3. Much of the accessible signal is already present in the inexpensive fused
   state. Adding history to the fused linear probe improves MAE by only 0.043;
   these results alone do not justify another large context encoder sweep.
4. Before another policy-training sweep, test the analogous within-request and
   cross-prompt structure of **actual block-conditioned outcomes** on matched
   prefixes. B16 A=15 is censored, and changing draft width changes candidates;
   these plots cannot infer B24/B32 outcomes or gains from clipping B16 labels.

See `report.md` in the output directory for all five figure sets, and
`summary.json`, `probe_predictions.npz`, and `plot_coordinates.npz` for the
machine-readable evidence. Saved decoded recent-text snippets are local research
artifacts; no manually labeled semantic states were introduced.
