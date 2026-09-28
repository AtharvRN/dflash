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
