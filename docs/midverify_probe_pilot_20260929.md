# Intermediate-target acceptance probe: bounded feasibility pilot

Authorized 2026-09-29 after reviewing related mechanisms in FASER,
Speculative Streaming and Lever. This is **not a claim of a novel mechanism**.
No SGLang serving code is modified by this pilot.

## Question and fixed data

Does intermediate target computation give a useful retention/work frontier
over replayed drafter-confidence trimming on unchanged B16 candidates?

- Target Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- DFlash B16 revision `b74e3a329c4d963783143b1e970d95b002be72bd`.
- First 2,000 eligible rows in the audited, immutable 10k training selection.
- All existing 342 calibration and 1,416 assessment states, prompt groups fixed.
- Input sources: `runs/actual_block_predictor_10k_finish_20260928/cache`
  and `runs/policy_granularity_20260927/cache` beneath the workstation data root
  `/data/scratch/zekaili/atharv/dflash`.

Saved committed prefixes, anchors and 15 B16 candidate IDs are immutable.
Recompute target labels with intermediate features in the SAME target forward;
old labels are an audit comparison, never supervision for new features.
The saved states are conditioned states, not fresh closed-loop trajectories.

## Collection and correctness scope

BF16 SDPA, TF32 off, Transformers 4.57.1. Length-sorted batches of eight with
explicit left-padding masks and per-request logical positions. Final arrays
retain original row ordering. Independent single-state replays check numerical
drift on spread-out batches. Greedy-only; no stochastic exactness claim.

Capture after 6, 9, 12, 18 and 24 completed target decoder layers, before
final RMSNorm. Candidate j is predicted by query row j-1. Query row zero is
the known anchor. Retaining K query rows permits K-1 proposals and one bonus;
the fixed-candidate acceptance is min(A_B16, K-1), and a hindsight keep-all-
accepted oracle needs A_B16+1 rows.

Capture the candidate's frozen LM-head weight vector. An untuned logit lens
(target final RMSNorm + original LM head) supplies rank and margin baselines;
this diagnostic full-vocabulary projection is NOT assumed free at deployment.

Draft baseline: recompute the B16 draft distribution using full-prefix target
features and score the **saved** candidate IDs. This is not identical to the
original incremental cached draft computation; report candidate-argmax drift.
All policies use exactly the same candidates and fresh target labels. Report
anchor and acceptance replay differences rather than silently dropping them.
Fresh accepted-EOS cases halt automatic training pending a censoring review.

## Probe and evaluation

Candidate-conditioned linear and 128-wide GELU MLP heads. Inputs are separately
RMS-normalized target h and candidate vector e, concatenated with h*e (7,680
dimensions). MLP dropout .05. Binary match loss covers only the at-risk prefix
through the first rejection; no fictional rejection beyond position 15.

Eight epochs, AdamW 3e-4, weight decay .01, clip1, 128 states/batch,
FP32/TF32-off training, seeds913/914/915. Select each checkpoint on calibration
kept rows at >=96% retention only. Exact score thresholds calibrated at
90/95/96/98/99%; apply frozen settings to assessment. Do not choose a winner
from assessment outcomes. Report all seeds and whole-prompt bootstrap intervals.

Baseline families: replayed draft candidate logprob and entropy; intermediate
untuned Top-K/rank and margin; fixed truncation of unchanged B16 candidates;
candidate-preserving 100%-retention oracle. These are mechanism controls,
**not full reproductions** of FASER or Lever systems.

Primary work proxy: [L*16 + (36-L)*K]/36, together with retained accepted tokens
and committed tokens A_trim+1. Neither this proxy nor acceptance ratio is a
throughput result. Probe, compaction, graph dispatch, cache bookkeeping,
nonlinear attention and full B16 drafting costs still need measurement.

## Execution and next gate

Run a 24-state collection/training smoke before the full 3,758-state replay.
Use only idle authorized GPU4; verify again immediately before launch.
Temporary collection uses a fresh `/tmp/dflash-midverify.*` directory and
checksummed per-batch durable backups. Never overwrite source evidence.
Both collection and training have explicit time bounds. No recurring monitor.

Only if the diagnostic frontier is promising should we implement and measure
segmented verification. No 10k expansion, pre-draft hybrid, or serving-engine
integration is automatically launched by this wrapper.

Related mechanisms reviewed:

- https://arxiv.org/html/2604.20503v1 (FASER, token-wise early exiting).
- https://arxiv.org/html/2402.11131 (Speculative Streaming, parallel tree pruning).
- https://arxiv.org/html/2605.16786v1 (Lever, predictor-based verification pruning).
