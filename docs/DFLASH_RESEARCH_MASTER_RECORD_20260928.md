# DFlash adaptive drafting — detailed research record

Updated 2026-09-28 Pacific. Covers the supplied historical handoff, repository
experiment notes, completed workstation experiments, and the latest recovered
ragged-runtime diagnostics. This is an internal research/handoff document, not
a paper draft or a claim that publication-level validation is complete.

## Contents

- [Executive summary](#1-executive-summary) and [comparison rules](#2-evidence-and-comparability-rules)
- [Research direction](#3-research-direction-and-paper-context)
- [Repositories, models and datasets](#4-repositories-hardware-and-data-locations)
- [Historical predictors](#5-historical-predictor-exploration) and [controlled September experiments](#6-controlled-september-predictor-experiments)
- [Actual lengths, oracle headroom and masks](#7-actual-block-lengths-oracle-headroom-and-the-mask-branch)
- [Fused-vector, text and verifier signal analysis](#8-signal-analysis-fused-vectors-recent-text-and-verifier-probabilities)
- [Request versus cycle selection](#9-frozen-100k-predictor-request-versus-cycle-decisions)
- [Completed actual-block predictor and collection recovery](#10-new-actual-block-predictor-collection-recovery-and-results)
- [Measured serving costs and conditional gains](#11-serving-cost-and-possible-gain)
- [Recovered ragged implementation and numerical audit](#12-recovering-and-auditing-the-true-ragged-implementation)
- [Separate multi-candidate project](#13-earlier-multi-candidate-work--separate-track)
- [Conclusions and next work](#14-what-we-learned-what-we-have-not-established-and-next-work)
- [Full plot/artifact paths](#15-full-plot-paths-and-artifact-navigation) and [current handoff](#16-current-handoff)

## 1. Executive summary

The objective is a substantially stronger DFlash-based speculative decoding
system for an MLSys submission, initially through **pre-draft, per-cycle integer
block-size selection**. The target/drafter pair remains original Qwen3-4B plus
the original B16-trained DFlash checkpoint. DFlash 2 was discussed separately;
the name `dflashv2_data` does not mean these experiments use DFlash 2.

What the evidence currently supports:

1. **The last fused vector contains useful acceptance information.** Overlapping
   PCA/t-SNE colors do not imply absence of signal. Cross-prompt neighborhood and
   held-out probe results show real predictive structure.
2. **Cycle-level adaptation has a meaningful offline motivation.** Acceptance
   varies within requests. A frozen small MLP refreshed each cycle saves about
   15.4% of proposed-token work versus using only its initial-request feature in
   an exploratory near-matched-retention common-state comparison.
3. **B16 clipping is not a reliable label for actual shorter drafting.** Changing
   B changes the bidirectional drafter's candidate tokens. Controlled B4/B8/B12
   comparisons found approximately 20–23% label disagreement.
4. **More elaborate predictors have not produced a large, robust improvement.**
   Attention from scratch lost; a 63M-parameter residual context encoder added
   little policy value; raw target features did not beat fused controls; causal
   verifier-probability additions have weak incremental evidence.
5. **The new actual-block supervision experiment is complete.** It trained six
   matched heads on 10,000 fresh training cycles: actual versus clipped labels,
   three paired seeds. Actual supervision gives a small, seed-dependent average
   ratio improvement, with lower assessment retention. It does not meet the
   requested joint **70% ratio / 96% retention** target.
6. **Measured fixed-width speedups are modest in the tested serving setup.** At
   C64, B12 is 1.53% faster than B16 despite 12.97% cheaper decode cycles, because
   smaller blocks also make less progress. These are cached-image SGLang
   spec-v1/Triton measurements, not the recovered ragged engine.
7. **True historical packed drafting/verification has been recovered.** Primitive
   acceptance, packing, padding, slot-isolation and bounded integration checks
   pass. Runtime source was preserved, not patched to make tests pass.
8. **Numerical parity is a separate unresolved boundary.** A severe hidden-state
   discrepancy and an actual output-changing discrepancy both reproduce under
   fixed-width same-state controls in the recovered engine. Neither captured
   case is specific to mixed-length packing. Exact graph/eager/independent
   greedy parity nevertheless does not pass.
9. **No measured end-to-end adaptive speedup exists from the current predictor
   in the recovered engine.** Conditional gain calculations are planning aids,
   not results. No claim of a strong MLSys contribution is justified yet.

September 28 follow-up: the matched-assessment CPU oracle is now complete, and a
bounded GPU-4-only dense/MoE fixed-width screen has been launched. See
[matched oracle and MoE screen](matched_oracle_and_moe_screen_20260928.md) for
the new scope, pinned models, run paths and status caveats. The earlier numerical
diagnostics are complete; no recurring monitor or predictor sweep was started.

## 2. Evidence and comparability rules

This document distinguishes four kinds of evidence:

- **Current artifact-backed results:** completed local JSON reports, audits and
  source were inspected while preparing this record; corresponding remote run
  directories are identified below.
- **Older documented experiments:** supported by dedicated repository notes.
  Older PVCs/checkpoints were not all re-inspected live for this document.
- **Historical handoff values:** explicitly identified tables inherited from
  the supplied handoff/screenshots; no missing configs or numbers are invented.
- **Proposals and hypotheses:** labeled as such, never presented as launched
  experiments or established mechanisms.

Essential definitions:

| Quantity | Meaning |
| --- | --- |
| B | Block size including one known anchor |
| d = B−1 | Number of proposed draft tokens |
| A_B | Consecutive accepted proposed tokens for an actual draft at B |
| B16 label | A_16 in [0,15]; A_16=15 is capped/right-censored beyond 15 |
| Aggregate accept ratio | sum(A_policy) / sum(d_policy), not mean(A/d) |
| Retention | sum(A_policy) / sum(A_16) on matched states |
| Clipped proxy | min(A_16,d); valid for truncating unchanged candidates, not generally for redrafting |
| Nonterminal progress | A+1 tokens advanced, including the target bonus |
| Expected-length MAE | Error of the predicted expectation, not necessarily error of the selected budget |
| Common-state policy evaluation | Every policy scored on states from the same reference trajectory |
| Closed-loop evaluation | Each policy changes its own subsequent states and scheduling |

Calibration targeting 96% retention does not guarantee 96% on assessment.
Lower MAE does not prove a better budget/retention policy. Acceptance ratio does
not equal throughput. Exact retention matching, hardware, workload, candidate
semantics, graph use, concurrency, warmup and timing boundaries all matter.

Math500, HumanEval and the recent assessment prompts have been repeatedly
inspected: they are **development evaluations**, not untouched final tests.
Prompt-bootstrap intervals hold fitted weights/thresholds fixed unless stated
otherwise; they do not substitute for training-seed uncertainty.

## 3. Research direction and paper context

The preferred direction is to predict B before the draft forward using information
already available from committed text, target features and previous verification.
The policy should support arbitrary integer lengths in its tested range, not be
silently reduced to B4/B8/B12/B16. Predictor weights may be target/drafter-specific,
but the intended scientific signal should not be a hardware-specific timing
reward or a manually labeled collection of semantic states.

The base DFlash paper describes parallel block drafting conditioned on target
context features. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036).
BlockPilot describes predicting a request's block size once from the prefilling
representation. [BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding](https://arxiv.org/abs/2606.31315).

Our question is whether refreshing the decision within a request produces enough
additional benefit to pay for its execution overhead. The initial-feature
request control below is **not a reproduction of trained BlockPilot**. Neither
an acceptance histogram nor within-request variability alone contradicts its
theory or reported results. They motivate a finer-granularity comparison.

AdaEAGLE, AdaFlash, LibraSpec, DFlash 2 and work outside speculative decoding were
raised in discussion. This document does not reconstruct an unaudited literature
comparison or claim novelty against them. A separate primary-source novelty and
baseline audit is still required before framing a submission.

## 4. Repositories, hardware and data locations

### 4.1 Current workspace

| Role | Location |
| --- | --- |
| Active local research checkout | `/Users/atharvramesh/Projects/MLSys/dflash-headroom` |
| Current branch | `codex/block-headroom-20260926` |
| Project remote | `git@github.com:AtharvRN/dflash.git` |
| Workstation SSH | `zekaili@tianhaowang-gpu0.ucsd.edu` |
| Host-reported name | `wth-gpu-01` |
| Workstation code | `/home/zekaili/atharv/dflash` (resolves under `/data/users/zekaili`) |
| Durable scratch root | `/data/scratch/zekaili/atharv/dflash` |
| Imported original data | `/data/scratch/zekaili/atharv/dflash/data/dflashv2_data` |
| New workstation runs | `/data/scratch/zekaili/atharv/dflash/runs` |
| Local result copies | `/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs` |

The shared workstation has eight RTX PRO 6000 Blackwell Server Edition GPUs,
approximately 96 GB each. The designated GPU for these runs is GPU 4,
UUID `GPU-2b489243-113f-1e33-ee0b-e4d28423e006`, driver 595.71.05. It is not an
A100, and an idle observation is not a reservation. Preflight checks and a
cooperative task lock are used. Other users' jobs are not stopped or reset.

Home and scratch share the same `/data` filesystem: scratch is durable for this
workflow, not an independent backup. Temporary `/tmp` copies are disposable.
Code deployment uses git push/pull; raw outputs are commonly untracked, so git
alone cannot reproduce the experiments.

The workstation model/collection environment is Python 3.12.12,
Torch 2.13.0+cu130, Transformers 4.57.1. Serving diagnostics use a separate pinned
container with Torch 2.11.0+cu129 and Transformers 5.8.1. Do not conflate them.
The original environment and historical dirty worktrees were preserved.

### 4.2 Pinned models and feature timing

- Target: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Draft: `z-lab/Qwen3-4B-DFlash-b16`, revision
  `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Target layers `[1,9,17,25,33]`, hidden tuple indices `[2,10,18,26,34]`.
- Raw predictor representation: concatenate five 2,560-vectors at the latest
  committed token, giving 12,800 dimensions.
- Fused representation: frozen `draft.hidden_norm(draft.fc(raw))`, 2,560 dimensions.

The latest committed target token is at `start−1`. The anchor at `start` is known
from the previous target result but has not yet been target-processed. The latest
available target distribution predicts that anchor, not the first proposed draft
token conditional on it. This distinction is important when evaluating verifier
probabilities as pre-draft inputs.

### 4.3 Original prompt dataset

Source recorded in experiment notes: `z-lab/qwen3-4b-instruct-100k`.
The supplied inspection found 99,987 conversations, 182,924,248 bytes
(approximately 174.45 MiB), one user and one assistant message per row.
Columns: `conversations, source, split, thinking, status`; all rows have
thinking off and status success.

| Source label | Conversations | Share |
| --- | ---: | ---: |
| nemotron | 60,881 | 60.89% |
| opencodeinstruct | 19,364 | 19.37% |
| openr1_math | 12,429 | 12.43% |
| evol_codealpaca | 7,313 | 7.31% |

These are labels in the supplied parquet, not independently established upstream
dataset revisions. Its `split` column contains upstream labels such as chat/stem/
train, **not our predictor train/validation assignment**.

Original local parquet:
`/Users/atharvramesh/Projects/dflash-fresh-zlab-main/train-00000-of-00001.parquet`.
Workstation copy:
`/data/scratch/zekaili/atharv/dflash/data/train-00000-of-00001.parquet`.
Verified transfer SHA256:
`0b66428455f6637c0ad6e6c9bd975f4b7dddff2657e09b3adb4036482cfe9c87`.

Original prompt manifest:
`/workspace/dflashv2_data/manifests/qwen3_4b_instruct_100k_messages.jsonl`.
`/workspace` refers to the old cluster PVC `atharv-rwx-storage-1ti`, namespace
`wenglab-interpretable-ai`, not the Mac or current workstation.

### 4.4 Dataset/cache inventory — these are distinct populations

| Dataset | Training cycles | Calibration cycles | Assessment/validation cycles | Status / purpose |
| --- | ---: | ---: | ---: | --- |
| Historical canonical traces | 3,308,764 | Subset below | 172,794 total validation | Large original B16 trace pool |
| Historical 100k context experiment | 100,000 | 34,293 | 138,501 assessment | Sample from audited historical training pool |
| Completed paired raw/fused pilot | 13,935 | 1,373 | 5,662 assessment | Fresh matched B16 features and labels |
| Paired raw/fused 100k expansion | 94,790 last durable count | 1,373 | 5,662 assessment | Incomplete; no expansion training results |
| Verifier-signal fresh replay | 1,812 | 184 | 732 assessment | Label-independent plot sample, fresh paired labels |
| Actual-block policy-granularity evaluation | None | 342 | 1,416 assessment | Fresh B2–B16 outcomes, 234 validation prompts |
| Completed actual-block training | 10,000 selected | Same 342 | Same 1,416 | New six-model supervision comparison |

Millions of trace rows are decoding cycles, **not millions of prompts**.
The historical canonical split is by prompt. Full validation has 4,187 prompts;
its context-experiment split is 837 calibration and 3,350 assessment prompts.

Historical source paths:

- Trace manifest: `/workspace/dflashv2_data/traces/dflashv2_qwen3_4b_b16_instruct100k_full_4a100_recovered_20260717/manifest.json`.
- Last-vector cache: `/workspace/dflashv2_data/compact_cache/qwen3_4b_instruct100k_full_canonical_last_feature_20260719`.
- Canonical split: `/workspace/dflashv2_data/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719`.
- Completed paired pilot: `/workspace/dflashv2_data/runs/prefusion_pilot_20260915/cache`.
- Incomplete expansion: `/workspace/dflashv2_data/runs/prefusion_100k_20260915/cache`.

The original historical traces contain 16-position fused windows. The later
last-vector cache selects the newest vector. A subsequent audit found 101
inconsistent rows in eight training shards; the context experiment conservatively
excluded all 19,970 rows in those shards, leaving 3,288,794 eligible training rows.
Neither fixed validation nor the earlier 20k sample contained affected rows.

The completed paired pilot uses FP16 stored raw/fused features, A in [0,15], and
prompt/cycle/prefix/anchor/hash/trajectory metadata. Greedy B16, SDPA, BF16,
TF32 off, thinking off, max 32 cycles/prompt, 512 output tokens, 2,048 prompt
tokens. Raw and fused features come from the same forward before drafting;
terminal/capped states and cross-group exact duplicate content are excluded.

The partial 100k paired expansion last had 94,790 durable training rows and
3,735 receipts; 94,822 in the log was not the verified durable count. The pod
hit its six-hour deadline. Final materialization/audit/training was not complete.
It is separate from the **completed new 10k actual-block dataset**.

During workstation setup, the original parquet, canonical manifests/splits,
completed paired pilot, partial expansion, and B2–B20 headroom data were migrated.
The huge historical trace pool/full fused cache were not migrated. A transfer
stream reset on the last results directory; that directory was retransferred
and audited. No permanent SSH/global Git configuration or credentials were copied.

## 5. Historical predictor exploration

### 5.1 Families already explored

Historical work includes last-fused MLPs with arm/integer classification and
survival outputs; CE/distance, soft-CE, ordinal/EMD, hazard and censored likelihood
losses; auxiliary length penalties; calibration and tail weighting; GRU/stateful
summaries; token embeddings/GRUs; verifier uncertainty/history; post-draft
confidence/hidden heads, early-layer probes and teacher distillation; and
predicting future draft entropy from pre-draft context.

These should not be proposed as entirely untried ideas. Exact variants differ
in target model, split, loss, action space and evaluation convention. The long
[`axis2_block_size_policy.md`](axis2_block_size_policy.md) mixes Qwen3-8B and
Qwen3-4B generations of experiments; its statements of “best” are time-local.

### 5.2 Historical Qwen3-4B policy values from the handoff

These values were recorded in earlier conversations/screenshots. They are not
newly checkpoint-audited results from the recent workstation experiments.

| Method | Math500 ratio | Retention | HumanEval ratio | Retention |
| --- | ---: | ---: | ---: | ---: |
| Fixed B16 | 0.421 | 1.000 | 0.360 | 1.000 |
| Older fused survival head | 0.508 | ~0.956 | 0.468 | ~0.957 |
| Predicted draft entropy | 0.533 | 0.961 | 0.432 | 0.987 |
| True current draft entropy threshold | 0.618 | 0.973 | 0.535 | 0.986 |
| Post-draft confidence only | 0.635 | 0.951 | 0.578 | 0.962 |
| Post-draft hidden + confidence | 0.642 | 0.951 | 0.604 | 0.954 |

The predicted-entropy model maps the latest fused vector through an MLP to 15
future per-position entropy predictions, trained with SmoothL1 on normalized
targets. A validation-tuned threshold chooses a prefix. It was promising on
Math500, not universally best. True draft entropy and post-draft heads pay the
drafting cost before making their decision.

Another direct-head point has Math500 ratio 0.513, retention 0.960, expected MAE
3.136, with hazard NLL + 0.05 SmoothL1; B16 mean A was 6.322. Historical expected
MAEs for integer objectives were 3.120 (CE+0.2 distance), 3.134 (CE+0.5 EMD+0.1
distance), 3.149 (soft CE tau=1+0.2 distance), and 3.238 (soft CE tau=2+EMD+
distance). Matched policy ratios for those four exact checkpoints were not
established by the supplied table. Do not infer a policy winner from MAE alone.

The July 18 note uses an older split of 3,273,552 train / 208,006 validation rows
with 5,000 fixed validation prompts, not the later canonical counts in section 4.

Separate **Qwen3-8B** matched 100k teacher-distillation controls in the historical
notes were essentially neutral/slightly negative: strong/weak teacher ratio
0.349/0.349 versus hard-only 0.350, at approximately 95.1–95.2% retention. This
does not establish a Qwen3-4B numerical result, but explains why generic teacher
distillation was not treated as a fresh breakthrough proposal.

## 6. Controlled September predictor experiments

### 6.1 Common inexpensive survival MLP

The recent matched fused baseline has 1,478,415 parameters:

`2560 → Linear512 → GELU → LayerNorm → Dropout0.05 → Linear256 → GELU → Dropout0.05 → Linear128 → GELU → Dropout0.05 → Linear15`.

Its outputs are conditional success probabilities `q_k = sigmoid(logit_k)`.
Survival is `S_k = product_{j<=k} q_j`; expected B16 A is `sum_k S_k`.
First-rejection negative log likelihood is:

```text
A < 15:  −sum_{k=1..A} log(q_k) − log(1−q_{A+1})
A = 15:  −sum_{k=1..15} log(q_k)
```

There is no invented failure beyond the observed cap. The budget policy chooses
the smallest d whose cumulative survival sum retains an alpha fraction of the
predicted total. Alpha=1 explicitly selects all 15 proposals. Recent controlled
experiments below do not add an auxiliary distance term.

### 6.2 Attention from scratch, 20k rows

Source: [context-attention pilot](context_attention_pilot_20260913.md).
20,000 train cycles / 17,243 prompts; historical full validation split.
One learned query or 15 position queries attend over 16 fused vectors through
two width-2560, 16-head cross-attention/FFN blocks, FFN 1024, approximately 63M
parameters. Six epochs, AdamW 3e-4, batch 128, first-rejection NLL. Select checkpoint
by calibration expected MAE, then calibrate retention.

| Model | Assessment MAE | Mean budget | Ratio | Retention |
| --- | ---: | ---: | ---: | ---: |
| Last MLP | 2.646 | 9.746 | 0.4851 | 0.9639 |
| One query | 2.769 | 10.038 | 0.4708 | 0.9635 |
| 15 queries | 2.783 | 9.998 | 0.4716 | 0.9613 |

Attention lost in this single-seed pilot. Capacity/optimization remain confounded;
it does not prove context can never help. No online or held-out-task claim.

### 6.3 Frozen MLP plus residual context, 100k rows

Source: [residual-context experiment](context_residual_100k_20260913.md).
100,000 audited cycles / 53,262 training prompts. Train and freeze the MLP, then
add zero-initialized conditional-logit corrections with the same approximately
63M-parameter architecture, using either the last vector alone or all 16 vectors.
Six epochs; checkpoint and alpha selected by calibration ratio subject to 96%
retention. Every selected model was epoch 4.

| Model | Assessment MAE | Mean accepted | Budget | Ratio | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 2.6023 | 4.7206 | 9.3263 | 0.50616 | 0.96252 |
| Last-only correction | 2.5664 | 4.7213 | 9.2813 | 0.50869 | 0.96266 |
| 16-context correction | 2.5653 | 4.7277 | 9.3304 | 0.50670 | 0.96395 |

Most MAE benefit is present without earlier context. Full context does not beat
the matched last-only correction in policy ratio. Some tiny bootstrap deltas
exclude zero, but this is not a large practical or multi-seed improvement.
Frozen weights/provenance/checkpoint reload tests passed. These are clipped-B16
policy metrics, not actual shorter drafting.

### 6.4 Raw target features versus fused features

Source: [paired pre-fusion pilot](prefusion_acceptance_pilot_20260915.md).
Fresh paired data, 13,935 train / 1,373 calibration / 5,662 assessment cycles.
Raw input 12800-512-256-128-15 has 6.72M parameters. Fused 2560-512-256-128-15
has 1.48M; capacity control 2560-2372-256-128-15 has approximately 6.72M.
GELU/LayerNorm/dropout 0.05, six epochs, batch 128, AdamW 3e-4, weight decay 0.01,
clip 1.0; first-rejection NLL with no auxiliary term.

| Model | Assessment MAE | Mean accepted | Budget | Ratio | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 2.7917 | 5.4624 | 10.1309 | 0.5392 | 0.9618 |
| Fused | 2.8374 | 5.4520 | 9.9558 | 0.5476 | 0.9599 |
| Fused parameter-matched | 2.7487 | 5.4520 | 10.0079 | 0.5448 | 0.9599 |

Raw did not improve policy efficiency; capacity-matched fused had best MAE.
The 100k expansion was authorized to test scaling, not because raw already won.
Its incomplete status is unchanged. The full paired pilot audit passed, including
approximately 0.337% relative RMSE for FP32 frozen-fusion replay versus BF16 capture.

## 7. Actual block lengths, oracle headroom and the mask branch

### 7.1 Why clipping the B16 label is not enough

Source: [paired-length diagnostic](paired_length_diagnostic_20260913.md).
2,000 matched states / 212 validation prompts; 1,991 nonterminal states scored.
Restore the same target/draft caches and actually execute B4/B8/B12/B16.

| B | Mean clipped B16 estimate | Actual mean A | Label disagreement |
| --- | ---: | ---: | ---: |
| 4 | 2.349 | 2.166 | 20.14% |
| 8 | 4.384 | 4.133 | 22.60% |
| 12 | 5.760 | 5.555 | 21.09% |
| 16 | 6.727 | 6.727 | Reference |

Holding draft candidates fixed and changing only verification width gives much
smaller disagreement: 0.70%, 1.16%, 1.21%. Changed candidates account for most
of the short-draft discrepancy in this experiment.

Frozen direct head: clipped retention 98.79%, actual 96.45%; predicted entropy:
97.77% versus 96.16%. Actual aggregate ratios are 0.5849/0.5883 respectively.
These validation diagnostic states are not authorized training data. The test
motivates actual-B supervision but did not itself demonstrate a new policy.

### 7.2 Fresh B2–B20 headroom diagnostic

Source: [protocol](block_headroom_20260926.md), locally retrieved completed
[`headroom_report.md`](../outputs/block_headroom_128p_20260926/headroom_report.md).
Source-balanced training development sample: 32 planned prompts/source, 128 total;
875 collected states, 871 eligible, 117 prompts with eligible states. All integer
B2–B20 are executed on shared B16 reference states, max eight progress-spaced
states/prompt, prompt cap 2048, output cap 256. This is not population-weighted.

Exact integer dynamic programming minimizes budget under retained-acceptance
constraints for global, per-prompt and per-cycle **hindsight oracles**.

| Retention target | Global budget / actual retention | Prompt-oracle budget / retention | Cycle-oracle budget / retention | Cycle budget saving vs prompt |
| --- | --- | --- | --- | ---: |
| 90% | 13.000 / 90.10% | 9.830 / 90.00% | 6.443 / 90.00% | 34.45% |
| 95% | 15.000 / 100% | 10.933 / 95.01% | 6.794 / 95.01% | 37.86% |
| 96% | 15.000 / 100% | 11.178 / 96.01% | 6.865 / 96.01% | 38.59% |
| 98% | 15.000 / 100% | 11.723 / 98.00% | 7.005 / 98.00% | 40.25% |
| 100% | 15.000 / 100% | 12.287 / 100% | 7.145 / 100% | 41.85% |

Mean actual A for B16/17/18/19/20 is 7.0184/7.0781/6.6188/5.7600/5.0448.
Bigger blocks do not monotonically preserve acceptance with this frozen drafter.
Eight reverse-order checks and 304 checked canonical comparisons passed;
12,250 same-candidate truncated-verification controls were collected.

This establishes hindsight headroom, not learnability, closed-loop policy benefit,
or a throughput bound. A request oracle on sampled common states is not a
full-response BlockPilot oracle. No learned controller gets to see these outcomes
before deciding.

### 7.3 Post-hoc causal draft mask, including B24/B32

Source: [causal-mask diagnostic](causal_mask_probe_20260927.md).
16 training-development prompts, 64 shared states, frozen weights, workstation
BF16/SDPA. Bidirectional drafting allows each draft query to see committed context
and the whole draft block. The intervention explicitly permits all context columns
but only a lower-triangular draft-to-draft block. Target attention is unchanged.

| B | Bidirectional block-verified mean A | Causal mean A |
| --- | ---: | ---: |
| 4 | 2.09375 | 1.96875 |
| 8 | 4.203125 | 3.609375 |
| 12 | 5.75000 | 4.90625 |
| 16 | 7.078125 | 5.65625 |
| 20 | 4.421875 | 6.109375 |
| 24 | 3.296875 | 6.28125 |
| 32 | 3.140625 | 6.265625 |

For B4–B20 there were zero canonical disagreements across 640 outcomes. The
B24/B32 extension had one disagreement among 512 outcomes: bidirectional B32
accepted 4 versus 1 under token-by-token reference; its reference-based mean is
3.09375, not 3.140625. Causal B24→B32 also had one actual acceptance regression
(2→0) and 19/64 candidate-prefix changes. Do not claim exact prefix invariance
or monotonicity in BF16.

At B16, causal masking loses 1.421875 accepted tokens; paired prompt-bootstrap
95% interval [−2.140625, −0.78125]. It stabilizes some length comparisons but
plateaus near 6.3 accepted tokens, below original B16's 7.08 despite larger budgets.
Only 44/64 historical B16 states replayed exactly under fresh full-prefix execution;
use the fresh paired comparison. No throughput measured. **The user dropped this
mask direction; it is not the current plan.**

## 8. Signal analysis: fused vectors, recent text and verifier probabilities

### 8.1 Acceptance histograms

The supplied histogram shows historical canonical train/validation means 4.89/4.91
and medians 3, with approximately 16% zero acceptance and 10.4%/10.6% at A=15.
Its Math500 sample has 32,768 cycles, mean 5.72; HumanEval 12,288, mean 4.49.
These are descriptive screenshot values; those task-row populations were not
reconciled to the older policy table whose Math500 mean is 6.322.

The endpoint at 15 is censored, not proof that those states would support B24/B32.
Marginal histograms cannot establish the optimal per-request B, the causal signal
available to a predictor, or contradiction of another paper's request-level policy.

### 8.2 Geometry and recent-text probes

Source: [acceptance geometry](acceptance_geometry_20260927.md).
Completed paired pilot: 13,935/1,373/5,662 rows across 464/47/187 prompts.
Plots use 2,766 label-independently selected cycles, at most four/prompt, not all
20,970 rows. The collected vectors are row-L2-normalized, followed by train-fit
PCA64. t-SNE uses the first 50 PCs, perplexities 15/50 and more than one seed.
Colors and sixteen-panel grouping use observed A=0..15; labels do not construct
the embedding. No averaging into “one vector per acceptance length.”

Key descriptive findings:

- Fused PC1/PC2 explain 5.78% of variance, PCA64 explains 34.63%; raw equivalents
  are 7.79%/39.52%. A mixed two-dimensional plot cannot rule out high-dimensional signal.
- Excluding same-prompt neighbors, fused-PCA 15-neighbor mean absolute A difference
  is 4.183 versus 5.789 under source/cycle-stratified label shuffling. Raw: 4.288/5.755.
- Within-prompt label sums of squares account for 66.17%, between-prompt means 33.83%.
  This is descriptive, not a noise-corrected variance-component estimate.
- Lag 1 correlation is 0.543 before prompt demeaning; demeaned lags 1/2/4/8/16 are
  0.305/0.165/0.069/−0.038/−0.143. By lag 8 it is near the within-prompt shuffled
  reference (about −0.033). Full-prompt means are used for analysis only, not inputs.

Linear probes include common source/progress controls. PCA, vocabulary/scaling
fit only training; Ridge alpha is selected on calibration MSE.

| Added information | Assessment MAE | R² |
| --- | ---: | ---: |
| None: source/progress | 3.7111 | 0.2599 |
| Original-prompt lexical features | 3.7412 | 0.2600 |
| Most recent 16 tokens | 3.5915 | 0.3284 |
| Most recent 64 tokens | 3.6893 | 0.3021 |
| Most recent 256 tokens | 3.7309 | 0.2654 |
| Causal previous-acceptance summaries | 3.3287 | 0.3778 |
| Raw PCA64 | 3.0693 | 0.4621 |
| Fused PCA64 | 2.9641 | 0.4959 |
| Fused PCA64 + previous acceptance | 2.9211 | 0.5056 |

Feature-only cross-prompt k=50 retrieval: fused MAE 3.0716, raw 3.2022. Training mean
constant MAE 4.6092; median constant 4.3207. Recent 16 improves MAE by 0.1195 over
controls (95% paired interval [−0.1803, −0.0564]); history adds 0.0429 MAE improvement
to fused PCs ([−0.0616, −0.0255]). Text features here are token unigram/bigram
TF-IDF, not a test of every possible semantic encoder.

Conclusion: useful fused and short-range text/history signals exist, but these
tests do not justify another expensive context-architecture sweep or prove
predictability of the actual optimal B.

### 8.3 Fresh verifier-probability replay

Source: [verifier-signal analysis](verifier_signal_analysis_20260927.md).
The old paired cache did not contain verifier probabilities, so 2,766 saved states
were replayed with fresh features **and fresh labels**. Full 151,936-way FP32
target probabilities, final target state, anchor embedding and committed-position
confidence history were saved. Current draft/verification confidence was retained
only as explicitly future-information diagnostics.

2,728 states preserve the original greedy anchor: 1,812 train/184 calibration/732
assessment. Original/fresh A agrees on 2,647 of 2,766; all 119 label changes are
among anchor-matched states. Mean fused cosine 0.9998406, minimum 0.9449522; four
rows below 0.99 were retained. The original incremental execution differs from
this full-prefix replay; causes were not isolated or attributed solely to hardware.

Assessment Spearman correlations with A: entropy −0.45388, anchor probability
+0.44685, top1/top2 probability margin +0.44173. Confidence has marginal signal,
but target confidence alone does not describe draft–target agreement.

| Ridge probe, common controls | Assessment MAE | Delta vs fused |
| --- | ---: | ---: |
| Fused PCA64 | 2.97954 | Reference |
| +latest verifier confidence | 2.99041 | +0.01087 |
| +verifier history | 2.97155 | −0.00799 |
| +full-probability PCA64 | 3.00205 | +0.02251 |
| +sorted probability shape | 3.00859 | +0.02905 |
| +anchor embedding | 2.97174 | −0.00780 |
| +final target hidden state | 3.01884 | +0.03930 |
| +anchor +confidence | 2.97771 | −0.00183 |

All listed incremental MAE intervals include zero. Confidence-only MAE 3.47220;
probability PCs-only 3.56625; controls 3.61223. Current **post-draft** confidence
MAE 1.80444 and **post-verification** confidence 1.71628 are not pre-draft results.
Full verification plus candidate identities already determines A exactly.

Probability geometry uses square-root probabilities before PCA. First 2 PCs
explain 4.92%, first 64 explain 39.33%. A descriptive t-SNE neighbor check suggests
much stronger same-anchor clustering than fused geometry, not better acceptance
grouping. Neither visually cleaner clusters nor correlation proves incremental
policy value.

Exploratory boosted-tree follow-ups did not establish gains from confidence,
history or anchor+confidence. Conditional-success logistic heads showed some
MAE benefits for history/anchor within that weaker family, but all added-feature
NLL/Brier intervals included zero. Thus “no effect on every metric” would be
incorrect; a large robust missing signal was not found.

## 9. Frozen 100k predictor: request versus cycle decisions

Source: [policy-granularity test](policy_granularity_20260927.md).
The exact historical 100k-trained last-fused MLP, epoch 4, was recovered through a
temporary CPU-only, read-only PVC mount; GPU work stayed on the workstation.
Checkpoint SHA256:
`84a37d00f61b1c8ae5ecd76e1ad49b4cd00850613e783edf72ecdf875663af35`.

Workstation checkpoint:
`/data/scratch/zekaili/atharv/dflash/checkpoints/context_residual_100k_20260913/last_mlp_best_epoch_4.pt`.

Fresh B16 trajectories from 47 calibration and 187 assessment prompts; every
integer B2–B16 independently drafted/verified at up to eight progress-spaced
states/prompt. 1,764 states, six common accepted-EOS exclusions, 1,758 eligible.
Calibration 342; assessment 1,416. Four reverse-order checks and 120 canonical
comparisons across eight states passed. Collection took 1,691 s excluding load.

Policy inputs are either the cycle-zero feature reused throughout a request or
the latest feature every cycle. The primary alpha sweep is 0.5–1.0 by 0.001,
selected only on calibration for 96% retained acceptance.

| Primary policy | Setting | Assessment A | Budget | Ratio | Retention |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed | B15 | 6.3249 | 14.0000 | 0.45178 | 96.146% |
| Initial-request feature | alpha 1 → B16 | 6.5784 | 15.0000 | 0.43856 | 100% |
| Current-cycle feature | alpha 0.990 | 6.3157 | 11.7083 | 0.53942 | 96.006% |

Cycle saves 16.369% proposed work versus fixed B15, paired prompt interval
[13.744%, 19.014%]. Retention point estimates are close, not proven equivalent.
The request model's collapse to B16 arose because calibration retention at alpha 0.999
was 93.58%; it is not evidence that request-level information is absent.

A separately named, post-hoc calibration-resolution check enumerates exact
decision breakpoints from calibration inputs only:

| Exploratory policy | Assessment A | Budget | Ratio | Retention |
| --- | ---: | ---: | ---: | ---: |
| Fixed B15 | 6.3249 | 14.0000 | 0.45178 | 96.146% |
| Request, alpha 0.9997000825731314 | 6.3072 | 13.7811 | 0.45767 | 95.878% |
| Cycle, alpha 0.9894556174155835 | 6.3051 | 11.6568 | 0.54089 | 95.845% |

Cycle saves 15.415% work versus request, interval [12.642%, 18.096%]. Retention
difference −0.032 percentage points has interval [−1.885, +1.755]; both policies
slightly miss 96% assessment retention. This remains common-state development
evidence, not throughput, closed-loop performance or a BlockPilot comparison.

### 9.1 Matched assessment oracle, added September 28

The exact budget oracle was recomputed on these same 1,416 assessment cycles /
187 prompts, not the older 871-state diagnostic. Prediction/outcome/prompt-order
bindings were reverified, including equality with the six trained heads' saved
assessment arrays. Existing calibration-selected settings remain unchanged.

At >=96% aggregate retention, request/cycle hindsight mean **proposed budgets**
are 10.7331/6.4633; ratios are 0.58843/0.97716, both at 0.960064 retention.
The frozen primary cycle MLP still uses budget 11.7083 at that same retained
accepted-token total. At 100% retention the oracle budgets are 12.3948/6.7804.
Thus the cycle oracle's 96% **mean B is 7.4633**, including the anchor.

These are minimum-token-work hindsight bounds, not latency-optimal policies,
learnability guarantees or throughput ceilings. The request oracle also sees
future sampled outcomes. Detailed audited tables, including the separate
exact-calibration sensitivity, are linked in the new screen note above.

## 10. New actual-block predictor: collection, recovery and results

### 10.1 Question and exact model

Source: [fixed protocol](actual_block_predictor_20260928.md).
Test whether actual-B supervision improves decisions over clipped B16 supervision,
without changing the input/trunk or repeating another survival-loss sweep.

- Input: latest 2,560-dimensional pre-draft fused vector, paired with labels in
  this new collection through a separate causal fusion replay. No current draft,
  future verifier output, token IDs or anchor embedding is an input.
- Trunk: same 1.48M architecture in section 6.1.
- Output: 15 independent means `mu_d = d * sigmoid(logit_d)`, d=1..15. These are
  expected actual accepted counts for B=d+1, **not a survival curve**.
- Actual labels: `(A_2,...,A_16)` from independent real drafts at each B.
- Control labels: `min(A_16,d)` on precisely the same rows.
- Loss: unweighted mean squared error in accepted-token counts, averaged across
  all rows/actions. No monotonicity imposed between different B values.
- Three paired seeds 913/914/915, identical initialization, minibatch order and
  dropout random streams within each actual/clipped pair.
- Six epochs, batch 128, AdamW 3e-4, weight decay 0.01, clip 1.0, no scheduler;
  deterministic FP32 training, TF32 off. Offline scoring is CPU FP32, dropout off.
- Decision: choose d maximizing `mu_d − lambda*d`, ties toward smaller d.
  This is a calibrated budget/accepted-count Lagrangian, not a hardware timing reward.
- Calibration searches upper-envelope breakpoints/intervals, with fixed B16 fallback.
  Select minimum budget meeting 96% calibration retention, then higher retention,
  then earlier epoch; freeze all selections before assessment.

### 10.2 What was collected

Canonical training prompts shuffled with seed 929; exact duplicate content and
cross-split duplicates excluded. Original pinned frozen models; greedy,
thinking off, SDPA/BF16, TF32 off; prompt cap 2048, output cap 256, at most eight
progress-spaced states/prompt. Each state restores both caches and executes all
integer B2–B16 in randomized order, preserving prefix IDs/hashes, candidates,
actual labels, aligned fused features and immutable receipts.

Exclude the whole paired state if any action accepts EOS; do not sample
insufficient-output-budget states. The fixed 342/1,416 evaluation states remain
read-only. The source-balanced B2–B20 training diagnostic is not assessment data.

Final audit: 10,102 collected states, 10,054 eligible, **10,000 selected training
cycles from 1,317 prompts**, 54 eligible tail rows retained but unused. Training
source counts: nemotron 5,957; opencodeinstruct 2,238; openr1_math 1,042;
evol_codealpaca 763. Canonical disjointness, hashes, alignment and checked greedy
outcomes passed. At B2–B15, actual/clipped label disagreement ranges 11.07–21.41%.
The target was 10k cycles, not 10k prompts or the old raw/fused 100k expansion.

### 10.3 Interruptions and efficient restart history

Source: [collection recovery log](efficient_collection_recovery_20260928.md).

| Attempt | Durable eligible cycles at stop | Observed event |
| --- | ---: | --- |
| `actual_block_predictor_10k_20260928` | 1,901 | SIGTERM/exit 143, before original four-hour bound; sender unidentified |
| `actual_block_predictor_10k_recovered_20260928` | 5,408 | Worker disappearance / BrokenProcessPool; cause not established |
| `actual_block_predictor_10k_diagnostics_20260928` | 8,912 | Worker SIGTERM recorded before parent cleanup; no observed OOM-counter increase |
| `actual_block_predictor_10k_finish_20260928` | 10,054 eligible; 10,000 used | Complete audit and all six trainings |

Every restart preserved and re-audited the prior committed prefix; no source
evidence was overwritten. Original prompt order, validation membership and
cumulative four-hour collection allowance were retained. No automatic restart
or recurring monitor was installed.

To improve GPU use without changing numerical labels, benchmarked 1/2/4
independent process workers on the same GPU and same 12 prompts/95 states.
This is concurrent single-request collection, not padded tensor batching.

| Workers | Eligible cycles/s, first matched benchmark | Mean GPU utilization | Sampled peak memory |
| --- | ---: | ---: | ---: |
| 1 | 0.97861 | 54.96% | 10,805 MiB |
| 2 | 1.67474 | 94.39% | 20,872 MiB |
| 4 | 1.81244 | 98.11% | 41,503 MiB |

Every prefix, candidate, acceptance label and stored fused feature matched
exactly. Four workers passed the predeclared performance/memory gate, about 1.85×
collection throughput. Later diagnostic/finish replay gates confirmed it. This
is not a serving speedup. Model load/warmup is outside this collection timing.

Worker diagnostics record signals, fault stacks, prompt identity, resource and
accessible cgroup counters. One worker received SIGTERM before peer cleanup;
the shared `oom_kill` counter stayed 17. A baseline shared counter is not evidence
that this run OOMed, and the signal sender remains unknown. SSH also temporarily
timed out before authentication, then recovered; no cause or configuration change
was established. All failed/incomplete runs were prevented from training.

### 10.4 Completed predictor results

Authoritative local reports:
`outputs/actual_block_predictor_10k_finish_20260928/training/{audit.json,config.json,summary.json,COMPLETE.json,report.md}`.
The completion file binds six checkpoints, row index, selection, predictions,
curves and logs; paired initialization/order and checkpoint reload checks passed.

| Model | Epoch | Assessment ratio | Retention | Mean budget |
| --- | ---: | ---: | ---: | ---: |
| Frozen historical 100k survival MLP | 4, frozen | 0.53942 | 0.96006 | 11.7083 |
| Fixed B15 | — | 0.45178 | 0.96146 | 14.0000 |
| Clipped seed 913 | 2 | 0.53522 | 0.95674 | 11.7592 |
| Actual seed 913 | 2 | 0.54113 | 0.95620 | 11.6243 |
| Clipped seed 914 | 3 | 0.53453 | 0.95813 | 11.7917 |
| Actual seed 914 | 2 | 0.55551 | 0.95513 | 11.3107 |
| Clipped seed 915 | 3 | 0.54816 | 0.95309 | 11.4379 |
| Actual seed 915 | 3 | 0.54237 | 0.94815 | 11.5000 |

Across-seed means: actual ratio 0.54634, retention 0.95316, budget 11.4783;
clipped 0.53930, 0.95598, 11.6629. Actual improves ratio in two seeds and loses in
one. Paired prompt-bootstrap ratio intervals for actual−clipped are
[0.00140, 0.01051], [0.01272, 0.02934], and [−0.01143, −0.00028]. Retention differs;
these are not matched-retention superiority claims. The frozen 100k baseline is
not training-size/data-matched to the new 10k models; only each actual/clipped
pair isolates supervision.

No model reaches joint 70% ratio/96% retention. Seed 914's highest ratio is not a
justification to select it using assessment. The frozen 100k small MLP remains
the practical established control near 96% retention; there is no universal
“best predictor” across all historical datasets and availability regimes.

## 11. Serving cost and possible gain

Source: [full latency results](high_concurrency_latency_results_20260928.md).
This completed study used the cached image's native DFlash spec-v1 path,
SGLang 0.5.13 with image-baked Slime modifications, Triton attention, TP1,
BF16/greedy, graph-enabled target and draft, TF32 off, radix reuse off. It is
neither a pristine upstream-default build nor the recovered historical ragged
spec-v2/FlashInfer implementation.

Image digest:
`sha256:a7317182c71d35712ee4edc86a5d1c313dc969efdf0026d339673299c186ea75`.
Image base SGLang commit:
`28b095c01005d4a3a2a5b637b7d028b07fba31b2`.

187 development prompts, lengths 81–1,997; saved anchor removed for prefill.
At C64: 384 requests cycling through this pool, output cap 256, EOS allowed;
warmup 64 requests capped 96. Three clean repetitions per B/C, separate event
runs and short profiler runs. Clocks/power not locked; shared host, not an
isolated publication benchmark.

### 11.1 Clean HTTP throughput

Mean ± sample SD output tokens/s; includes prefill and drain.

| C | B16 | B12 | B8 |
| ---: | ---: | ---: | ---: |
| 1 | 407.94±0.08 | 373.22±0.19 | 320.67±0.01 |
| 16 | 2,410.26±14.20 | 2,417.84±33.93 | 2,224.25±26.40 |
| 32 | 2,945.02±5.23 | 2,865.63±4.09 | 2,825.91±4.97 |
| 64 | 3,327.56±6.64 | 3,378.57±7.68 | 3,349.79±8.81 |

B12/B16 at C64 is 1.0153×. C64 target-only AR is 2,962.77±7.38 tokens/s, so
B16/B12 are 1.1231×/1.1403× that AR configuration. AR keeps its native overlap
scheduler while spec-v1 disables overlap; these are practical engine comparisons,
not isolated algorithmic gains. Different C values cover different request counts.

### 11.2 Full-C64 decode components

CUDA-stream elapsed milliseconds, **not pure kernel time**. Non-synchronizing
events, checked disjoint accounting; inclusive parent/child spans are not summed.

| Component | B16 | B12 | B8 |
| --- | ---: | ---: | ---: |
| Draft setup/allocation | 0.360 | 0.343 | 0.336 |
| Draft transformer | 7.096 | 6.247 | 5.600 |
| Draft vocabulary projection/argmax | 2.930 | 2.198 | 1.460 |
| Verify preparation | 0.269 | 0.263 | 0.264 |
| Target verify including logits | 52.942 | 46.205 | 40.955 |
| Acceptance/target KV commit | 2.860 | 2.532 | 2.285 |
| Draft KV upkeep/feature handling | 1.194 | 1.081 | 1.064 |
| Other in-worker work | 0.080 | 0.078 | 0.077 |
| Total mean | 67.730 | 58.948 | 52.042 |
| Cycle p95 | 69.603 | 61.301 | 54.216 |
| Full-batch cycles | 67 | 83 | 102 |

Mean prefixes are approximately 954–962 tokens. Target verify is 78.17% of B16's
cycle. Accepted proposed tokens average 4.984/4.425/3.702 respectively on those
different trajectories. Cheaper cycles alone do not imply proportionate throughput.

Instrumented B16 C64 wall 28.141 s comprises 9.332 s prefill worker spans, 18.192 s
decode, 0.617 s outside-worker residual. The residual is not isolated scheduler
CPU time or predictor overhead. Event-run throughput is 1.05% below the clean B16
mean; clean repetitions remain authoritative.

### 11.3 Kernel observations and overhead traps

Short traces attribute only three full-C64 cycles per B, not a robust population
distribution. B16→B12 GPU activity: GEMM 31.203→23.699 ms; target+draft attention
25.980→25.919 ms; target attention 23.006→22.950 ms; explicit copies 0.0134→0.0118 ms.
These overlapping categories must not be summed as separate components.

Pinned Triton source uses query tile `BLOCK_M=64`, key tile `BLOCK_N=128` for this
SM120/head shape; both B12 and B16 have attention grid (64, 32, 1). The observed
attention plateau is consistent with shared query-tile work, not evidence that
all attention backends cannot exploit variable lengths. A controlled backend/
tile intervention would be needed to establish the mechanism causally.

Host `cudaMemcpyAsync` can take 55.63 ms while explicit GPU copy activity is only
about 0.013 ms: the host call waits for prior work. Likewise host acceptance/commit
63.30 ms versus stream span 2.86 ms. Adding those host and GPU times double-counts.

### 11.4 Standalone predictor cost and conditional headroom

Latest fused vectors already on GPU; FP32 model with TF32 off, input cast included,
decision included. Feature gathering, packing, KV remapping and bucket waste excluded.

| C64 predictor | Eager ms | Graph ms | Graph+CPU length list ms |
| --- | ---: | ---: | ---: |
| Exact frozen 100k survival MLP | 0.23145 | 0.09846 | 0.11319 |
| Actual-block head, prespecified seed 913 | 0.14540 | 0.08665 | 0.10088 |

The planning model uses `((mean_A_policy+1)/(mean_A_B16+1))*T16/(T(mean_B)+extra)`.
It interpolates **uniform-B** costs and assumes they transfer to mixed batches,
graph padding and policy trajectories. Those assumptions are unverified.

| Scenario | Mean B | Retention | Conditional decode gain, no extra cost | Conditional workload gain |
| --- | ---: | ---: | ---: | ---: |
| Frozen 100k MLP | 12.708 | 96.01% | 8.06% | 5.07% |
| Actual-block three-seed mean | 12.478 | 95.32% | 8.30% | 5.21% |
| Hypothetical 70% ratio/96% retention | 10.022 | 96.00% | 17.73% | 10.79% |

The 70% point was not achieved. For the current actual-block scenario, 2 ms extra
per cycle leaves 4.80% decode/3.05% workload gain; approximately 4.98 ms erases the
conditional gain. Preserving 5% decode gain requires less than 1.88 ms additional
cost under these assumptions. Existing baseline upkeep is already in T(B).

These are neither measured adaptive speedups nor upper bounds. The offline B16
mean A=6.578 differs from the serving sample; eight sampled states/prompt are not
all serving cycles. Numerically, 70% ratio at 96% retention on that offline sample
would require mean proposal budget about 9.022, not the current roughly 11.5–11.7.

### 11.5 Correctness limits of the cost study

At C64, each later fixed B16 repeat differs from its first run on 138/384 output
sequences; B12/B8 differ from B16 on 196/193. Causes were not isolated in that run.
No bitwise losslessness claim follows from the throughput table. Small C4 smoke
parity did pass. A B12 trace initially failed because derived gRPC ports exceeded
65535; bounded HTTP ports 18000–29999 fixed the retry. This was not an OOM or loss
of collection data. Both failed wrapper and successful retry were preserved.

## 12. Recovering and auditing the true ragged implementation

Sources: [recovery report](ragged_recovery_20260928.md),
[latest numerical follow-up](ragged_numerical_followup_20260928.md).

### 12.1 Three serving codebases must not be conflated

| Codebase | What it represents |
| --- | --- |
| Cached image spec-v1/Triton | The completed fixed-width performance study above |
| Newer dirty `sglang-dflash-ragged` checkout | At inspection, full-width drafting with verification compaction; not a substitute for true shorter drafting |
| Recovered `sglang-dflash-pr23000` historical implementation | Packed variable-length draft and verify, spec-v2/FlashInfer, now under correctness audit |

Historical source:
`/Users/atharvramesh/Projects/sglang-dflash-pr23000`, branch
`codex/dflash-dynamic-policy`, base `e5add3ba0f20d46b5bcb7a9a356df7cb102c49ae`.
Dirty source changes were preserved verbatim. Recovery package:
`vendor/sglang_ragged_20260723`, a checksummed 6.98 MB base archive plus 16 overlay
files. A called-but-missing ragged acceptance kernel was recovered from the saved
deployed `.tmp_sglang_opt/current_pod/dflash_accept_bonus.py` snapshot; provenance
is explicit. This is not a claim of byte identity to the whole old live pod.

Local restored source:
`/Users/atharvramesh/Projects/MLSys/sglang-dflash-recovered-20260928`.
Serving uses this source first on `PYTHONPATH` inside the pinned dependency image,
not the image's built-in SGLang implementation. No original source worktree was
edited or reset. All 16 overlay-origin hashes were rechecked unchanged.

### 12.2 Major runtime components

1. Obtain per-request integer B and construct a packed block with offsets;
   desired token work is sum(B_i), not C×max(B_i).
2. Draft with the correct block-dependent bidirectional attention; compact
   vocabulary projection excludes known-anchor positions.
3. Prepare packed target verification and causal attention metadata; optionally
   append dummy suffix tokens to fill a **total-token graph bucket**.
4. Compare draft tokens to target argmax, find the consecutive accepted prefix,
   select bonus, and use consistent gather indices for hidden states and KV.
5. Commit accepted target slots/features, materialize draft KV, free/reuse rejected
   suffix allocations, and preserve per-request state through batch changes.

The recovered worker explicitly leaves drafting eager; **target verification can
use graphs**. Historical `draft_graph_total_tokens` fields describe planned
budgets, not proof of actual draft graph replay. This is a substantial systems
path with attention/graph/allocator interactions, not simply an MLP plus a slice.

### 12.3 Primitive and bounded integration results

CPU and actual recovered CUDA/Triton acceptance passed 1,456 request-cases in 49
batches, including all 136 (B, A) combinations for B1–16. GPU block construction,
packing/projection/commit/padding checks passed 54 batches. Silent fallback to
eager acceptance was forbidden. B1 is an edge primitive, not a claim the current
learned policy normally selects it.

Small-batch tests complete 32 requests each, EOS respected and caps 1/7/8/16/32/
63/64/96, strict busy KV accounting. Independent reference forwards restore the
same committed prefix and really use the chosen B; no clipped-B16 substitution.

| Control | Matched request-cycles | Draft-token differences | Target top1 differences | Changed emission vs independent | Changed emission vs eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rotating ragged | 54 | 4/441 | 2/495 | 0/54 | 0/54 |
| Fixed B16 | 54 | 12/810 | 5/864 | 0/54 | 1/54 |
| Ragged, batch-invariant flag | 54 | 5/441 | 4/495 | 0/54 | 0/54 |

All tested cross-request isolation and packed restorations were bitwise. The
different policies visit different states; this is not an error-rate ranking.
The existing batch-invariant flag did not establish cross-shape parity.

### 12.4 High-concurrency numerical investigation

Hidden-error metrics below concern the concatenated five selected target-layer
outputs (12,800 dimensions), not the normalized fused predictor vector (2,560).
Relative L2 uses the independent/single-request hidden norm as denominator
unless explicitly stated otherwise.

An initial C64 run stopped on a B15/prefix 409 raw-target-hidden relative L2 gap
0.9082. The selected row's argmax/acceptance agreed, but full tensors were not
saved, so that exact state cannot now be replayed. Other C64 controls complete
256 requests and bounded checks but do not directly explain it.

A separately saved eager B7/prefix 1543 case had relative L2 gap 0.06214. Same-runtime
fixed C63/B10 exactly reproduced its packed output; uniform ragged metadata was
bitwise equal to fixed metadata. All seven verifier predictions and A6/bonus 31610
agreed. It demonstrated that the smaller discrepancy was not mixed-length-specific.

Further target-only audit results:

| Run | Matched cycles | Max hidden relative L2 | Target top1 changes | Changed emissions vs independent |
| --- | ---: | ---: | ---: | ---: |
| `ragged_extreme_capture_20260928` | 2,015 | 0.470654 | 103/18,286 | 26/2,015 |
| `ragged_extreme_preserved_20260928` | 1,827 | 0.850557 | 113/16,402 | 26/1,827 |

Every audited forward had unique suffix slots disjoint from live prefixes,
bitwise protected-row isolation, independent repeat and packed restoration.
Investigation mode retains/logs the ordinary 5% diagnostic gate but stops/saves a
severe recurrence at 50%; it is not a relaxed “pass.” The second run deliberately
stops after saving full 36-layer KV state.

### 12.5 Severe 85.1% recurrence: exact same-state controls

Preserved B10/prefix 1491, 63 real requests, 567 physical queries, captured C64×9:
**576 executed rows**. Matching physical rather than executed row count would
give the wrong matrix shape. Largest raw-feature change is target layer 17,
channel 4: 14.75 versus 1456.0 at query position 8. This is a hidden norm discrepancy,
not 85% wrong tokens; all ten target argmax tokens agree.

| Replay | Result on the ten real queries |
| --- | --- |
| Fixed C1/B10 or B12 | Bitwise equal to saved independent |
| Fixed C48/B12, eager, 576 rows | Bitwise equal to saved packed-eager output; 0.850749 relative L2 vs C1 |
| Uniform ragged C48/B12 | Bitwise equal to fixed C48/B12 |
| Fixed C64/B16, eager | 0.136535 relative L2 vs C1 |
| Original physical layout, eager | Bitwise equal to saved eager |
| Original physical layout, graph | Bitwise equal to saved graph |

Every decision is A9/bonus 10343. Prefix KV is unchanged, and allocator availability
returns to 64 request/131,072 token slots. Other requests use copies of the
protected prefix; the whole original batch was not preserved.

A separate Transformers precision control gives dense C48/B12-versus-C1/B10
hidden L2 0.00825435 in BF16 and 0.000001562 in FP32, no argmax changes. However,
FP32 dense outputs still differ substantially from saved SGLang outputs. This
is not a demonstrated FP32 fix for SGLang; backend/fused-operation differences
remain confounded and the exact arithmetic kernel was not isolated.

### 12.6 Actual emitted-token discrepancy: latest completed result

Captured B9/prefix 793, row 31 of C63, 693 physical queries, graph C64×11=704 rows.
Hidden relative L2 is only 0.007722, but graph verification accepts A5 with bonus 525;
independent and same-batch eager accept A0 with bonus 8253.

Exact saved-state replay found:

- Fixed C64/B11 **eager** is bitwise equal to saved ragged-eager hidden/logits.
- Fixed C64/B11 **graph** is bitwise equal to saved ragged-graph hidden/logits.
- Uniform ragged/fixed metadata agree under eager execution.
- Original-layout eager/graph each exactly reproduce their own saved result.
- Prefix KV and allocator counts remain unchanged.

The first prediction explains the acceptance jump:

| Execution | Logit for token 8253 | Logit for token 11625 | Selected token |
| --- | ---: | ---: | ---: |
| Independent/same-batch eager | 33.25 | 33.25 | 8253 |
| Graph | 33.0 | 33.25 | 11625 |

The first proposed token is 11625. A true eager `argmax` tie chooses 8253, rejecting
immediately; the graph accepts it and four following tokens. This is an actual
output difference, **not** the superseded audit's topk/argmax-counting mistake.
It is also not specific to mixed-length packing: fixed-width graph replay
reproduces the full saved tensors. It does not explain every other mismatched
state or certify full transcripts.

### 12.7 Harness fixes and sign-off boundary

Three earlier failed/superseded harness runs are preserved:

1. Draft input embeddings were cloned after an in-place fused normalization;
   fixed by preserving pre-forward inputs and testing packed restoration.
2. Comparing `topk(...)[0]` to `argmax` counted tied identical logits differently;
   fixed by using the verifier's actual argmax rule on both sides.
3. Shadow projection grew persistent inference tensors later written outside
   inference mode; fixed test-side buffer creation in a normal no-grad context.

These are harness errors, not fixes to the recovered runtime, and do not explain
the subsequent independently preserved numeric cases.

Current conclusion: **bounded packing/acceptance/KV/isolation checks pass; no
ragged-specific defect is demonstrated in the preserved cases; exact numerical
and transcript parity is not certified.** A large hidden gap can coexist with
unchanged decisions, while a small gap can change the first accepted token.
The drafter consumes target-derived features, so unchanged current argmax does
not imply unchanged future draft candidates.

Ten focused recovery/profiling-helper tests pass. Greedy-only, TP1, page 1, B2–16
are the relevant runtime-tested scope. Arbitrary stochastic decoding, other
page sizes/TP settings, long-run allocator stress and full target-only transcript
parity remain unvalidated. No clean recovered-engine performance numbers exist yet.

## 13. Earlier multi-candidate work — separate track

Historical handoff evidence, not rerun for this document. Repo:
`/Users/atharvramesh/Projects/dflash`, including its `third_party/sglang`.
One draft forward provides logits for a greedy candidate plus sampled
alternatives; this is not K independent draft forwards. Dense verification
repeats suffix/prefix work; packed-tree verification shares candidate prefixes,
uses custom masks/logical positions, selects a branch and commits matching KV.

Recorded Transformers AIME25 result:

| Method | Tokens/s | Mean tau | Cycle ms |
| --- | ---: | ---: | ---: |
| B16 baseline | 129.252 | 6.435 | 48.793 |
| Batch K4 | 133.140 | 6.793 | 49.914 |
| Tree K4 | 137.345 | 6.879 | 49.046 |
| Tree K16 | 146.597 | 7.581 | 50.602 |

The +13.4% throughput result belongs to this **Transformers** run, not SGLang.
Candidate realizations differ; it is not an identical-candidate kernel A/B.
`tau` conventions must be inspected before comparison with proposed-token A.

Separate SGLang shared-prefix compaction reduced verify positions 60.37→48.15,
but cycle 25.357→25.737 ms and throughput 243.73→241.59 tokens/s. Graphs disabled,
AIME25, C1, K4, prefix 2. Another focused profile measured target verify 8.0345→10.2877 ms
and draft 1.6630→1.6141 ms; attention trace aggregates grew about 82%, GEMM 7%.
These are different runs/denominators and must not be merged. Greedy checks
do not certify arbitrary stochastic sampling.

Recorded references in that older repo:

- `outputs/aime25_transformers_redo_20260315_190259/transformers_redo_summary.md`
- `docs/SGLANG_SHARED_PREFIX_AB_20260314.md`
- `docs/SGLANG_FOCUSED_KERNEL_PROFILE_20260315.md`

## 14. What we learned, what we have not established, and next work

### 14.1 Grounded findings

- Acceptance and useful block choice cannot be reduced to one dataset-level
  mean. There is within-request structure and useful pre-draft fused signal.
- Current evidence favors a small fused control over costly context models.
- Correcting the counterfactual labels is necessary for actual shorter drafting,
  but the 10k experiment shows it is not sufficient for a large predictor jump.
- Future draft/verification information predicts better, but paying to obtain it
  changes the system design. Offline post-draft gains do not imply saved draft cost.
- High-concurrency cost is dominated by target verification in the measured
  spec-v1 setup. Attention query tiling and dynamic bookkeeping can limit savings.
- Cheap predictor compute is not the main unresolved cost. The packed execution,
  graph padding, accept/commit/free path and target features matter.
- Numeric execution modes can change labels and trajectories in fixed as well
  as ragged batching. Same-prefix, same-engine controls are essential.

### 14.2 Claims not currently supported

- Achieving 70% acceptance ratio at 96% retained acceptance.
- A substantial measured adaptive DFlash serving speedup from the new predictor.
- Beating BlockPilot or other named papers in a matched implementation.
- Exact hardware/concurrency-independent logits, acceptance labels or transcripts.
- That overlapping PCA/t-SNE means fused features are unrelated to acceptance.
- That larger B must preserve the smaller B candidate prefix/accepted count.
- That a low scalar hidden error alone establishes correctness.
- That the unfinished September 15 raw/fused expansion trained 100k models.
- That the source of collection SIGTERM or temporary SSH outages is known.

### 14.3 Next executable sequence

The user has now authorized a narrower regime screen before the integration
sequence below: matched CPU oracles (completed), then fixed B8/B16 at C64/C128
and controlled 512/1,024-token prompts on dense and MoE pairs in the recovered
v2 engine. Separate clean/event processes and low-C model smokes are required.
This does not authorize training another predictor or launching a hybrid. The
following remains the longer-term sequence, contingent on measured headroom.

1. Preserve the current numerical fixtures and define the comparison contract:
   same runtime/backend/dtype/graph mode, token-level decisions as well as hidden
   errors. Extend target-only transcript and allocator/lifecycle checks without
   hiding graph/eager differences or relaxing thresholds into a claimed pass.
2. Port non-synchronizing component instrumentation to the recovered **v2** worker.
   The existing fixed-study hook targets v1 and cannot be reused unchanged. Keep
   uninstrumented throughput separate. Built-in synchronized timers are diagnostic.
3. Measure same-engine fixed versus packed variable-length execution, retaining
   actual per-request B, physical/executed token counts, actual graph use and prefix
   distributions. Separate draft forward, projection, verify, allocation/packing,
   accept/commit, feature/KV upkeep and outside-worker gaps.
4. Add the frozen 100k and new actual-block heads as explicitly named controls;
   distinguish a forced length pattern from a learned policy and from full-B16
   drafting followed only by shorter verification.
5. Evaluate closed-loop outputs and throughput against the best fixed B and
   target-only AR on matched workloads. Then decide whether predictor improvements,
   runtime work or a changed drafter-training objective offer sufficient headroom.
6. Before a paper claim, add genuinely held-out workloads, repeated seeds/runs,
   supported hardware/concurrency coverage, and a verified literature/baseline
   comparison. Do not select favorable historical tables across unmatched regimes.

No new large training sweep, renewed 100k paired collection, mask training,
hardware-specific reward, recurring monitor or broad cluster change is implicit
in this plan. The unfinished paired expansion remains a separate recover/audit
task if returned to; its partial evidence must not be overwritten.

## 15. Full plot paths and artifact navigation

All following local files were located while preparing this record. PNGs also
have corresponding PDF files in the same directories.

### 15.1 Collected fused vectors grouped by observed acceptance

- [Overlay, A0–15](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/fused_acceptance_groups_20260927/fused_acceptance_groups_overlay.png)
- [PCA, sixteen acceptance panels](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/fused_acceptance_groups_20260927/fused_pca_by_acceptance.png)
- [t-SNE, sixteen acceptance panels](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/fused_acceptance_groups_20260927/fused_tsne_by_acceptance.png)

These are individual collected fused vectors, not text embeddings or model
predictions. Shared coordinates/axes; gray points are other acceptance groups.

### 15.2 Broader geometry, temporal and diagnostic probes

- [Feature geometry](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/acceptance_geometry_pilot_20260927/feature_geometry.png)
- [Raw versus fused PCA](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/acceptance_geometry_pilot_20260927/raw_fused_pca.png)
- [t-SNE sensitivity](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/acceptance_geometry_pilot_20260927/tsne_sensitivity.png)
- [Temporal structure](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/acceptance_geometry_pilot_20260927/temporal_structure.png)
- [Held-out probes](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/acceptance_geometry_pilot_20260927/heldout_probes.png)

### 15.3 Verifier probabilities

- [Verifier versus fused geometry](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/verifier_signal_analysis_20260927/verifier_vs_fused_geometry.png)
- [Verifier-probability PCA by A](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/verifier_signal_analysis_20260927/verifier_pca_by_acceptance.png)
- [Verifier-probability t-SNE by A](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/verifier_signal_analysis_20260927/verifier_tsne_by_acceptance.png)
- [Causal confidence by A](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/verifier_signal_analysis_20260927/causal_verifier_confidence_by_acceptance.png)
- [Incremental signal probes](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/verifier_signal_analysis_20260927/incremental_signal_probes.png)

### 15.4 Key current machine-readable reports

- [10k training audit](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/actual_block_predictor_10k_finish_20260928/training/audit.json)
- [All six predictor results and uncertainty](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/actual_block_predictor_10k_finish_20260928/training/summary.json)
- [Primary request/cycle policy evaluation](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/policy_granularity_20260927/analysis/summary.json)
- [Exact-breakpoint sensitivity](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/policy_granularity_20260927/analysis_exact_calibration/summary.json)
- [Completed headroom diagnostic](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/block_headroom_128p_20260926/headroom_summary.json)
- [Severe saved-state replay](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/ragged_extreme_replay_20260928/graph/FIXTURE.json)
- [BF16/FP32 precision control](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/ragged_extreme_precision_20260928/summary.json)
- [Output-changing fixed/ragged replay](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/ragged_emission_replay_20260928/graph/FIXTURE.json)

For newer experiments, remote directories use the same run name beneath
`/data/scratch/zekaili/atharv/dflash/runs/`. The imported original headroom run
instead sits under `/data/scratch/zekaili/atharv/dflash/data/dflashv2_data/runs/`.
Large KV snapshots, models, full probability arrays, training checkpoints and
compiler caches generally remain remote rather than duplicated locally.

### 15.5 Key scripts and deployment provenance

| Purpose | Code / references |
| --- | --- |
| Actual block collection / oracle | `scripts/collect_block_headroom.py`, `scripts/block_headroom.py`, `scripts/audit_block_headroom.py` |
| Mask intervention | `scripts/causal_mask_probe.py` |
| Geometry / grouped plots | `scripts/analyze_acceptance_geometry.py`, `scripts/plot_fused_acceptance_groups.py` |
| Verifier replay / probes | `scripts/collect_verifier_signal_replay.py`, `scripts/analyze_verifier_signals.py`, `scripts/probe_verifier_nonlinear.py`, `scripts/probe_verifier_survival.py` |
| Request/cycle comparison | `scripts/collect_policy_granularity.py`, `scripts/analyze_policy_granularity.py` |
| Actual-block heads | `dflash/block_response.py`, `scripts/train_actual_block_predictor.py` |
| Efficient recovery / worker diagnostics | `scripts/run_efficient_recovery.sh`, `scripts/worker_diagnostics.py` |
| Fixed-width profiles / analysis | `scripts/profile_sglang_latency.py`, `scripts/summarize_latency_study.py`, `scripts/summarize_cuda_trace.py`, `scripts/analyze_latency_headroom.py` |
| Source recovery / primitives | `scripts/recover_legacy_ragged.py`, `scripts/check_ragged_primitives.py`, `scripts/run_ragged_primitive_check.py` |
| Recovered-engine model audit | `scripts/run_ragged_model_check.py`, `scripts/ragged_audit_hook/`, `scripts/summarize_ragged_audit.py` |
| Dense precision control | `scripts/diagnose_saved_ragged_state.py` |

Important run-pinned commits include geometry `d4232ff`, verifier replay `279c815`,
policy granularity `87985ea`, exact calibration `22f6c90`, efficient recovery `14ab486`,
diagnostics `46d1d71`, completed actual-block launch `ca09995`, fixed profiling
`6b785fd`, severe-state replay `5ed4f23`, precision control `e0ab25e`, emission
capture `759e1b5`, and fixed-graph emission replay `b3cbaa0`. Individual config files
and source hashes remain authoritative; a short commit alone does not pin data
or all runtime dependencies.

## 16. Current handoff

This section records the September 28 handoff; see the dated addendum in
sections 17–19 for the later intermediate-target probe pilot, controlled
data/update scaling study, and confidence-first cascade test.

The newest completed analysis at that handoff was the matched-state oracle in section 9.1. A
bounded dense/MoE fixed-width screen is launched on GPU 4, with C1/C4 model
smokes before C64/C128; see the dedicated screen note for durable paths and
completion markers. No recurring monitor is active. The earlier fixed-width
graph reproduction in section 12.6 remains a numerical diagnosis, not universal
parity. The recovered source evidence is unchanged.

The project has real cycle-level signal and actual-block datasets, but the
present predictor gain is small and the strong-paper outcome remains unachieved.
The immediate question is whether a supported MoE/high-concurrency regime has
a meaningfully steeper fixed-width cost curve. Favorable measurements would
justify matched MoE acceptance headroom and then **same-engine, output-validated
packed adaptive execution**. The 4B oracle must not be presented as MoE headroom.

## 17. September 29 addendum: intermediate-target probe pilot

The bounded feasibility test is complete, not running. See
[the full protocol, audit, results and paths](midverify_probe_pilot_20260929.md).
No serving-engine changes, larger collection or recurring monitor were launched.

Frozen saved B16 candidates were replayed on Qwen3-4B. The dataset contains
2,000 training cycles (263 prompts), unchanged 342 calibration cycles (47
prompts), and unchanged 1,416 assessment cycles (187 prompts). Capture layers
6/9/12/18/24; train candidate-conditioned linear and 128-wide MLP probes,
three seeds each. Fresh labels and hidden states come from the same forward.

At calibration-96% settings, the replayed post-draft candidate-logprob control
achieves 95.25% assessment retention with 9.049 full-depth-equivalent work rows.
The L12 MLP seed mean achieves 95.14% with 12.360 work rows; the L24 MLP achieves
95.21% with 13.885. These are work proxies, NOT throughput measurements. The
comparison does not support integrating this pilot into SGLang.

A matched-progress clairvoyant L12 probe needs at least 10.181 work rows at
the confidence control's exact accepted-token total. Break-even depth in this
proxy is only 7.33 layers at that operating point. This is not a universal
hardware/latency ceiling; at higher retention it changes. The control remains
stronger in diagnostic subsets with stable replayed candidate identities.

Use `outputs/midverify_probe_2k_20260929/assessment_fp64` for authoritative
metrics/plots and `matched_progress_and_sensitivity_fp64` for the bound and
replay checks. Original `training` checkpoints/history are preserved. A
float32 threshold-comparison edge case was fixed by CPU rescoring; checkpoint
and calibration selection did not change. Final focused tests: 30 passed.
The detailed note reports all numerical drift and the replayed-confidence
baseline limitation. No pre-draft 70%-ratio or online-speedup claim follows.

## 18. September 29 addendum: controlled intermediate-probe scaling

The subsequently user-authorized scaling study is complete. See
[protocol, audits, full results and paths](midverify_scaling_20260929.md).
Same frozen target/drafter, five depths, linear/128-wide MLP architectures and
three seeds. Nested 2k/5k/9,999 training cycles; all original 2k feature bytes and
all 342 calibration/1,416 assessment states preserved. The raw 10k replay found
one newly accepted-EOS training state outside the 5k prefix. Its evidence was
preserved and it was excluded in a separate fully audited nonterminal cache;
no validation change, label repair or replacement state was introduced.

All 90 independent models completed with 128/1,024/extended update-budget
checkpoints, 270 policies total. Full 128-state batches match exposures across
sizes; initialization is identical per architecture/seed. Selection uses only
calibration. All 270 same-shape checkpoint reloads and all final checksum
audits passed. GPU4 was idle afterward; no recurring monitor or integration.

At the primary 1,024-update budget, L6 MLP three-seed means:

| Train cycles | Assessment retention | Equivalent full-depth work rows |
| ---: | ---: | ---: |
| 2,000 | 95.72% | 12.038 |
| 5,000 | 95.18% | 11.071 |
| 9,999 | 95.48% | 10.782 |
| Replayed draft-confidence control | 95.25% | 9.049 |

More data helps: estimated work falls 10.43% from 2k to 9,999 with a -0.23pp
change in retention. These are calibration-matched, not exactly assessment-
matched operating points. The larger L6 probe still needs 19.15% more work than
the confidence control. Its 71.95% acceptance ratio is post-draft AND after six
target layers; it is not the original pre-draft 70% objective. Later layers
still lose on this work proxy. None of these values is measured throughput.

Extended training chooses the same L6 MLP checkpoints as the 1,024 budget.
Training loss continues falling while calibration loss rises; more data reduces
but does not eliminate overfitting. The improvement argues against declaring
an intrinsic information limit from the 2k pilot, but still does not justify
segmented-engine integration or an automatic larger sweep.

## 19. September 29 addendum: confidence-first L6 cascade

The subsequently authorized bounded complementarity test is complete. See
[full protocol, audits, results and paths](midverify_cascade_20260929.md).
No new data, target/drafter forward passes or fine-tuning, engine changes,
or recurring monitor. Same 9,999 training / 342 calibration / 1,416 assessment
states, all frozen. Only L6 is used.

First trim with draft candidate logprob to K0 query rows; then an L6 probe may
shorten this to K1 <= K0. Joint exact threshold calibration minimizes
`[6*K0+30*K1]/36` at 96% or 99% accepted-token retention. Both target-free learned
controls get the same two-threshold search, with cost K1 because both decisions
are available before target verification. All lengths remain integers 1–16.

Three feature ablations, seeds 913/914/915: confidence-only (641 parameters),
candidate-confidence (328,321), and target-candidate-confidence (983,681).
Same 128-wide GELU MLP, dropout 0.05, masked BCE, AdamW 3e-4 and 1,024-update budget.
Nine models, 18 calibration-selected checkpoints. Also evaluate the frozen
target-only L6 heads under the new cascade policy, with no weight changes.

| Calibration target | Policy | Assessment retention | Full-depth-equivalent work rows |
| --- | --- | ---: | ---: |
| 96% | Calibration-chosen target-free controls | 94.805% | 8.895 |
| 96% | Confidence-first L6 fusion cascade | 95.012% | 8.787 |
| 99% | Calibration-chosen target-free controls | 98.395% | 10.666 |
| 99% | Confidence-first L6 fusion cascade | 98.209% | 9.999 |

These are three-seed means, not ensembles. The 96% setting offers only about
1.22% work reduction; two seed-specific paired work intervals span zero. At 99%,
work falls 6.25%, with 0.186pp lower retention. Each seed's paired work-saving
interval excludes zero, but retention equivalence is not established. All
values are offline work proxies, not latency/throughput; target-feature head
overfitting, unmatched parameter counts and calibration uncertainty remain.

Forty focused tests, CPU smoke, all 18 exact checkpoint reloads and the 26-file
remote completion audit passed. Main launch commit `db47d71`; GPU4 completed in
19.14 seconds excluding initial source audit and was idle afterward. The initial
SSH connection reset started no run, verified before retry. Reports and plots
are copied under `outputs/midverify_cascade_20260929`.

Current finding: evidence of modest complementary target signal, especially
at the higher-retention point, not yet a matched-retention systems win. No
automatic 100k collection or segmented-engine integration. A further gate would
be retention-controlled confirmation versus the strongest target-free controls,
followed only then by bounded overhead and actual-forward checks.
