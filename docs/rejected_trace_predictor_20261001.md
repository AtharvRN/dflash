# Previous rejected-trace information for pre-draft block selection

Prepared 2026-10-01. This document describes the implemented pilot protocol,
not a completed research result.

## Question and current status

Does the immediately preceding cycle's rejected verification trace improve the
choice of the next actual draft block size, beyond the current fused feature,
the new anchor, the previously rejected anchor, and the previous acceptance
count?

The implementation covers collection, cache auditing, matched training, policy
calibration, assessment, and isolated timing. Local synthetic CPU checks test
the implementation. No Qwen3-4B collection or trained-model result for this new
experiment is established yet. At the workstation inspection for this
implementation, every GPU was occupied, including GPU 4; the real GPU run was
blocked by resource availability. The launch code refuses an occupied device.
There is no speedup result or active recurring monitor implied by this document.

The experiment tests information value on existing B16 reference trajectories.
It does not implement the proposed two-cycle value-of-information planner.
Rejected-state reuse should be treated as existing prior-art territory, not a
new contribution by itself. Any later priority claim requires a verified
literature review. A successful pilot would establish that this particular
information improves actual block-size decisions in our controlled setting.

## Definitions and causal boundary

- B includes one known anchor and B-1 proposed tokens.
- A_B is the consecutive number of accepted proposed tokens, between zero and
  B-1. The anchor and correction/bonus token are excluded from acceptance ratios.
- The collection trajectory always runs original Qwen3-4B/DFlash B16 with greedy
  decoding, thinking off, BF16, SDPA, TF32 disabled, and pinned Transformers
  4.57.1. Alternative B2 through B16 outcomes are measured at the same state.
- The immediately preceding cycle is cycle t-1, even when that cycle was not
  selected as a dataset row. The previous saved row may be much older.
- Every input to the cycle-t predictor is copied before any cycle-t draft is
  executed. The alternative current-block forwards supply labels only.
- Rejected states are predictor evidence. They are not valid target KV and are
  never committed into the target's continuation cache.

Let a previous B16 block be `[anchor, d1, ..., d15]`, with A accepted draft tokens.
The target posterior at block row A supplies the new correction token, or the
bonus token when A=15. The next prefix, including its new anchor, must equal:

```text
previous prefix including old anchor
    + previous draft tokens[:A]
    + [previous target posterior[A]]
```

For A<15, draft token d_(A+1) is the first rejected token. Its embedding is a
shared input to all model arms. The optional trace contains only tokens
**strictly after** that first rejected token: d_(A+2) through d15. Thus its length
is `max(14-A, 0)`. A=14 has a rejected anchor but no later trace; A=15 has neither.

The draft output row for token d_j is block row j. The target hidden state
predicting d_j is block row j-1. For the retained suffix, this means draft rows
`A+2..15` paired with target rows `A+1..14`. These target states can depend on the
previous wrong continuation. That is intentional: the experiment asks whether
that old, already-computed evidence is useful after supplying the actual new
anchor. It does not treat these states as hidden states of the corrected path.

## Collection and row selection

The collector is `scripts/collect_rejected_trace.py`.

Training prompts come from the canonical training split with the existing
uniformly shuffled, exact-content-filtered selection. The default plan permits
up to 600 training prompts and samples at most eight states per prompt along a
256-token output trajectory, with a 2,048-token prompt cap. Collection stops
submitting new training prompts once it has at least 2,000 eligible rows;
bounded in-flight prompts may finish and produce a small excess. Training uses
exactly the **first 2,000 eligible training rows in immutable receipt order**.
These are cycles, not 2,000 prompts.

Evaluation preserves the source experiment's prompt partition: 47 calibration
prompts and 187 assessment prompts, whose source caches contain 342 and 1,416
states respectively. The source cache identifies the cycle numbers and prefix
hashes to replay. Its previous labels are never joined to newly generated
features. Every retained replay row obtains fresh B2 through B16 outcomes.

The collector reports, rather than hides, changes in usable evaluation coverage:

- Overlength prompts, early EOS, or unavailable output slots can leave source
  cycle IDs uncollected. These appear in `missing_evaluation_states`.
- A newly replayed prefix differing from the source prefix is recorded with
  `reference_prefix_drift` and excluded from training/evaluation.
- Rows excluded in the source experiment remain excluded (`source_excluded`).
- If any actual-B outcome accepts an EOS token, the current row is excluded
  (`accepted_eos`).

Consequently, the final assessment size is **not assumed to remain 1,416**. A
completed collection means all planned evaluation prompts were processed and
the training target was reached; it does not mean every source evaluation
state survived replay. Report planned, collected, eligible, and missing counts
by partition before interpreting a result. All arms use the identical retained
row set. Calibration and assessment membership never exchange prompts.

At each selected state, independent target/draft cache snapshots are restored
for alternative block sizes. Actual B16 uses the reference forward. B2-B15
are evaluated in randomized order. Unchanged-candidate B4/B8/B12 verification
controls measure the effect of target verification width separately. The first
audit prompts in each partition additionally check reversed execution order
and greedy autoregressive verification. A canonical disagreement stops the run
before training and preserves the evidence for investigation.

The collection seed defaults to 1001. The frozen evaluation partition is
recovered using the existing seed-928 selection; the new seed does not replace
that partition.

## Stored inputs and provenance

The schema is `dflash_rejected_trace_v1`. Each row has:

| Field | Shape and stored dtype | Role |
|---|---|---|
| `features` | 2560, FP16 | Latest pre-draft fused target-context vector |
| `anchor_embedding` | 2560, FP16 | Embedding of the actual current anchor/correction |
| `rejected_anchor_embedding` | 2560, FP16 | Previous first rejected token embedding; zero when absent |
| `previous_meta` | 3, FP32 | `[has_previous, previous_B/16, previous_A/15]`; all zero at cycle zero |
| `trace_draft` | 15 x 2560, FP16 | Previous drafter final hidden rows for retained suffix tokens |
| `trace_target` | 15 x 2560, FP16 | Previous target final hidden rows predicting those suffix tokens |
| `trace_mask` | 15, uint8 | Leading valid slots followed by zeros |
| `trace_token_ids` | 15, int64 | Retained old suffix token IDs; padding -1 |
| `trace_offsets` | 15, int16 | Relative successor offsets 1..count; padding zero |
| `actual` | 15, int16 | Fresh `[A_2, ..., A_16]` labels |

The current fused vector uses `hidden_norm(fc(latest_pending_target_feature))`
before current drafting. Previous trace hidden states come directly from the
same previous B16 draft and target verification forwards, before cropping.
The token IDs support auditing; this first model does not feed suffix token-ID
embeddings to its trace encoder.

Each prompt has JSON records, an NPZ array shard, and a hash receipt. JSON proof
records include the previous cycle number, previous prefix and hash, drafted
IDs, all previous target posterior IDs, accepted length, correction token,
same-forward declaration, and current prefix/hash. This lets the audit verify
the immediate predecessor relationship rather than assuming adjacent saved
rows are consecutive cycles.

Shards are durably backed up before their receipt. The final completion marker
binds `config.json`, `receipts.json`, and `collection_summary.json`. The audit in
`scripts/audit_rejected_trace_cache.py` rejects incomplete production caches,
missing/orphan shards or receipts, hash mismatches, wrong shapes/dtypes,
nonfinite values, invalid padding, label/record disagreement, split leakage,
prefix drift retained as eligible, and invalid predecessor/correction evidence.
The trainer hashes the audited inputs again after training.

These are serialized integrity checks, not an independent proof that every
activation came from the claimed GPU forward. Collector indexing tests and
later real-model smoke/replay checks remain part of validation.

## Three arms and donor construction

All arms instantiate `dflash.rejected_trace.RejectedTraceResponseModel` with
1,546,767 parameters, width 128, and dropout 0.05. All receive the same common
inputs: fused vector, actual anchor embedding, previous rejected-anchor
embedding, and previous metadata. This intentionally tests the incremental
value of the longer trace beyond these inexpensive inputs.

| Arm | Optional trace |
|---|---|
| `control` | Every trace slot masked; only the common inputs and learned null-memory path remain |
| `aligned` | The actual immediately preceding trace, provided a valid matched donor exists for this row |
| `shuffled` | A different prompt's trace from the same partition and previous B/A stratum |

The donor mapping is frozen with seed 913. Donors are sampled **with replacement**
from another prompt, within the same train/calibration/assessment partition,
and with the same previous block size and previous acceptance count. This is a
sampled donor control, not necessarily a permutation: some strata cannot admit
a cross-prompt permutation. Entire traces are substituted together, preserving
their internal draft/target pairing and valid length.

Only optional trace fields are replaced. The row's actual and wrong-anchor
embeddings, previous metadata, current fused feature, and current labels stay
unchanged. No current outcome participates in donor selection. Evaluation
donors are an offline negative-control construction, not a deployable policy
mechanism or additional training examples.

If a row has no nonempty suffix, or no different-prompt donor in its stratum,
its optional memory is empty in **all three arms**. The row and common inputs
remain in the experiment. Report original-memory rows, effective-memory rows,
and nonempty rows that lost memory because no donor existed, separately by
partition. This rule ensures that the aligned/shuffled comparison uses the
same memory availability, while restricting the conclusion to the covered
rows. Overall and memory-availability subgroup metrics are both saved.

Equal parameter counts are not equal effective capacities: the control's
masked trace projections do not learn from varying memory. The learned null
memory supplies a constant path. Therefore, a gain over the null-memory
control alone cannot establish that semantic alignment is useful. The aligned
versus shuffled comparison is essential; the pilot must report both.

## Model, loss, and actual block choice

The shared encoder projects the fused vector and the two anchor embeddings to
128 dimensions; the token projection is shared between the actual and wrong
anchors. Their concatenation with the three metadata values becomes a learned
query. Each optional memory slot combines projected draft hidden, projected
target hidden, and a learned relative-offset embedding, then maps to width
128. A single query attends over these slots and an always-valid learned null
slot. The null slot makes first-cycle/empty-memory attention well-defined.

The query and pooled memory are concatenated into a 256-256-128-15 MLP. Its
outputs are independently bounded:

\[
\hat\mu_B = (B-1)\,\sigma(f_B(x,M)), \qquad B=2,\ldots,16.
\]

There is no monotonicity constraint across B because changing B changes the
actual proposed candidates. This is an actual-response model, not a survival
head or a clipped-B16 prediction.

The loss is unweighted MSE on accepted-token counts over all rows and all 15
actual block sizes:

\[
\mathcal L = \frac{1}{15N}\sum_{i=1}^{N}\sum_{B=2}^{16}
  (\hat\mu_B(x_i,M_i)-A_{i,B})^2.
\]

Production training uses seeds 913/914/915, six epochs, batch size 128, AdamW
with learning rate 3e-4 and weight decay 0.01, gradient clipping at 1.0, and no
learning-rate scheduler. Every arm within a seed uses identical initial
weights, epoch row order, model shapes, and dropout RNG consumption. Initial
weight and ordering hashes are saved and checked. All calibration and
assessment scoring uses CPU FP32, including when optimization runs on a GPU.

At inference, the selected proposed-token budget d=B-1 is:

\[
d^* = \operatorname*{argmax}_{d\in\{1,\ldots,15\}}
       [\hat\mu_{d+1} - \lambda d], \qquad B^*=d^*+1.
\]

The multiplier is calibrated against an accepted-token retention constraint;
it is not a hardware timing reward. Ties choose the smaller budget. Candidate
penalties use calibration prediction-envelope breakpoints and their intervals,
with a fixed-B16 fallback. The same actual-block policy code is reused from
`dflash/block_response.py`.

## Checkpoint selection and assessment

For every epoch, calibration jointly selects the penalty and checkpoint that
minimize mean proposed budget subject to at least 96% aggregate retention
relative to actual B16 outcomes. Ties prefer higher calibration retention,
then the earlier epoch. The code also saves calibration-selected operating
points for 90%, 95%, 98%, and 100% retention. It does not select the checkpoint
by assessment MAE, acceptance ratio, or speed.

All nine checkpoint selections are frozen before assessment. Reloaded weights
must reproduce the saved calibration selections. Each model is then evaluated
on the same eligible assessment rows. A calibration-selected fixed width and
fixed B16 are additional controls.

For selected actual budgets d_i, report:

\[
\text{ratio}=\frac{\sum_i A_{i,d_i+1}}{\sum_i d_i},\qquad
\text{retention}=\frac{\sum_i A_{i,d_i+1}}{\sum_i A_{i,16}}.
\]

The primary scientific comparison is proposed work at comparable assessment
retention. The nominal 96% calibration gate does **not** guarantee identical
assessment retention; report actual retention alongside every budget or ratio
comparison. Full assessment retention/work curves use already generated
calibration settings. They are descriptive curves, not permission to retune
the primary policy on assessment.

Saved results include individual seed metrics, mean/std/range across seeds,
per-block MAE/MSE, mean accepted length, aggregate ratio, mean budget,
memory-coverage subgroups, donor identities, predictions and selected budgets,
and 2,000 paired whole-prompt bootstrap resamples. Bootstrap draws use the same
prompt samples across policies and freeze checkpoints, penalties, and the
donor map. They do not replace training-seed uncertainty or measure donor-seed
robustness. The existing assessment tasks have already been inspected in this
project and remain development evidence, not an untouched final benchmark.

There is no automated scientific go/no-go threshold beyond integrity and
calibration checks. A useful result requires aligned memory to improve actual
decisions over both controls across seeds. Lower prediction error alone, or a
ratio increase explained by lower retention, does not establish success.

## Cost measurements and limits

Collection records CUDA-event time around prior-memory allocation/gather/copy.
It excludes dataset device-to-host transfer, JSON proof recording, predictor
execution, and serving integration. It is diagnostic stream elapsed time, not
an end-to-end per-cycle overhead measurement. The two full FP16 trace buffers
alone occupy 153,600 bytes per retained request before metadata and embeddings;
serving lifetime and allocation costs still need measurement.

The trainer can benchmark the full predictor after ten warmups, by default
50 repetitions, for batches of 1, 64, and 128 rows. Inputs are preallocated;
timing includes the model's trace encoding and dtype conversion, but excludes
input gathering/H2D, target trace capture/retention, scheduling, KV handling,
and engine integration. CPU uses `perf_counter`; GPU uses CUDA events. These
isolated timings cannot be added to historic engine numbers and called a
speedup. The implementation's full-shape masked control is an experimental
capacity/shape control, not an optimized serving implementation.

Further limitations:

- All labels are counterfactual choices on fixed-B16 trajectories. Deploying a
  learned length policy changes cycle boundaries and the previous trace
  distribution; offline gains require a closed-loop test.
- Previous B is always 16. This pilot cannot establish that deliberately
  choosing a longer prior block has positive future information value.
- The small training set may be insufficient, but scaling is conditional on a
  decision gain from the new information rather than another small MAE gain.
- Retention counts accepted proposed tokens. Throughput also depends on the
  anchor/correction progress, draft/verify costs, contexts, batch packing,
  concurrency, graph mode, and trace lifetime overhead.

If the aligned arm shows a meaningful, reproducible decision improvement, the
next stage is a closed-loop rollout with a correctness check and matched
engine timing. The more ambitious two-cycle controller then needs genuinely
paired prior-length interventions and future outcomes; this cache alone cannot
train or validate that causal value-of-information claim.

## Entry points and review artifacts

```text
scripts/collect_rejected_trace.py
scripts/audit_rejected_trace_cache.py
scripts/train_rejected_trace_predictor.py
scripts/compare_rejected_trace_caches.py
scripts/run_rejected_trace_predictor.sh
dflash/rejected_trace.py
tests/test_collect_rejected_trace.py
tests/test_rejected_trace_cache.py
tests/test_rejected_trace_predictor.py
tests/test_compare_rejected_trace_caches.py
```

The bounded workstation launcher takes an explicit idle GPU and a new run ID:

```bash
GPU=4 WORKERS=4 MODE=pilot RUN_ID=rejected_trace_2k_20261001 \
  bash scripts/run_rejected_trace_predictor.sh
```

This is a ready command, not a claim that GPU 4 is available or that it has
been launched. The launcher acquires cooperative experiment locks and exits
if the GPU is occupied. It first runs a real-model serial smoke, then an
identical parallel smoke and exact cache comparison before enabling multiple
workers for the pilot. It then tests training on the smoke cache. Any failed
gate prevents the full collection. `MODE=smoke` stops after these gates;
`WORKERS=1` skips the parallel gate and collects serially. Collection, smoke,
and training each have hard timeout bounds. There is no retry or monitoring
loop, and existing destinations are never overwritten.

The default durable destination is
`/data/scratch/zekaili/atharv/dflash/runs/rejected_trace_2k_20261001`;
temporary collection data goes to `/tmp/rejected_trace_2k_20261001`. Each smoke
uses one training prompt and two prompts from each frozen evaluation group,
a 64-token output cap, and fewer selected states. Its labels and activations
must match across worker counts; it is not a statistical test of the method.
The one-row training smoke tests ingestion and optimization with empty memory;
nonempty-memory gradients are tested by the synthetic CPU suite, not by that
one-row real-data training smoke. The collector's internal `--max-seconds`
stops submissions but can wait for a running prompt; the launcher's external
process-group timeout supplies the hard execution bound. The collector also
supports `--preflight` to audit source files and planned membership without
loading a model, accessing a GPU, or writing outputs.

The audit CLI is read-only:

```bash
python scripts/audit_rejected_trace_cache.py /absolute/path/to/completed/cache
```

The production training CLI defaults to the exact 2,000-row, three-seed,
six-epoch protocol and requires a fresh destination:

```bash
python scripts/train_rejected_trace_predictor.py \
  --cache /absolute/path/to/completed/cache \
  --output /absolute/path/to/new/training/output
```

Omitting `--gpu` uses CPU. A real GPU launch must first use an available device
and a completed audited cache. `--smoke` is explicitly for reduced/synthetic
validation; its outputs are labeled as smoke results and are not evidence for
the research hypothesis.

Collection artifacts include the immutable plan, receipts, per-prompt proofs
and arrays, collection summary, completion binding, and audit. Training
artifacts include `config.json`, `row_index_and_donors.json`,
`memory_coverage.json`, nine checkpoints and calibration histories,
`selection_frozen.json`, `summary.json`, `assessment_curves.json`,
`assessment_predictions.npz`, `report.md`, and a completion file binding the
result files.
