# Bounded native-engine segmented-verification test

Status: completed. Final amended run and audited analysis are authoritative;
the initial run and both smoke runs remain preserved separately.

Authorized 2026-09-29. This is an execution-feasibility test, not a new training
or collection sweep and not a claim of closed-loop serving throughput.

Use the recovered, hash-verified SGLang spec-v2 source and pinned dependency
container. Qwen3-4B and original DFlash B16, BF16, FlashInfer, TF32 off, GPU4 only.
Read existing cache/checkpoints without changes. Freeze seed913 and the existing
99%-calibrated candidate-confidence and target/candidate-confidence L6 policies;
do not select a seed or threshold using these test outputs.

First smoke on four distinct assessment prompts. Then first eligible cycle of
each of 128 distinct assessment prompts in canonical order, without selection
by acceptance, scored as two C64 batches and the same states as one C128 batch.
Current policies use fresh same-engine draft confidence and L6 target features;
record numerical and policy drift rather than treating offline features as free.

Compare fixed full B16, fixed post-draft B8 truncation (same B16 candidates, not
actual shorter drafting), actual B8/B16 redrafting, target-free learned trimming, the same target-free
policy with a no-prune L6 split, and the L6 cascade. Prefix tokens are preserved;
only the explicitly labeled redraft controls change candidates. Measure both eager execution and manually captured
exact-shape target graphs, with eager drafting for every case. These graph
captures use native attention planning and native model layers but are NOT a
general graph-bucket scheduler integration. Misses must fail, not silently pad.

Prefill and graph capture are untimed. Timed replay includes draft setup/model/
projection, confidence computation where used, actual probe inference, D2H
length decisions, packed-row construction, native attention metadata, model
segments, acceptance/bonus selection and draft-KV materialization. Repeat
snapshots rather than advancing trajectories. Slots are preallocated; native
reserve/free microcost is reported separately. No request scheduler, HTTP,
prefill, live allocator commits, graph misses or workload drain is measured.

Use disjoint CUDA-event phase spans plus synchronized wall-clock cycle times;
also measure the cycle with phase instrumentation disabled. Alternate cases in
a seeded random order across repeats. Report time per actually committed token,
actual same-engine B16 retention, prefix lengths and all numerical differences.
Do not interpret reduced cycle latency alone as a throughput win.

Correctness gate: retained prefix against an independent all-layer forward on
that exact shorter prefix; target keys/values at layers0/5/35; native model
dispatch versus explicit layer traversal; unchanged sampled committed-prefix
keys at every layer; request/KV pool capacity restored on cleanup. The eager
no-prune split must be bitwise identical. Shape-changing/graph controls report
all top1/acceptance/bonus differences; a 2% hidden-relative-L2 stop guard is a
diagnostic guard, NOT a general correctness certificate.

No auto-training extension, new trajectory collection, runtime-source edits,
recurring monitor, other GPU use or subsequent serving integration.

Protocol amendment after the first completed run: the original fixed16 replay
retains saved candidates while the actual B8 control recomputes them. Add actual
B16 redrafting in a fresh bounded rerun to make the two actual fixed-width
controls comparable. Do not change models, states, thresholds or the main
saved-candidate target-free/cascade comparison. Keep all initial evidence.
Also retain individual uninstrumented timing samples, not just their summaries.

## Completed execution and provenance

- Initial implementation: `05340ad`; actual-B8 control: `3ae720b`.
- Initial C4 smoke and expanded C4 smoke passed before C64/C128 measurement.
- Initial main run: `midverify_latency_20260929`, 36 cells, 86.77 seconds inside
  the loaded worker. Preserved, not overwritten.
- Final amended run: `midverify_latency_v2_20260929`, launch commit `a11bbbc`,
  42 cells (seven policies × two modes × three matched batches). 12 instrumented
  and 12 uninstrumented repetitions per cell; 98.83 seconds inside the worker,
  excluding container/model startup and initial file-hash audit.
- Serving dependency image uses Torch **2.11.0+cu129**. This is not the host
  Torch 2.13 replay/training environment; model revisions remain identical.
- Final analysis code `002bd18`; use `analysis_audited`, which also distinguishes
  the target-free policy's physical target-input length from its earlier raw
  confidence gate. This reporting correction changes no recorded timings.
- 45 focused tests passed. Raw completion binds four files; final analysis
  binds four files. Remote audits and all eight downloaded-file hashes passed.
- GPU 4 returned to 0 MiB / 0% and no benchmark container remained running.
  No training, new trajectory collection, background monitor or subsequent
  serving integration was started.

One saved cycle per each of 128 distinct assessment prompts was used, with
prefix lengths 81–1,920 (mean 842.52). These are first eligible cycles, NOT a
representative random sample of all 1,416 assessment cycles. Repetitions measure
timing noise on the same snapshots, not new independent workload examples.

## Primary result: the L6 increment is small

Primary comparison is candidate-confidence target-free trimming versus the L6
cascade, with the SAME saved B16 candidates. Seed 913 and both 99%-calibrated
checkpoints/thresholds were frozen before testing. No assessment retuning.

| Actual C | Mode | Policy | Clean cycle ms | Observed retention | ms/committed token | Rate relative to target-free |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 64 | graph | Target-free | 40.432 | 98.451% | 0.14112 | 1.0000× |
| 64 | graph | No-prune L6 split | 41.391 | 98.451% | 0.14447 | 0.9768× |
| 64 | graph | L6 cascade | 39.544 | 98.673% | 0.13778 | **1.0242×** |
| 128 | graph | Target-free | 68.264 | 98.451% | 0.11913 | 1.0000× |
| 128 | graph | No-prune L6 split | 69.505 | 98.451% | 0.12130 | 0.9821× |
| 128 | graph | L6 cascade | 67.308 | 96.903% | 0.11892 | **1.0018×** |
| 64 | eager | Target-free | 39.007 | 97.168% | 0.13591 | 1.0000× |
| 64 | eager | L6 cascade | 38.218 | 97.603% | 0.13270 | 1.0242× |
| 128 | eager | Target-free | 66.508 | 98.475% | 0.11467 | 1.0000× |
| 128 | eager | L6 cascade | 65.682 | 95.861% | 0.11564 | **0.9916×** |

Retention denominator is accepted proposed tokens from same-engine verification
of the SAVED B16 candidates, within the same concurrency/execution mode.
Time/committed token uses the sum of batch times divided by the sum of actual
`A+1`, never the mean of per-request ratios. It is a replay rate, not HTTP or
closed-loop serving throughput. C64 rows aggregate the two 64-request batches.

At C128 graph, cycle time falls 1.40%, but total committed tokens fall 573→566;
almost all of the apparent cycle benefit disappears after accounting for progress.
The no-prune split costs 1.24 ms at identical lengths and outputs, directly
measuring splitting/extra planning/identity compaction overhead in this harness.

The initial run found approximately +2.53% at C64 graph and +0.14% at C128 graph.
The amended run reproduces this qualitative result: a small C64 increment,
essentially flat C128 graph, and a small C128 eager regression.

Conditional repeated-timing 95% intervals for cascade/target-free replay rate:
C64 graph 1.0232–1.0252; C128 graph 1.0012–1.0025; C128 eager 0.9905–0.9930.
These narrow intervals hold the snapshots and their token counts fixed. They
do not establish workload generalization. Paired prompt-bootstrap retention
delta intervals are much wider: C64 graph −0.495 to +0.998 percentage points;
C128 graph −5.608 to +0.798 points. Retention equivalence is NOT established.

## C128 component timing

Means of disjoint CUDA-stream spans from separate instrumented repeats, ms.
These include device idle gaps inside a span and are **not kernel-only times**.
They need not sum exactly to the clean, uninstrumented wall-clock cycle above.

| Component | Target-free | No-prune split | L6 cascade |
| --- | ---: | ---: | ---: |
| Draft setup / metadata | 1.197 | 1.191 | 1.224 |
| Draft transformer | 8.348 | 8.353 | 8.344 |
| Draft projection + confidence statistics | 13.866 | 13.879 | 13.870 |
| First policy / initial packing | 0.547 | 0.550 | 0.400 |
| Target layers [0, 36), including projection/argmax | 42.573 | — | — |
| Target layers [0, 6) | — | 6.593 | 7.479 |
| Target layers [6, 36), including projection/argmax | — | 36.002 | 32.556 |
| All target attention-metadata planning | 0.944 | 1.840 | 1.836 |
| L6 probe + feature construction | — | <0.001 | 0.616 |
| Suffix compaction + second packing | — | 0.417 | 0.395 |
| Acceptance / commit indices | 0.299 | 0.286 | 0.275 |
| Draft KV upkeep | 0.536 | 0.539 | 0.508 |

Target computation falls 42.573→40.034 ms, saving 2.538 ms, while the extra probe,
compaction and planning consume much of that. The row-layer proxy falls about
15.8% (7.859→6.616 equivalent rows), but measured target computation only falls
about 5.96%. Do not convert proportional token work into proportional latency.

The confidence path also has substantial cost: B16 projection/argmax without
the three statistics costs 5.648 ms, versus 13.870 ms with them (roughly 8.22 ms extra
in this implementation). That cost is common to BOTH learned policies; it is
not an L6-only overhead. This is a PyTorch statistics implementation, not a
fused/optimized confidence kernel or a lower bound on the necessary cost.

Native suffix reserve/free microcheck, reported separately: C128 median 0.0165 ms,
mean 0.0248 ms over 20 calls. This does NOT include scheduler commit/remapping or
represent their full costs. Prefix/suffix slots are otherwise preallocated in
the timed snapshot replay.

## Actual fixed-width controls

Fresh native candidates for BOTH widths, same prefixes, same backend/mode.
These differ from the saved-candidate predictor comparison and must be labeled.

| Actual C | Mode | Actual draft B | Cycle ms | ms/committed token |
| ---: | --- | ---: | ---: | ---: |
| 64 | graph | 16 | 46.938 | 0.16020 |
| 64 | graph | 8 | 31.318 | **0.13557** |
| 128 | graph | 16 | 84.765 | 0.14465 |
| 128 | graph | 8 | 53.909 | **0.11927** |

Actual B8 is about 1.18×/1.21× the actual B16 replay rate at C64/C128. The cascade
is about 1.6% slower than actual B8 at C64 and only 0.29% faster at C128. Thus a
large-looking comparison against fixed B16 alone is not evidence of a useful
new L6 contribution. Only B8/B16 were screened; this is not an exhaustive
best-fixed-width claim, nor a matched-retention B8 comparison.

Fresh B16 accepts 458 proposals versus 452 for saved B16 in graph mode. Therefore
the fresh-control accepted-total ratio relative to saved B16 can exceed 100%;
the plot's 101.3% is not probability or retention of an unchanged candidate set.

## Correctness, numerical transfer and boundaries

- Explicit native-layer traversal matches the engine's ordinary eager model
  dispatch exactly in auxiliary hidden features and argmax, all tested cells.
- The no-prune split matches the unsplit target-free path **bitwise within
  each mode**, all C4/C64/C128 batches. Both smoke suites had zero changed
  acceptance/bonus decisions against their independent shorter-prefix checks.
- At high concurrency, pruning/graph comparisons are not bitwise exact. At C128
  graph, cascade versus independent eager verification at the same final lengths
  has 7 top-1 differences, 4 changed acceptance lengths and 5 changed bonus IDs.
  Fixed saved-B16 graph versus eager has 24/4/6 respectively. These are different
  query sets: neither comparison proves equivalence or excuses the other.
- C128 eager cascade versus an independent all-layer shorter-prefix forward has
  5 top-1 differences, 2 changed acceptance lengths and 2 changed bonus IDs. This
  requires accounting for shape-dependent numerical behavior before deployment.
  Do not call the serving implementation lossless based on this test.
- The 2% hidden-relative-L2 diagnostic guard was not exceeded. Sampled committed
  prefix keys at every layer were unchanged; suffix-KV discrepancies were
  recorded and pool-capacity restoration checks passed. This is not an
  exhaustive correctness proof.
- Native prefill argmax differs from the saved anchor on 6/128 states. Replayed
  B16 draft argmax differs on 74/1,920 candidate positions. Prefixes/candidates
  remain fixed for the main matched comparison; they are not certified native
  greedy trajectories.
- Frozen cached-policy versus native-runtime selected lengths differ on 14/128
  states for target-free and 9/128 for graph cascade. Cached-cohort retentions
  were 98.455% and 98.896%; runtime label/shape/policy drift must be measured,
  not assumed away. The 1,416-row offline assessment table is a different cohort.
- Graphs are exact-shape manually captured native target segments with planning
  outside capture. They are NOT the recovered scheduler's total-token bucket
  implementation. No padding-to-B16 disguises the actual packed work, but graph
  misses, scheduler integration, prefill/drain and HTTP costs are excluded.

## Decision / handoff

The tests support a modest complementary signal, not a strong systems result.
Do not extend L6 training on the same data or scale serving integration simply
because cycle time fell. Keep the inexpensive target-free alternative as the
control. Further investment would need stable retention/correctness in the
actual execution path and a larger net improvement after confidence, probe,
planning and compaction costs. These measurements do not prove the method is
unhelpful on other workloads, targets, hardware or larger training datasets.

No additional work has been launched after this bounded screen.

## Full paths

Remote final run:
`/data/scratch/zekaili/atharv/dflash/runs/midverify_latency_v2_20260929`

Use `analysis_audited/`, not the earlier automatically generated `analysis/`,
for corrected front accounting and the conditional intervals. Raw observations
are identical in both analyses.

- [Final comparison plot](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_latency_v2_20260929/analysis_audited/latency_comparison.png)
- [Audited report](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_latency_v2_20260929/analysis_audited/report.md)
- [Aggregated metrics and intervals](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_latency_v2_20260929/analysis_audited/summary.json)
- [Frozen/native policy comparison](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_latency_v2_20260929/analysis_audited/frozen_vs_native.json)
- [Raw phase timings and correctness audits](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_latency_v2_20260929/summary.json)
