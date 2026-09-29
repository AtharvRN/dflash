# Recovered ragged DFlash: severe-discrepancy follow-up

2026-09-28, continuing [the recovery audit](ragged_recovery_20260928.md).

## Scope and current interpretation

The recovered runtime remains unchanged. All 16 original overlay source hashes
were rechecked. Only bounded test harnesses, regression tests and documentation
have changed. GPU 4 is the designated workstation RTX PRO 6000 Blackwell, not an
A100. No training, collection sweep or recurring monitor was launched.

A preserved severe hidden-state discrepancy can be reproduced **without mixed
lengths or ragged verification metadata** in the same runtime. This is evidence
against attributing that case to ragged packing. It is not a guarantee of exact
greedy transcripts. One preserved emitted-token discrepancy also reproduces
under fixed-width graph execution, as detailed below; other changed states in
the audits have not all received same-state controls.

The measured hidden states here are the concatenated five selected **target
layer outputs, width 12,800**, not the drafter's normalized fused 2,560-vector.
Relative L2 below uses the independent/single-request hidden norm as denominator.
It is not a percentage of wrong tokens or a task-accuracy loss.

## Capturing a comparable severe recurrence

Both runs request concurrency 64, 128 development prompts in two orders, a
256-token cap, and ignore EOS for stress only. They audit up to 32 target forwards
with actual batch size >=48, using the rotating B2–16 pattern. Draft shadows are
disabled to focus on the target. Each target request is independently verified
at its actual B on the same committed KV prefix.

| Run | Audited target forwards | Matched request-cycles | Maximum hidden relative L2 | Verifier top-1 changes | Emitted-decision changes vs independent | Emitted-decision changes vs same-batch eager |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ragged_extreme_capture_20260928` | 32 | 2,015 | 0.470654 | 103 / 18,286 | 26 / 2,015 | 29 / 2,015 |
| `ragged_extreme_preserved_20260928` | 29 | 1,827 | 0.850557 | 113 / 16,402 | 26 / 1,827 | 23 / 1,827 |

These are different trajectories/schedules, not a matched policy comparison.
The first run completed 256 requests; the second stopped deliberately after
saving the severe state, not from OOM. Investigation mode logs the normal 5%
hidden gate but continues to a 50% capture/stop threshold. It is **not** a relaxed
correctness pass. Up to four record-breaking >10% near misses can also be saved.

Every audited forward in both runs had unique output KV slots, no output overlap
with any live committed prefix, bitwise worst-row isolation under changes to
other requests, repeatable independent execution, and bitwise packed restoration.
These are bounded checks, not a proof for arbitrary schedules.

The preserved severe state is target forward 29, row 10, actual B10, committed
prefix length 1,491. The graph has 63 real requests, 567 physical input tokens,
and a C64 x 9 capture: **576 executed token rows**. Matching only the 567 physical
rows would use the wrong linear-operation shape.

At query position 8 and concatenated channel 5,124 (target layer 17, channel 4),
the saved packed output is 14.75 versus independent 1,456.0. Relative error is
small in the first two captured layer groups, then increases sharply in the
third group. All ten verifier argmax tokens nevertheless agree for this state;
both decisions are A=9, bonus token 10343.

Remote preserved state:

`/data/scratch/zekaili/atharv/dflash/runs/ragged_extreme_preserved_20260928/graph/discrepancy_target_29_585.pt`

SHA256: `67a824832e85515a1aeb8ef8b9002295171d7d8f53c835d334f7f8dec053bfc9`.

This is a comparable recurrence, **not** a reconstruction of the earlier unsaved
B15/prefix-409 case with 0.9082 relative L2.

## Exact saved-state replay in the recovered runtime

`ragged_extreme_replay_20260928`, code `5ed4f23`, same image, FlashInfer backend,
weights and BF16 settings. The full 36-layer committed prefix is loaded into new
slots. Replicas share that immutable prefix and use disjoint suffix slots.
References use the same ten real input tokens. B12/B16 append causal target-only
dummy queries; they do **not** rerun drafting or create new candidates.

| Replay | Real-prefix hidden result |
| --- | --- |
| Fixed C1/B10 | Bitwise equal to saved independent output |
| Fixed C1/B12 | Bitwise equal to C1/B10 |
| Fixed C48/B12, 576 query rows | Bitwise equal to saved same-batch eager output; relative L2 0.850749 vs C1 |
| Ragged metadata, uniform C48/B12 | Hidden and logits bitwise equal to fixed C48/B12 |
| Fixed C64/B16 | Relative L2 0.136535 vs C1 |
| Original physical length layout, eager | Bitwise equal to saved same-batch eager output |
| Original physical length layout, graph | Hidden and logits bitwise equal to saved original graph output; relative L2 0.850560 vs C1 |

All ten top-1 predictions and A=9/bonus=10343 agree in every replay. The committed
prefix remains bitwise unchanged after all cases; request/KV availability returns
exactly to 64 / 131,072 slots. The graph control confirms actual graph replay.

The original-layout control preserves the protected row index and all physical
query lengths, but repeats the protected prefix for other requests; it is not a
snapshot of the entire original batch. Fixed eager versus graph still differs
by approximately 0.0218 relative L2 (using the packed/eager norm), so reproducing
the large independent gap does not identify the exact arithmetic kernel.

Local report:

`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/ragged_extreme_replay_20260928/graph/FIXTURE.json`

## Precision control outside SGLang

`ragged_extreme_precision_20260928`, code `e0ab25e`, uses Transformers 4.57.1,
Torch 2.13.0+cu130, dense SDPA, TF32 off. It holds the saved BF16 prefix fixed and
upcasts both the pinned BF16 weights and that prefix for FP32. It does not
recompute a full FP32 prefill. The comparison is C1/B10 versus C48/B12, scoring
only the ten real query positions. Every replica has bitwise-identical logits
within its batch.

| Dtype | Dense batched-vs-single hidden relative L2 | Maximum absolute hidden difference | Top-1 changes |
| --- | ---: | ---: | ---: |
| BF16 | 0.00825435 | 32.0 | 0 / 10 |
| FP32 | 0.00000156202 | 0.00154114 | 0 / 10 |

This supports numerical batch-shape sensitivity within the dense control. It
does **not** demonstrate that FP32 fixes the SGLang 85.1% gap: FP32 dense hidden
states still differ from saved SGLang independent/packed states by 1.50043/4.91371
relative L2, respectively. Cross-engine/backend/fused-operation differences are
not isolated. All ten argmax tokens and A=9/bonus=10343 still agree across them.

Local report:

`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/ragged_extreme_precision_20260928/summary.json`

## Emitted-token case: reproduced in fixed-width graph execution

`ragged_emission_capture_20260928`, code `759e1b5`, stops after preserving an
actual accepted-prefix/bonus mismatch, even when hidden relative L2 is small.
Its first C63 target forward has 2/63 changed emitted decisions and 6/671 changed
real-position argmax tokens versus independent execution. It saves row 31,
B9, prefix length 793, with 693 physical input rows and a C64 x 11 graph capture
(704 executed rows). The saved graph decision is A=5/bonus=525, versus independent
and same-batch eager A=0/bonus=8253. Hidden relative L2 is only 0.007722.

The test protects the selected row in the isolation check and restores the
original packed execution before its deliberate stop. Slot overlap, isolation,
independent repeat and packed restoration checks all pass. A CPU regression
verifies snapshot row selection and restoration.

`ragged_emission_replay_20260928`, code `b3cbaa0`, adds a **fixed-width graph**
control when the exact matching shape is captured, in addition to eager controls:

| Replay | A | Bonus | Exact correspondence |
| --- | ---: | ---: | --- |
| Fixed C1/B9 or C1/B11, eager | 0 | 8253 | Saved independent hidden/logits |
| Fixed C64/B11, eager | 0 | 8253 | Saved same-batch eager hidden/logits |
| Ragged metadata, uniform C64/B11, eager | 0 | 8253 | Fixed C64/B11 eager |
| Fixed C64/B16, eager | 0 | 8253 | Same decisions, not bitwise hidden equality |
| Original physical layout, eager | 0 | 8253 | Saved same-batch eager hidden/logits |
| Fixed C64/B11, graph | 5 | 525 | Saved original ragged graph hidden/logits |
| Original physical layout, graph | 5 | 525 | Saved original ragged graph hidden/logits |

Only the first argmax differs. Its two competing logits are:

| Execution | Token 8253 | Token 11625 | Argmax |
| --- | ---: | ---: | ---: |
| Independent / same-batch eager | 33.25 | 33.25 | 8253 |
| Original graph | 33.0 | 33.25 | 11625 |

The actual verifier uses `argmax`, selecting the smaller token index at the
eager tie; the difference is not the earlier harness's `topk`/`argmax` mismatch.
The first proposed draft token is 11625, so this one decision changes the
accepted prefix from zero tokens to five. The fixed graph control reproduces
the entire saved real-prefix hidden/logit tensors, not merely this decision.

This case is **not specific to mixed-length packing**. The same recovered
graph engine exhibits it under fixed-width replay. Its exact arithmetic-kernel
cause has not been isolated, and it does not explain every changed state in the
population. Both hidden-state and token-decision tests are necessary: the severe
hidden case had no token change, whereas this small hidden difference did.

Both the committed prefix and allocator counts are unchanged after replay.
The original-layout controls still repeat one preserved prefix for other rows;
they do not reconstruct the entire original batch.

Remote state:

`/data/scratch/zekaili/atharv/dflash/runs/ragged_emission_capture_20260928/graph/discrepancy_target_1_584.pt`

Local replay:

`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/ragged_emission_replay_20260928/graph/FIXTURE.json`

## Sign-off boundary and current execution status

There is no demonstrated ragged-specific packing/indexing defect in these
preserved cases, and bounded primitive/slot/isolation checks pass. Exact
graph/eager/independent greedy parity does **not** pass, including fixed-width
controls. Do not call the implementation unconditionally lossless based on
these results. Full target-only transcript comparisons, longer allocator stress,
other page sizes/TP settings and arbitrary stochastic sampling remain uncertified.

Next: establish the numerical comparison contract and retain fixed-width
same-state controls while moving toward same-engine timing. Any exploratory
profile must be labeled diagnostic until output-level validation is adequate.
No inference that current raw-hidden discrepancies leave future draft candidates
unchanged is warranted; the drafter consumes target-derived features.

All bounded GPU diagnostics in this follow-up have finished. At 2026-09-29
01:26 UTC (2026-09-28 Pacific), GPU 4 reported 0 MiB / 0% utilization and only
the unrelated existing container remained. No task server or monitor is running.

Ten local recovery/profiling-helper tests pass. The default 5% hidden gate remains
unchanged; neither investigation mode is reported as a parity pass. No clean
recovered-runtime timing run has been performed. These shadow-forward runs must
not be interpreted as latency or throughput measurements.
