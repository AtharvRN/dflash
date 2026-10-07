# Post-draft trimming: actual SGLang serving (2026-10-06)

## Scope

Implement and measure full successive-cycle serving, not saved-state replay.
The earlier C128 raw-confidence 10,010 tokens/s was replay-derived throughput;
it did not include complete requests, prefill or the real scheduler/allocator.
Do not label that number end-to-end serving throughput.

## Implementation

`scripts/prepare_postdraft_serving.py` first verifies/restores the historical
`vendor/sglang_ragged_20260723` package, then applies a separate extension. The
historical archive and overlay remain unchanged. Each changed runtime file has
before/after hashes in `POSTDRAFT_EXTENSION.json`.

New CLI: `--speculative-dflash-postdraft-logprob-threshold`.
Policy: `dflash/serving_raw_confidence.py`, copied into the serving source tree.
No learned checkpoint is required for this raw-confidence control. This is NOT
a reproduction of DSpark or AdaFlash.

1. Execute the original full B16 drafter, including one anchor and 15 proposals.
2. Reuse the same LM-head projection used for greedy draft sampling. Compute
   each proposal's log probability in FP32; do not compute unused entropy/top-k
   features or perform another projection.
3. Keep the consecutive prefix whose individual proposal log probabilities
   exceed the frozen threshold `-0.8720796704292296`. This is NOT a cumulative
   probability threshold. No threshold is fitted on serving results.
4. Verify the anchor plus that prefix, using the existing packed ragged worker.
   Per-request verification lengths are arbitrary integers 1 through 16.
5. Fully verify every committed token, generate the bonus, materialize committed
   target features in drafter KV, and continue the request through normal
   SGLang spec-v2 scheduling and allocation.

Greedy/TP1/FlashInfer only. No extra target forward is added. CUDA graphs capture
integer total-token buckets; the draft stays full B16. The historical serving
worker disables draft CUDA graphs for every speculative arm.

## Workload and measurement

`scripts/benchmark_postdraft_serving.py` uses original user prompts from the fixed
canonical validation membership, shuffled with seed 934, thinking off. Prompts
over 2,048 tokens are excluded, not truncated. Exact duplicate tokenized prompts
are removed. The frozen workload includes disjoint warmup prompts and all hashes.
This is a development workload, NOT a new untouched task-accuracy benchmark.

Planned main comparison: 512 measured requests, 128 disjoint warmup prompts,
512 max output tokens, natural EOS, C128 and C64, two repeats per arm/concurrency.
Controls: fixed B16, actual shorter drafting B12/B8, raw post-draft B16 trimming.
The same prompts/order, sampler, precision, attention backend and graph settings
are used. Prefix caching is disabled. Client runs in the same pod.

Reported throughput = sum of returned completion-token counts divided by time
from submitting the workload until every request completes. Includes prefill,
all consecutive decode cycles, scheduling, allocator/KV work, HTTP and output
detokenization. Input tokenization, server startup, graph capture and warmup are
outside the timer. This is aggregate output tokens/s, not per-request tokens/s.
Request latencies, raw responses/output IDs, GPU identity, revisions, server
arguments and environment are saved. Server speculative acceptance denominators
can still count full draft proposals; do not interpret them as a trimmed-verify
acceptance ratio without a separate counter audit.

## Tests and initial status

- GPU-environment CPU regression suite: 13 tests passed before the later summary
  test was added. Tests cover unchanged recovery hashes and all 136 possible
  (block length, accepted prefix) combinations for lengths 1–16.
- First smoke (`smoke_r1`) stopped before GPU measurement: Transformers 5 returns
  a BatchEncoding by default. Fixed by explicit `return_dict=False` with type
  validation. Preserve failed artifacts.
- `smoke_r2`: eager C4, 16 requests × up to 128 output tokens, four warmup prompts.
  Target-only and fixed16 completed. Full-width post-draft (`threshold=-1000`)
  matched fixed16 output token IDs exactly: 16/16 requests, 2,048 tokens.
- Fixed16 versus target-only matched 5/16 entire outputs at C4. This comparison
  changes batching/query shapes; it is NOT proof of a new trimming bug or proof
  of numerical exactness. Retain actual outputs for analysis. Do not claim task
  accuracy or universal token-level parity from this smoke.
- Trimming smoke completed: 480.13 tokens/s versus fixed16 574.30 at eager C4.
  This tiny smoke is not the high-concurrency performance comparison.

Remote root: `/workspace/dflashv2_data/runs/postdraft_serving_20261006`.
Pod: `dflash-a100-gpu-test`, UID `f082c30f-5097-4f14-b03a-673950da323e`.
GPU: A100-SXM4-80GB, UUID `GPU-c7c79d6a-ab60-23ac-9d6b-5a5ef2aa6cf0`.
Runtime: `/tmp/predraft-latency-env/bin/python` (Torch 2.11.0 + cu130).
All result/log files are durable on the PVC; temporary restored source is not.

## Completed actual-serving results

Run: `main_r1`, benchmark commit `c0dc23c`. All 16 timed runs completed:
four methods × two concurrency levels × two repeats, 512 requests per run.
No serving request failures or KV retractions were observed. Each completion-token
count was checked against its returned `output_ids` length. The server and bounded
GPU telemetry process stopped after completion; the GPU was released.

Aggregate output tokens/s (arithmetic mean of two whole-workload repeats):

| Method | C64 | C128 |
| --- | ---: | ---: |
| Fixed B16 | 3,772.88 | 4,163.63 |
| Fixed B12, actual shorter drafting | 3,899.78 | 4,494.12 |
| Fixed B8, actual shorter drafting | 4,659.65 | 5,552.43 |
| B16 draft + raw-confidence post-draft trim | 4,522.25 | 5,550.29 |

Post-draft versus B16: **1.1986× at C64; 1.3330× at C128**.
Post-draft versus best tested fixed width (B8): **0.9705× at C64;
0.9996× at C128**. Thus no demonstrated win over the strongest tested fixed
control. Do not frame the B16-relative gain as a new adaptive-policy advantage.

Raw-confidence repeat ranges: C64 4,501.76–4,542.74; C128 5,500.41–5,600.17.
B8 ranges: C64 4,615.82–4,703.47; C128 5,535.05–5,569.81. Only two repeats,
sequential method order, one GPU/model/workload; no statistical-significance or
universal-speedup claim. B8 is best among B8/B12/B16, not an exhaustive optimum.

The 512 measured prompts have mean input length 195.14 tokens; source counts:
315 nemotron, 103 opencodeinstruct, 60 openr1_math, 34 evol_codealpaca.
Each timed run produced approximately 213k output tokens, with natural stops.
This differs from the earlier saved-state replay prefix distribution and timing
boundary; the absolute replay TPS and serving TPS are not directly comparable.

Five-second GPU-utilization samples restricted to timed request windows:
raw C128 mean 84.0% (15 samples); raw C64 75.68% (19 samples). Thus the earlier
80% utilization preference was met on average at C128, not at C64. Other arms'
means were 81.89–91.71%. Startup/capture/warmup are excluded from these averages.

### Output stability caveat

These are real performance measurements, not a quality-equivalence certification.
Fixed16 repeat-0 versus repeat-1 output token IDs matched for 302/512 requests at
C128 and 387/512 at C64. Raw trimming repeat agreement was 107/512 and 109/512,
respectively. Raw versus fixed16 matched only 99–126/512 at C128 and 96–106/512
at C64. The adaptive path is substantially less repeat-stable in this setup.
Different BF16 batching/query shapes are a possible contributor, but this run
does not prove they explain every difference. Do not dismiss a correctness issue
or claim unchanged task quality without a separate audit. The no-trim C4 control
matching 16/16 is encouraging but does not certify high-concurrency trimming.

### Evidence and reproduction

Local full evidence:
`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/postdraft_serving_20261006/main_r1`.
Durable remote equivalent:
`/workspace/dflashv2_data/runs/postdraft_serving_20261006/main_r1`.
GPU telemetry is `main_r1.gpu.csv` alongside that directory. Small smoke evidence
is under the sibling `smoke_r2` directory in both locations.

SHA256 of the canonical sorted JSON mapping of filename → SHA256 for
`config.json`, `workload.json`, `COMPLETE.json` and all 16 timed response files:
`71f07c381040b7574440186ded7cb4d9b3f3067cf57e859640da70e00b0d7b77`.

Use `scripts/run_postdraft_serving_nrp.sh` with `DFLASH_SERVING_OUTPUT` and
`DFLASH_SERVING_SCRATCH` pointing to new destinations. It records telemetry only
for the run. `scripts/summarize_postdraft_serving.py` recomputes measured TPS,
checks output counts, and compares against both B16 and the best tested fixed
control. The later CLI validation guards reject unsupported spec-v1, TP>1 or
non-FlashInfer configurations; those guards do not change this run's policy or
worker execution. Historical recovered engine files remain unchanged.

Final regression suite on the remote Torch environment: **16 passed** (4.68 s).
Includes exact greedy-candidate/logprob preservation by the optimized confidence
projection and fail-fast checks for unsupported CLI configurations. Local/PVC
19-file aggregate evidence digests match exactly. All 8,192 timed responses had
valid stop/length finishes; total KV retractions were zero. The pre-existing edit
to `docs/DFLASH_RESEARCH_MASTER_RECORD_20260928.md` was left untouched.

## Verification-length instrumentation rerun (2026-10-07 UTC)

The original serving run did not record selected verification lengths. These
are **fresh instrumented measurements**, not reconstructed statistics of that
run and not a new matched throughput comparison.

Same frozen 512-prompt workload, 128 disjoint warmup prompts, two repeats per
concurrency, natural EOS, 512 output-token cap, same model revisions and raw
logprob threshold. Exact workload SHA256:
`023fccdeb19c250f4413d3cbf580e9b4d4a2481deba9c0e22caaa8ffd72d869c`.
Only raw-confidence adaptive verification was rerun; drafting remains full B16.

The replacement pod UID is `1e9b0dc5-f87b-4876-9a17-30488948eeba`, with an
**A100-PCIE-40GB**, UUID `GPU-e9fb957c-4f38-6389-7886-c99a9370d6f9`.
The original benchmark used an A100-SXM4-80GB. The new run uses static memory
fraction 0.60 and a 98,304-token capacity rather than the old capacity; no KV
retractions were observed. Instrumented timing must not replace the original
uninstrumented throughput. Output-equivalence caveats above remain unresolved.

| Pooled metric across two repeats | C64 | C128 |
| --- | ---: | ---: |
| Counted request-cycles | 107,141 | 106,395 |
| Selected verification positions | 565,295 | 563,361 |
| Accepted draft proposals | 322,085 | 320,767 |
| **Mean selected verification length, anchor included** | **5.276178** | **5.294995** |
| Mean selected proposals, anchor excluded | 4.276178 | 4.294995 |
| Mean accepted draft proposals | 3.006179 | 3.014869 |
| Accepted / selected proposals | 70.3006% | 70.1949% |
| Packed rows / counted cycle | 5.813321 | 5.877974 |
| Executed graph rows / counted cycle | 5.990050 | 6.293172 |

The selected-length mean excludes warmup, finished/retracted overlap results,
and graph padding. Physical row totals include padding and discarded overlap
work, so the last two rows are work per counted cycle, not mean policy choices.
All observed verification batches replayed CUDA graphs and none mixed phases.
Fixed B8/B12/B16 select 8/12/16 verification positions by definition; this new
run did not remeasure those controls. The approximately 70% ratio divides by
*selected verification proposals*, not the 15 proposals still generated by each
full B16 draft. It is not matched-state retention or a new predictor result.

Instrumentation adds optional `GenerationBatchResult` fields, transfers the
selected lengths with the existing asynchronous result copy, and logs after
the normal synchronization. `scripts/summarize_verify_lengths.py` asserts exact
per-request agreement with `spec_verify_ct` and `spec_num_correct_drafts` for
all 2,048 timed responses. All assertions passed. The no-trim graph-enabled C4
smoke measured exactly 16 positions on all 247 counted cycles; adaptive smoke
measured 5.172161 positions on 273 cycles (not a representative benchmark).
The expanded regression suite passed **18 tests** in the remote pinned runtime.

Benchmark code: `1266bc0` (instrumentation introduced in `1a88aec`).
Runtime setup: `scripts/setup_postdraft_nrp_runtime.sh`; Torch 2.11.0+cu130 and
kernel 0.4.3+cu130. The replacement environment used NumPy 2.3.5 rather than the
original 2.5.2. PVC checkpoint reads stalled startup; exact-revision public
snapshots were downloaded to `/tmp/postdraft-hf-cache`. All four weight SHA256
hashes matched the original PVC blob hashes. No model/threshold changes.
Earlier `smoke_r1`/`smoke_r2` attempts were stopped before measurement because
of slow PVC reads; preserve their evidence. Successful smoke is `smoke_r3`.

Durable full evidence (including raw responses):
`/workspace/dflashv2_data/runs/postdraft_verify_lengths_20261007/main_r1`.
Local aggregate-only evidence:
`outputs/postdraft_verify_lengths_20261007/verify_length_summary.json`.
The local export of raw prompts/outputs was blocked by approval review and was
not performed. Raw evidence remains on the PVC. The server and bounded telemetry
were stopped after completion; no GPU process remained.

### C32/C16 extension (same pod, 2026-10-07 UTC)

Completed `c32_c16_r1`: 512 prompts, 128 disjoint warmup prompts, two repeats at
each concurrency, unchanged raw-confidence threshold and full B16 drafting.
Compared configuration records directly: identical workload hash, models,
environment, code commit (`1266bc0`), GPU and memory settings to `main_r1` above.
Only output/scratch paths and concurrency arguments differed. The server's
maximum request count and largest captured request-count bucket were 32
(rather than 128 in the C64/C128 run); integer length buckets remain 1–16.

| Pooled metric across two repeats | C16 | C32 |
| --- | ---: | ---: |
| Counted request-cycles | 106,585 | 106,881 |
| Selected verification positions | 567,331 | 566,051 |
| Accepted draft proposals | 323,173 | 322,310 |
| **Mean selected verification length, anchor included** | **5.322803** | **5.296086** |
| Mean selected proposals, anchor excluded | 4.322803 | 4.296086 |
| Mean accepted draft proposals | 3.032068 | 3.015597 |
| Accepted / selected proposals | 70.1412% | 70.1940% |
| Packed rows / counted cycle | 5.853310 | 5.846876 |
| Executed graph rows / counted cycle | 5.927213 | 6.259840 |

All 2,048 responses passed exact per-request cycle/accepted-count checks.
All recorded verification batches used CUDA graphs; no mixed-phase batches,
KV retractions, or runtime assertions/errors were found. Selected averages
exclude warmup and graph padding; physical totals include discarded overlap
work as in the previous audit. No new fixed-width controls were run, and
instrumented HTTP timings are not a matched speedup result. Differences of
this size between concurrency levels do not establish a systematic trend.

Full evidence remains at:
`/workspace/dflashv2_data/runs/postdraft_verify_lengths_20261007/c32_c16_r1`.
Local aggregate-only evidence:
`outputs/postdraft_verify_lengths_20261007/c32_c16_verify_length_summary.json`.
The server exited and run-scoped telemetry was stopped; the GPU was released.

### Matched C32/C16 fixed-width throughput sweep launched

At 2026-10-07 01:24 UTC, launched a separate **uninstrumented** sweep on the
same idle A100-PCIE-40GB and pod UID as the length audit. Cases, in order:
`fixed8 fixed12 fixed16 raw`; concurrencies `32 16`; two repeats each,
512 measured prompts, 128 disjoint warmup prompts, 512 max output tokens with
natural EOS. All cases use the same frozen workload hash, pinned models,
runtime/code (`1266bc0`), FlashInfer, CUDA graphs, static memory fraction 0.60,
and 98,304-token capacity. No verification-length audit logging is enabled.
The fresh adaptive arm makes the throughput comparison instrumentation-matched;
do not use the earlier instrumented adaptive timing as its speed baseline.

Launch wrapper: `scripts/run_postdraft_serving_nrp.sh`, bounded to 5,400 seconds,
with run-scoped GPU telemetry, automatic server cleanup, and final throughput
summary. This is not a recurring monitor. Launch PID: 3017.
Durable run:
`/workspace/dflashv2_data/runs/postdraft_serving_20261007/c32_c16_r1`.
Launch log and GPU CSV are adjacent (`c32_c16_r1.launch.log`,
`c32_c16_r1.gpu.csv`). Scratch: `/tmp/postdraft-serving-c32-c16-r1`.
At 01:26 UTC the first fixed-B8 C32 workload was actively serving, with 88%
GPU utilization sampled; no completed timed result yet. Read `progress.json`,
`COMPLETE.json`/`FAILED.json`, and `summary.json`
before reporting completion or a speedup. Raw prompts/outputs remain on PVC.
