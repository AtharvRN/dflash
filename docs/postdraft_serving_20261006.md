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
