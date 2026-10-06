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
- Trimming smoke and high-concurrency serving measurements are pending as of
  this initial note. No actual-serving speedup is claimed yet.

Remote root: `/workspace/dflashv2_data/runs/postdraft_serving_20261006`.
Pod: `dflash-a100-gpu-test`, UID `f082c30f-5097-4f14-b03a-673950da323e`.
GPU: A100-SXM4-80GB, UUID `GPU-c7c79d6a-ab60-23ac-9d6b-5a5ef2aa6cf0`.
Runtime: `/tmp/predraft-latency-env/bin/python` (Torch 2.11.0 + cu130).
All result/log files are durable on the PVC; temporary restored source is not.
