# Frozen history-priced policy: actual serving measurement

## Question

Do the small matched-state gains over the calibration-selected fixed block
survive actual request-level serving, after entropy, lookup, ragged execution,
KV management, prefill, and scheduling costs?

The earlier 1,319-question / 61,568-state assessment was **offline replay**.
Its best-fixed comparisons were cost-model choices, not measured end-to-end
throughput rankings. The instrumented fixed-B HTTP numbers from that run are
diagnostics, not clean speedup evidence.

## Implementation

- Frozen table and C-specific rho from `history_priced_gsm8k_20261008_r2_finish`.
  No training, test-set tuning, or retention constraint added.
- Inputs: last two completed-cycle mean verifier entropies, previous B,
  previous full-accept flag, history count. Cold start uses frozen cold/global
  backoff. The policy is **pre-draft**, not the older raw-confidence post-draft
  threshold.
- Select B from {4,8,12,16} by the exact frozen `V_B - rho*T_C(B)` score.
  Compile the finite histogram action table once using Python float/tie
  semantics. GPU lookup uses the original certainty-bin edges.
- State resides on GPU indexed by request-pool slot, reset at first prefill.
  Radix cache is disabled; greedy TP1 only. State updates on actual chosen
  blocks, unlike the fixed-B16 replay trajectory used in calibration.
- Mean entropy uses target logit rows 0..A inclusive: accepted proposal
  distributions plus correction/bonus. Never includes rows conditioned on
  the first rejected token. A chunked Triton FP32 reduction avoids allocating
  full FP32 probability/log-probability matrices. History averaging uses FP64.
- Existing uniform worker path for uniform decisions; existing packed ragged
  draft and verification paths for mixed decisions. Target CUDA graph buckets
  4/8/12/16, draft eager, BF16, FlashInfer. Suffix candidates are actually
  redrafted at chosen B, not full-B16 proposals post-hoc clipped.
- Historical vendored sources remain immutable. `prepare_history_serving.py`
  applies anchored, SHA256-recorded extensions to a fresh restored runtime.

## Validation and measurements

CPU tests compare GPU-style lookup to the original Python selector across all
history counts, arms, flags, certainty bins and boundaries. Additional tests
cover cold start, slot reuse, malformed artifact hashes, packed-row entropy
alignment and rejected-prefix exclusion. The A100 executes the fused entropy
test against a PyTorch reference.

Serving smoke: 8 prompts, max64 output tokens, history-C64 policy, fixed B8,
target-only AR. This is a functional check, not a throughput conclusion or a
full-concurrency stress result.

Main study: all 1,319 GSM8K test questions, disjoint 128 training-question
warmup, two repeats, greedy natural EOS/max512, same frozen workload and
limits across methods. C64 then C32 then C16/C8 if pod lifetime permits. Each
concurrency compares adaptive against **all** fixed B4/B8/B12/B16 arms.
The frozen policy for nominal C stays fixed during request ramp-up/drain.

Throughput = sum returned completion tokens / client workload wall seconds.
Includes prefill, actual closed-loop decoding, entropy/lookup, packing, KV,
scheduler, detokenization and HTTP client overhead. Input tokenization and
server startup/capture are outside timing. Server logs are node-local and
copied to PVC between phases. No per-cycle profiler or length audit enabled
during throughput measurements. Save every complete phase durably; bounded
deadline leaves partial evidence rather than fabricating completion.

Numeric boxed-answer accuracy and cap fraction are checked using the existing
GSM8K scorer. This is our zero-shot chat protocol, not standard few-shot
leaderboard evaluation. Greedy BF16 batch/graph shape can affect near ties;
small smoke agreement is not a universal bitwise-parity guarantee.

## Artifacts

Cluster PVC (`/workspace`, not local Mac):

- `/workspace/dflashv2_data/runs/history_serving_20261008_smoke`
- Planned main root: `/workspace/dflashv2_data/runs/history_serving_20261008`
- Original frozen workload:
  `/workspace/dflashv2_data/runs/gsm8k_serving_20261007/workload.json`

Results are pending; no adaptive tokens/s claim is made in this note.
