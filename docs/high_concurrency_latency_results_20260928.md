# DFlash latency and attainable-gain diagnostic — 2026-09-28

## Bottom line

There is measurable block-size sensitivity at C64, but the present predictor
does not yet imply a large serving improvement. Fixed B12 is 1.53% faster than
fixed B16 in the clean workload. A deliberately conditional calculation using
the new predictor's offline outcomes gives about 8.3% decode / 5.2% workload
gain before incremental integration cost, falling to 4.8% / 3.1% with 2 ms of
added cost per cycle. These are planning scenarios, NOT measured adaptive
speedups, upper bounds, or a reason to claim a paper result.

The new predictor's standalone graph-replayed compute plus decision costs
0.087 ms at C64. Attention tiling and actual variable-length execution are
more consequential uncertainties than this MLP's compute.

All bounded jobs completed except the initial two-block trace wrapper, whose
B16 trace succeeded before B12 failed at server argument validation. The
missing B12 trace succeeded in a separate retry. GPU 4 was checked afterward:
0 MiB / 0% utilization; no profiling server remains running. No models were
trained, no cluster changes made, and no recurring monitor installed.

## Exact scope

- Workstation: `zekaili@tianhaowang-gpu0.ucsd.edu` (`wth-gpu-01`).
- GPU 4: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB;
  UUID `GPU-2b489243-113f-1e33-ee0b-e4d28423e006`; driver 595.71.05.
- Original Qwen3-4B target revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Original B16-trained DFlash revision `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Torch 2.11.0+cu129, Transformers 5.8.1, SGLang 0.5.13, pinned image
  `sha256:a7317182c71d35712ee4edc86a5d1c313dc969efdf0026d339673299c186ea75`.
- Image SGLang base `28b095c01005d4a3a2a5b637b7d028b07fba31b2`, with image-baked
  Slime modifications. The digest, not the base git SHA alone, pins this runtime.
- BF16, greedy, thinking-off saved tokenized prompts, TP1, Triton attention,
  CUDA graphs enabled, TF32 disabled, radix reuse disabled.
- This is spec-v1, which disables overlap scheduling. It is NOT the historical
  optimized ragged fork and NOT an integrated adaptive-policy benchmark.
- Main harness commit `6b785fd`; trace B16 `2adadb0`; B12 retry and AR `d9b0c16`;
  predictor probe `b1a83a6`. Analysis scripts subsequently improved attribution
  and reporting without changing the completed measurements.
- 187 development prompts, initial prompt lengths 81–1,997 tokens, shuffled
  with seed 930. The saved anchor is removed before serving prefill. C64 uses
  384 requests, cycling through this pool; cap 256 new tokens, EOS allowed.
  C32/C16 use 192/96 requests; C1 uses 8. Comparisons within C share input order.
- Warmup: C requests, cap 96. Three clean repetitions per B/C; separate event
  repetitions. No throughput claim uses profiler-enabled timing.
- Other users' GPUs/jobs were left untouched. GPU clocks/power were not locked;
  before/after telemetry is saved. This is not an isolated-host publication run.

## Clean HTTP throughput

Output tokens/s, mean ± sample SD of three runs. Includes prefill and drain.
Different C values cover different amounts of the workload; do not infer a
precise universal concurrency crossover from this table.

| Concurrency | B16 | B12 | B8 | B12 / B16 |
|---:|---:|---:|---:|---:|
| 1 | 407.94 ± 0.08 | 373.22 ± 0.19 | 320.67 ± 0.01 | 0.9149× |
| 16 | 2,410.26 ± 14.20 | 2,417.84 ± 33.93 | 2,224.25 ± 26.40 | 1.0031× |
| 32 | 2,945.02 ± 5.23 | 2,865.63 ± 4.09 | 2,825.91 ± 4.97 | 0.9730× |
| 64 | 3,327.56 ± 6.64 | 3,378.57 ± 7.68 | 3,349.79 ± 8.81 | 1.0153× |

The C64 target-only AR control is **2,962.77 ± 7.38 tokens/s**. Thus B16 and
B12 are 1.1231× and 1.1403× this AR configuration. AR retains its native overlap
scheduler, unlike spec-v1 DFlash; this is a practical engine comparison, not an
identical-scheduler algorithm-only experiment. These ratios are not adaptive gains.

## Full-batch C64 decode-cycle components

CUDA-stream elapsed milliseconds, not pure kernel duration. No synchronization
is inserted between components. Inclusive parents and children are not added;
repeated upkeep calls are aggregated, with disjoint accounting checked in code.

| Component | B16 | B12 | B8 |
|---|---:|---:|---:|
| Draft setup / allocation | 0.360 | 0.343 | 0.336 |
| Draft transformer forward | 7.096 | 6.247 | 5.600 |
| Draft vocabulary projection / argmax | 2.930 | 2.198 | 1.460 |
| Verification preparation | 0.269 | 0.263 | 0.264 |
| Target verification, including logits | 52.942 | 46.205 | 40.955 |
| Acceptance / target KV commit | 2.860 | 2.532 | 2.285 |
| Draft KV upkeep / committed-feature handling | 1.194 | 1.081 | 1.064 |
| Other work inside worker cycle | 0.080 | 0.078 | 0.077 |
| **Total mean** | **67.730** | **58.948** | **52.042** |
| Cycle SD | 1.431 | 1.467 | 1.436 |
| Cycle p95 | 69.603 | 61.301 | 54.216 |
| Full-batch cycles | 67 | 83 | 102 |
| Mean prefix length | 961.9 | 953.9 | 957.2 |

Both draft and target graph-use fractions are 1.0 for these rows. They are
correlated cycles on different policy trajectories, not paired identical states
or independent statistical replicates. Target verification is 78.17% of B16's
cycle. Draft forward plus projection is 10.026 ms; the remaining in-worker
setup, acceptance, KV upkeep and other work totals 4.763 ms.

Shorter blocks reduce useful progress too. Mean accepted proposed tokens per
request-cycle in the full-batch event samples is 4.984 / 4.425 / 3.702 for
B16/B12/B8; these are not matched-state retention estimates. The larger number
of cycles is why 12.97% cheaper B12 cycles produce only 1.53% higher throughput.

The C64 instrumented wall accounting is:

| B | Client wall (s) | Prefill worker spans (s) | Decode worker spans (s) | Outside-worker residual (s) |
|---:|---:|---:|---:|---:|
| 16 | 28.141 | 9.332 | 18.192 | 0.617 |
| 12 | 27.743 | 9.402 | 17.722 | 0.619 |
| 8 | 27.930 | 9.472 | 17.777 | 0.681 |

The residual includes scheduler/client/serialization gaps and instrumentation;
it is not isolated scheduler CPU time or predictor overhead. Decode is 64.64%
of the B16 measured wall interval. Event-instrumented C64 B16 throughput is
1.05% below its clean mean, and B12 is 0.76% below; keep the clean table authoritative.

## Short CUDA kernel traces

Each warmed trace captured 20 scheduler steps, including prefill and partial
batches. Only **three full-C64 decode cycles per B** are attributed below,
through CUDA runtime AND driver launch correlations, including graph launches.
This is a bottleneck diagnostic, not a precise distribution or CI. Mean prefixes
for those cycles are 856.3 (B16) and 853.4 (B12). Their stage times should not be
substituted for the larger event-run means.

| GPU activity, ms per selected cycle | B16 | B12 |
|---|---:|---:|
| GEMM kernels, all stages | 31.203 | 23.699 |
| Attention `_fwd_kernel`, target + draft | 25.980 | 25.919 |
| Target attention alone | 23.006 | 22.950 |
| Target forward GPU activity, including logits | 50.455 | 43.862 |
| Draft forward GPU activity | 6.450 | 5.634 |
| Draft projection / argmax GPU activity | 2.917 | 2.189 |
| Explicit GPU memcpy activity, all stages | 0.0134 | 0.0118 |
| GPU active-time union inside selected-cycle envelope | 60.820 | 52.580 |
| Gaps inside that envelope | 5.907 | 5.799 |

Rows overlap conceptually (e.g. GEMM is contained in stage totals): **do not sum
this table**. Exact kernel names, stage ownership, runtime/driver calls and all
unattributed/outside-selected-root activity counts are retained in kernel JSON.

The pinned source explains a plausible mechanism for nearly flat attention:
`sglang/srt/layers/attention/triton_ops/extend_attention.py` selects
`BLOCK_M=64, BLOCK_N=128` for SM120/head dimension ≤128, independently of B.
Its grid is `(batch_size, head_num, ceil(max_len_extend/BLOCK_M))`. Both traces
show `_fwd_kernel` grid `(64,32,1)` and block `(256,1,1)` for all 123 selected
attention launches. B12 and B16 occupy the same query tile. The observed plateau
is consistent with that geometry; a controlled tile/backend intervention would
be needed to establish the exact causal saving available. This is not a claim
that attention cannot scale with draft length in other implementations.

Host `cudaMemcpyAsync` duration is about 55.63 ms at B16, despite ~0.013 ms of
explicit GPU memcpy. The long host call waits for preceding GPU work. Likewise,
the event-run host acceptance/commit call is 63.30 ms, whereas its stream span
is 2.86 ms. Adding host-call time to GPU-stage time would double-count computation.

## Predictor and decision cost, separately measured

Latest fused vectors already on GPU, FP16 input cast to FP32; FP32 trained
weights; TF32 off. Ten samples of 100 calls after warmup, synchronized wall
time per invocation. Graph/eager policy outputs agree. No feature extraction,
gather, packing, graph-bucket padding or KV remapping is included.

| C64 model | Eager (ms) | Graph (ms) | Graph + CPU length list (ms) |
|---|---:|---:|---:|
| Exact frozen 100k survival MLP | 0.23145 | 0.09846 | 0.11319 |
| Actual-block MLP, seed 913 | 0.14540 | 0.08665 | 0.10088 |

Seed 913 was selected in advance for timing, not by assessment performance.
Full C1/16/32/64/128 results, sample SDs and checkpoint hashes are in
`highc_predictor_cost_20260928/predictor_cost.json`.

## Conditional gain calculation

For nonterminal cycles, advancement is A+1, not A. The scenario uses

`decode_speedup = ((mean_A_policy+1)/(mean_A_B16+1)) * T16 / (T(mean_B)+added_ms)`.

`T(mean_B)` linearly interpolates the measured **uniform-B** C64 costs. This
assumes mixed batches are equally cheap, unchanged graph padding, representative
offline states, and retention transferring to the policy's own trajectories.
None is established. The offline sample contains eight progress-spaced states
per prompt, not all serving cycles; its B16 mean A=6.578 differs from the serving
sample. Treat all the following as sensitivity calculations, not predictions.

| Offline policy / scenario | Mean B | Accepted retention | Decode gain, 0 added ms | Workload gain, 0 added ms | Workload gain, 2 added ms |
|---|---:|---:|---:|---:|---:|
| Frozen 100k MLP | 12.708 | 96.01% | 8.06% | 5.07% | 2.93% |
| Actual-block MLP, three-seed mean | 12.478 | 95.32% | 8.30% | 5.21% | 3.05% |
| Hypothetical 70% ratio at 96% retention | 10.022 | 96.00% | 17.73% | 10.79% | 8.41% |

The new model's lower retention means these are not matched-retention predictor
rankings. The 70% point has NOT been achieved. Workload estimates additionally
hold prefill/outside-worker time fixed and apply full-C64 savings to all decode.

For the current actual-block scenario, 1 / 2 / 4 ms added per cycle leaves
6.52% / 4.80% / 1.53% decode gain. About **4.98 ms** eliminates the entire
conditional gain; to preserve 5% decode gain, added cost must be below
**1.88 ms**. These are engineering budgets under stated assumptions, not measured
ragged overhead. Existing baseline upkeep is already in T(B); do not add it again.

## Correctness, provenance and reporting limits

At C64, B16 repeat 1 and repeat 2 each differ from B16 repeat 0 on 138/384 output
sequences. B12 repeat 0 differs from B16 repeat 0 on 196/384. B8 differs on
193/384. Dynamic scheduling and numerical effects are possible explanations,
but the cause has not been isolated. No bitwise losslessness claim is supported
by this serving run. Prompt order/caps are matched; realized outputs are not.
Small C4 clean/event smoke outputs did agree exactly. A deterministic replay
correctness gate is needed before publication-level engine comparisons.

The initially failed B12 trace launch used an HTTP port whose derived gRPC port
exceeded 65535. This was fixed by choosing HTTP ports in 18000–29999. No OOM,
model failure, lost collection data or change to the main completed measurements
was observed. The failed run and successful retry are both preserved.

Eight focused analysis/accounting tests pass. Neither the serving backend nor
the predictor was optimized during this study. SGLang/speculative-decoding
skill guidance informed separate clean throughput, event and kernel boundaries;
generic speedup claims from those guides were not used as empirical evidence.

## Recommended next gate — not launched

1. Reproduce deterministic greedy outputs and recover a clean, correct intended
   serving implementation. Do not present this cached spec-v1/Triton setup as
   the historical optimized ragged baseline.
2. On fixed saved states, compare native variable-length attention/packing costs
   with B16, including all integer B and C32/64/128. Verify whether fewer tokens
   reduce executed attention work; preserve candidate semantics. Test a supported
   optimized attention backend before proposing a custom kernel.
3. Integrate true pre-draft integer B selection with packed drafting AND
   verification. Time predictor, gathering, packing, graph bucket waste, accept
   selection, KV commit/free and scheduler gaps separately. Full-B16 drafting
   with only shorter verification is a distinct policy, not this integration.
4. Run closed-loop serving against best fixed B, exact frozen 100k predictor,
   and target-only AR, on matched workloads and timing boundaries. Only then
   decide whether another predictor training sweep can produce a meaningful
   paper-level gain. The present offline predictor difference alone is tiny.

## Full artifact locations

Local evidence is under
`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/`:

- `highc_latency_20260928/`: raw clean requests, event cycles, config, launch
  arguments, server logs, `analysis.json`, `report.md`, `headroom_scenarios.json`.
- `highc_kernel_trace_20260928/trace_b16/`: B16 trace and `kernel_analysis.json`;
  the run-level FAILED flag refers to the later B12 port error.
- `highc_kernel_trace_b12_retry_20260928/trace_b12/`: successful B12 trace and
  `kernel_analysis.json`.
- `highc_target_only_20260928/`: AR raw results and report.
- `highc_predictor_cost_20260928/`: predictor timings, hashes and launch config.
- `actual_block_predictor_10k_finish_20260928/training/summary.json`: exact
  offline outcomes used in the sensitivity calculation.

The corresponding persistent remote roots are
`/data/scratch/zekaili/atharv/dflash/runs/<same-directory-name>/`.
Compiler caches remain remote and were not unnecessarily copied locally.

Analysis is reproducible with `scripts/summarize_latency_study.py`,
`scripts/summarize_cuda_trace.py` and `scripts/analyze_latency_headroom.py`.
Raw files are untracked evidence: git alone does not reproduce these runs.
