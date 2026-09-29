# Matched oracle and bounded dense/MoE screen

Started 2026-09-28 Pacific. Internal development study; not a publication benchmark.

## 1. Completed matched-state CPU analysis

`scripts/matched_policy_oracle.py` reverified the five granularity-analysis artifact
hashes and the included head-report/config/audit/prediction hashes. The head and
granularity artifacts match **exactly** on actual outcomes, prompt IDs, row order,
and the frozen MLP's survival predictions. This report does not reload the six
trained checkpoints; it recomputes decisions from their bound saved predictions.

All comparisons use **1,416 eligible assessment cycles from 187 prompts**, with
actual independently drafted B2--B16 outcomes on common B16 reference states.
Reference accepted-token sum: 9,315; mean A16: 6.5783898305. B includes the anchor.

The oracle uses exact multiple-choice integer dynamic programming to minimize
sum(B−1) subject to the aggregate accepted-token constraint. Request groups must
choose one B for all their sampled states; cycle groups can choose independently.
These are hindsight choices, unavailable before drafting, not trained policies.

| Policy | Mean A | Mean proposed budget | Ratio | Retention |
| --- | ---: | ---: | ---: | ---: |
| Fixed B16 | 6.5784 | 15.0000 | 0.43856 | 1.00000 |
| Fixed B15, calibration-selected | 6.3249 | 14.0000 | 0.45178 | 0.96146 |
| Frozen 100k cycle MLP, primary alpha grid | 6.3157 | 11.7083 | 0.53942 | 0.96006 |
| Request hindsight, 96% constraint | 6.3157 | 10.7331 | 0.58843 | 0.96006 |
| Cycle hindsight, 96% constraint | 6.3157 | 6.4633 | 0.97716 | 0.96006 |
| Request hindsight, 100% constraint | 6.5784 | 12.3948 | 0.53074 | 1.00000 |
| Cycle hindsight, 100% constraint | 6.5784 | 6.7804 | 0.97021 | 1.00000 |

The 96% cycle oracle's **mean block size is 7.4633**, not its 6.4633 proposed
budget. The 96% integer constraint is ceil(0.96 × 9,315) = 8,943 accepted tokens.
These oracles are on the policy assessment rows, not the older 871-state sample.

The learned request control on the original .001 alpha grid selects B16 at this
operating point. Keep the existing exact-calibration-breakpoint sensitivity
analysis separate: its request/cycle budgets are 13.7811/11.6568 with assessment
retentions 0.95878/0.95845. Neither is exactly 96% on assessment. No thresholds or
checkpoints were changed by this report. The six actual/clipped 10k heads were
also reproduced on these same rows; see the full tables below.

Local artifacts:

- `outputs/matched_policy_oracle_20260928/{report.md,summary.json,COMPLETE.json}`
- `outputs/matched_policy_oracle_exact_calibration_20260928/{report.md,summary.json,COMPLETE.json}`

Interpretation: there is substantial **token-work** headroom beyond the current
predictor, and decision granularity matters even with clairvoyance. This proves
neither learnability nor a throughput ceiling. Applying T(mean B) from a fixed-B
curve ignores mixed-batch costs, graph buckets, scheduling, candidate changes,
and closed-loop trajectory shifts. At mean B=7.4633, using only B8/B12 timings
would additionally require extrapolation, not interpolation.

## 2. Authorized screening scope

GPU 4 only, `zekaili@tianhaowang-gpu0.ucsd.edu`, RTX PRO 6000 Blackwell Server
Edition, 97,887 MiB. No other jobs are interrupted. Cooperative GPU lock and a
two-hour job deadline are used; containers stop on normal exit and exceptions.
There is no recurring monitor or automatic restart.

- Models: existing Qwen3-4B/original DFlash B16 and Qwen3-Coder-30B-A3B-Instruct
  with its matching DFlash checkpoint.
- First run C1/C4 B16 load-and-generate smoke for each pair. This is a support
  test, **not** an AR-parity certificate.
- Fixed B8/B16 × requested C64/C128 × exact 512/1,024 prompt tokens.
- One warmup and two measured waves per cell; 256 output tokens, greedy, EOS
  ignored to sustain a bounded stress workload. HTTP wall time includes prefill
  and drain. It is not natural-workload task throughput.
- C64 uses the first 64 selected prompts; C128 uses all 128. B8/B16 and clean/event
  comparisons have identical input identities within each C/length cell. The
  across-C comparison is a nested input cohort, not an identical-request-set
  concurrency ablation; do not conflate these two comparisons.
- Same 128 canonical-training message identities for both pairs; model-specific
  tokenizer/chat wrapper, thinking disabled where supported. Truncate only the
  user body to reach the specified total input length. Saved assistant answers
  are **never** model inputs. Source mixture: 76 nemotron, 49 evol_codealpaca,
  3 opencodeinstruct, no openr1_math. This long-prompt-selected pool is not a
  source-balanced or task-representative quality benchmark.
- BF16, TF32 off, TP1, radix cache disabled, page size 1. Recovered historical
  spec-v2 source (all 16 overlays checksum-verified) over the pinned dependency
  image. FlashInfer target/draft attention, target CUDA graphs, **eager drafter**.
- MoE expert backend: Triton BF16. This tests a supported baseline on SM120,
  not the best possible MoE kernel across hardware/backends.
- Clean and event-instrumented measurements run in separate server processes.
  Instrumentation replaces existing v2 timer boundaries with non-synchronizing
  CUDA events; runtime source is not patched. Existing native scalar syncs
  remain in both configurations. CUDA stream elapsed time is not kernel time.
  The new hook decomposes the decode worker only. Prefill, client/scheduler work
  and bookkeeping outside that worker remain included in HTTP wall time but are
  not separately attributed; this is not a complete end-to-end kernel profile.
- Report actual full-C occupancy and actual device-side prefix lengths,
  component distributions, graph use, and accepted drafts. A requested C128
  alone is not evidence of full-C128 execution. Cells with fewer than ten
  full-batch measured cycles are flagged unusable for full-batch timing.

The SGLang/speculative-decoding guides informed the separation of smoke tests,
clean measurements and instrumentation. No adaptive policy, hybrid, new
predictor training, quantization, or second GPU is part of this screen.

## 3. Checkpoint staging and provenance

Dedicated registry: `/data/scratch/zekaili/atharv/dflash/models_moe_screen_20260928.json`.
The existing dense `models.json` is unchanged. About 62.0 GB of selected files
were staged in the existing HF cache; all file sizes and LFS SHA256 values passed.
Free storage after staging: 237,069,504,512 bytes (about 220.8 GiB).

| Role | Repository | Pinned revision |
| --- | --- | --- |
| Target | Qwen/Qwen3-Coder-30B-A3B-Instruct | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` |
| Drafter | z-lab/Qwen3-Coder-30B-A3B-DFlash | `98ca0e3e2e6a372f2789d3a5e146566194084317` |

Staging helper: `scripts/prepare_moe_screen_models.py`, first deployed at
`cc66f84`. Workspace-adjusted retry code: `8645c2f`. Active retry root:
`/data/scratch/zekaili/atharv/dflash/runs/fixed_regime_screen_v4_20260928`.
Launch record and stdout log are the sibling `.launch.json` and `.log` files.
Retry launch PID 1992178, deadline 6,600 seconds; PID alone must not be treated
as persistent identity. Earlier v3 root with PID 1957963 is preserved separately.

Two earlier run directories preserve preparation failures: manifest contained a
saved assistant answer, then the sampled candidate pool had too few long prompts.
An additional CPU preflight rejected an oversized equal-per-source quota. These
were loader/pool assumptions, not GPU/model failures. None loaded model weights
onto GPU. Full-pool CPU preflight passed before the v3 launch.

## 4. Interpretation boundary

The fixed-width screen is a cost-sensitivity/feasibility test. Do not transfer the
Qwen3-4B oracle or 2,560-dimensional predictor to this MoE as if they were matched
MoE evidence: the MoE drafter's fused width is 2,048 and its acceptance function
differs. A favorable curve would justify a separate small MoE paired-state
headroom experiment, not an immediate adaptive-speedup claim. An unfavorable
two-point curve does not prove that every block size/backend/workload fails.

The v3 MoE C1/C4 B16 smoke **passed**. It loaded both BF16 checkpoints, captured
target graphs, and generated the requested 32-token continuations. This only
establishes load/generate support, not AR parity or C128 feasibility.

The next v3 C128-configured B16 server stopped during target graph capture:
FlashInfer needed 534,773,760 bytes for `batch_prefill_tmp_v`, exceeding its
402,653,184-byte default scratch buffer. About 17.42 GiB VRAM was still available;
this was a workspace-capacity error, not a demonstrated total-VRAM OOM. No
high-C measurements completed. All v3 evidence is preserved and the GPU released.
The bounded retry raises **only the MoE FlashInfer workspace to 1 GiB** through
the supported `SGLANG_FLASHINFER_WORKSPACE_SIZE` setting; graphs stay enabled.
Dense uses its backend's existing class-specific 512 MiB override. This is
scratch capacity, not a changed attention algorithm or new runtime source.

The v4 workspace-adjusted retry also passed its C1/C4 smoke, completed C128-capable
graph startup, and began clean C64 measurements with graphs enabled. Approximately
84.3 GiB GPU memory was in use during that stage. No completed B8/B16 comparison
exists at this update. Consult completion markers and detailed event summaries,
not the existence of a folder.
Twenty-two focused CPU/helper tests pass, including phase accounting, exact
oracle DP, runtime recovery, user-only inputs and neutral greedy parameters.

Event accepted-draft counters describe the internally verified blocks and may
include terminal-cap overshoot discarded from HTTP output. They are not matched
retention measurements; HTTP throughput uses actual capped completion counts.

## 5. First completed clean baseline: MoE B16

The v4 **clean MoE B16 stage is complete**. Server logs explicitly show actual
128-request decode batches with target CUDA graphs, at both tested context
lengths. This establishes C128 feasibility for this configuration, not C256 or
longer contexts. Each request returned exactly 256 completion tokens.

| Requested C | Prompt tokens | Output tokens/s, mean ± sample SD | Mean HTTP wave seconds |
| ---: | ---: | ---: | ---: |
| 64 | 512 | 1,224.10 ± 1.17 | 13.3846 |
| 64 | 1,024 | 1,086.46 ± 1.09 | 15.0802 |
| 128 | 512 | 1,785.47 ± 0.18 | 18.3526 |
| 128 | 1,024 | 1,542.25 ± 0.01 | 21.2468 |

Two measured waves after a same-cell warmup; these SDs describe only those two
waves, not a confidence interval or general robustness. Timing includes prefill
and drain; do not compare against the older v1/different-prompt dense table.
The separate MoE B16 event stage is next, followed by B8 event/clean runs and the
dense control. **No adaptive or B8/B16 speedup can be claimed from this table.**

Completed baseline artifacts are also local under
`outputs/fixed_regime_screen_v4_20260928/moe_b16_clean/`.
