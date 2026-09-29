# Historical packed DFlash recovery — 2026-09-28

Status: runtime recovered; primitive and small-batch integration checks completed.
C64 ragged stress found a large same-state target-hidden discrepancy and stopped.
Focused diagnostics explain a separate 6.2% eager discrepancy as ordinary
batch-shape sensitivity, but the extreme 90.8% graph-run case remains unresolved.
Profiling remains gated; no new ragged throughput/speedup claim. No task jobs or
recurring monitor remain running at handoff.

## Recovered runtime

The complete historical parent worktree was located at
`/Users/atharvramesh/Projects/sglang-dflash-pr23000`, branch
`codex/dflash-dynamic-policy`, base commit
`e5add3ba0f20d46b5bcb7a9a356df7cb102c49ae`, with dirty runtime changes.
Neither that worktree nor the newer `sglang-dflash-ragged` worktree was changed.

`vendor/sglang_ragged_20260723` contains a 6.98 MB git-authored Python-source
archive, 16 verbatim overlay files, and checksummed provenance. It is source,
not an environment, model or dataset. The archive includes the upstream license.
The manifest attributes each overlay to its original absolute path. The only
cross-snapshot addition is `triton_ops/dflash_accept_bonus.py`, recovered from
the saved deployed snapshot in `dflash-fresh-zlab-main/.tmp_sglang_opt/current_pod`.
The parent worktree already calls that ragged kernel but only contains the
fixed-width kernel, falling back to eager acceptance when the import is absent.
The deployed block-preparation helper is byte-identical to the parent worktree;
the parent worker additionally has later post-draft features, which are disabled
for these tests. This is a provenance-explicit reconstruction, not a claim of
bitwise identity to the old live pod.

Local reconstructed source:
`/Users/atharvramesh/Projects/MLSys/sglang-dflash-recovered-20260928`.

## Scope and correctness gates

1. Reference acceptance checks, all B=1..16 and A=0..B-1, and heterogeneous,
   changing batches. B=1 is a primitive/forced-path edge case; the learned
   pre-draft path normally requires B>=2.
2. Actual CUDA/Triton helpers: packed blocks, projection/commit indices, physical
   graph padding, KV locations, bonus selection and repeated buffer reuse.
   Silent fallback to eager acceptance is a failure, not a passing Triton test.
3. Full-model greedy tests, actual short drafting, graph on/off, different
   lengths/order/termination and per-request output accounting. A B16-clipped
   oracle is not a valid reference for actual shorter drafting.
4. Only after relevant correctness gates pass: same-engine fixed-vs-ragged
   component timing and clean throughput, with separate graph/backend flags.

CPU eager acceptance passed 1,456 request-cases in 49 batches, including the
136 exhaustive (B,A) cases. This does not test attention, GPU kernels, scheduling,
or KV lifetime. Artifact: `outputs/ragged_recovery_20260928/cpu_primitives.json`.

The actual recovered CUDA class and Triton kernels passed the same 1,456
acceptance cases plus 54 packing/padding batches. These test all integer lengths,
all first-rejection positions, non-power-of-two/changing batch sizes, request
permutations, projection/commit indices, dummy suffixes and buffer reuse. No
silent fallback to eager acceptance was allowed. GPU artifact:
`outputs/ragged_primitives_20260928/primitives.json`.

## Full-model checks and numerical controls

Hardware: workstation GPU 4, RTX PRO 6000 Blackwell Server Edition. Runtime:
recovered source, cached image digest recorded in each run's `config.json`, Torch
2.11.0+cu129, BF16, TF32 off, TP=1, page size 1, FlashInfer for target and draft,
spec-v2 overlap, greedy decoding, original pinned Qwen3-4B and DFlash-B16 snapshots.
All inputs are already-inspected development prompts, not fresh final test data.

The independent reference restores the **same committed KV prefix** and actually
forwards each request at its chosen B. It never substitutes clipped B16 labels.
Shadow forwards are executed in reverse request order. Each packed forward is
rerun afterwards to restore suffix KV writes before normal decoding resumes.
These reference passes and synchronization make this an audit, not a benchmark.

The final small-batch controls each complete 32 requests (16 prompts in two
orders), with output caps 1/7/8/16/32/63/64/96. Two requests in each run stop on
token 151645 before their caps; the rest stop at their length caps. Strict busy
KV-pool accounting is enabled. No KV-leak assertion or scheduler exception was
observed in these completed controls.

| Control | Audited request-cycles | Draft-token differences vs independent actual-B | Verifier top-1 differences vs independent | Different emitted prefix+bonus vs independent | Different emitted prefix+bonus vs same-batch eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rotating ragged B2–16 | 54 | 4 / 441 | 2 / 495 | 0 / 54 | 0 / 54 |
| Fixed B16, same recovered engine | 54 | 12 / 810 | 5 / 864 | 0 / 54 | 1 / 54 |
| Rotating ragged, batch-invariant flag | 54 | 5 / 441 | 4 / 495 | 0 / 54 | 0 / 54 |

These are diagnostic counts, not error-rate rankings: fixed and dynamic policies
do not visit exactly the same decoding states. Differences after first rejection
can change neither the accepted prefix nor its bonus; hence the two distinct
metrics. Fixed B16 also exhibiting differences does **not** prove that every
ragged discrepancy is benign. SGLang's existing batch-invariant flag did not
establish exact cross-shape numerical parity in this setup.

For each of these three controls:

- Eight draft and eight target forwards were audited; observed batch sizes were
  6–7. Target graph replay actually occurred in 8/8 target forwards. The draft
  remained eager, as specified by the recovered worker.
- Cross-request isolation passed bit-for-bit in all eight checks per role:
  changing other requests' draft embeddings or verify-input tokens did not
  change the protected request's hidden states (or target logits).
- Repeating the original packed forward after the shadow passes reproduced its
  hidden states and target logits bit-for-bit.
- The rotating runs exercised every integer B2–16.

Artifacts, relative to this repository:

- `outputs/ragged_isolation_check_20260928/`
- `outputs/ragged_fixed16_control_20260928/`
- `outputs/ragged_deterministic_check_20260928/`

Each includes launch/configuration records, raw audit JSONL, full server logs,
and output token IDs. Remote roots use the same run names under
`/data/scratch/zekaili/atharv/dflash/runs/`.

### C64 stress: unresolved discrepancy

`ragged_c64_stress_buffers_20260928` uses 128 development prompts, concurrency
64, and a fixed 128-token cap with EOS ignored solely to sustain concurrency.
It stopped deliberately at the second audited C64 target forward (128 matched
request-cycles), not from an OOM. One request at B15/prefix length 409 had target
hidden relative L2 error 0.9082 and maximum absolute error 2660.5625 versus its
independent actual-B forward. The same request's verifier top-1 tokens and emitted
decision agreed; the hidden discrepancy is still unresolved and is not dismissed
as rounding. All other requests in that cycle had relative hidden error <=0.0128.

Across those two cycles, verifier top-1 differed at 9/1,168 real positions and
emitted prefix+bonus differed in 3/128 comparisons against independent forwards.
Graph versus same-batch eager changed 2/128 emitted decisions. Packed replay was
bitwise reproducible. The rotating protected-row isolation check passed, but it
did not protect the large-discrepancy row and cannot rule out its isolation bug.
The follow-up harness checks output/prefix slot overlap, protects the worst row,
repeats its independent forward, and preserves large-discrepancy tensors and KV.

`ragged_c64_fixed16_control_20260928` completed 256 requests (two orders). It
audited three actual C64 forwards per role: 192 request-cycles, all B16. Maximum
target hidden relative L2 error was 0.0228, verifier top-1 differed at 15/3,072
positions, and 0/192 emitted decisions differed versus independent forwards.
Graph versus same-batch eager changed 2/192 emitted decisions. Isolation and
packed restoration were bitwise equal for all three checks per role. This is a
control in the same engine, not an identical-state comparison to the ragged run.

`ragged_c64_diagnostic_20260928` completed 256 requests and audited four C64
forwards per role (256 request-cycles). It did **not** reproduce the extreme
0.9082 discrepancy. Maximum target hidden relative L2 was 0.0247, verifier
top-1 differed at 18/2,536 real positions, and 2/256 emitted decisions differed
from independent execution (5/256 versus same-batch eager). Every diagnostic
found unique output slots disjoint from all live committed prefixes. Changing
other requests left the worst-discrepancy request unchanged, and independently
repeating that request was bitwise stable. These additional checks passed for
all four forwards per role. Different scheduling/trajectories mean this rerun
does not resolve the earlier extreme case.

`ragged_eager_diagnostic_20260928` disables graphs and checks more states. It
stopped at its fourth audited forward, C63/B7/prefix length 1,543, on a different
case with target hidden relative L2 0.06214. The largest difference was at the
last of seven query positions (per-position relative L2 0.2073). Error was small
at early captured layers and larger at later layers. All seven target top-1
tokens and the emitted decision agreed for this case. Across all four forwards,
1/252 emitted decisions differed from independent execution; same-batch eager
and packed restoration were bitwise identical, with no slot overlap or failure
of worst-row request isolation. Thus a >5% hidden discrepancy can arise without
graphs, but this is **not** a reproduction of the original 90.8% case.

The exact failing eager state, including all 36 layers of committed target KV,
was preserved at:
`/data/scratch/zekaili/atharv/dflash/runs/ragged_eager_diagnostic_20260928/eager/discrepancy_target_4_584.pt`.
It is an audit artifact, not training data. Only its logs/metadata were copied
locally; the 230 MiB KV snapshot remains on the workstation.

### Saved-state numerical control outside SGLang

`ragged_saved_state_precision_20260928` replays that B7 state with Transformers
4.57.1 / Torch 2.13.0+cu130, dense SDPA, TF32 off, and the same pinned target
weights. Its saved BF16 prefix KV is held fixed, not recomputed. Duplicating the
same request 90 times gives 630 query tokens, matching the eager failing batch's
total query count without introducing ragged packing. Only one representative
request is compared; every replica's logits are bitwise equal within each run.

| Dense dtype | C90 versus C1 hidden relative L2 | Maximum absolute hidden difference | Target top-1 differences |
| --- | ---: | ---: | ---: |
| BF16 | 0.0132478 | 3.0 | 0 / 7 |
| FP32 | 0.000001465 | 0.0002022 | 0 / 7 |

This establishes ordinary batch-shape numerical sensitivity outside the ragged
implementation. It does **not** explain every SGLang discrepancy: the FP32 dense
output still differs from saved SGLang packed/independent hidden states by
0.1375/0.1170 relative L2 respectively, although all seven top-1 predictions and
the accepted-prefix/bonus decision agree. Cross-engine/backend/version/fused
operation differences remain confounded; this is not an FP32 SGLang reference
or a correctness certificate. The original extreme graph-run case remains open.
The FP32 C90 run logged an allocator retry after a memory-allocation warning,
then completed successfully. No throughput interpretation is made.

### Same-runtime fixed-width replay explains the saved eager case

`ragged_saved_state_runtime_20260928` loads the saved prefix KV into newly
allocated slots of the **same recovered SGLang/image runtime**, before serving
any requests. Replicas share the immutable prefix but have disjoint suffix slots.
It compares fixed-width metadata (`DFlashVerifyInput`) with the ragged metadata
class. No changes were made to either implementation. Every comparison below
uses the first seven real query tokens; B10/B16 append causal dummy suffix tokens.

| Replay | Hidden result for the real B7 prefix |
| --- | --- |
| Fixed C1/B7 | Bitwise identical to saved independent reference |
| Fixed C1/B10 or B16 | Bitwise identical to fixed C1/B7 |
| Fixed C63/B10 (630 query tokens) | Bitwise identical to saved ragged packed output; reproduces 0.06214 relative L2 versus C1 |
| Ragged metadata C63, all B10 | Bitwise identical to fixed C63/B10 |
| Fixed C63/B16 | 0.05769 relative L2 versus C1 |

All seven target top-1 predictions, acceptance A=6, and bonus token 31610 agree
in every case. Replicated requests' logits are bitwise identical within each
replay. Request/KV allocator availability is unchanged before and after
(64 request slots, 131,072 token slots). The normal server warmup subsequently
completed, and the task-owned container was stopped.

This is stronger than the unrelated fixed-B16 workload control: it reproduces
the **same saved eager discrepancy without mixed lengths or ragged metadata**.
Thus that discrepancy is not evidence of a ragged indexing/packing defect. It
does not identify the exact arithmetic kernel responsible, and does not resolve
the original 0.9082 graph-run case, whose full tensors were not saved. It would
be incorrect to discard that extreme case merely because this smaller case is
now explained. The next correctness step is a preserved same-state replay of
that extreme case (or a comparable recurrence), followed by target-only output
and longer allocator checks before clean profiling.

### Preserved harness failures / superseded evidence

`ragged_model_check_20260928` deliberately stops on its first large discrepancy.
This was a **harness error**: DFlash's fused residual normalization mutates its
input embeddings in place; the original shadow control reused the modified
buffer. The harness now snapshots inputs before the first forward, has a
regression test, and exact packed replay verifies restoration. The runtime itself
was not patched to make this test pass.

`ragged_model_check_inputs_20260928` then completed 32 eager and 32 graph-enabled
requests, but its top-1 counters compared `topk(...)[0]` to `argmax`. Their tie
rules differ, so those counters are **superseded**, not usable acceptance evidence.
The final controls use `argmax` on both sides, as the actual verifier does. A
regression test checks identical tied logits. All original runs remain intact.

`ragged_c64_stress_20260928` initially failed because the audit grew persistent
projection buffers inside `inference_mode`, while the normal worker updates them
outside that mode. Its normal projection then rejected an in-place write to an
inference tensor. The harness now creates those shadow-induced buffers in a
normal no-grad context, with a regression test. The recovered runtime remains
unchanged. The subsequent `_buffers_` run above is a distinct failure and is not
explained by this fixed harness issue.

## Remaining sign-off boundary

This establishes recovered code, passing primitive checks, functioning graph
replay, and bounded request isolation/acceptance/KV checks. It is **not** an
unconditional correctness certificate. Target-only end-to-end transcript parity,
long-run allocator stress, other page sizes/TP settings, and arbitrary stochastic
sampling have not been certified. The recovered ragged path explicitly supports
greedy decoding only. Numerical differences must remain visible in any report.

Do not use the previous stock-image/spec-v1/Triton timings as measurements of
this recovered spec-v2/FlashInfer implementation. Same-engine fixed-versus-ragged
clean timing is a separate step; no timings from shadow-forward audits are valid
throughput measurements.

The recovered code disables draft CUDA graphs explicitly; target verification
can use graph buckets. Historical fields named `draft_graph_total_tokens` are
planned budgets, not proof of actual draft graph replay. Existing synchronized
per-component timers perturb execution and are not clean throughput results.

GPU use is limited to the designated workstation's free GPU 4, under the existing
cooperative lock. Checks use a new task-owned container and pinned cached image,
with no changes to the main environment or unrelated running containers. Launches
are bounded, logged durably, and do not create a recurring monitor.

Eight local recovery/profiling-helper tests pass. All 16 original source overlay
hashes were rechecked at handoff and remain unchanged. Runtime source stays
verbatim; only test harnesses, recovery tooling and this report were changed.
