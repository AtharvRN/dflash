# Historical packed DFlash recovery — 2026-09-28

Status: recovery and correctness checks in progress; no new ragged speedup claim.

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

The recovered code disables draft CUDA graphs explicitly; target verification
can use graph buckets. Historical fields named `draft_graph_total_tokens` are
planned budgets, not proof of actual draft graph replay. Existing synchronized
per-component timers perturb execution and are not clean throughput results.

GPU use is limited to the designated workstation's free GPU 4, under the existing
cooperative lock. Checks use a new task-owned container and pinned cached image,
with no changes to the main environment or unrelated running containers. Launches
are bounded, logged durably, and do not create a recurring monitor.
