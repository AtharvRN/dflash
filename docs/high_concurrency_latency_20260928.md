# High-concurrency DFlash component-cost study

User requested precise draft, verify, overhead/upkeep and kernel measurements
before choosing the next predictor experiment. This is a systems diagnostic,
not another predictor-training run and not permission to modify other jobs.

## First gate: fixed-width serving costs

Use the existing immutable Slime image
`sha256:a7317182c71d35712ee4edc86a5d1c313dc969efdf0026d339673299c186ea75`.
It contains Torch 2.11.0+cu129, Transformers 5.8.1, SGLang 0.5.13, base source
`28b095c01005d4a3a2a5b637b7d028b07fba31b2` with image-baked Slime changes.
This is NOT the collection environment or the historical ragged fork. Pin the
image digest and record source hashes. Leave the existing experiment venv intact.

Original target/draft model snapshot paths and revisions come from `models.json`.
BF16, greedy, thinking-off tokenized inputs, TP1, one explicitly idle GPU. Bind
HTTP only to localhost. Use an isolated named container; stop only that container.
No recurring monitor, broad pkill, GPU reset, model training, or cluster access.
The image retains its built-in user because editable dependencies live under
`/root`. Host code and model/data mounts are read-only; only the individual run's
output/cache directory is writable. Drop capabilities except DAC_OVERRIDE and
disable privilege escalation. Do not mount Docker's socket or unrelated paths.

First measure fixed B8/B12/B16 at concurrency 16/32/64, plus C1 control. These
profiling grid points do NOT restrict the predictor's integer action space.
CUDA graphs stay enabled and their use is logged. Record backend, output cap,
actual batch occupancy and prefix lengths; stratify components by actual batch
and prefix length, rather than silently averaging warmup/prefill/draining tails.

Use saved cycle-zero prompt-token prefixes from the existing development cache,
not random tokens. Keep prompt order identical across B and repetitions. Disable
radix reuse so repeated workload passes do not benefit from earlier completions.
Allow EOS; report actual generated token counts. This is a development workload.

Separate uninstrumented wall-throughput repetitions from instrumented repetitions.
Events enqueue on the current CUDA stream and are read after completion without
per-stage synchronization. Preserve nested spans and report exclusive subtraction;
do not sum inclusive parents and children. CUDA-event spans include GPU idle gaps
while the CPU submits work, and are not pure kernel durations. Host call spans can
include synchronization already present in the implementation; CPU and GPU times
must not be added as if disjoint. End-to-end HTTP timing includes prefill and drain;
decode-cycle profiles are a different denominator.

Boundaries: draft setup/allocation, draft transformer forward, draft vocabulary
projection/argmax, target verify preparation, target forward (including its logits
projection), acceptance/target KV commit, committed-feature projection/draft KV
materialization, remaining cycle work. Scheduler/HTTP gaps outside the worker
are not mislabeled as predictor overhead. The fixed runs have no predictor.

At C64, capture short warmed CPU/CUDA traces for B16 and B12. Inspect attention,
GEMM/logits projection, copies, synchronization, and bookkeeping kernels. Kernel
duration sums, stream elapsed spans and wall throughput are distinct quantities.
Keep unprofiled measurements authoritative for throughput.

## Boundaries on interpretation

Uniform B12 is NOT a measured substitute for a mixed batch with mean B=12.
The cached v1 runtime does not implement the current actual-block predictor or
true packed variable-B drafting. It measures fixed-width headroom and baseline
upkeep only. Adaptive predictor compute may be microbenchmarked separately, but
feature extraction, graph-bucket padding, request remapping and ragged KV costs
remain unmeasured until the correct serving integration is restored and tested.
Do not claim a measured dynamic-policy speedup from these fixed-width results.

The local newer ragged fork is dirty and its current pre-draft policy selects a
verification cutoff while still drafting the full block. It cannot silently stand
in for true shorter drafting. Preserve this worktree and all historical evidence.

Report repeated measurements and dispersion, exact hardware/software/workload
provenance, correctness checks and limitations before using costs in a planning
calculation. No hardware-specific training reward is introduced by this study.
