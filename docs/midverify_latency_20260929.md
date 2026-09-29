# Bounded native-engine segmented-verification test

Authorized 2026-09-29. This is an execution-feasibility test, not a new training
or collection sweep and not a claim of closed-loop serving throughput.

Use the recovered, hash-verified SGLang spec-v2 source and pinned dependency
container. Qwen3-4B and original DFlash B16, BF16, FlashInfer, TF32 off, GPU4 only.
Read existing cache/checkpoints without changes. Freeze seed913 and the existing
99%-calibrated candidate-confidence and target/candidate-confidence L6 policies;
do not select a seed or threshold using these test outputs.

First smoke on four distinct assessment prompts. Then first eligible cycle of
each of 128 distinct assessment prompts in canonical order, without selection
by acceptance, scored as two C64 batches and the same states as one C128 batch.
Current policies use fresh same-engine draft confidence and L6 target features;
record numerical and policy drift rather than treating offline features as free.

Compare fixed full B16, fixed post-draft B8 truncation (same B16 candidates, not
actual shorter drafting), actual B8 redrafting, target-free learned trimming, the same target-free
policy with a no-prune L6 split, and the L6 cascade. Prefix tokens are preserved;
only the explicitly labeled B8 redraft control changes candidates. Measure both eager execution and manually captured
exact-shape target graphs, with eager drafting for every case. These graph
captures use native attention planning and native model layers but are NOT a
general graph-bucket scheduler integration. Misses must fail, not silently pad.

Prefill and graph capture are untimed. Timed replay includes draft setup/model/
projection, confidence computation where used, actual probe inference, D2H
length decisions, packed-row construction, native attention metadata, model
segments, acceptance/bonus selection and draft-KV materialization. Repeat
snapshots rather than advancing trajectories. Slots are preallocated; native
reserve/free microcost is reported separately. No request scheduler, HTTP,
prefill, live allocator commits, graph misses or workload drain is measured.

Use disjoint CUDA-event phase spans plus synchronized wall-clock cycle times;
also measure the cycle with phase instrumentation disabled. Alternate cases in
a seeded random order across repeats. Report time per actually committed token,
actual same-engine B16 retention, prefix lengths and all numerical differences.
Do not interpret reduced cycle latency alone as a throughput win.

Correctness gate: retained prefix against an independent all-layer forward on
that exact shorter prefix; target keys/values at layers0/5/35; native model
dispatch versus explicit layer traversal; unchanged sampled committed-prefix
keys at every layer; request/KV pool capacity restored on cleanup. The eager
no-prune split must be bitwise identical. Shape-changing/graph controls report
all top1/acceptance/bonus differences; a 2% hidden-relative-L2 stop guard is a
diagnostic guard, NOT a general correctness certificate.

No auto-training extension, new trajectory collection, runtime-source edits,
recurring monitor, other GPU use or subsequent serving integration.
