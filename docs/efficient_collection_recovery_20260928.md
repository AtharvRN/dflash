# Efficient collection and recovery amendment

The original 10k collection was terminated with SIGTERM / exit 143 at
2026-09-28 08:15:26 UTC, not its four-hour bound. No agent stop command was
issued; available account logs do not identify the sender. A read-only audit
verified 281 prompt receipts, 1,915 cycle states, 1,901 eligible rows and all
532 referenced shard files. No canonical disagreements or orphan shard files
were found. There is no completed dataset or training result from that run.

The user requested more efficient GPU use on the restart. Preserve all original
files and use new temporary/durable destinations. Scientific labels, input,
prompt ordering, selection of the first 10,000 eligible rows, validation bytes,
and six-model training protocol remain unchanged.

## Execution change and gate

Benchmark 1, 2 and 4 independent process workers on ONE previously idle GPU.
Each loads its own frozen models and executes the original single-request
sampler with identical shapes, dtypes, masks, random order and cache semantics.
This is concurrent prompt collection, not tensor batching or larger B.
No padding-mask intervention and no change to reference trajectories.

Use the same first 12 nonempty completed training prompts at all three worker
counts, preserving each original global state offset for the early audit checks.
Warm up each worker before timing. Time the same prompt workload, including
result transfer and replay checking but excluding model load/warmup. Compare
every saved prefix, draft token, acceptance outcome, metadata record and fused
FP16 feature exactly. A serial mismatch stops recovery. Exclude concurrent
settings with any mismatch or sampled peak GPU use over 85% of capacity.
Require at least 10% measured throughput improvement to use multiple workers;
prefer fewer workers if within 5% of the fastest valid setting. These are
collection-throughput measurements, not serving or predictor gains.

## Recovery and durability

Verify every original receipt and source input hash, canonical group and prefix
continuity. Copy verified shards and receipts byte-for-byte; preserve the
original configuration separately and record its hash and source-file inventory.
Do not import unreceipted evidence as valid data or delete anything at the source.
Require an uninterrupted prefix of planned prompt receipts. Resume at the next
planned prompt, not a newly sampled subset.

Keep at most twice the worker count queued, commit results and receipts in
original prompt order, and copy each committed prompt atomically to durable
scratch storage. Finish in-flight prompts at the 10k boundary; extra eligible
tail rows are preserved but excluded by the existing first-10k training rule.
Deduct prior collection elapsed time from the original four-hour collection
allowance. Model warmup and the bounded in-flight tail can add startup/tail time.

Record SIGTERM/SIGINT requests and stop starting new prompts. When only the
controller receives the signal, drain the bounded pending work and save an
incomplete final manifest. If workers are also terminated, preserve the last
committed prefix and record the failure. No automatic restart or recurring
monitor is installed. Incomplete/error runs never launch training.

After complete collection, the original audit and matched three-seed training
pipeline runs unchanged. The frozen baseline and fixed evaluation remain intact.

## Locations

- Original evidence: /data/scratch/zekaili/atharv/dflash/runs/actual_block_predictor_10k_20260928
- Recovery run: /data/scratch/zekaili/atharv/dflash/runs/actual_block_predictor_10k_recovered_20260928
- Temporary recovery: /tmp/actual_block_predictor_10k_recovered_20260928_cache
- Benchmark results: recovery run/benchmark/summary.json
- Durable data: recovery run/cache
- Training results: recovery run/training
- Single bounded batch: scripts/run_efficient_recovery.sh

## Measured gate and restart

Implementation commit: 14ab486. All 30 targeted tests passed locally and on the
workstation. GPU 4 benchmarked the same 12 prompts / 95 states at each setting:

| Workers | Eligible cycles/s | Timed seconds | Mean GPU utilization | Sampled peak MiB |
|---|---:|---:|---:|---:|
| 1 | 0.97861 | 97.076 | 54.96% | 10,805 |
| 2 | 1.67474 | 56.725 | 94.39% | 20,872 |
| 4 | 1.81244 | 52.415 | 98.11% | 41,503 |

All three settings matched every stored record and fused feature exactly.
Four workers were selected by the predeclared rule: approximately 1.85x the
single-worker collection rate on this matched sample. This is a small collection
benchmark, not a serving or predictor speedup. It excludes model load/warmup.
The single bounded recovery-and-training batch restarted collection on GPU 4
with the 1,901 eligible rows preserved; final collection/training is still pending.
The original interrupted directory remains unchanged.

Local benchmark artifact:
/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/actual_block_predictor_10k_recovered_20260928/benchmark/summary.json

## Second failure and diagnostic restart

The recovered run stopped at 2026-09-28 09:08:27 UTC (02:08:27 PDT), exit 1,
after a worker disappeared and `ProcessPoolExecutor` raised `BrokenProcessPool`.
The application log does not identify the signal, sender or an OOM. The account
cannot see system kernel journal entries. The semaphore warning followed the
worker failure and is not established as its cause. No training started.

Read-only verification found 775 prompt receipts, 5,436 saved states and 5,408
eligible training cycles. All 2,267 receipt/config/shard hashes and alignment
checks passed, as did the three final metadata hash bindings. There were zero
canonical disagreements. The resumed portion added 3,507 eligible cycles at
approximately 1.84 cycles/s with four workers; the final GPU sample showed 100%
utilization and 44,007 MiB used. These are collection measurements only.

On the user's explicit request, add diagnostics and restart from this latest
verified cache, not the original 1,901-row prefix. Use a new run directory and
preserve both previous attempts. The collection/training protocol is unchanged.
The wrapper accepts `SOURCE_CACHE` and repeats the matched 1/2/4-worker replay
gate with the instrumented code before appending. The original four-hour
collection allowance remains cumulative across attempts: use the larger saved
progress/summary elapsed time, not a fresh four-hour budget.

Diagnostics in `scripts/worker_diagnostics.py` use the same standard spawned
processes and executor. Each worker writes a durable stdout/stderr/crash log and
JSONL lifecycle/task events. Events include active prompt identity, host RSS,
resource limits, and accessible Linux cgroup-v2 memory/OOM/pid counters. Parent
events record exit codes BEFORE executor cleanup terminates peers, then final
joined exit statuses; this distinguishes an already-dead worker from cleanup
SIGTERMs. SIGTERM/SIGINT/SIGHUP are logged with a stack dump and retain their
signal exit status. Fatal native faults use Python faulthandler. SIGKILL cannot
be caught in the child, and exit -9 alone does not prove OOM or identify a sender.
Cgroup counters can include other processes in a shared cgroup.

Snapshots are tied to startup, prompt boundaries, errors and shutdown; there is
no background monitoring service, automatic restart loop, or scheduler change.
The short GPU sampler remains confined to the existing bounded benchmark.
Diagnostics are stored on durable scratch, outside the numerical dataset cache.
The benchmark binds both driver and diagnostic-module hashes. Incomplete or
failed collection still blocks training.

Planned new run:
`/data/scratch/zekaili/atharv/dflash/runs/actual_block_predictor_10k_diagnostics_20260928`

Important files beneath that root:

- `diagnostics/collection/parent_lifecycle.jsonl`: spawn, pre-cleanup termination, final exit.
- `diagnostics/collection/controller.jsonl`: controller resources, commits and failure context.
- `diagnostics/collection/worker_<pid>.jsonl`: task identity, resources and caught signals/errors.
- `diagnostics/collection/worker_<pid>.log`: worker stdout/stderr and fault stacks.
- `cache/progress.json`: durable eligible-cycle progress.
- `pipeline_exit.txt`: final wrapper exit status when the bounded batch ends.

Local validation: 37 targeted tests pass, including normal spawned execution,
Python exceptions, caught SIGTERM, and a disposable two-worker SIGKILL test that
distinguishes the killed worker from the peer terminated during executor cleanup.
Launch and GPU replay results are pending at the time of this code amendment.
