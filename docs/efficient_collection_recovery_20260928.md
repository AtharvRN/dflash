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

New diagnostic run:
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
The same 37 tests also passed on the workstation before launch.

### Diagnostic restart verification

Implementation/deployment commit: `46d1d71`. The new wrapper launched in tmux
`dflash-diagnostics-20260928` on GPU 4. The instrumented matched replay again
passed every state/feature comparison for all 95 states at each worker count:

| Workers | Eligible cycles/s | Timed seconds | Mean GPU utilization | Sampled peak MiB |
|---|---:|---:|---:|---:|
| 1 | 1.01734 | 93.381 | 56.85% | 10,805 |
| 2 | 1.68828 | 56.270 | 94.80% | 20,872 |
| 4 | 1.81820 | 52.250 | 98.51% | 41,559 |

Four workers were selected by the unchanged rule. Collection started at
2026-09-28 18:19:24 UTC, with models ready at 18:19:35 UTC. Verified all 2,267
recovery source files remained unchanged and their copied bytes matched (source
config preserved separately). At 18:20:34 UTC, durable progress reached 5,508
eligible cycles / 5,536 total states across 789 processed prompt receipts:
5,408 preserved + 100 new eligible cycles. GPU 4 sampled at 100%, 42,345 MiB.
Worker PIDs were 654862, 654871, 654877 and 654882; all four were starting new
prompts and writing diagnostic events. Training remains pending collection/audit.

The tmux session's shared cgroup already had `oom_kill=17` at collection startup;
it was still 17 at the above progress check. This is a historical shared-cgroup
counter, NOT a diagnosis of either previous failure. Compare deltas and worker
exit records if another failure occurs; do not attribute the baseline to this run.

Local replay summary:
`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/actual_block_predictor_10k_diagnostics_20260928/benchmark/summary.json`

## Diagnostic failure and explicit completion relaunch

The diagnostic collection also stopped, at 2026-09-28 18:51:23 UTC (11:51 PDT),
with `BrokenProcessPool` / wrapper exit 1. It saved 8,912 eligible training cycles
(8,956 total states, 1,297 processed prompt receipts). A subsequent read-only
audit verified all 3,774 config/receipt/shard files and the final metadata hash
bindings, with zero canonical disagreements. Training never started.

This time the initiating event is visible: worker 654862 recorded SIGTERM at
18:51:22.389455 UTC, BEFORE the parent's first cleanup termination record at
18:51:22.529965 UTC. The other three workers recorded SIGTERM after controller
cleanup began. Their final exit statuses were all -15. No Python/CUDA OOM
exception was recorded, and the shared-cgroup `oom_kill` counter remained 17.
The sender of the first SIGTERM remains unidentified. The roughly 32-minute
duration repeats earlier failures; it does not establish a specific scheduler,
timeout, user action or resource policy as the cause. No automatic retries or
changes to system limits were introduced.

SSH later timed out before authentication; DNS and GitHub TCP controls worked.
SSH was reachable again at the user's next check. No SSH configuration was
changed, and these observations do not establish a cause for the access outage.

The user explicitly requested another relaunch. Its source is the full 8,912-row
diagnostic cache, leaving 1,088 eligible rows to the training target. The existing
four-hour cumulative collection bound has 8,647.3257 seconds remaining. The same
source/input audit, matched replay gate, diagnostics, first-10k selection and
three-seed training pipeline apply. Previous runs remain untouched.

- New root: `/data/scratch/zekaili/atharv/dflash/runs/actual_block_predictor_10k_finish_20260928`
- Temporary cache: `/tmp/actual_block_predictor_10k_finish_20260928_cache`
- Source cache: `/data/scratch/zekaili/atharv/dflash/runs/actual_block_predictor_10k_diagnostics_20260928/cache`
- tmux session: `dflash-finish-20260928`
- Deployment commit at launch: `ca09995` (numerical/diagnostic code unchanged from `46d1d71`).

GPU 4 had no listed GPU processes or open device users at preflight, but reported
100% utilization with zero allocated memory. No device reset or other-job action
was performed. After the replay began, the GPU reported normal single-worker
activity (53% utilization, 10,189 MiB allocated) and the first replayed states
matched.

### Completion relaunch verified

All 95 replay states matched exactly at each setting. The 1/2/4-worker rates were
0.97085 / 1.66730 / 1.80965 eligible cycles/s; four workers were selected by the
unchanged gate. Its sampled mean utilization was 98.41%, peak GPU memory 41,559
MiB. This is collection throughput, not a predictor or serving result.

Collection resumed at 2026-09-28 21:05:20 UTC. At 21:07:01 UTC it had durably
saved 9,069 eligible cycles: 8,912 preserved plus 157 new. All 3,774 source files
were rechecked unchanged at the old location and byte-identical in the new cache
(source configuration stored as `recovery_source_config.json`). Four workers
(916179, 916187, 916191, 916196) were active, GPU 4 sampled at 100% with 42,187 MiB
allocated, and there were no controller failure/signal events or wrapper exit
file. Training had not yet started; it remains gated on complete collection and
the original audit. The underlying first-SIGTERM sender is still unresolved.

Local replay summary:
`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/actual_block_predictor_10k_finish_20260928/benchmark/summary.json`

## Completed collection and training (supersedes progress-only status above)

The finish run subsequently completed its audit and all six predictor trainings.
The verified training audit records 10,102 collected states, 10,054 eligible
states, exactly 10,000 selected training cycles from 1,317 prompts, and 54 unused
eligible tail rows. Canonical split separation, receipt/prefix hashes and feature
alignment passed; checked canonical disagreements were zero. Evaluation remains
342 calibration and 1,416 assessment states on the unchanged prompt membership.

`training/COMPLETE.json` records `success=true`, `smoke=false` and binds the
configuration, audit, selected index, six checkpoints, histories, frozen
selection, predictions, operating curves and report. Reload and paired
initialization/order checks passed. This is distinct from the still-unfinished
September 15 paired raw/fused 100k expansion.

At the 96% calibration target, the three actual-supervision seeds average ratio
0.54634 at assessment retention 0.95316; clipped-supervision controls average
0.53930 at 0.95598. The frozen historical 100k MLP reaches 0.53942 at 0.96006.
These are unequal-retention operating points. No model reaches the requested
joint 70% ratio / 96% retention point. Gains are small and seed-dependent, not
a serving speedup. Full per-seed results and paired prompt intervals are in:

`/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/actual_block_predictor_10k_finish_20260928/training/summary.json`
