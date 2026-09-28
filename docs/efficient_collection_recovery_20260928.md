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
