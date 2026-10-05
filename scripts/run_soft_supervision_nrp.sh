#!/usr/bin/env bash
# One bounded experiment; no scheduler changes, no automatic resume or overwrite.
set -euo pipefail
: "${DFLASH_CODE_ROOT:?}"
: "${DFLASH_CODE_COMMIT:?}"
: "${DFLASH_POD_UID:?}"
: "${DFLASH_PYTHON:?}"
: "${DFLASH_SOURCE:?}"
: "${DFLASH_EVALUATION:?}"
: "${DFLASH_MODELS:?}"
: "${RUN_ID:?}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
cd "$DFLASH_CODE_ROOT"
[[ "$(git rev-parse HEAD)" == "$DFLASH_CODE_COMMIT" && -z "$(git status --porcelain)" ]]
export PYTHONPATH="$DFLASH_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
TASK_ROOT=/workspace/dflashv2_data
TASK_RUN="$TASK_ROOT/runs/$RUN_ID"
TASK_WORK="/tmp/$RUN_ID"
[[ ! -e "$TASK_RUN" && ! -e "$TASK_WORK" ]]
TASK_GPU=$("$DFLASH_PYTHON" -c 'from scripts.gpu_runtime import configure_gpu_runtime; print(configure_gpu_runtime(use_container_gpu=True,require_gpu=True)["nvidia_smi_query_id"])')
[[ "$TASK_GPU" =~ ^GPU-[a-zA-Z0-9-]+$ ]]
exec 9>"$TASK_ROOT/gpu_${TASK_GPU}_actual_block.lock"
exec 8>"$TASK_ROOT/gpu_${TASK_GPU}_midverify.lock"
exec 7>"$TASK_ROOT/gpu_${TASK_GPU}_rejected_trace.lock"
flock -n 9
flock -n 8
flock -n 7
mkdir -p "$TASK_RUN" "$TASK_WORK"
exec > >(tee "$TASK_RUN/pipeline.log") 2>&1
date -u
printf 'pod_uid=%s code=%s\n' "$DFLASH_POD_UID" "$DFLASH_CODE_COMMIT"
"$DFLASH_PYTHON" -m pip freeze > "$TASK_RUN/environment.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$TASK_WORK/gpu_utilization.csv" &
TASK_MONITOR=$!
finish() {
    local code=$?
    kill "$TASK_MONITOR" 2>/dev/null || true
    cp "$TASK_WORK/gpu_utilization.csv" "$TASK_RUN/gpu_utilization.csv"
    printf 'exit_code=%s\n' "$code" > "$TASK_RUN/pipeline_exit.txt"
}
trap finish EXIT
# NVML utilization is a recent sampling window and can stay high briefly after
# a completed worker releases all memory. Keep the strict entry guard; wait for
# three quiet readings between our stages, never override occupied-device checks.
wait_for_quiet_gpu() {
    "$DFLASH_PYTHON" -c '
import subprocess,time
quiet=0
for attempt in range(30):
    line=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=10)
    if len(line.strip().splitlines())!=1:
        raise RuntimeError("Ambiguous GPU inventory between stages")
    memory,util=map(int,line.strip().split(","))
    quiet=quiet+1 if 0<=memory<=1024 and 0<=util<=10 else 0
    if quiet>=3:
        print("GPU_QUIET",memory,util,flush=True)
        break
    time.sleep(1)
else:
    raise RuntimeError("GPU did not become idle within bounded inter-stage check")
'
}
TASK_COMMON=(--source "$DFLASH_SOURCE" --evaluation "$DFLASH_EVALUATION" --models "$DFLASH_MODELS" --workers 4)
timeout --signal=TERM --kill-after=30s 600s "$DFLASH_PYTHON" -u scripts/collect_soft_supervision.py \
    "${TASK_COMMON[@]}" --smoke --output "$TASK_WORK/smoke_cache" --backup "$TASK_RUN/smoke_cache"
wait_for_quiet_gpu
timeout --signal=TERM --kill-after=30s 600s "$DFLASH_PYTHON" -u scripts/train_soft_supervision.py \
    --cache "$TASK_WORK/smoke_cache" --smoke --output "$TASK_WORK/smoke_training" --backup "$TASK_RUN/smoke_training"
wait_for_quiet_gpu
timeout --signal=TERM --kill-after=30s 1800s "$DFLASH_PYTHON" -u scripts/collect_soft_supervision.py \
    "${TASK_COMMON[@]}" --output "$TASK_WORK/cache" --backup "$TASK_RUN/cache"
wait_for_quiet_gpu
timeout --signal=TERM --kill-after=30s 3600s "$DFLASH_PYTHON" -u scripts/train_soft_supervision.py \
    --cache "$TASK_WORK/cache" --output "$TASK_WORK/training" --backup "$TASK_RUN/training"
echo "EXPERIMENT_COMPLETE $TASK_RUN/training/summary.json"
