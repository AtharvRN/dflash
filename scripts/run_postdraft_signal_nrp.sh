#!/usr/bin/env bash
# Bounded cache-only predictor audit. Does not collect or change any candidates.
set -euo pipefail
: "${DFLASH_CODE_ROOT:?}"
: "${DFLASH_CODE_COMMIT:?}"
: "${DFLASH_POD_UID:?}"
cd "$DFLASH_CODE_ROOT"
[[ "$(git rev-parse HEAD)" == "$DFLASH_CODE_COMMIT" && -z "$(git status --porcelain)" ]]
export PYTHONPATH="$DFLASH_CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
task_python=/opt/sglang/bin/python
task_run=/workspace/dflashv2_data/runs/postdraft_signal_20261007/r1
task_work=/tmp/postdraft_signal_20261007_r1
[[ ! -e "$task_run" && ! -e "$task_work" ]]
task_gpu=$("$task_python" -c 'from scripts.gpu_runtime import wait_gpu_runtime; print(wait_gpu_runtime(use_container_gpu=True,require_gpu=True)["nvidia_smi_query_id"])')
[[ "$task_gpu" =~ ^GPU-[a-zA-Z0-9-]+$ ]]
exec 9>"/workspace/dflashv2_data/gpu_${task_gpu}_actual_block.lock"
exec 8>"/workspace/dflashv2_data/gpu_${task_gpu}_midverify.lock"
exec 7>"/workspace/dflashv2_data/gpu_${task_gpu}_rejected_trace.lock"
flock -n 9
flock -n 8
flock -n 7
mkdir -p "$task_run" "$task_work"
exec > >(tee "$task_run/pipeline.log") 2>&1
date -u
printf 'pod_uid=%s commit=%s\n' "$DFLASH_POD_UID" "$DFLASH_CODE_COMMIT"
"$task_python" -m pip freeze > "$task_run/environment.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv -l 5 > "$task_run/gpu_utilization.csv" &
task_monitor=$!
finish() {
    local status=$?
    kill "$task_monitor" 2>/dev/null || true
    printf 'exit_code=%s\n' "$status" > "$task_run/pipeline_exit.txt"
}
trap finish EXIT
"$task_python" -m pytest -q -p no:cacheprovider tests/test_postdraft_signal.py tests/test_soft_supervision.py
cp -a /workspace/dflashv2_data/runs/soft_supervision_2k_r2_20261005/cache "$task_work/cache"
cp -a /workspace/dflashv2_data/runs/soft_supervision_2k_r2_20261005/training "$task_work/old_training"
task_common=(--cache "$task_work/cache" --old-training "$task_work/old_training")
timeout --signal=TERM --kill-after=30s 1200s "$task_python" -u scripts/train_postdraft_signal.py \
    "${task_common[@]}" --smoke --output "$task_work/smoke" --backup "$task_run/smoke"
timeout --signal=TERM --kill-after=30s 7200s "$task_python" -u scripts/train_postdraft_signal.py \
    "${task_common[@]}" --output "$task_work/training" --backup "$task_run/training"
echo "EXPERIMENT_COMPLETE $task_run/training/summary.json"
