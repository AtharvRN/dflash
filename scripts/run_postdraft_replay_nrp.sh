#!/usr/bin/env bash
# Frozen existing post-draft heads: matched graph-mode replay, not serving TPS.
set -euo pipefail
: "${DFLASH_POD_UID:?}"
: "${DFLASH_CODE_COMMIT:?}"
: "${DFLASH_RUN_ROOT:?}"
: "${DFLASH_POSTDRAFT_BUNDLE:?}"
task_python="${DFLASH_PYTHON:-/tmp/predraft-latency-env/bin/python}"
[[ "$(git rev-parse HEAD)" == "$DFLASH_CODE_COMMIT" && -z "$(git status --porcelain --untracked-files=no)" ]]
[[ ! -e "$DFLASH_RUN_ROOT" ]]
mkdir -p "$DFLASH_RUN_ROOT"
exec > >(tee "$DFLASH_RUN_ROOT/pipeline.log") 2>&1
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1
task_scratch="/tmp/postdraft-replay-${DFLASH_POD_UID}-${DFLASH_RUN_ROOT##*/}"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv -l 5 > "$DFLASH_RUN_ROOT/gpu_utilization.csv" &
task_monitor=$!
trap 'kill "$task_monitor" 2>/dev/null || true' EXIT
task_common=(--use-container-gpu --data-root /workspace/dflashv2_data
  --models-config configs/rejected_trace_nrp_models.json --predraft-bundle "$DFLASH_POSTDRAFT_BUNDLE"
  --correctness-reference same-mode --gpu-idle-wait-seconds 30)
"$task_python" scripts/run_midverify_latency.py "${task_common[@]}" --smoke \
  --output "$DFLASH_RUN_ROOT/smoke" --scratch-dir "$task_scratch/smoke" --max-seconds 1800
"$task_python" scripts/run_midverify_latency.py "${task_common[@]}" --modes graph --concurrencies 128 64 32 \
  --output "$DFLASH_RUN_ROOT/high" --scratch-dir "$task_scratch/high" --max-seconds 3600
"$task_python" scripts/summarize_predraft_latency.py --run "$DFLASH_RUN_ROOT/high"
"$task_python" scripts/run_midverify_latency.py "${task_common[@]}" --modes graph --concurrencies 16 8 \
  --output "$DFLASH_RUN_ROOT/low" --scratch-dir "$task_scratch/low" --max-seconds 3600
"$task_python" scripts/summarize_predraft_latency.py --run "$DFLASH_RUN_ROOT/low"
date -u +%FT%TZ
