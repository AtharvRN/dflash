#!/usr/bin/env bash
# Bounded closed-loop serving benchmark; no replay hooks or persistent monitor.
set -euo pipefail
: "${DFLASH_SERVING_OUTPUT:?}"
: "${DFLASH_SERVING_SCRATCH:?}"
task_python="${DFLASH_PYTHON:-/tmp/predraft-latency-env/bin/python}"
test ! -e "$DFLASH_SERVING_OUTPUT"
test ! -e "$DFLASH_SERVING_SCRATCH"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv -l 5 > "$DFLASH_SERVING_OUTPUT.gpu.csv" &
task_telemetry=$!
trap 'kill "$task_telemetry" 2>/dev/null || true' EXIT
"$task_python" scripts/benchmark_postdraft_serving.py --output "$DFLASH_SERVING_OUTPUT" --scratch "$DFLASH_SERVING_SCRATCH" "$@"
"$task_python" scripts/summarize_postdraft_serving.py "$DFLASH_SERVING_OUTPUT"
