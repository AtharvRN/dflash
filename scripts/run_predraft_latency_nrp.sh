#!/usr/bin/env bash
# Existing single-GPU pod only. Frozen bundle/checkpoints; no training or fitting.
set -euo pipefail
: "${DFLASH_POD_UID:?Downward API UID required}"
: "${DFLASH_CODE_COMMIT:?Pinned checkout required}"
: "${DFLASH_RUN_ROOT:?Fresh durable run directory required}"
if [[ "$(git rev-parse HEAD)" != "$DFLASH_CODE_COMMIT" || -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Code provenance mismatch' >&2
  exit 1
fi
if [[ -e "$DFLASH_RUN_ROOT" ]]; then
  echo 'Fresh run root required' >&2
  exit 1
fi
mkdir -p "$DFLASH_RUN_ROOT"
exec > >(tee "$DFLASH_RUN_ROOT/pipeline.log") 2>&1
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
task_python="${DFLASH_PYTHON:-python3}"
task_bundle=/workspace/dflashv2_data/runs/predraft_latency_20261005/bundle
task_scratch="/tmp/predraft-latency-${DFLASH_POD_UID}-${DFLASH_RUN_ROOT##*/}"
date -u +%FT%TZ
"$task_python" -m pip freeze
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv
# Bounded run-local telemetry, not a recurring external monitor.
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv -l 5 > "$DFLASH_RUN_ROOT/gpu_utilization.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
"$task_python" -m pytest -q -p no:cacheprovider tests/test_predraft_latency.py tests/test_midverify_latency.py tests/test_predraft_same_head_bce.py tests/test_gpu_runtime.py
"$task_python" scripts/run_midverify_latency.py --use-container-gpu --data-root /workspace/dflashv2_data \
  --models-config configs/rejected_trace_nrp_models.json --predraft-bundle "$task_bundle" \
  --output "$DFLASH_RUN_ROOT/smoke" --scratch-dir "$task_scratch/smoke" --smoke --max-seconds 1800
"$task_python" scripts/run_midverify_latency.py --use-container-gpu --data-root /workspace/dflashv2_data \
  --models-config configs/rejected_trace_nrp_models.json --predraft-bundle "$task_bundle" \
  --output "$DFLASH_RUN_ROOT/full" --scratch-dir "$task_scratch/full" --max-seconds 3600
"$task_python" scripts/summarize_predraft_latency.py --run "$DFLASH_RUN_ROOT/full"
