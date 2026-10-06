#!/usr/bin/env bash
# Frozen primary r96 comparison on all assessment cycles; no training/retuning.
set -euo pipefail
: "${DFLASH_POD_UID:?Verified pod UID required}"
: "${DFLASH_CODE_COMMIT:?Pinned code commit required}"
: "${DFLASH_RUN_ROOT:?Fresh durable output required}"
: "${DFLASH_PREDRAFT_BUNDLE:?Verified full-cycle confidence bundle required}"
task_python="${DFLASH_PYTHON:-/tmp/predraft-latency-env/bin/python}"
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
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1
task_scratch="/tmp/confidence-replay-${DFLASH_POD_UID}-${DFLASH_RUN_ROOT##*/}"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv -l 5 > "$DFLASH_RUN_ROOT/gpu_utilization.csv" &
task_monitor=$!
trap 'kill "$task_monitor" 2>/dev/null || true' EXIT
date -u +%FT%TZ
"$task_python" scripts/run_midverify_latency.py --use-container-gpu --data-root /workspace/dflashv2_data \
  --models-config configs/rejected_trace_nrp_models.json --predraft-bundle "$DFLASH_PREDRAFT_BUNDLE" \
  --output "$DFLASH_RUN_ROOT/smoke" --scratch-dir "$task_scratch/smoke" --smoke --max-seconds 1800 \
  --gpu-idle-wait-seconds 30
# A failed correctness smoke exits above; no full run is launched.
"$task_python" scripts/run_midverify_latency.py --use-container-gpu --data-root /workspace/dflashv2_data \
  --models-config configs/rejected_trace_nrp_models.json --predraft-bundle "$DFLASH_PREDRAFT_BUNDLE" \
  --output "$DFLASH_RUN_ROOT/full" --scratch-dir "$task_scratch/full" --concurrencies 32 64 128 \
  --max-seconds 3600 --gpu-idle-wait-seconds 30
"$task_python" scripts/summarize_predraft_latency.py --run "$DFLASH_RUN_ROOT/full"
date -u +%FT%TZ
