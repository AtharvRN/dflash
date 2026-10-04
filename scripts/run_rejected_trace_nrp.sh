#!/usr/bin/env bash
# Entrypoint for a bounded, single-A100 Kubernetes batch job.
set -euo pipefail
: "${DFLASH_CODE_ROOT:?Pinned checkout is required}"
: "${DFLASH_CODE_COMMIT:?Exact code revision is required}"
: "${DFLASH_POD_UID:?Downward API pod UID is required}"
: "${DFLASH_PYTHON:?Prepared Python environment is required}"
: "${DFLASH_TASK_ROOT:?Durable task root is required}"
: "${RUN_ID:?Unique run ID is required}"
if [[ ! "$DFLASH_CODE_COMMIT" =~ ^[0-9a-f]{40}$ || ! "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Invalid pinned revision or run ID" >&2
  exit 1
fi
cd "$DFLASH_CODE_ROOT"
if [[ "$(git rev-parse HEAD)" != "$DFLASH_CODE_COMMIT" || -n "$(git status --porcelain)" ]]; then
  echo "Code revision or clean-checkout check failed" >&2
  exit 1
fi
TASK_LOG_DIR="$DFLASH_TASK_ROOT/nrp_logs/${RUN_ID}_${DFLASH_POD_UID}"
mkdir -p "$TASK_LOG_DIR"
exec > >(tee "$TASK_LOG_DIR/pipeline.log") 2>&1
date -u +%FT%TZ
printf 'pod_uid=%s commit=%s\n' "$DFLASH_POD_UID" "$DFLASH_CODE_COMMIT"
"$DFLASH_PYTHON" -m pip freeze > "$TASK_LOG_DIR/environment.txt"
"$DFLASH_PYTHON" -c 'import torch, transformers; assert torch.__version__ == "2.13.0+cu130"; assert transformers.__version__ == "4.57.1"; print("Pinned framework versions passed")'
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DFLASH_MODELS="$DFLASH_CODE_ROOT/configs/rejected_trace_nrp_models.json"
bash scripts/run_rejected_trace_predictor.sh
