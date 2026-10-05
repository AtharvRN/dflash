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
"$DFLASH_PYTHON" -c 'from scripts.gpu_runtime import configure_gpu_runtime; print(configure_gpu_runtime(use_container_gpu=True, require_gpu=True)); import torch, transformers; assert torch.__version__ == "2.13.0+cu130"; assert transformers.__version__ == "4.57.1"; assert torch.cuda.device_count() == 1; x=torch.ones((64,64),device="cuda"); assert (x@x)[0,0].item() == 64; print("Pinned framework versions and CUDA compute smoke passed")'
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DFLASH_MODELS="${DFLASH_MODELS:-$DFLASH_CODE_ROOT/configs/rejected_trace_nrp_models.json}"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5 > "$TASK_LOG_DIR/gpu_utilization.csv" &
TASK_GPU_MONITOR=$!
trap 'kill "$TASK_GPU_MONITOR" 2>/dev/null || true' EXIT
bash scripts/run_rejected_trace_predictor.sh
