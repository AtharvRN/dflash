#!/usr/bin/env bash
# Bounded, no-resume pilot. Never waits for or preempts an occupied GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT="/data/scratch/zekaili/atharv/dflash"
TASK_DATA="$TASK_ROOT/data/dflashv2_data"
TASK_PYTHON="$TASK_ROOT/envs/main/bin/python"
TASK_RUN_ID="${RUN_ID:-rejected_trace_2k_20261001}"
TASK_MODE="${MODE:-pilot}"
TASK_WORKERS="${WORKERS:-4}"
if [[ ! "$TASK_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Invalid run ID" >&2
  exit 1
fi
if [[ "$TASK_MODE" != pilot && "$TASK_MODE" != smoke ]]; then
  echo "MODE must be pilot or smoke" >&2
  exit 1
fi
if [[ "$TASK_WORKERS" != 1 && "$TASK_WORKERS" != 2 && "$TASK_WORKERS" != 4 ]]; then
  echo "WORKERS must be 1, 2, or 4" >&2
  exit 1
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -n "${GPU:-}" || ! "${SLURM_JOB_GPUS:-}" =~ ^[0-7]$ ||
        -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Slurm requires one allocated GPU and inherited CUDA_VISIBLE_DEVICES; do not set GPU" >&2
    exit 1
  fi
  TASK_PHYSICAL_GPU="$SLURM_JOB_GPUS"
  TASK_GPU_ARGS=(--use-visible-gpu)
else
  : "${GPU:?Choose an idle GPU explicitly, or submit through Slurm}"
  if [[ ! "$GPU" =~ ^[0-7]$ ]]; then
    echo "Invalid GPU index" >&2
    exit 1
  fi
  TASK_PHYSICAL_GPU="$GPU"
  TASK_GPU_ARGS=(--gpu "$GPU")
fi
TASK_RUN="$TASK_ROOT/runs/$TASK_RUN_ID"
TASK_WORK="/tmp/${TASK_RUN_ID}"
if [[ -e "$TASK_RUN" || -e "$TASK_WORK" ]]; then
  echo "Refusing existing temporary or durable destination; preserve previous evidence" >&2
  exit 1
fi
# Queued jobs recheck storage at start; never clean up other runs automatically.
for TASK_STORAGE in "$TASK_ROOT" /tmp; do
  TASK_FREE_KB=$(df -Pk "$TASK_STORAGE" | awk 'END {print $4}')
  if [[ ! "$TASK_FREE_KB" =~ ^[0-9]+$ ]] || (( TASK_FREE_KB < 10485760 )); then
    echo "Need at least 10 GiB free on $TASK_STORAGE; refusing to start" >&2
    exit 1
  fi
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Existing manual launchers key locks by NVML index, not /dev/nvidia minor.
  # Resolve the Slurm device through its UUID; this never initializes CUDA.
  TASK_PHYSICAL_GPU=$("$TASK_PYTHON" -c \
    'from scripts.gpu_runtime import configure_gpu_runtime; print(configure_gpu_runtime(use_visible_gpu=True, require_gpu=True)["nvidia_smi_index"])')
  if [[ ! "$TASK_PHYSICAL_GPU" =~ ^[0-7]$ ]]; then
    echo "Cannot resolve allocated GPU to a cooperative lock index" >&2
    exit 1
  fi
fi
# Cooperate with all our known pipelines; these are not scheduler reservations.
exec 9>"$TASK_ROOT/gpu_${TASK_PHYSICAL_GPU}_actual_block.lock"
exec 8>"$TASK_ROOT/gpu_${TASK_PHYSICAL_GPU}_midverify.lock"
exec 7>"$TASK_ROOT/gpu_${TASK_PHYSICAL_GPU}_rejected_trace.lock"
if ! flock -n 9 || ! flock -n 8 || ! flock -n 7; then
  echo "An existing experiment holds this GPU lock" >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  TASK_GPU_QUERY=$(nvidia-smi --id="$TASK_PHYSICAL_GPU" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  IFS=, read -r TASK_MEMORY TASK_UTIL <<< "$TASK_GPU_QUERY"
  TASK_MEMORY="${TASK_MEMORY//[[:space:]]/}"
  TASK_UTIL="${TASK_UTIL//[[:space:]]/}"
  if [[ ! "$TASK_MEMORY" =~ ^[0-9]+$ || ! "$TASK_UTIL" =~ ^[0-9]+$ || "$TASK_GPU_QUERY" == *$'\n'* ]]; then
    echo "Invalid GPU occupancy query; refusing to guess" >&2
    exit 1
  fi
  if (( TASK_MEMORY > 1024 || TASK_UTIL > 10 )); then
    echo "GPU $TASK_PHYSICAL_GPU is occupied: ${TASK_MEMORY}MiB, ${TASK_UTIL}%; nothing launched" >&2
    exit 1
  fi
fi
# Under Slurm the Python runtime resolves the allocated device minor to its UUID
# and checks occupancy before loading models; CUDA/NVML ordinals may differ.
mkdir -p "$TASK_RUN" "$TASK_WORK"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/pipeline_exit.txt"' EXIT
printf 'job_id=%s nvidia_smi_index=%s CUDA_VISIBLE_DEVICES=%s commit=%s\n' \
  "${SLURM_JOB_ID:-manual}" "$TASK_PHYSICAL_GPU" "${CUDA_VISIBLE_DEVICES:-unset}" "$(git rev-parse HEAD)"
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
TASK_COMMON=(
  --manifest "$TASK_DATA/manifests/qwen3_4b_instruct_100k_messages.jsonl"
  --pilot-manifest "$TASK_DATA/runs/prefusion_pilot_20260915/cache/manifest.json"
  --split-dir "$TASK_DATA/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719"
  --eval-cache "$TASK_ROOT/runs/policy_granularity_20260927/cache"
  --models "$TASK_ROOT/models.json" "${TASK_GPU_ARGS[@]}" --seed 1001
)

# Real-model same-forward and canonical AR checks precede the full pilot.
# One training prompt avoids worker-dependent in-flight training overshoot.
timeout --signal=TERM --kill-after=30s 2100s "$TASK_PYTHON" -u scripts/collect_rejected_trace.py \
  "${TASK_COMMON[@]}" --output "$TASK_WORK/serial_smoke" --backup "$TASK_RUN/serial_smoke" \
  --workers 1 --smoke --training-rows 1 --limit-training-prompts 1 \
  --states-per-prompt 2 --max-new-tokens 64 --max-seconds 1800 \
  2>&1 | tee "$TASK_RUN/serial_smoke.log"
if [[ "$TASK_WORKERS" != 1 ]]; then
  timeout --signal=TERM --kill-after=30s 2100s "$TASK_PYTHON" -u scripts/collect_rejected_trace.py \
    "${TASK_COMMON[@]}" --output "$TASK_WORK/parallel_smoke" --backup "$TASK_RUN/parallel_smoke" \
    --workers "$TASK_WORKERS" --smoke --training-rows 1 --limit-training-prompts 1 \
    --states-per-prompt 2 --max-new-tokens 64 --max-seconds 1800 \
    2>&1 | tee "$TASK_RUN/parallel_smoke.log"
  timeout --signal=TERM --kill-after=30s 300s "$TASK_PYTHON" scripts/compare_rejected_trace_caches.py \
    --serial "$TASK_RUN/serial_smoke" --parallel "$TASK_RUN/parallel_smoke" \
    --output "$TASK_RUN/parallel_gate.json"
fi
timeout --signal=TERM --kill-after=30s 300s "$TASK_PYTHON" -u scripts/train_rejected_trace_predictor.py \
  --cache "$TASK_RUN/serial_smoke" --output "$TASK_RUN/training_smoke" \
  --smoke --training-rows 1 --seeds 913 --epochs 1 --bootstrap 100 --benchmark-repeats 0 \
  2>&1 | tee "$TASK_RUN/training_smoke.log"
if [[ "$TASK_MODE" == smoke ]]; then
  echo "Smoke gates passed; no full pilot requested"
  exit 0
fi

# Every prompt shard is checksum-backed up before its receipt is recorded.
timeout --signal=TERM --kill-after=30s 8100s "$TASK_PYTHON" -u scripts/collect_rejected_trace.py \
  "${TASK_COMMON[@]}" --output "$TASK_WORK/cache" --backup "$TASK_RUN/cache" \
  --workers "$TASK_WORKERS" --training-rows 2000 --limit-training-prompts 600 --max-seconds 7200 \
  2>&1 | tee "$TASK_RUN/collection.log"
# The trainer re-audits the complete cache, freezes calibration choices, and
# reports every seed/arm. CPU FP32 scoring is matched across all variants.
timeout --signal=TERM --kill-after=30s 3600s "$TASK_PYTHON" -u scripts/train_rejected_trace_predictor.py \
  --cache "$TASK_RUN/cache" --output "$TASK_RUN/training" "${TASK_GPU_ARGS[@]}" \
  2>&1 | tee "$TASK_RUN/training.log"
echo "Pilot complete: $TASK_RUN/training/report.md"
