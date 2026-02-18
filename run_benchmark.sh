#!/usr/bin/env bash
set -euo pipefail

gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if [[ "${count}" -gt 0 ]]; then
      echo "${count}"
      return
    fi
  fi
  echo "1"
}

FAST="${FAST:-0}"
if [[ "${FAST}" == "1" ]]; then
  DEFAULT_MAX_NEW_TOKENS=256
  DEFAULT_TASKS=("aime25:8" "gsm8k:8" "humaneval:8")
else
  DEFAULT_MAX_NEW_TOKENS=2048
  DEFAULT_TASKS=(
    "gsm8k:128"
    "math500:128"
    "aime24:30"
    "aime25:30"
    "humaneval:164"
    "mbpp:128"
    "livecodebench:128"
    "swe-bench:128"
    "mt-bench:80"
    "alpaca:128"
  )
fi

MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${DEFAULT_MAX_NEW_TOKENS}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SAVE_OUTPUTS_DIR="${SAVE_OUTPUTS_DIR:-}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

if [[ -n "${BLOCK_SIZE:-}" && -z "${BLOCK_SIZES:-}" ]]; then
  BLOCK_SIZES="${BLOCK_SIZE}"
fi

if [[ -n "${BLOCK_SIZES:-}" ]]; then
  read -r -a BLOCK_SIZE_LIST <<< "${BLOCK_SIZES}"
else
  BLOCK_SIZE_LIST=("")
fi

if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_LIST <<< "${TASKS}"
else
  TASK_LIST=("${DEFAULT_TASKS[@]}")
fi

if [[ -n "${NPROC:-}" ]]; then
  NPROC_RESOLVED="${NPROC}"
elif [[ "${FAST}" == "1" ]]; then
  NPROC_RESOLVED=1
else
  NPROC_RESOLVED="$(gpu_count)"
fi

mkdir -p "${LOG_DIR}"
if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
  mkdir -p "${SAVE_OUTPUTS_DIR}"
fi

echo "========================================================"
echo "DFlash benchmark run"
echo "run_tag=${RUN_TAG}"
echo "fast=${FAST}"
echo "nproc=${NPROC_RESOLVED}"
echo "model=${MODEL}"
echo "draft=${DRAFT}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "temperature=${TEMPERATURE}"
echo "tasks=${TASK_LIST[*]}"
echo "skip_baseline=${SKIP_BASELINE}"
if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
  echo "save_outputs_dir=${SAVE_OUTPUTS_DIR}"
fi
if [[ ${#BLOCK_SIZE_LIST[@]} -eq 1 && -z "${BLOCK_SIZE_LIST[0]}" ]]; then
  echo "block_sizes=default_from_model"
else
  echo "block_sizes=${BLOCK_SIZE_LIST[*]}"
fi
echo "logs_dir=${LOG_DIR}"
echo "========================================================"

run_idx=0
for task in "${TASK_LIST[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "${task}"
  if [[ -z "${DATASET_NAME}" || -z "${MAX_SAMPLES}" ]]; then
    echo "Invalid TASKS entry: '${task}' (expected dataset:max_samples)" >&2
    exit 1
  fi

  for bs in "${BLOCK_SIZE_LIST[@]}"; do
    master_port=$((MASTER_PORT_BASE + run_idx))
    run_idx=$((run_idx + 1))
    bs_tag="auto"
    if [[ -n "${bs}" ]]; then
      bs_tag="${bs}"
    fi

    log_path="${LOG_DIR}/${RUN_TAG}_${DATASET_NAME}_bs${bs_tag}.log"
    cmd=(
      torchrun
      --nproc_per_node="${NPROC_RESOLVED}"
      --master_port="${master_port}"
      benchmark.py
      --dataset "${DATASET_NAME}"
      --max-samples "${MAX_SAMPLES}"
      --model-name-or-path "${MODEL}"
      --draft-name-or-path "${DRAFT}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature "${TEMPERATURE}"
    )
    if [[ -n "${bs}" ]]; then
      cmd+=(--block-size "${bs}")
    fi
    if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
      cmd+=(--save-outputs-path "${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET_NAME}_bs${bs_tag}.jsonl")
    fi
    if [[ "${SKIP_BASELINE}" == "1" ]]; then
      cmd+=(--skip-baseline)
    fi

    echo "--------------------------------------------------------" | tee "${log_path}"
    printf "Launch: " | tee -a "${log_path}"
    printf "%q " "${cmd[@]}" | tee -a "${log_path}"
    printf "\n" | tee -a "${log_path}"
    echo "log=${log_path}" | tee -a "${log_path}"
    echo "--------------------------------------------------------" | tee -a "${log_path}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    set +e
    "${cmd[@]}" 2>&1 | tee -a "${log_path}"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "${status}" -ne 0 ]]; then
      echo "Run failed (dataset=${DATASET_NAME}, block_size=${bs_tag}, status=${status})." | tee -a "${log_path}"
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit "${status}"
      fi
    fi
  done
done
