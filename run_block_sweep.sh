#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-aime25}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
RUN_TAG="${RUN_TAG:-sweep_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_DIR}/summary.csv}"

if [[ -n "${BLOCK_SIZES:-}" ]]; then
  read -r -a BS_LIST <<< "${BLOCK_SIZES}"
else
  BS_LIST=(4 8 12 16)
fi

if [[ -n "${NPROC:-}" ]]; then
  NPROC_RESOLVED="${NPROC}"
else
  NPROC_RESOLVED=1
fi

mkdir -p "${LOG_DIR}"
echo "dataset,max_samples,block_size,speedup,tau,acceptance_histogram,log_path" > "${SUMMARY_CSV}"

echo "Running block-size sweep"
echo "dataset=${DATASET} max_samples=${MAX_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}"
echo "block_sizes=${BS_LIST[*]} nproc=${NPROC_RESOLVED} logs=${LOG_DIR}"

run_idx=0
for bs in "${BS_LIST[@]}"; do
  master_port=$((MASTER_PORT_BASE + run_idx))
  run_idx=$((run_idx + 1))
  log_path="${LOG_DIR}/${DATASET}_bs${bs}.log"
  cmd=(
    torchrun
    --nproc_per_node="${NPROC_RESOLVED}"
    --master_port="${master_port}"
    benchmark.py
    --dataset "${DATASET}"
    --max-samples "${MAX_SAMPLES}"
    --model-name-or-path "${MODEL}"
    --draft-name-or-path "${DRAFT}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --block-size "${bs}"
  )

  echo "--------------------------------------------------------" | tee "${log_path}"
  printf "Launch: " | tee -a "${log_path}"
  printf "%q " "${cmd[@]}" | tee -a "${log_path}"
  printf "\n" | tee -a "${log_path}"
  echo "--------------------------------------------------------" | tee -a "${log_path}"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "${DATASET},${MAX_SAMPLES},${bs},ERROR,ERROR,ERROR,${log_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  speedup="$(grep -Eo 'Decoding speedup: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  tau="$(grep -Eo 'Average Acceptance length: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $4}')"
  histogram="$(grep -E 'Acceptance length histogram:' "${log_path}" | tail -1 | sed 's/^.*Acceptance length histogram: //')"
  speedup="${speedup:-NA}"
  tau="${tau:-NA}"
  histogram="${histogram:-NA}"
  histogram="${histogram//\"/\"\"}"
  echo "${DATASET},${MAX_SAMPLES},${bs},${speedup},${tau},\"${histogram}\",${log_path}" >> "${SUMMARY_CSV}"
done

echo "Sweep complete. Summary: ${SUMMARY_CSV}"
