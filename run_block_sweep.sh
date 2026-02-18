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
SAVE_OUTPUTS_DIR="${SAVE_OUTPUTS_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

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
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}"
if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
  mkdir -p "${SAVE_OUTPUTS_DIR}"
fi
echo "dataset,max_samples,block_size,speedup,tau,baseline_tpot,speculative_tpot,acceptance_histogram,log_path,output_jsonl_path" > "${SUMMARY_CSV}"

echo "Running block-size sweep"
echo "dataset=${DATASET} max_samples=${MAX_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}"
echo "block_sizes=${BS_LIST[*]} nproc=${NPROC_RESOLVED} logs=${LOG_DIR}"
echo "skip_baseline=${SKIP_BASELINE}"
if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
  echo "save_outputs_dir=${SAVE_OUTPUTS_DIR}"
fi

run_idx=0
BASELINE_TPOT="NA"
if [[ "${SKIP_BASELINE}" == "1" ]]; then
  baseline_log_path="${LOG_DIR}/${DATASET}_baseline_bs1.log"
  baseline_output_path=""
  if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
    baseline_output_path="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_baseline_bs1.jsonl"
  fi
  if [[ "${NPROC_RESOLVED}" -eq 1 ]]; then
    baseline_cmd=(
      "${PYTHON_BIN}"
      benchmark.py
      --dataset "${DATASET}"
      --max-samples "${MAX_SAMPLES}"
      --model-name-or-path "${MODEL}"
      --draft-name-or-path "${DRAFT}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature "${TEMPERATURE}"
      --block-size 1
    )
  else
    baseline_cmd=(
      torchrun
      --nproc_per_node="${NPROC_RESOLVED}"
      --master_port="${MASTER_PORT_BASE}"
      benchmark.py
      --dataset "${DATASET}"
      --max-samples "${MAX_SAMPLES}"
      --model-name-or-path "${MODEL}"
      --draft-name-or-path "${DRAFT}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature "${TEMPERATURE}"
      --block-size 1
    )
  fi
  if [[ -n "${baseline_output_path}" ]]; then
    baseline_cmd+=(--save-outputs-path "${baseline_output_path}")
  fi

  echo "--------------------------------------------------------" | tee "${baseline_log_path}"
  printf "Launch baseline: " | tee -a "${baseline_log_path}"
  printf "%q " "${baseline_cmd[@]}" | tee -a "${baseline_log_path}"
  printf "\n" | tee -a "${baseline_log_path}"
  echo "--------------------------------------------------------" | tee -a "${baseline_log_path}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    BASELINE_TPOT="DRY_RUN"
  else
    set +e
    "${baseline_cmd[@]}" 2>&1 | tee -a "${baseline_log_path}"
    baseline_status=${PIPESTATUS[0]}
    set -e
    if [[ "${baseline_status}" -ne 0 ]]; then
      echo "Baseline run failed (status=${baseline_status})." | tee -a "${baseline_log_path}"
      exit "${baseline_status}"
    fi
    BASELINE_TPOT="$(grep -Eo 'Baseline TPOT: [0-9.]+$' "${baseline_log_path}" | tail -1 | awk '{print $3}')"
    if [[ -z "${BASELINE_TPOT}" ]]; then
      BASELINE_TPOT="$(grep -Eo 'Speculative TPOT: [0-9.]+$' "${baseline_log_path}" | tail -1 | awk '{print $3}')"
    fi
    if [[ -z "${BASELINE_TPOT}" ]]; then
      echo "Could not parse baseline TPOT from ${baseline_log_path}" >&2
      exit 1
    fi
    echo "Shared baseline TPOT=${BASELINE_TPOT}"
  fi
  run_idx=1
fi

for bs in "${BS_LIST[@]}"; do
  master_port=$((MASTER_PORT_BASE + run_idx))
  run_idx=$((run_idx + 1))
  log_path="${LOG_DIR}/${DATASET}_bs${bs}.log"
  output_path=""
  if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
    output_path="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_bs${bs}.jsonl"
  fi
  if [[ "${NPROC_RESOLVED}" -eq 1 ]]; then
    cmd=(
      "${PYTHON_BIN}"
      benchmark.py
      --dataset "${DATASET}"
      --max-samples "${MAX_SAMPLES}"
      --model-name-or-path "${MODEL}"
      --draft-name-or-path "${DRAFT}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature "${TEMPERATURE}"
      --block-size "${bs}"
    )
  else
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
  fi
  if [[ -n "${output_path}" ]]; then
    cmd+=(--save-outputs-path "${output_path}")
  fi
  if [[ "${SKIP_BASELINE}" == "1" ]]; then
    cmd+=(--skip-baseline)
  fi

  echo "--------------------------------------------------------" | tee "${log_path}"
  printf "Launch: " | tee -a "${log_path}"
  printf "%q " "${cmd[@]}" | tee -a "${log_path}"
  printf "\n" | tee -a "${log_path}"
  echo "--------------------------------------------------------" | tee -a "${log_path}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    output_csv_path="${output_path:-NA}"
    echo "${DATASET},${MAX_SAMPLES},${bs},DRY_RUN,DRY_RUN,${BASELINE_TPOT},DRY_RUN,DRY_RUN,${log_path},${output_csv_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    output_csv_path="${output_path:-NA}"
    echo "${DATASET},${MAX_SAMPLES},${bs},ERROR,ERROR,${BASELINE_TPOT},ERROR,ERROR,${log_path},${output_csv_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  speedup="$(grep -Eo 'Decoding speedup: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  tau="$(grep -Eo 'Average Acceptance length: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $4}')"
  speculative_tpot="$(grep -Eo 'Speculative TPOT: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  baseline_tpot_from_log="$(grep -Eo 'Baseline TPOT: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  if [[ -z "${baseline_tpot_from_log}" ]]; then
    baseline_tpot_from_log="${BASELINE_TPOT}"
  fi
  if [[ "${SKIP_BASELINE}" == "1" && -n "${speculative_tpot}" && "${BASELINE_TPOT}" != "NA" && "${BASELINE_TPOT}" != "DRY_RUN" ]]; then
    speedup="$(awk -v b="${BASELINE_TPOT}" -v s="${speculative_tpot}" 'BEGIN { if (s > 0) printf "%.2f", b / s; else print "NA" }')"
  fi
  histogram="$(grep -E 'Acceptance length histogram:' "${log_path}" | tail -1 | sed 's/^.*Acceptance length histogram: //')"
  speedup="${speedup:-NA}"
  tau="${tau:-NA}"
  speculative_tpot="${speculative_tpot:-NA}"
  baseline_tpot_from_log="${baseline_tpot_from_log:-NA}"
  histogram="${histogram:-NA}"
  histogram="${histogram//\"/\"\"}"
  output_csv_path="${output_path:-NA}"
  echo "${DATASET},${MAX_SAMPLES},${bs},${speedup},${tau},${baseline_tpot_from_log},${speculative_tpot},\"${histogram}\",${log_path},${output_csv_path}" >> "${SUMMARY_CSV}"
done

echo "Sweep complete. Summary: ${SUMMARY_CSV}"
