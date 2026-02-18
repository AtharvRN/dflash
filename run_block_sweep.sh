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
echo "dataset,max_samples,block_size,speedup,tau,gpu_name,cuda_version,torch_version,baseline_total_wall_s,speculative_total_wall_s,baseline_tokens_per_sec,speculative_tokens_per_sec,baseline_tpot,speculative_tpot,baseline_ttft,speculative_ttft,acceptance_histogram,log_path,output_jsonl_path" > "${SUMMARY_CSV}"

echo "Running block-size sweep"
echo "dataset=${DATASET} max_samples=${MAX_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}"
echo "block_sizes=${BS_LIST[*]} nproc=${NPROC_RESOLVED} logs=${LOG_DIR}"
echo "skip_baseline=${SKIP_BASELINE}"
if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
  echo "save_outputs_dir=${SAVE_OUTPUTS_DIR}"
fi

run_idx=0
BASELINE_TPOT="NA"
BASELINE_TTFT="NA"
BASELINE_TOTAL_WALL_S="NA"
BASELINE_TOKS_PER_SEC="NA"
BASELINE_GPU_NAME="NA"
BASELINE_CUDA_VERSION="NA"
BASELINE_TORCH_VERSION="NA"
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
    BASELINE_TTFT="$(grep -Eo 'Baseline TTFT: [0-9.]+$' "${baseline_log_path}" | tail -1 | awk '{print $3}')"
    BASELINE_TOTAL_WALL_S="$(grep -Eo 'Baseline total_wall_s: [0-9.]+$' "${baseline_log_path}" | tail -1 | awk '{print $3}')"
    BASELINE_TOKS_PER_SEC="$(grep -Eo 'Baseline tokens_per_sec: [0-9.]+$' "${baseline_log_path}" | tail -1 | awk '{print $3}')"
    BASELINE_GPU_NAME="$(grep -E '^Hardware GPU:' "${baseline_log_path}" | tail -1 | sed 's/^Hardware GPU: //')"
    BASELINE_CUDA_VERSION="$(grep -E '^Hardware CUDA:' "${baseline_log_path}" | tail -1 | sed 's/^Hardware CUDA: //')"
    BASELINE_TORCH_VERSION="$(grep -E '^Hardware Torch:' "${baseline_log_path}" | tail -1 | sed 's/^Hardware Torch: //')"
    BASELINE_TTFT="${BASELINE_TTFT:-NA}"
    BASELINE_TOTAL_WALL_S="${BASELINE_TOTAL_WALL_S:-NA}"
    BASELINE_TOKS_PER_SEC="${BASELINE_TOKS_PER_SEC:-NA}"
    BASELINE_GPU_NAME="${BASELINE_GPU_NAME:-NA}"
    BASELINE_CUDA_VERSION="${BASELINE_CUDA_VERSION:-NA}"
    BASELINE_TORCH_VERSION="${BASELINE_TORCH_VERSION:-NA}"
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
    echo "${DATASET},${MAX_SAMPLES},${bs},DRY_RUN,DRY_RUN,NA,NA,NA,${BASELINE_TOTAL_WALL_S},DRY_RUN,${BASELINE_TOKS_PER_SEC},DRY_RUN,${BASELINE_TPOT},DRY_RUN,${BASELINE_TTFT},DRY_RUN,DRY_RUN,${log_path},${output_csv_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    output_csv_path="${output_path:-NA}"
    echo "${DATASET},${MAX_SAMPLES},${bs},ERROR,ERROR,NA,NA,NA,${BASELINE_TOTAL_WALL_S},ERROR,${BASELINE_TOKS_PER_SEC},ERROR,${BASELINE_TPOT},ERROR,${BASELINE_TTFT},ERROR,ERROR,${log_path},${output_csv_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  speedup="$(grep -Eo 'Decoding speedup: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  tau="$(grep -Eo 'Average Acceptance length: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $4}')"
  speculative_total_wall_s="$(grep -Eo 'Speculative total_wall_s: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  speculative_tokens_per_sec="$(grep -Eo 'Speculative tokens_per_sec: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  speculative_tpot="$(grep -Eo 'Speculative TPOT: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  speculative_ttft="$(grep -Eo 'Speculative TTFT: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  gpu_name="$(grep -E '^Hardware GPU:' "${log_path}" | tail -1 | sed 's/^Hardware GPU: //')"
  cuda_version="$(grep -E '^Hardware CUDA:' "${log_path}" | tail -1 | sed 's/^Hardware CUDA: //')"
  torch_version="$(grep -E '^Hardware Torch:' "${log_path}" | tail -1 | sed 's/^Hardware Torch: //')"
  baseline_total_wall_s="$(grep -Eo 'Baseline total_wall_s: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  baseline_tokens_per_sec="$(grep -Eo 'Baseline tokens_per_sec: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  baseline_tpot_from_log="$(grep -Eo 'Baseline TPOT: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  baseline_ttft_from_log="$(grep -Eo 'Baseline TTFT: [0-9.]+$' "${log_path}" | tail -1 | awk '{print $3}')"
  if [[ -z "${baseline_tpot_from_log}" ]]; then
    baseline_tpot_from_log="${BASELINE_TPOT}"
  fi
  if [[ -z "${baseline_ttft_from_log}" ]]; then
    baseline_ttft_from_log="${BASELINE_TTFT}"
  fi
  if [[ -z "${baseline_total_wall_s}" ]]; then
    baseline_total_wall_s="${BASELINE_TOTAL_WALL_S}"
  fi
  if [[ -z "${baseline_tokens_per_sec}" ]]; then
    baseline_tokens_per_sec="${BASELINE_TOKS_PER_SEC}"
  fi
  if [[ -z "${gpu_name}" ]]; then
    gpu_name="${BASELINE_GPU_NAME}"
  fi
  if [[ -z "${cuda_version}" ]]; then
    cuda_version="${BASELINE_CUDA_VERSION}"
  fi
  if [[ -z "${torch_version}" ]]; then
    torch_version="${BASELINE_TORCH_VERSION}"
  fi
  if [[ "${SKIP_BASELINE}" == "1" && -n "${speculative_tpot}" && "${BASELINE_TPOT}" != "NA" && "${BASELINE_TPOT}" != "DRY_RUN" ]]; then
    speedup="$(awk -v b="${BASELINE_TPOT}" -v s="${speculative_tpot}" 'BEGIN { if (s > 0) printf "%.2f", b / s; else print "NA" }')"
  fi
  histogram="$(grep -E 'Acceptance length histogram:' "${log_path}" | tail -1 | sed 's/^.*Acceptance length histogram: //')"
  speedup="${speedup:-NA}"
  tau="${tau:-NA}"
  speculative_total_wall_s="${speculative_total_wall_s:-NA}"
  speculative_tokens_per_sec="${speculative_tokens_per_sec:-NA}"
  speculative_tpot="${speculative_tpot:-NA}"
  speculative_ttft="${speculative_ttft:-NA}"
  gpu_name="${gpu_name:-NA}"
  cuda_version="${cuda_version:-NA}"
  torch_version="${torch_version:-NA}"
  baseline_total_wall_s="${baseline_total_wall_s:-NA}"
  baseline_tokens_per_sec="${baseline_tokens_per_sec:-NA}"
  baseline_tpot_from_log="${baseline_tpot_from_log:-NA}"
  baseline_ttft_from_log="${baseline_ttft_from_log:-NA}"
  histogram="${histogram:-NA}"
  gpu_name="${gpu_name//\"/\"\"}"
  cuda_version="${cuda_version//\"/\"\"}"
  torch_version="${torch_version//\"/\"\"}"
  histogram="${histogram//\"/\"\"}"
  output_csv_path="${output_path:-NA}"
  echo "${DATASET},${MAX_SAMPLES},${bs},${speedup},${tau},\"${gpu_name}\",\"${cuda_version}\",\"${torch_version}\",${baseline_total_wall_s},${speculative_total_wall_s},${baseline_tokens_per_sec},${speculative_tokens_per_sec},${baseline_tpot_from_log},${speculative_tpot},${baseline_ttft_from_log},${speculative_ttft},\"${histogram}\",${log_path},${output_csv_path}" >> "${SUMMARY_CSV}"
done

echo "Sweep complete. Summary: ${SUMMARY_CSV}"
