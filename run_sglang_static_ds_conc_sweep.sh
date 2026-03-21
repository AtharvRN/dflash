#!/usr/bin/env bash
set -euo pipefail

# Static-only SGLang sweep for theoretical analysis:
# - baseline (once) + static bs sweep per (dataset, concurrency)
# - no adaptive policy
#
# Produces:
# - per-(dataset,concurrency) matrix summaries
# - one global summary across all runs
#
# Example:
#   RUN_TAG_PREFIX=sg_static_multi_$(date +%Y%m%d_%H%M%S) \
#   DATASETS=gsm8k,math500,humaneval,mbpp \
#   CONCURRENCIES=1,4,8,16,32 \
#   STATIC_BLOCK_SIZES=8,12,16 \
#   FIXED_QUESTION_COUNT=128 \
#   FIXED_QUESTION_OFFSET=0 \
#   bash run_sglang_static_ds_conc_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-sglang_static_ds_conc_$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-gsm8k,math500,humaneval,mbpp}"
CONCURRENCIES="${CONCURRENCIES:-1,4,8,16,32}"
STATIC_BLOCK_SIZES="${STATIC_BLOCK_SIZES:-8,12,16}"

FIXED_QUESTION_COUNT="${FIXED_QUESTION_COUNT:-128}"
FIXED_QUESTION_OFFSET="${FIXED_QUESTION_OFFSET:-0}"
QUESTIONS_PER_CONCURRENCY_BASE="${QUESTIONS_PER_CONCURRENCY_BASE:-8}"
MAX_QUESTIONS_PER_CONFIG="${MAX_QUESTIONS_PER_CONFIG:-128}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
TP_SIZE="${TP_SIZE:-1}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
ENABLE_DFLASH_CYCLE_TRACE="${ENABLE_DFLASH_CYCLE_TRACE:-1}"
ENABLE_DFLASH_STAGE_TIMING="${ENABLE_DFLASH_STAGE_TIMING:-1}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
CPUSET="${CPUSET:-}"

_parse_csv_to_array() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a _parts <<< "${raw}"
  for _p in "${_parts[@]}"; do
    _p="$(echo "${_p}" | xargs)"
    if [[ -n "${_p}" ]]; then
      out_ref+=("${_p}")
    fi
  done
}

run_with_affinity() {
  local cpuset="$1"
  shift
  if [[ -n "${cpuset}" ]]; then
    taskset -c "${cpuset}" "$@"
  else
    "$@"
  fi
}

declare -a _DATASET_LIST
declare -a _CONC_LIST
declare -a _BS_LIST
_parse_csv_to_array "${DATASETS}" _DATASET_LIST
_parse_csv_to_array "${CONCURRENCIES}" _CONC_LIST
_parse_csv_to_array "${STATIC_BLOCK_SIZES}" _BS_LIST

if [[ "${#_DATASET_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: DATASETS is empty." >&2
  exit 1
fi
if [[ "${#_CONC_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: CONCURRENCIES is empty." >&2
  exit 1
fi
if [[ "${#_BS_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: STATIC_BLOCK_SIZES is empty." >&2
  exit 1
fi

for conc in "${_CONC_LIST[@]}"; do
  if ! [[ "${conc}" =~ ^[0-9]+$ ]] || [[ "${conc}" -lt 1 ]]; then
    echo "ERROR: invalid concurrency '${conc}'." >&2
    exit 1
  fi
done
for bs in "${_BS_LIST[@]}"; do
  if ! [[ "${bs}" =~ ^[0-9]+$ ]] || [[ "${bs}" -lt 1 ]]; then
    echo "ERROR: invalid block size '${bs}'." >&2
    exit 1
  fi
done

echo "Running static-only dataset x concurrency sweep"
echo "run_tag_prefix=${RUN_TAG_PREFIX}"
echo "datasets=${DATASETS}"
echo "concurrencies=${CONCURRENCIES}"
echo "static_block_sizes=${STATIC_BLOCK_SIZES}"
echo "fixed_subset=count=${FIXED_QUESTION_COUNT},offset=${FIXED_QUESTION_OFFSET}"
echo "cpuset=${CPUSET:-<none>}"

run_case() {
  local run_tag="$1"
  local log_dir="$2"
  shift 2

  echo "============================================================"
  echo "run_tag=${run_tag}"
  echo "log_dir=${log_dir}"
  echo "============================================================"

  run_with_affinity "${CPUSET}" \
    env \
      RUN_TAG="${run_tag}" \
      LOG_DIR="${log_dir}" \
      TP_SIZE="${TP_SIZE}" \
      DATASET_NAME="${DATASET_NAME}" \
      TARGET_MODEL="${TARGET_MODEL}" \
      DRAFT_MODEL="${DRAFT_MODEL}" \
      CONCURRENCY="${CONCURRENCY}" \
      QUESTIONS_PER_CONCURRENCY_BASE="${QUESTIONS_PER_CONCURRENCY_BASE}" \
      MAX_QUESTIONS_PER_CONFIG="${MAX_QUESTIONS_PER_CONFIG}" \
      FIXED_QUESTION_COUNT="${FIXED_QUESTION_COUNT}" \
      FIXED_QUESTION_OFFSET="${FIXED_QUESTION_OFFSET}" \
      MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
      TIMEOUT_S="${TIMEOUT_S}" \
      ENABLE_DFLASH_CYCLE_TRACE="${ENABLE_DFLASH_CYCLE_TRACE}" \
      ENABLE_DFLASH_STAGE_TIMING="${ENABLE_DFLASH_STAGE_TIMING}" \
      ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR}" \
      "$@" \
      bash "${SCRIPT_DIR}/run_sglang_dynamic_c16.sh"
}

for dataset in "${_DATASET_LIST[@]}"; do
  for conc in "${_CONC_LIST[@]}"; do
    DATASET_NAME="${dataset}"
    CONCURRENCY="${conc}"
    case_prefix="${RUN_TAG_PREFIX}_${dataset}_c${conc}"

    first_bs="${_BS_LIST[0]}"
    run_case \
      "${case_prefix}_baseline_and_static_bs${first_bs}" \
      "logs/${case_prefix}_baseline_and_static_bs${first_bs}" \
      RUN_BASELINE=1 \
      ADAPTIVE_ENABLED=0 \
      DFLASH_BLOCK_SIZE="${first_bs}"

    for bs in "${_BS_LIST[@]:1}"; do
      run_case \
        "${case_prefix}_static_bs${bs}" \
        "logs/${case_prefix}_static_bs${bs}" \
        RUN_BASELINE=0 \
        ADAPTIVE_ENABLED=0 \
        DFLASH_BLOCK_SIZE="${bs}"
    done

    python "${SCRIPT_DIR}/scripts/summarize_sglang_policy_matrix.py" \
      --run-tag-prefix "${case_prefix}" \
      --logs-root logs \
      --output-md "logs/${case_prefix}_matrix_summary.md" \
      --output-csv "logs/${case_prefix}_matrix_summary.csv"
  done
done

echo "Sweep complete. Building global summary..."
python "${SCRIPT_DIR}/scripts/summarize_sglang_static_ds_conc.py" \
  --run-tag-prefix "${RUN_TAG_PREFIX}" \
  --logs-root logs \
  --output-md "logs/${RUN_TAG_PREFIX}_global_summary.md" \
  --output-csv "logs/${RUN_TAG_PREFIX}_global_summary.csv"

echo "Done."
echo "Global summary markdown: logs/${RUN_TAG_PREFIX}_global_summary.md"
echo "Global summary csv: logs/${RUN_TAG_PREFIX}_global_summary.csv"
