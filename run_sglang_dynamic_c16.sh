#!/usr/bin/env bash
set -euo pipefail

# True server-side adaptive DFLASH block-size experiment focused on concurrency=16.
# This uses one SGLang DFLASH server and lets the worker adapt runtime block size
# per request from acceptance history.
#
# Example:
#   RUN_TAG=sg_dynamic_c16_$(date +%Y%m%d_%H%M%S) bash run_sglang_dynamic_c16.sh

DATASET_NAME="${DATASET_NAME:-gsm8k}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
CONCURRENCY="${CONCURRENCY:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
QUESTIONS_PER_CONCURRENCY_BASE="${QUESTIONS_PER_CONCURRENCY_BASE:-16}"
MAX_QUESTIONS_PER_CONFIG="${MAX_QUESTIONS_PER_CONFIG:-256}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
TIMEOUT_S="${TIMEOUT_S:-3600}"

RUN_BASELINE="${RUN_BASELINE:-1}"
BATCH_REQUESTS="${BATCH_REQUESTS:-1}"

DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-16}"
ADAPTIVE_ENABLED="${ADAPTIVE_ENABLED:-1}"
ADAPTIVE_RHO="${ADAPTIVE_RHO:-0.30}"
ADAPTIVE_DELTA="${ADAPTIVE_DELTA:-1.0}"
ADAPTIVE_K_MIN="${ADAPTIVE_K_MIN:-1}"
ADAPTIVE_K_MAX="${ADAPTIVE_K_MAX:-16}"
ADAPTIVE_LOW_ACCEPT_THRESHOLD="${ADAPTIVE_LOW_ACCEPT_THRESHOLD:-0.35}"
ADAPTIVE_LOW_ACCEPT_STREAK="${ADAPTIVE_LOW_ACCEPT_STREAK:-2}"

RUN_TAG="${RUN_TAG:-sglang_dynamic_c16_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

OUT_MD="${LOG_DIR}/${RUN_TAG}.md"
OUT_TRACE="${LOG_DIR}/${RUN_TAG}_calls.jsonl"
OUT_LOG="${LOG_DIR}/${RUN_TAG}.log"

cmd=(
  python benchmark_sglang.py
  --dataset-name "${DATASET_NAME}"
  --target-model "${TARGET_MODEL}"
  --draft-model "${DRAFT_MODEL}"
  --tp-size "${TP_SIZE}"
  --attention-backends "${ATTENTION_BACKEND}"
  --concurrencies "${CONCURRENCY}"
  --questions-per-concurrency-base "${QUESTIONS_PER_CONCURRENCY_BASE}"
  --max-questions-per-config "${MAX_QUESTIONS_PER_CONFIG}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --timeout-s "${TIMEOUT_S}"
  --speculative-algorithm DFLASH
  --speculative-dflash-block-size "${DFLASH_BLOCK_SIZE}"
  --speculative-dflash-adaptive-rho "${ADAPTIVE_RHO}"
  --speculative-dflash-adaptive-delta "${ADAPTIVE_DELTA}"
  --speculative-dflash-adaptive-k-min "${ADAPTIVE_K_MIN}"
  --speculative-dflash-adaptive-k-max "${ADAPTIVE_K_MAX}"
  --speculative-dflash-adaptive-low-accept-threshold "${ADAPTIVE_LOW_ACCEPT_THRESHOLD}"
  --speculative-dflash-adaptive-low-accept-streak "${ADAPTIVE_LOW_ACCEPT_STREAK}"
  --enable-server-metrics
  --enable-dflash-stage-timing
  --save-call-trace-path "${OUT_TRACE}"
  --output-md "${OUT_MD}"
)

if [[ "${ADAPTIVE_ENABLED}" == "1" ]]; then
  cmd+=(--speculative-dflash-adaptive-block-size)
fi

if [[ "${BATCH_REQUESTS}" == "1" ]]; then
  cmd+=(--batch-requests)
fi
if [[ "${RUN_BASELINE}" == "0" ]]; then
  cmd+=(--skip-baseline)
fi

{
  echo "Running dynamic SGLang DFLASH c=${CONCURRENCY}"
  echo "run_tag=${RUN_TAG}"
  printf "command: "
  printf "%q " "${cmd[@]}"
  printf "\n"
} | tee "${OUT_LOG}"

"${cmd[@]}" 2>&1 | tee -a "${OUT_LOG}"

echo "Done."
echo "Markdown: ${OUT_MD}"
echo "Call trace: ${OUT_TRACE}"
echo "Log: ${OUT_LOG}"
