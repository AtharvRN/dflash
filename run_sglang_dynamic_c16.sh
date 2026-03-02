#!/usr/bin/env bash
set -euo pipefail

# Dynamic DFLASH block-size experiment focused on concurrency=16.
# Designed for a 2xA100 pod where each block-size server can live on a separate GPU.
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

DYNAMIC_BLOCK_SIZES="${DYNAMIC_BLOCK_SIZES:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}"
# Single-server dynamic mode: only max block-size entry is used if provided.
DYNAMIC_GPU_MAP="${DYNAMIC_GPU_MAP:-}"
DYNAMIC_EWMA_ALPHA="${DYNAMIC_EWMA_ALPHA:-0.20}"
DYNAMIC_SWITCH_MARGIN="${DYNAMIC_SWITCH_MARGIN:-0.02}"
DYNAMIC_REQUIRED_STREAK="${DYNAMIC_REQUIRED_STREAK:-2}"
DYNAMIC_WARMUP_CHUNKS="${DYNAMIC_WARMUP_CHUNKS:-4}"
DYNAMIC_PROBE_INTERVAL="${DYNAMIC_PROBE_INTERVAL:-8}"
DYNAMIC_SCORE_METRIC="${DYNAMIC_SCORE_METRIC:-output_toks_per_s}"

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
  --dynamic-single-server
  --dynamic-block-sizes "${DYNAMIC_BLOCK_SIZES}"
  --dynamic-gpu-map "${DYNAMIC_GPU_MAP}"
  --dynamic-ewma-alpha "${DYNAMIC_EWMA_ALPHA}"
  --dynamic-switch-margin "${DYNAMIC_SWITCH_MARGIN}"
  --dynamic-required-streak "${DYNAMIC_REQUIRED_STREAK}"
  --dynamic-warmup-chunks "${DYNAMIC_WARMUP_CHUNKS}"
  --dynamic-probe-interval "${DYNAMIC_PROBE_INTERVAL}"
  --dynamic-score-metric "${DYNAMIC_SCORE_METRIC}"
  --enable-server-metrics
  --enable-dflash-stage-timing
  --save-call-trace-path "${OUT_TRACE}"
  --output-md "${OUT_MD}"
)

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
