#!/usr/bin/env bash
set -euo pipefail

# True server-side adaptive DFLASH block-size experiment focused on concurrency=16.
# This uses one SGLang DFLASH server and lets the worker adapt runtime block size
# per request from acceptance history.
#
# Example:
#   RUN_TAG=sg_dynamic_c16_$(date +%Y%m%d_%H%M%S) bash run_sglang_dynamic_c16.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_NAME="${DATASET_NAME:-gsm8k}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
CONCURRENCY="${CONCURRENCY:-16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
QUESTIONS_PER_CONCURRENCY_BASE="${QUESTIONS_PER_CONCURRENCY_BASE:-16}"
MAX_QUESTIONS_PER_CONFIG="${MAX_QUESTIONS_PER_CONFIG:-256}"
FIXED_QUESTION_COUNT="${FIXED_QUESTION_COUNT:-0}"
FIXED_QUESTION_OFFSET="${FIXED_QUESTION_OFFSET:-0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

RUN_BASELINE="${RUN_BASELINE:-1}"
BATCH_REQUESTS="${BATCH_REQUESTS:-1}"

DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-16}"
ADAPTIVE_ENABLED="${ADAPTIVE_ENABLED:-1}"
ADAPTIVE_ALGO="${ADAPTIVE_ALGO:-ewma}"
ADAPTIVE_RHO="${ADAPTIVE_RHO:-0.30}"
ADAPTIVE_DELTA="${ADAPTIVE_DELTA:-1.0}"
ADAPTIVE_REWARD_MODE="${ADAPTIVE_REWARD_MODE:-accept_length}"
ADAPTIVE_UCB_C="${ADAPTIVE_UCB_C:-1.0}"
ADAPTIVE_UCB_DELTA="${ADAPTIVE_UCB_DELTA:-0.05}"
ADAPTIVE_LINUCB_ALPHA="${ADAPTIVE_LINUCB_ALPHA:-1.0}"
ADAPTIVE_LINUCB_LAMBDA="${ADAPTIVE_LINUCB_LAMBDA:-1.0}"
ADAPTIVE_PROXY_CYCLE_MS="${ADAPTIVE_PROXY_CYCLE_MS:-}"
ADAPTIVE_PROXY_POWERLAW_A="${ADAPTIVE_PROXY_POWERLAW_A:-0.0}"
ADAPTIVE_PROXY_POWERLAW_C_EXP="${ADAPTIVE_PROXY_POWERLAW_C_EXP:-0.430}"
ADAPTIVE_PROXY_POWERLAW_K_EXP="${ADAPTIVE_PROXY_POWERLAW_K_EXP:-0.160}"
ADAPTIVE_PROXY_TAU_EXP="${ADAPTIVE_PROXY_TAU_EXP:-1.0}"
ADAPTIVE_PROXY_TIME_EXP="${ADAPTIVE_PROXY_TIME_EXP:-0.2}"
ADAPTIVE_ALLOW_PROXY_FALLBACK="${ADAPTIVE_ALLOW_PROXY_FALLBACK:-0}"
ADAPTIVE_K_MIN="${ADAPTIVE_K_MIN:-1}"
ADAPTIVE_K_MAX="${ADAPTIVE_K_MAX:-16}"
ADAPTIVE_K_START="${ADAPTIVE_K_START:-}"
ADAPTIVE_BLOCK_BUCKETS="${ADAPTIVE_BLOCK_BUCKETS:-}"
ADAPTIVE_LOW_ACCEPT_THRESHOLD="${ADAPTIVE_LOW_ACCEPT_THRESHOLD:-0.35}"
ADAPTIVE_LOW_ACCEPT_STREAK="${ADAPTIVE_LOW_ACCEPT_STREAK:-2}"
ADAPTIVE_HIGH_ACCEPT_THRESHOLD="${ADAPTIVE_HIGH_ACCEPT_THRESHOLD:-0.90}"
ADAPTIVE_HIGH_ACCEPT_STREAK="${ADAPTIVE_HIGH_ACCEPT_STREAK:-2}"
ADAPTIVE_COOLDOWN_CYCLES="${ADAPTIVE_COOLDOWN_CYCLES:-1}"
ENABLE_DFLASH_CYCLE_TRACE="${ENABLE_DFLASH_CYCLE_TRACE:-1}"
ENABLE_DFLASH_STAGE_TIMING="${ENABLE_DFLASH_STAGE_TIMING:-1}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_MONITOR_INTERVAL_S="${GPU_MONITOR_INTERVAL_S:-1}"

RUN_TAG="${RUN_TAG:-sglang_dynamic_c16_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

OUT_MD="${LOG_DIR}/${RUN_TAG}.md"
OUT_TRACE="${LOG_DIR}/${RUN_TAG}_calls.jsonl"
OUT_LOG="${LOG_DIR}/${RUN_TAG}.log"
OUT_TRACE_SUMMARY_MD="${LOG_DIR}/${RUN_TAG}_calls_summary.md"
OUT_GPU_METRICS_CSV="${LOG_DIR}/${RUN_TAG}_gpu_metrics.csv"
OUT_GPU_SUMMARY_MD="${LOG_DIR}/${RUN_TAG}_gpu_metrics_summary.md"
OUT_GPU_SUMMARY_JSON="${LOG_DIR}/${RUN_TAG}_gpu_metrics_summary.json"

GPU_MONITOR_PID=""

start_gpu_monitor() {
  if [[ "${ENABLE_GPU_MONITOR}" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-monitor] nvidia-smi not found; skipping GPU metrics capture." | tee -a "${OUT_LOG}"
    return
  fi
  bash "${SCRIPT_DIR}/scripts/record_gpu_metrics.sh" \
    "${OUT_GPU_METRICS_CSV}" \
    "${GPU_MONITOR_INTERVAL_S}" >> "${OUT_LOG}" 2>&1 &
  GPU_MONITOR_PID=$!
  echo "[gpu-monitor] started pid=${GPU_MONITOR_PID}, csv=${OUT_GPU_METRICS_CSV}, interval_s=${GPU_MONITOR_INTERVAL_S}" | tee -a "${OUT_LOG}"
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    echo "[gpu-monitor] stopped pid=${GPU_MONITOR_PID}" | tee -a "${OUT_LOG}"
    GPU_MONITOR_PID=""
  fi
}

cleanup() {
  stop_gpu_monitor
}

trap cleanup EXIT

if ! [[ "${FIXED_QUESTION_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: FIXED_QUESTION_COUNT must be a non-negative integer. Got '${FIXED_QUESTION_COUNT}'." >&2
  exit 1
fi
if ! [[ "${FIXED_QUESTION_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: FIXED_QUESTION_OFFSET must be a non-negative integer. Got '${FIXED_QUESTION_OFFSET}'." >&2
  exit 1
fi
if [[ "${FIXED_QUESTION_COUNT}" -eq 0 && "${FIXED_QUESTION_OFFSET}" -ne 0 ]]; then
  echo "ERROR: FIXED_QUESTION_OFFSET requires FIXED_QUESTION_COUNT > 0." >&2
  exit 1
fi

# Guard against silent legacy proxy fallback:
# reward_mode=throughput_proxy with no cycle-ms map and no power-law estimator
# degrades to proxy reward tau/k (block-size units), which is usually unintended.
if [[ "${ADAPTIVE_ENABLED}" == "1" && "${ADAPTIVE_REWARD_MODE}" == "throughput_proxy" ]]; then
  _powerlaw_enabled="$(awk "BEGIN { print (${ADAPTIVE_PROXY_POWERLAW_A} > 0.0) ? 1 : 0 }")"
  if [[ -z "${ADAPTIVE_PROXY_CYCLE_MS}" && "${_powerlaw_enabled}" != "1" ]]; then
    if [[ "${ADAPTIVE_ALLOW_PROXY_FALLBACK}" != "1" ]]; then
      echo "ERROR: throughput_proxy reward requires either:" >&2
      echo "  1) ADAPTIVE_PROXY_CYCLE_MS (recommended, explicit lookup), or" >&2
      echo "  2) ADAPTIVE_PROXY_POWERLAW_A > 0 (power-law estimator)." >&2
      echo "Current config would silently fall back to tau/k proxy units." >&2
      echo "If you really want legacy fallback, set ADAPTIVE_ALLOW_PROXY_FALLBACK=1." >&2
      exit 1
    fi
    echo "WARNING: Using legacy throughput_proxy fallback (tau/k units)." | tee -a "${OUT_LOG}"
  fi
fi

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
  --speculative-dflash-adaptive-algo "${ADAPTIVE_ALGO}"
  --speculative-dflash-adaptive-rho "${ADAPTIVE_RHO}"
  --speculative-dflash-adaptive-delta "${ADAPTIVE_DELTA}"
  --speculative-dflash-adaptive-reward-mode "${ADAPTIVE_REWARD_MODE}"
  --speculative-dflash-adaptive-ucb-c "${ADAPTIVE_UCB_C}"
  --speculative-dflash-adaptive-ucb-delta "${ADAPTIVE_UCB_DELTA}"
  --speculative-dflash-adaptive-linucb-alpha "${ADAPTIVE_LINUCB_ALPHA}"
  --speculative-dflash-adaptive-linucb-lambda "${ADAPTIVE_LINUCB_LAMBDA}"
  --speculative-dflash-adaptive-proxy-powerlaw-a "${ADAPTIVE_PROXY_POWERLAW_A}"
  --speculative-dflash-adaptive-proxy-powerlaw-c-exp "${ADAPTIVE_PROXY_POWERLAW_C_EXP}"
  --speculative-dflash-adaptive-proxy-powerlaw-k-exp "${ADAPTIVE_PROXY_POWERLAW_K_EXP}"
  --speculative-dflash-adaptive-proxy-tau-exp "${ADAPTIVE_PROXY_TAU_EXP}"
  --speculative-dflash-adaptive-proxy-time-exp "${ADAPTIVE_PROXY_TIME_EXP}"
  --speculative-dflash-adaptive-k-min "${ADAPTIVE_K_MIN}"
  --speculative-dflash-adaptive-k-max "${ADAPTIVE_K_MAX}"
  --speculative-dflash-adaptive-low-accept-threshold "${ADAPTIVE_LOW_ACCEPT_THRESHOLD}"
  --speculative-dflash-adaptive-low-accept-streak "${ADAPTIVE_LOW_ACCEPT_STREAK}"
  --speculative-dflash-adaptive-high-accept-threshold "${ADAPTIVE_HIGH_ACCEPT_THRESHOLD}"
  --speculative-dflash-adaptive-high-accept-streak "${ADAPTIVE_HIGH_ACCEPT_STREAK}"
  --speculative-dflash-adaptive-cooldown-cycles "${ADAPTIVE_COOLDOWN_CYCLES}"
  --enable-server-metrics
  --save-call-trace-path "${OUT_TRACE}"
  --output-md "${OUT_MD}"
)

if [[ "${FIXED_QUESTION_COUNT}" -gt 0 ]]; then
  cmd+=(
    --fixed-question-count "${FIXED_QUESTION_COUNT}"
    --fixed-question-offset "${FIXED_QUESTION_OFFSET}"
  )
fi

if [[ "${ADAPTIVE_ENABLED}" == "1" ]]; then
  cmd+=(--speculative-dflash-adaptive-block-size)
fi
if [[ -n "${ADAPTIVE_K_START}" ]]; then
  cmd+=(--speculative-dflash-adaptive-k-start "${ADAPTIVE_K_START}")
fi
if [[ -n "${ADAPTIVE_BLOCK_BUCKETS}" ]]; then
  cmd+=(--speculative-dflash-adaptive-block-buckets "${ADAPTIVE_BLOCK_BUCKETS}")
fi
if [[ -n "${ADAPTIVE_PROXY_CYCLE_MS}" ]]; then
  cmd+=(--speculative-dflash-adaptive-proxy-cycle-ms "${ADAPTIVE_PROXY_CYCLE_MS}")
fi

if [[ "${BATCH_REQUESTS}" == "1" ]]; then
  cmd+=(--batch-requests)
fi
if [[ "${RUN_BASELINE}" == "0" ]]; then
  cmd+=(--skip-baseline)
fi
if [[ "${ENABLE_DFLASH_CYCLE_TRACE}" == "1" ]]; then
  cmd+=(--enable-dflash-cycle-trace)
fi
if [[ "${ENABLE_DFLASH_STAGE_TIMING}" == "1" ]]; then
  cmd+=(--enable-dflash-stage-timing)
fi
if [[ -n "${SERVER_EXTRA_ARGS}" ]]; then
  cmd+=(--server-extra-args="${SERVER_EXTRA_ARGS}")
fi

{
  echo "Running dynamic SGLang DFLASH c=${CONCURRENCY}"
  echo "run_tag=${RUN_TAG}"
  printf "command: "
  printf "%q " "${cmd[@]}"
  printf "\n"
} | tee "${OUT_LOG}"

start_gpu_monitor
"${cmd[@]}" 2>&1 | tee -a "${OUT_LOG}"
stop_gpu_monitor

if [[ -f "${OUT_TRACE}" ]]; then
  python "${SCRIPT_DIR}/scripts/summarize_sglang_calls.py" \
    --input "${OUT_TRACE}" \
    --output-md "${OUT_TRACE_SUMMARY_MD}" | tee -a "${OUT_LOG}"
else
  echo "Call trace not found at ${OUT_TRACE}; skipping summary generation." | tee -a "${OUT_LOG}"
fi

if [[ -s "${OUT_GPU_METRICS_CSV}" ]]; then
  python "${SCRIPT_DIR}/scripts/summarize_gpu_metrics.py" \
    --input "${OUT_GPU_METRICS_CSV}" \
    --output-md "${OUT_GPU_SUMMARY_MD}" \
    --output-json "${OUT_GPU_SUMMARY_JSON}" | tee -a "${OUT_LOG}"
else
  echo "GPU metrics not found at ${OUT_GPU_METRICS_CSV}; skipping GPU summary generation." | tee -a "${OUT_LOG}"
fi

echo "Done."
echo "Markdown: ${OUT_MD}"
echo "Call trace: ${OUT_TRACE}"
echo "Call trace summary: ${OUT_TRACE_SUMMARY_MD}"
echo "GPU metrics: ${OUT_GPU_METRICS_CSV}"
echo "GPU summary: ${OUT_GPU_SUMMARY_MD}"
echo "GPU summary JSON: ${OUT_GPU_SUMMARY_JSON}"
echo "Log: ${OUT_LOG}"
