#!/usr/bin/env bash
set -euo pipefail

# Run the C=16 fixed-subset comparison matrix:
# - baseline
# - static block sizes: 8,10,12,14,16,18
# - adaptive: ewma/ucb/linucb x reward variants
#
# All runs are delegated to run_sglang_dynamic_c16.sh so outputs stay consistent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-sglang_c16_policy_matrix_$(date +%Y%m%d_%H%M%S)}"
FIXED_QUESTION_COUNT="${FIXED_QUESTION_COUNT:-256}"
FIXED_QUESTION_OFFSET="${FIXED_QUESTION_OFFSET:-0}"
CONCURRENCY="${CONCURRENCY:-16}"
QUESTIONS_PER_CONCURRENCY_BASE="${QUESTIONS_PER_CONCURRENCY_BASE:-8}"
MAX_QUESTIONS_PER_CONFIG="${MAX_QUESTIONS_PER_CONFIG:-256}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
ENABLE_DFLASH_CYCLE_TRACE="${ENABLE_DFLASH_CYCLE_TRACE:-1}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"

# Adaptive defaults for all adaptive matrix runs.
ADAPTIVE_K_MIN="${ADAPTIVE_K_MIN:-1}"
ADAPTIVE_K_MAX="${ADAPTIVE_K_MAX:-18}"
ADAPTIVE_K_START="${ADAPTIVE_K_START:-12}"
ADAPTIVE_BLOCK_BUCKETS="${ADAPTIVE_BLOCK_BUCKETS:-8,10,12,14,16,18}"
ADAPTIVE_RHO="${ADAPTIVE_RHO:-0.30}"
ADAPTIVE_DELTA="${ADAPTIVE_DELTA:-1.0}"
ADAPTIVE_UCB_C="${ADAPTIVE_UCB_C:-1.0}"
ADAPTIVE_UCB_DELTA="${ADAPTIVE_UCB_DELTA:-0.05}"
ADAPTIVE_LINUCB_ALPHA="${ADAPTIVE_LINUCB_ALPHA:-1.0}"
ADAPTIVE_LINUCB_LAMBDA="${ADAPTIVE_LINUCB_LAMBDA:-1.0}"
ADAPTIVE_LOW_ACCEPT_THRESHOLD="${ADAPTIVE_LOW_ACCEPT_THRESHOLD:-0.35}"
ADAPTIVE_LOW_ACCEPT_STREAK="${ADAPTIVE_LOW_ACCEPT_STREAK:-2}"
ADAPTIVE_HIGH_ACCEPT_THRESHOLD="${ADAPTIVE_HIGH_ACCEPT_THRESHOLD:-0.90}"
ADAPTIVE_HIGH_ACCEPT_STREAK="${ADAPTIVE_HIGH_ACCEPT_STREAK:-2}"
ADAPTIVE_COOLDOWN_CYCLES="${ADAPTIVE_COOLDOWN_CYCLES:-1}"

echo "Running C=16 policy matrix"
echo "run_tag_prefix=${RUN_TAG_PREFIX}"
echo "fixed_subset=count=${FIXED_QUESTION_COUNT},offset=${FIXED_QUESTION_OFFSET}"
echo "concurrency=${CONCURRENCY}"
echo "dataset=${DATASET_NAME}"
echo "static_block_sizes=8,10,12,14,16,18"
echo "adaptive_algos=ewma,ucb,linucb"
echo "adaptive_reward_modes=accept_length,throughput"

run_case() {
  local case_name="$1"
  shift
  local case_tag="${RUN_TAG_PREFIX}_${case_name}"
  local case_log_dir="logs/${case_tag}"

  echo "============================================================"
  echo "case=${case_name}"
  echo "run_tag=${case_tag}"
  echo "log_dir=${case_log_dir}"
  echo "============================================================"

  env \
    RUN_TAG="${case_tag}" \
    LOG_DIR="${case_log_dir}" \
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
    ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR}" \
    "$@" \
    bash "${SCRIPT_DIR}/run_sglang_dynamic_c16.sh"
}

# Baseline + one static run in the same invocation (DFLASH block size 8).
run_case \
  "baseline_and_static_bs8" \
  RUN_BASELINE=1 \
  ADAPTIVE_ENABLED=0 \
  DFLASH_BLOCK_SIZE=8

# Remaining static runs (skip baseline to avoid re-running baseline each case).
for bs in 10 12 14 16 18; do
  run_case \
    "static_bs${bs}" \
    RUN_BASELINE=0 \
    ADAPTIVE_ENABLED=0 \
    DFLASH_BLOCK_SIZE="${bs}"
done

# Adaptive runs: ewma/ucb/linucb x reward variants.
for algo in ewma ucb linucb; do
  for reward in accept_length throughput; do
    run_case \
      "adaptive_${algo}_${reward}" \
      RUN_BASELINE=0 \
      ADAPTIVE_ENABLED=1 \
      DFLASH_BLOCK_SIZE=18 \
      ADAPTIVE_ALGO="${algo}" \
      ADAPTIVE_REWARD_MODE="${reward}" \
      ADAPTIVE_K_MIN="${ADAPTIVE_K_MIN}" \
      ADAPTIVE_K_MAX="${ADAPTIVE_K_MAX}" \
      ADAPTIVE_K_START="${ADAPTIVE_K_START}" \
      ADAPTIVE_BLOCK_BUCKETS="${ADAPTIVE_BLOCK_BUCKETS}" \
      ADAPTIVE_RHO="${ADAPTIVE_RHO}" \
      ADAPTIVE_DELTA="${ADAPTIVE_DELTA}" \
      ADAPTIVE_UCB_C="${ADAPTIVE_UCB_C}" \
      ADAPTIVE_UCB_DELTA="${ADAPTIVE_UCB_DELTA}" \
      ADAPTIVE_LINUCB_ALPHA="${ADAPTIVE_LINUCB_ALPHA}" \
      ADAPTIVE_LINUCB_LAMBDA="${ADAPTIVE_LINUCB_LAMBDA}" \
      ADAPTIVE_LOW_ACCEPT_THRESHOLD="${ADAPTIVE_LOW_ACCEPT_THRESHOLD}" \
      ADAPTIVE_LOW_ACCEPT_STREAK="${ADAPTIVE_LOW_ACCEPT_STREAK}" \
      ADAPTIVE_HIGH_ACCEPT_THRESHOLD="${ADAPTIVE_HIGH_ACCEPT_THRESHOLD}" \
      ADAPTIVE_HIGH_ACCEPT_STREAK="${ADAPTIVE_HIGH_ACCEPT_STREAK}" \
      ADAPTIVE_COOLDOWN_CYCLES="${ADAPTIVE_COOLDOWN_CYCLES}"
  done
done

echo "Matrix complete."
echo "Run tag prefix: ${RUN_TAG_PREFIX}"
echo "Logs under: logs/${RUN_TAG_PREFIX}_*"

python "${SCRIPT_DIR}/scripts/summarize_sglang_policy_matrix.py" \
  --run-tag-prefix "${RUN_TAG_PREFIX}" \
  --logs-root logs \
  --output-md "logs/${RUN_TAG_PREFIX}_matrix_summary.md" \
  --output-csv "logs/${RUN_TAG_PREFIX}_matrix_summary.csv"
