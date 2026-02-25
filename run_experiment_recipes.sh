#!/usr/bin/env bash
set -euo pipefail

# One launcher for common benchmark commands used in this repo.
# Usage examples:
#   MODE=baseline_bs1 bash run_experiment_recipes.sh
#   MODE=vanilla_bs16 MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 bash run_experiment_recipes.sh
#   MODE=fixed_prefix_naive FIXED_PREFIX_LEN=2 BRANCH_TOP_K=4 MAX_CANDIDATES=4 bash run_experiment_recipes.sh
#   MODE=sparse_conservative bash run_experiment_recipes.sh
#   MODE=sparse_aggressive bash run_experiment_recipes.sh
#   MODE=dynamic_adl bash run_experiment_recipes.sh
#   MODE=block_sweep BLOCK_SIZES="12 13 14 15 16" SKIP_BASELINE=1 SHARED_BASELINE_IF_SKIP=0 bash run_experiment_recipes.sh
#   MODE=fixed_prefix_sweep FIXED_PREFIX_LENS="1 2 4 6 7" TOP_K_LIST="4" MAX_CANDIDATES_LIST="4" RUN_BASELINE=0 bash run_experiment_recipes.sh

MODE="${MODE:-help}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="${DATASET:-aime25}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3-4B-DFlash-b16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SAVE_OUTPUTS_DIR="${SAVE_OUTPUTS_DIR:-outputs}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${SAVE_OUTPUTS_DIR}" "${LOG_DIR}"

run_cmd() {
  echo "--------------------------------------------------------"
  printf "Launch: "
  printf "%q " "$@"
  printf "\n"
  echo "--------------------------------------------------------"
  "$@"
}

show_help() {
  cat <<'EOF'
Available MODE values:
  baseline_bs1
    Run target-only baseline (block-size 1) with profiling.

  vanilla_bs16
    Run standard DFlash benchmark.py at block size (default 16) with profiling.

  fixed_prefix_naive
    Run older simple multi-candidate method:
      candidate_mode=fixed_prefix_rank
    Tunables:
      FIXED_PREFIX_LEN (default 2), BRANCH_TOP_K (default 4), MAX_CANDIDATES (default 4)

  fixed_prefix_adaptive
    Run fixed_prefix_rank with adaptive per-cycle candidate budget.
    Tunables:
      FIXED_PREFIX_LEN (default 2), BRANCH_TOP_K (default 4), MAX_CANDIDATES (default 8),
      ADAPTIVE_BUDGETS (default 1,4,8), ADAPTIVE_ACCEPT_THRESHOLDS (default 0.85,0.65),
      ADAPTIVE_WARMUP_CYCLES (default 4), ADAPTIVE_PROBE_INTERVAL (default 16)

  sparse_conservative
    Run uncertainty_sparse_rank conservative config:
      fixed_prefix_len=2, sparse_max_positions=3, top_k=3, max_candidates=4, margin=0.08

  sparse_aggressive
    Run uncertainty_sparse_rank aggressive config:
      fixed_prefix_len=2, sparse_max_positions=4, top_k=4, max_candidates=8, margin=-1

  dynamic_adl
    Run benchmark_dynamic_schedule.py using adl_ewma mode.

  block_sweep
    Wrapper around run_block_sweep.sh.
    Tunables:
      BLOCK_SIZES (default: "12 13 14 15 16"), SKIP_BASELINE, SHARED_BASELINE_IF_SKIP, NPROC

  fixed_prefix_sweep
    Wrapper around run_fixed_prefix_sweep.sh.
    Tunables:
      FIXED_PREFIX_LENS, TOP_K_LIST, MAX_CANDIDATES_LIST, RUN_BASELINE, RUN_VANILLA_REF

Common env vars:
  CUDA_VISIBLE_DEVICES, DATASET, MAX_SAMPLES, MAX_NEW_TOKENS, MODEL, DRAFT,
  BLOCK_SIZE, TEMPERATURE, RUN_TAG, SAVE_OUTPUTS_DIR, LOG_DIR, PYTHON_BIN
EOF
}

case "${MODE}" in
  baseline_bs1)
    OUT_JSON="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_baseline_bs1_profiled.jsonl"
    OUT_CYCLE="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_baseline_bs1_cycle.jsonl"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" benchmark.py \
      --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL}" --draft-name-or-path "${DRAFT}" \
      --block-size 1 --max-new-tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" \
      --collect-profile \
      --save-outputs-path "${OUT_JSON}" \
      --save-cycle-trace-path "${OUT_CYCLE}"
    ;;

  vanilla_bs16)
    OUT_JSON="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_bs${BLOCK_SIZE}_vanilla_profiled.jsonl"
    OUT_CYCLE="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_bs${BLOCK_SIZE}_vanilla_cycle.jsonl"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" benchmark.py \
      --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL}" --draft-name-or-path "${DRAFT}" \
      --block-size "${BLOCK_SIZE}" --max-new-tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" \
      --skip-baseline --collect-profile \
      --save-outputs-path "${OUT_JSON}" \
      --save-cycle-trace-path "${OUT_CYCLE}"
    ;;

  fixed_prefix_naive)
    FIXED_PREFIX_LEN="${FIXED_PREFIX_LEN:-2}"
    BRANCH_TOP_K="${BRANCH_TOP_K:-4}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-4}"
    OUT_JSON="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_fprefix_p${FIXED_PREFIX_LEN}_k${BRANCH_TOP_K}_c${MAX_CANDIDATES}.jsonl"
    OUT_CYCLE="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_fprefix_p${FIXED_PREFIX_LEN}_k${BRANCH_TOP_K}_c${MAX_CANDIDATES}_cycle.jsonl"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" benchmark_candidate_solutions.py \
      --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL}" --draft-name-or-path "${DRAFT}" \
      --block-size "${BLOCK_SIZE}" --max-new-tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" \
      --candidate-mode fixed_prefix_rank \
      --fixed-prefix-len "${FIXED_PREFIX_LEN}" \
      --branch-top-k "${BRANCH_TOP_K}" \
      --max-candidates "${MAX_CANDIDATES}" \
      --skip-baseline --collect-profile \
      --save-outputs-path "${OUT_JSON}" \
      --save-cycle-trace-path "${OUT_CYCLE}"
    ;;

  fixed_prefix_adaptive)
    FIXED_PREFIX_LEN="${FIXED_PREFIX_LEN:-2}"
    BRANCH_TOP_K="${BRANCH_TOP_K:-4}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-8}"
    ADAPTIVE_BUDGETS="${ADAPTIVE_BUDGETS:-1,4,8}"
    ADAPTIVE_ACCEPT_THRESHOLDS="${ADAPTIVE_ACCEPT_THRESHOLDS:-0.85,0.65}"
    ADAPTIVE_WARMUP_CYCLES="${ADAPTIVE_WARMUP_CYCLES:-4}"
    ADAPTIVE_PROBE_INTERVAL="${ADAPTIVE_PROBE_INTERVAL:-16}"
    OUT_JSON="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_fprefix_adaptive_p${FIXED_PREFIX_LEN}_k${BRANCH_TOP_K}_c${MAX_CANDIDATES}.jsonl"
    OUT_CYCLE="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_fprefix_adaptive_p${FIXED_PREFIX_LEN}_k${BRANCH_TOP_K}_c${MAX_CANDIDATES}_cycle.jsonl"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" benchmark_candidate_solutions.py \
      --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL}" --draft-name-or-path "${DRAFT}" \
      --block-size "${BLOCK_SIZE}" --max-new-tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" \
      --candidate-mode fixed_prefix_rank \
      --fixed-prefix-len "${FIXED_PREFIX_LEN}" \
      --branch-top-k "${BRANCH_TOP_K}" \
      --max-candidates "${MAX_CANDIDATES}" \
      --adaptive-candidates \
      --adaptive-budgets "${ADAPTIVE_BUDGETS}" \
      --adaptive-accept-thresholds "${ADAPTIVE_ACCEPT_THRESHOLDS}" \
      --adaptive-warmup-cycles "${ADAPTIVE_WARMUP_CYCLES}" \
      --adaptive-probe-interval "${ADAPTIVE_PROBE_INTERVAL}" \
      --skip-baseline --collect-profile \
      --save-outputs-path "${OUT_JSON}" \
      --save-cycle-trace-path "${OUT_CYCLE}"
    ;;

  sparse_conservative)
    OUT_JSON="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_sparse_conservative.jsonl"
    OUT_CYCLE="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_sparse_conservative_cycle.jsonl"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" benchmark_candidate_solutions.py \
      --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL}" --draft-name-or-path "${DRAFT}" \
      --block-size "${BLOCK_SIZE}" --max-new-tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" \
      --candidate-mode uncertainty_sparse_rank \
      --fixed-prefix-len 2 \
      --sparse-max-positions 3 \
      --branch-top-k 3 \
      --max-candidates 4 \
      --branch-margin-threshold 0.08 \
      --skip-baseline --collect-profile \
      --save-outputs-path "${OUT_JSON}" \
      --save-cycle-trace-path "${OUT_CYCLE}"
    ;;

  sparse_aggressive)
    OUT_JSON="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_sparse_aggressive.jsonl"
    OUT_CYCLE="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_sparse_aggressive_cycle.jsonl"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" benchmark_candidate_solutions.py \
      --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL}" --draft-name-or-path "${DRAFT}" \
      --block-size "${BLOCK_SIZE}" --max-new-tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" \
      --candidate-mode uncertainty_sparse_rank \
      --fixed-prefix-len 2 \
      --sparse-max-positions 4 \
      --branch-top-k 4 \
      --max-candidates 8 \
      --branch-margin-threshold -1 \
      --skip-baseline --collect-profile \
      --save-outputs-path "${OUT_JSON}" \
      --save-cycle-trace-path "${OUT_CYCLE}"
    ;;

  dynamic_adl)
    OUT_JSON="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_dynamic_adl_ewma.jsonl"
    CANDIDATE_BLOCK_SIZES="${CANDIDATE_BLOCK_SIZES:-8,12,16}"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" benchmark_dynamic_schedule.py \
      --dataset "${DATASET}" --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL}" --draft-name-or-path "${DRAFT}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --candidate-block-sizes "${CANDIDATE_BLOCK_SIZES}" \
      --scheduler-mode adl_ewma \
      --warmup-cycles 2 \
      --ewma-alpha 0.10 \
      --switch-margin 0.05 \
      --required-streak 2 \
      --cooldown-cycles 2 \
      --probe-interval 12 \
      --low-accept-threshold 0.35 \
      --low-accept-streak 2 \
      --adl-rho 0.30 \
      --adl-delta 2 \
      --adl-k-min 8 \
      --adl-k-max 16 \
      --adl-neighborhood 4 \
      --save-outputs-path "${OUT_JSON}"
    ;;

  block_sweep)
    BLOCK_SIZES="${BLOCK_SIZES:-12 13 14 15 16}"
    NPROC="${NPROC:-1}"
    SKIP_BASELINE="${SKIP_BASELINE:-1}"
    SHARED_BASELINE_IF_SKIP="${SHARED_BASELINE_IF_SKIP:-0}"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      DATASET="${DATASET}" MAX_SAMPLES="${MAX_SAMPLES}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      MODEL="${MODEL}" DRAFT="${DRAFT}" TEMPERATURE="${TEMPERATURE}" \
      BLOCK_SIZES="${BLOCK_SIZES}" NPROC="${NPROC}" SKIP_BASELINE="${SKIP_BASELINE}" \
      SHARED_BASELINE_IF_SKIP="${SHARED_BASELINE_IF_SKIP}" \
      SAVE_OUTPUTS_DIR="${SAVE_OUTPUTS_DIR}" RUN_TAG="${RUN_TAG}" \
      PYTHON_BIN="${PYTHON_BIN}" bash run_block_sweep.sh
    ;;

  fixed_prefix_sweep)
    FIXED_PREFIX_LENS="${FIXED_PREFIX_LENS:-1 2 4 6 7}"
    TOP_K_LIST="${TOP_K_LIST:-4}"
    MAX_CANDIDATES_LIST="${MAX_CANDIDATES_LIST:-4}"
    RUN_BASELINE="${RUN_BASELINE:-0}"
    RUN_VANILLA_REF="${RUN_VANILLA_REF:-1}"
    CANDIDATE_MODE="${CANDIDATE_MODE:-fixed_prefix_rank}"
    SPARSE_MAX_POSITIONS="${SPARSE_MAX_POSITIONS:-4}"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      DATASET="${DATASET}" MAX_SAMPLES="${MAX_SAMPLES}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      MODEL="${MODEL}" DRAFT="${DRAFT}" BLOCK_SIZE="${BLOCK_SIZE}" TEMPERATURE="${TEMPERATURE}" \
      FIXED_PREFIX_LENS="${FIXED_PREFIX_LENS}" TOP_K_LIST="${TOP_K_LIST}" \
      MAX_CANDIDATES_LIST="${MAX_CANDIDATES_LIST}" \
      CANDIDATE_MODE="${CANDIDATE_MODE}" SPARSE_MAX_POSITIONS="${SPARSE_MAX_POSITIONS}" \
      RUN_BASELINE="${RUN_BASELINE}" RUN_VANILLA_REF="${RUN_VANILLA_REF}" \
      SAVE_OUTPUTS_DIR="${SAVE_OUTPUTS_DIR}" RUN_TAG="${RUN_TAG}" \
      PYTHON_BIN="${PYTHON_BIN}" bash run_fixed_prefix_sweep.sh
    ;;

  help|--help|-h)
    show_help
    ;;

  *)
    echo "Unknown MODE=${MODE}"
    echo "Run with MODE=help for available modes."
    exit 1
    ;;
esac
