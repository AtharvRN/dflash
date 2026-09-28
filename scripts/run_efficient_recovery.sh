#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT="/data/scratch/zekaili/atharv/dflash"
TASK_PYTHON="$TASK_ROOT/envs/main/bin/python"
TASK_SOURCE="${SOURCE_CACHE:-$TASK_ROOT/runs/actual_block_predictor_10k_20260928/cache}"
TASK_RUN_ID="${RUN_ID:-actual_block_predictor_10k_recovered_20260928}"
TASK_RUN="$TASK_ROOT/runs/$TASK_RUN_ID"
TASK_WORK="/tmp/${TASK_RUN_ID}_cache"
: "${GPU:?Choose one idle GPU}"
if [[ -e "$TASK_RUN" || -e "$TASK_WORK" ]]; then
  echo "Refusing existing recovery destinations" >&2
  exit 1
fi
exec 9>"$TASK_ROOT/gpu_${GPU}_actual_block.lock"
flock -n 9
mkdir -p "$TASK_RUN"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/pipeline_exit.txt"' EXIT
export PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
"$TASK_PYTHON" -u scripts/efficient_collection.py benchmark \
  --source "$TASK_SOURCE" --gpu "$GPU" --output "$TASK_RUN/benchmark" \
  --diagnostics "$TASK_RUN/diagnostics/benchmark" \
  2>&1 | tee "$TASK_RUN/benchmark.log"
"$TASK_PYTHON" -u scripts/efficient_collection.py collect \
  --source "$TASK_SOURCE" --gpu "$GPU" --output "$TASK_WORK" --backup "$TASK_RUN/cache" \
  --benchmark "$TASK_RUN/benchmark/summary.json" --diagnostics "$TASK_RUN/diagnostics/collection" \
  2>&1 | tee "$TASK_RUN/collection.log"
"$TASK_PYTHON" -u scripts/train_actual_block_predictor.py \
  --train-cache "$TASK_RUN/cache" --eval-cache "$TASK_ROOT/runs/policy_granularity_20260927/cache" \
  --split-dir "$TASK_ROOT/data/dflashv2_data/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719" \
  --checkpoint "$TASK_ROOT/checkpoints/context_residual_100k_20260913/last_mlp_best_epoch_4.pt" \
  --output "$TASK_RUN/training" --gpu "$GPU" 2>&1 | tee "$TASK_RUN/training.log"
