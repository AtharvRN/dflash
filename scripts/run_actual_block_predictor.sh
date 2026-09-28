#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT="/data/scratch/zekaili/atharv/dflash"
TASK_DATA="$TASK_ROOT/data/dflashv2_data"
TASK_PYTHON="$TASK_ROOT/envs/main/bin/python"
TASK_CHECKPOINT="$TASK_ROOT/checkpoints/context_residual_100k_20260913/last_mlp_best_epoch_4.pt"
TASK_RUN_ID="${RUN_ID:-actual_block_predictor_10k_20260928}"
TASK_RUN="$TASK_ROOT/runs/$TASK_RUN_ID"
TASK_WORK="/tmp/${TASK_RUN_ID}_cache"
: "${GPU:?Choose an idle GPU explicitly}"
if [[ -e "$TASK_RUN" || -e "$TASK_WORK" ]]; then
  echo "Refusing existing run destinations" >&2
  exit 1
fi
# Cooperates with our own experiment launches; other users remain untouched.
exec 9>"$TASK_ROOT/gpu_${GPU}_actual_block.lock"
if ! flock -n 9; then
  echo "Another actual-block pipeline holds this GPU lock" >&2
  exit 1
fi
mkdir -p "$TASK_RUN"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/pipeline_exit.txt"' EXIT
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
TASK_ROWS=10000
TASK_PROMPTS=2000
TASK_SECONDS=14400
TASK_EXTRA=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  TASK_ROWS=16
  TASK_PROMPTS=10
  TASK_SECONDS=180
  TASK_EXTRA=(--smoke --seeds 913 --epochs 1)
fi
"$TASK_PYTHON" -u scripts/collect_policy_granularity.py \
  --manifest "$TASK_DATA/manifests/qwen3_4b_instruct_100k_messages.jsonl" \
  --pilot-manifest "$TASK_DATA/runs/prefusion_pilot_20260915/cache/manifest.json" \
  --split-dir "$TASK_DATA/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719" \
  --models "$TASK_ROOT/models.json" --checkpoint "$TASK_CHECKPOINT" \
  --output "$TASK_WORK" --backup "$TASK_RUN/cache" --gpu "$GPU" --seed 929 \
  --training-rows "$TASK_ROWS" --limit-prompts "$TASK_PROMPTS" --max-seconds "$TASK_SECONDS" \
  2>&1 | tee "$TASK_RUN/collection.log"
"$TASK_PYTHON" -u scripts/train_actual_block_predictor.py \
  --train-cache "$TASK_RUN/cache" \
  --eval-cache "$TASK_ROOT/runs/policy_granularity_20260927/cache" \
  --split-dir "$TASK_DATA/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719" \
  --checkpoint "$TASK_CHECKPOINT" --output "$TASK_RUN/training" --gpu "$GPU" \
  "${TASK_EXTRA[@]}" 2>&1 | tee "$TASK_RUN/training.log"
