#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT="/data/scratch/zekaili/atharv/dflash"
TASK_DATA="$TASK_ROOT/data/dflashv2_data"
TASK_PYTHON="$TASK_ROOT/envs/main/bin/python"
TASK_CHECKPOINT="$TASK_ROOT/checkpoints/context_residual_100k_20260913/last_mlp_best_epoch_4.pt"
TASK_RUN_ID="${RUN_ID:-policy_granularity_20260927}"
TASK_RUN="$TASK_ROOT/runs/$TASK_RUN_ID"
TASK_WORK="/tmp/${TASK_RUN_ID}_cache"
: "${GPU:?Choose an idle GPU explicitly}"
if [[ -e "$TASK_RUN" || -e "$TASK_WORK" ]]; then
  echo "Refusing existing run destinations" >&2
  exit 1
fi
mkdir -p "$TASK_RUN"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/pipeline_exit.txt"' EXIT
export PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
"$TASK_PYTHON" -u scripts/collect_policy_granularity.py \
  --manifest "$TASK_DATA/manifests/qwen3_4b_instruct_100k_messages.jsonl" \
  --pilot-manifest "$TASK_DATA/runs/prefusion_pilot_20260915/cache/manifest.json" \
  --split-dir "$TASK_DATA/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719" \
  --models "$TASK_ROOT/models.json" --checkpoint "$TASK_CHECKPOINT" \
  --output "$TASK_WORK" --backup "$TASK_RUN/cache" --gpu "$GPU" \
  --max-seconds 1800 2>&1 | tee "$TASK_RUN/collection.log"
"$TASK_PYTHON" -u scripts/analyze_policy_granularity.py \
  --cache "$TASK_RUN/cache" --checkpoint "$TASK_CHECKPOINT" \
  --output "$TASK_RUN/analysis" 2>&1 | tee "$TASK_RUN/analysis.log"
