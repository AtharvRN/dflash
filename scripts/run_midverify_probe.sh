#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT=/data/scratch/zekaili/atharv/dflash
TASK_PYTHON="$TASK_ROOT/envs/main/bin/python"
: "${GPU:?Choose the authorized idle GPU}"
TASK_RUN_ID="${RUN_ID:-midverify_probe_2k_20260929}"
TASK_RUN="$TASK_ROOT/runs/$TASK_RUN_ID"
if [[ -e "$TASK_RUN" ]]; then
  echo "Refusing to overwrite existing run" >&2
  exit 1
fi
exec 9>"$TASK_ROOT/gpu_${GPU}_midverify.lock"
flock -n 9
mkdir -p "$TASK_RUN"
TASK_WORK=$(mktemp -d /tmp/dflash-midverify.XXXXXXXX)
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/pipeline_exit.txt"' EXIT
export PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
TASK_FLAGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then TASK_FLAGS+=(--smoke); fi
"$TASK_PYTHON" -u scripts/collect_midverify_probe.py \
  --train-cache "$TASK_ROOT/runs/actual_block_predictor_10k_finish_20260928/cache" \
  --eval-cache "$TASK_ROOT/runs/policy_granularity_20260927/cache" \
  --split-dir "$TASK_ROOT/data/dflashv2_data/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719" \
  --models "$TASK_ROOT/models.json" --output "$TASK_WORK/cache" --backup "$TASK_RUN/cache" \
  --gpu "$GPU" --batch-size "${BATCH_SIZE:-8}" --max-seconds 3600 "${TASK_FLAGS[@]}" \
  2>&1 | tee "$TASK_RUN/collection.log"
TASK_TRAIN_FLAGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then TASK_TRAIN_FLAGS+=(--epochs 1 --seeds 913); fi
"$TASK_PYTHON" -u scripts/train_midverify_probe.py \
  --cache "$TASK_RUN/cache" --output "$TASK_RUN/training" --gpu "$GPU" \
  --max-seconds 1800 "${TASK_FLAGS[@]}" "${TASK_TRAIN_FLAGS[@]}" \
  2>&1 | tee "$TASK_RUN/training.log"
