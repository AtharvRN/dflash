#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT=/data/scratch/zekaili/atharv/dflash
TASK_PYTHON="$TASK_ROOT/envs/main/bin/python"
: "${GPU:?Choose the authorized idle GPU}"
TASK_RUN_ID="${RUN_ID:-midverify_scaling_10k_20260929}"
TASK_RUN="$TASK_ROOT/runs/$TASK_RUN_ID"
TASK_STAGE="${1:?Choose collect or train}"
exec 9>"$TASK_ROOT/gpu_${GPU}_midverify.lock"
flock -n 9
export PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
case "$TASK_STAGE" in
  collect)
    if [[ -e "$TASK_RUN" ]]; then echo "Refusing existing expansion" >&2; exit 1; fi
    mkdir -p "$TASK_RUN"
    TASK_WORK=$(mktemp -d /tmp/dflash-midverify-scale.XXXXXXXX)
    trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/collection_exit.txt"' EXIT
    "$TASK_PYTHON" -u scripts/collect_midverify_probe.py \
      --train-cache "$TASK_ROOT/runs/actual_block_predictor_10k_finish_20260928/cache" \
      --eval-cache "$TASK_ROOT/runs/policy_granularity_20260927/cache" \
      --split-dir "$TASK_ROOT/data/dflashv2_data/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719" \
      --models "$TASK_ROOT/models.json" --seed-cache "$TASK_ROOT/runs/midverify_probe_2k_20260929/cache" \
      --train-rows "${TRAIN_ROWS:-10000}" --output "$TASK_WORK/cache" --backup "$TASK_RUN/cache" \
      --gpu "$GPU" --batch-size 8 --max-seconds 3600 2>&1 | tee "$TASK_RUN/collection.log"
    ;;
  train)
    TASK_TRAIN_CACHE="${TRAIN_CACHE:-$TASK_RUN/cache}"
    if [[ ! -f "$TASK_TRAIN_CACHE/COMPLETE.json" || -e "$TASK_RUN/training" ]]; then
      echo "Require completed cache and fresh training destination" >&2; exit 1
    fi
    trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/training_exit.txt"' EXIT
    "$TASK_PYTHON" -u scripts/train_midverify_scaling.py --cache "$TASK_TRAIN_CACHE" \
      --sizes 2000 5000 "${TRAIN_ROWS:-10000}" \
      --baseline "$TASK_ROOT/runs/midverify_probe_2k_20260929/assessment_fp64" \
      --output "$TASK_RUN/training" --gpu "$GPU" --max-seconds 3600 \
      2>&1 | tee "$TASK_RUN/training.log"
    ;;
  *) echo "Unknown stage" >&2; exit 1 ;;
esac
