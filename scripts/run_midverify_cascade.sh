#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT=/data/scratch/zekaili/atharv/dflash
: "${GPU:?Choose the authorized idle GPU}"
TASK_RUN="$TASK_ROOT/runs/${RUN_ID:-midverify_cascade_20260929}"
if [[ -e "$TASK_RUN" ]]; then echo "Refusing existing run" >&2; exit 1; fi
exec 9>"$TASK_ROOT/gpu_${GPU}_midverify.lock"
flock -n 9
mkdir -p "$TASK_RUN"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$TASK_RUN/exit.txt"' EXIT
export PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
"$TASK_ROOT/envs/main/bin/python" -u scripts/train_midverify_cascade.py \
  --cache "$TASK_ROOT/runs/midverify_scaling_10k_20260929/cache_nonterminal" \
  --previous "$TASK_ROOT/runs/midverify_scaling_10k_20260929/training" \
  --output "$TASK_RUN/training" --gpu "$GPU" --max-seconds 1800 \
  2>&1 | tee "$TASK_RUN/training.log"
