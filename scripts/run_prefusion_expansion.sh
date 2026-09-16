#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=""
export TOKENIZERS_PARALLELISM=false
PYTHON=${PYTHON:-/tmp/dflash-prefusion-venv/bin/python}
RUN_ID=${RUN_ID:-prefusion_100k_20260915}
BASE_CACHE=${BASE_CACHE:-/tmp/prefusion_pilot_20260915_cache}
PERSISTENT=/workspace/dflashv2_data/runs/$RUN_ID
mkdir -p "$PERSISTENT"
trap 'status=$?; printf "exit_code=%s\n" "$status" > "$PERSISTENT/pipeline_exit.txt"' EXIT
"$PYTHON" -u scripts/grow_prefusion_acceptance.py \
  --base-cache "$BASE_CACHE" \
  --output-dir "/tmp/${RUN_ID}_cache" \
  --persistent-dir "$PERSISTENT/cache" \
  --target-train-rows "${TARGET_TRAIN_ROWS:-100000}" \
  --candidate-prompts "${CANDIDATE_PROMPTS:-6000}" \
  2>&1 | tee "$PERSISTENT/collection.log"
