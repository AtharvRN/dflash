#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PYTHON="${PYTHON:-/tmp/dflash-headroom-venv-20260926/bin/python}"
RUN_ID="${RUN_ID:-block_headroom_20260926}"
DATA_ROOT="${DFLASH_DATA_ROOT:-/workspace/dflashv2_data}"
PERSIST="${DFLASH_RUN_ROOT:-/workspace/dflashv2_data/runs}/${RUN_ID}"
WORK_ROOT="${DFLASH_WORK_ROOT:-/tmp}"
TARGET="${DFLASH_TARGET_MODEL:-/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${DFLASH_DRAFT_MODEL:-/workspace/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/b74e3a329c4d963783143b1e970d95b002be72bd}"
STAGE_ARGS=()
if [[ -z "${DFLASH_ROOT:-}" ]]; then
  STAGE_ARGS=(--stage-models /tmp/dflash-headroom-models-20260926)
else
  : "${CUDA_VISIBLE_DEVICES:?Select an available shared GPU explicitly before starting a run}"
fi
if [[ -e "$PERSIST" ]]; then
  echo "Refusing existing run root: $PERSIST" >&2
  exit 1
fi
mkdir -p "$PERSIST"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$PERSIST/pipeline_exit.txt"' EXIT
"$PYTHON" scripts/collect_block_headroom.py \
  --manifest "$DATA_ROOT/manifests/qwen3_4b_instruct_100k_messages.jsonl" \
  --split-dir "$DATA_ROOT/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719" \
  --model "$TARGET" --draft-model "$DRAFT" "${STAGE_ARGS[@]}" \
  --output-dir "$WORK_ROOT/${RUN_ID}" --persistent-dir "$PERSIST/cache" \
  --per-source "${PER_SOURCE:-32}" --states-per-prompt "${STATES_PER_PROMPT:-8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" --max-seconds "${MAX_SECONDS:-10800}" \
  --reverse-check-states 8 --canonical-check-states 16 \
  2>&1 | tee "$PERSIST/collection.log"
