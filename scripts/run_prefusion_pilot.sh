#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=
export TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-/tmp/dflash-prefusion-venv/bin/python}"
RUN_ID="${RUN_ID:-prefusion_pilot_20260915}"
CACHE="${CACHE:-/tmp/${RUN_ID}_cache}"
RUN="${RUN:-/tmp/${RUN_ID}_train}"
PERSIST="${PERSIST:-/workspace/dflashv2_data/runs/${RUN_ID}}"
mkdir -p "$PERSIST"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$PERSIST/pipeline_exit.txt"' EXIT

"$PYTHON" scripts/collect_prefusion_acceptance.py \
  --manifest /workspace/dflashv2_data/manifests/qwen3_4b_instruct_100k_messages.jsonl \
  --split-dir /workspace/dflashv2_data/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719 \
  --reference-cache-manifest /workspace/dflashv2_data/runs/context_residual_100k_20260913/cache_provenance/manifest.json \
  --model /tmp/dflash-prefusion-target --draft-model /tmp/dflash-prefusion-draft \
  --output-dir "$CACHE" --persistent-dir "$PERSIST/cache" \
  --train-prompts "${TRAIN_PROMPTS:-512}" --val-prompts "${VAL_PROMPTS:-256}" \
  --max-cycles "${MAX_CYCLES:-32}" --max-new-tokens 512 \
  2>&1 | tee "$PERSIST/collection.log"

"$PYTHON" scripts/audit_prefusion_cache.py --cache-dir "$CACHE" \
  --persistent-dir "$PERSIST/cache" --device cuda 2>&1 | tee "$PERSIST/audit.log"

"$PYTHON" scripts/train_prefusion_acceptance.py --cache-dir "$CACHE" \
  --output-dir "$RUN" --persistent-dir "$PERSIST/training" \
  --epochs 6 --batch-size 128 --eval-batch-size 256 --workers 2 \
  --cpu-threads 4 --seed 913 --retention .96 \
  2>&1 | tee "$PERSIST/training.log"
