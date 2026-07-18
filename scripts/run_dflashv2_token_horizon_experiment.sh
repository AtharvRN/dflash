#!/usr/bin/env bash
set -euo pipefail

cd /workspace/dflash-fresh-zlab-main

# End-to-end token-ID horizon experiment:
# 1. Materialize last fused vector + recent predraft token ids into /tmp.
# 2. Train/evaluate fused-only and token-GRU horizon heads on Math500.
#
# This does not collect traces. TRACE and MATH must already have been collected
# with scripts/collect_dflashv2_horizon_traces.py --log-predraft-token-ids.

STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
TRACE=${TRACE:-/workspace/dflashv2_data/traces/dflashv2_qwen3_4b_b16_instruct100k_tokenids}
MATH=${MATH:-/workspace/dflashv2_data/traces/math500_qwen3_4b_b16_eval_tokenids}
VALIDS=${VALIDS:-/workspace/dflashv2_data/splits/qwen3_4b_instruct100k_val5k_seed0.json}
RUN_ID=${RUN_ID:-integer_horizon_token_qwen3_4b_${STAMP}}

CACHE=${CACHE:-/tmp/dflashv2_cache/${RUN_ID}}
ROOT=${ROOT:-/workspace/dflashv2_data/runs/${RUN_ID}}
EVALROOT=${EVALROOT:-/workspace/dflashv2_data/evals/${RUN_ID}}
LOGROOT=${LOGROOT:-/workspace/dflashv2_data/logs/${RUN_ID}}

mkdir -p "$CACHE" "$ROOT" "$EVALROOT" "$LOGROOT"

echo "RUN_ID=$RUN_ID"
echo "TRACE=$TRACE"
echo "MATH=$MATH"
echo "CACHE=$CACHE"
echo "ROOT=$ROOT"
echo "EVALROOT=$EVALROOT"
echo "LOGROOT=$LOGROOT"

python scripts/materialize_dflashv2_last_feature_cache.py \
  --trace-dir "$TRACE" \
  --output-dir "$CACHE" \
  --selection-mode prefix \
  --split-by-prompt-metadata \
  --prompt-key manifest_index \
  --val-prompt-count 5000 \
  --val-prompt-ids-path "$VALIDS" \
  --parallel-workers "${MATERIALIZE_WORKERS:-8}" \
  --max-read-rows "${MAX_READ_ROWS:-2048}" \
  --max-output-rows "${MAX_OUTPUT_ROWS:-2048}" \
  --copy-rows "${COPY_ROWS:-131072}" \
  --include-predraft-token-ids \
  --overwrite

CACHE="$CACHE" \
MATH="$MATH" \
ROOT="$ROOT" \
EVALROOT="$EVALROOT" \
LOGROOT="$LOGROOT" \
  bash scripts/run_dflashv2_integer_horizon_token_sweep.sh
