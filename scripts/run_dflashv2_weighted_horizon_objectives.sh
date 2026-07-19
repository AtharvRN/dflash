#!/usr/bin/env bash
set -euo pipefail

cd /workspace/dflash-fresh-zlab-main

TRACE_DIR=${TRACE_DIR:-/workspace/dflashv2_data/traces/dflashv2_qwen3_4b_b16_instruct100k_full_4a100_20260716_2000}
SPLIT_DIR=${SPLIT_DIR:-/workspace/dflashv2_data/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719}
VAL_PROMPT_IDS=${VAL_PROMPT_IDS:-$SPLIT_DIR/val_prompt_ids.json}
CACHE=${CACHE:-/workspace/dflashv2_data/compact_cache/qwen3_4b_instruct100k_full_canonical_last_feature_20260719}
MATH=${MATH:-/workspace/dflashv2_data/traces/math500_qwen3_4b_b16_eval_20260716_215958}
HUMANEVAL=${HUMANEVAL:-/workspace/dflashv2_data/traces/humaneval_qwen3_4b_b16_eval_20260718_185737}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}

ROOT=${ROOT:-/workspace/dflashv2_data/runs/qwen3_4b_weighted_horizon_objectives_${STAMP}}
EVALROOT=${EVALROOT:-/workspace/dflashv2_data/evals/qwen3_4b_weighted_horizon_objectives_${STAMP}}
LOGROOT=${LOGROOT:-/workspace/dflashv2_data/logs/qwen3_4b_weighted_horizon_objectives_${STAMP}}

mkdir -p "$ROOT" "$EVALROOT" "$LOGROOT"

echo "TRACE_DIR=$TRACE_DIR"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "CACHE=$CACHE"
echo "ROOT=$ROOT"
echo "EVALROOT=$EVALROOT"
echo "LOGROOT=$LOGROOT"

if [[ ! -f "$CACHE/train/features.npy" || ! -f "$CACHE/val/features.npy" ]]; then
  echo "BUILD_CACHE_START $(date -u)"
  python scripts/materialize_dflashv2_last_feature_cache.py \
    --trace-dir "$TRACE_DIR" \
    --output-dir "$CACHE" \
    --split-by-prompt-metadata \
    --prompt-key manifest_index \
    --val-prompt-ids-path "$VAL_PROMPT_IDS" \
    --parallel-workers "${MATERIALIZE_WORKERS:-8}" \
    --max-read-rows "${MAX_READ_ROWS:-1024}" \
    --max-output-rows "${MAX_OUTPUT_ROWS:-2048}" \
    --copy-rows "${COPY_ROWS:-65536}" \
    --progress-every "${PROGRESS_EVERY:-50000}" \
    > "$LOGROOT/materialize_cache.log" 2>&1
  echo "BUILD_CACHE_DONE $(date -u)"
else
  echo "CACHE_READY $(date -u)"
fi

run_eval() {
  local name="$1"
  local out="$2"
  local trace="$3"
  local label="$4"
  shift 4
  python scripts/train_dflashv2_integer_horizon.py \
    --trace-dir "$trace" \
    --output-dir "$out/eval_$label" \
    --checkpoint "$out/best.pt" \
    --eval-only \
    --batch-size "${EVAL_BATCH_SIZE:-4096}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --device cuda \
    "$@" >> "$LOGROOT/$name.log" 2>&1
  cp "$out/eval_$label/eval_metrics.json" "$EVALROOT/${name}_${label}.json"
}

run_one() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  local log="$LOGROOT/$name.log"

  echo "START $name $(date -u)"
  python scripts/train_dflashv2_integer_horizon.py \
    --train-dir "$CACHE/train" \
    --val-dir "$CACHE/val" \
    --output-dir "$out" \
    --epochs "${EPOCHS:-8}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --lr "${LR:-3e-4}" \
    --weight-decay "${WEIGHT_DECAY:-1e-4}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --seed 0 \
    --device cuda \
    --hidden-size "${HIDDEN_SIZE:-512}" \
    --proj-dim "${PROJ_DIM:-512}" \
    --dropout "${DROPOUT:-0.1}" \
    --checkpoint-selection "${CHECKPOINT_SELECTION:-rounded_expected_len_mae}" \
    "$@" > "$log" 2>&1

  echo "TRAIN_DONE $name $(date -u)"
  run_eval "$name" "$out" "$MATH" math500 "$@"
  run_eval "$name" "$out" "$HUMANEVAL" humaneval "$@"
  echo "EVAL_DONE $name $(date -u)"
}

run_one weighted_smoothl1_invsqrt \
  --loss-mode weighted_smooth_l1 \
  --class-weight-scheme inverse_sqrt \
  --class-weight-max 8.0 \
  --ce-weight 0.0 \
  --soft-ce-weight 0.0 \
  --distance-weight 0.0 \
  --emd-weight 0.0

run_one weighted_ce_dist0p2_invsqrt \
  --loss-mode ce_distance \
  --class-weight-scheme inverse_sqrt \
  --class-weight-max 8.0 \
  --ce-weight 1.0 \
  --soft-ce-weight 0.0 \
  --distance-weight 0.2 \
  --emd-weight 0.0

python - "$EVALROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.json")):
    metrics = json.loads(path.read_text())
    stem = path.stem
    name, split = stem.rsplit("_", 1)
    rows.append(
        {
            "name": name,
            "split": split,
            "expected_mae": metrics.get("expected_len_mae"),
            "rounded_mae": metrics.get("rounded_expected_len_mae"),
            "rounded_exact": metrics.get("rounded_expected_len_exact"),
            "argmax_mae": metrics.get("argmax_len_mae"),
            "argmax_exact": metrics.get("argmax_len_exact"),
            "mean_expected": metrics.get("mean_expected_len"),
            "mean_argmax": metrics.get("mean_argmax_len"),
            "mean_target": metrics.get("mean_target_len"),
        }
    )

summary = {"evalroot": str(root), "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
