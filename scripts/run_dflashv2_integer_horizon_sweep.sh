#!/usr/bin/env bash
set -euo pipefail

cd /workspace/dflash-fresh-zlab-main

# This sweep treats the target as an integer horizon H in [0, 15], not as a
# block-arm decision. It expects an already materialized compact last-feature
# cache for training/validation. If the cache split also contains
# predraft_token_ids.npy and predraft_token_mask.npy, the token-tower variant
# can use actual recent token identity.

CACHE=${CACHE:-/tmp/dflashv2_cache/horizon_qwen3_4b_integer_cache}
MATH=${MATH:-/workspace/dflashv2_data/traces/math500_qwen3_4b_b16_eval_20260716_215958}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}

ROOT=${ROOT:-/workspace/dflashv2_data/runs/qwen3_4b_integer_horizon_${STAMP}}
EVALROOT=${EVALROOT:-/workspace/dflashv2_data/evals/qwen3_4b_integer_horizon_${STAMP}}
LOGROOT=${LOGROOT:-/workspace/dflashv2_data/logs/qwen3_4b_integer_horizon_${STAMP}}

mkdir -p "$ROOT" "$EVALROOT" "$LOGROOT"

echo "CACHE=$CACHE"
echo "MATH=$MATH"
echo "ROOT=$ROOT"
echo "EVALROOT=$EVALROOT"
echo "LOGROOT=$LOGROOT"

if [[ ! -f "$CACHE/train/features.npy" || ! -f "$CACHE/val/features.npy" ]]; then
  echo "Missing compact cache under $CACHE/train and $CACHE/val" >&2
  exit 1
fi

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
    --epochs 8 \
    --batch-size 1024 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --seed 0 \
    --device cuda \
    --hidden-size 512 \
    --proj-dim 512 \
    --dropout 0.1 \
    --checkpoint-selection rounded_expected_len_mae \
    "$@" > "$log" 2>&1

  echo "TRAIN_DONE $name $(date -u)"
  python scripts/train_dflashv2_integer_horizon.py \
    --trace-dir "$MATH" \
    --output-dir "$out/eval_math500" \
    --checkpoint "$out/best.pt" \
    --eval-only \
    --batch-size 1024 \
    --num-workers 4 \
    --device cuda \
    "$@" >> "$log" 2>&1

  cp "$out/eval_math500/eval_metrics.json" "$EVALROOT/${name}_math500.json"
  echo "EVAL_DONE $name $(date -u)"
}

run_one ce_dist0p2 \
  --ce-weight 1.0 \
  --soft-ce-weight 0.0 \
  --distance-weight 0.2 \
  --emd-weight 0.0

run_one softce_tau1_dist0p2 \
  --ce-weight 0.0 \
  --soft-ce-weight 1.0 \
  --soft-tau 1.0 \
  --distance-weight 0.2 \
  --emd-weight 0.0

run_one ce_emd0p5_dist0p1 \
  --ce-weight 1.0 \
  --soft-ce-weight 0.0 \
  --distance-weight 0.1 \
  --emd-weight 0.5

run_one softce_tau2_emd0p5_dist0p1 \
  --ce-weight 0.0 \
  --soft-ce-weight 1.0 \
  --soft-tau 2.0 \
  --distance-weight 0.1 \
  --emd-weight 0.5

if [[ -f "$CACHE/train/predraft_token_ids.npy" && -f "$CACHE/train/predraft_token_mask.npy" ]] \
  && find "$MATH" -maxdepth 2 -name predraft_token_ids.npy | grep -q .; then
  run_one token_tower_softce_tau1_dist0p2 \
    --use-token-tower \
    --ce-weight 0.0 \
    --soft-ce-weight 1.0 \
    --soft-tau 1.0 \
    --distance-weight 0.2 \
    --emd-weight 0.0
else
  echo "SKIP token_tower_softce_tau1_dist0p2: train and eval both need predraft token ids"
fi

python - "$EVALROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*_math500.json")):
    metrics = json.loads(path.read_text())
    rows.append(
        {
            "name": path.name.replace("_math500.json", ""),
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
