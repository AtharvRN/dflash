#!/usr/bin/env bash
set -euo pipefail

cd /workspace/dflash-fresh-zlab-main

CACHE=${CACHE:-/tmp/dflashv2_cache/horizon_qwen3_4b_fixedval5k_hazard_partial323k_20260717_012251}
TRACE=${TRACE:-/workspace/dflashv2_data/traces/dflashv2_qwen3_4b_b16_instruct100k_partial_snapshot_20260716_2100}
MATH=${MATH:-/workspace/dflashv2_data/traces/math500_qwen3_4b_b16_eval_20260716_215958}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}

ROOT=${ROOT:-/workspace/dflashv2_data/runs/qwen3_4b_partial323k_ablation_${STAMP}}
EVALROOT=${EVALROOT:-/workspace/dflashv2_data/evals/qwen3_4b_partial323k_ablation_${STAMP}}
LOGROOT=${LOGROOT:-/workspace/dflashv2_data/logs/qwen3_4b_partial323k_ablation_${STAMP}}

mkdir -p "$ROOT" "$EVALROOT" "$LOGROOT"

echo "ROOT=$ROOT"
echo "EVALROOT=$EVALROOT"
echo "LOGROOT=$LOGROOT"

run_one() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  local log="$LOGROOT/$name.log"

  echo "START $name $(date -u)"
  python scripts/train_dflashv2_horizon_predictor.py \
    --trace-dir "$TRACE" \
    --compact-cache-dir "$CACHE" \
    --output-dir "$out" \
    --architecture last_mlp \
    --epochs 8 \
    --batch-size 1024 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --seed 0 \
    --device cuda \
    --monotonicize-eval \
    --arms 4,8,12,16 \
    --alphas 0.8,0.85,0.9,0.92,0.95 \
    --selection-min-retention 0.95 \
    --checkpoint-selection selected_accept_ratio \
    "$@" > "$log" 2>&1

  echo "TRAIN_DONE $name $(date -u)"
  local objective
  objective=$(python - "$out/config.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["objective"])
PY
)

  python scripts/train_dflashv2_horizon_predictor.py \
    --trace-dir "$MATH" \
    --output-dir "$out/eval_math500" \
    --checkpoint "$out/best.pt" \
    --eval-only \
    --architecture last_mlp \
    --objective "$objective" \
    --device cuda \
    --monotonicize-eval \
    --arms 4,8,12,16 \
    --alphas 0.8,0.85,0.9,0.92,0.95 \
    --selection-min-retention 0.95 \
    --checkpoint-selection selected_accept_ratio \
    "$@" \
    >> "$log" 2>&1

  cp "$out/eval_math500/eval_metrics.json" "$EVALROOT/${name}_math500.json"
  echo "EVAL_DONE $name $(date -u)"
}

run_one surv_len0 \
  --objective survival_bce \
  --length-loss-weight 0.0 \
  --monotonic-weight 0.02 \
  --boundary-weight 1.0 \
  --boundary-ks 4,8,12,15

run_one surv_len0p1 \
  --objective survival_bce \
  --length-loss-weight 0.1 \
  --monotonic-weight 0.02 \
  --boundary-weight 1.0 \
  --boundary-ks 4,8,12,15

run_one surv_bw2 \
  --objective survival_bce \
  --length-loss-weight 0.05 \
  --monotonic-weight 0.02 \
  --boundary-weight 2.0 \
  --boundary-ks 4,8,12,15

run_one surv_aux0p05 \
  --objective survival_bce \
  --length-loss-weight 0.05 \
  --monotonic-weight 0.02 \
  --boundary-weight 1.0 \
  --boundary-ks 4,8,12,15 \
  --aux-arm-weight 0.05

run_one hazard_len0p05 \
  --objective hazard \
  --length-loss-weight 0.05 \
  --monotonic-weight 0.0 \
  --boundary-weight 1.0 \
  --boundary-ks 4,8,12,15

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
            "alpha": metrics.get("selected_alpha"),
            "block": metrics.get("selected_mean_block"),
            "accepted": metrics.get("selected_mean_accepted"),
            "retention": metrics.get("selected_accept_retention"),
            "ratio": metrics.get("selected_accept_ratio"),
            "feasible": metrics.get("selected_feasible"),
            "mae": metrics.get("expected_len_mae"),
            "auroc8": metrics.get("auroc_h_ge_8"),
            "auroc12": metrics.get("auroc_h_ge_12"),
        }
    )

summary = {"evalroot": str(root), "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
