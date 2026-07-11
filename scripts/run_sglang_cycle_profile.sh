#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/workspace/dflash-axis2-runs/math500_qwen3_8b_cycle_profile/run_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-30000}
BASE_URL=http://127.0.0.1:${PORT}
MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT=${DRAFT:-z-lab/Qwen3-8B-DFlash-b16}
POLICY=${POLICY:-/workspace/dflash-axis2-runs/math500_qwen3_8b_survival/latest/models/internal_window_gru16_survival/best.pt}
CONCURRENCY=${CONCURRENCY:-64}
NUM_PROMPTS=${NUM_PROMPTS:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
SERVER_CUDA_VISIBLE_DEVICES=${SERVER_CUDA_VISIBLE_DEVICES:-0}
VARIANTS=${VARIANTS:-"fixed_b8 fixed_b12 fixed_b16 survival_alpha090"}

export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTHONPATH=/root/dflash-src/sglang-dflash-pr23000/python:/sgl-workspace/sglang/python:/workspace/dflash-fresh-zlab-main:${PYTHONPATH:-}
export SGLANG_ENABLE_SPEC_V2=0
export SGLANG_ENABLE_DFLASH_SPEC_V2=0

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/profiles"

kill_server() {
  pkill -TERM -f "sglang.launch_server" 2>/dev/null || true
  sleep 8
  pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
}

wait_server() {
  python3 - <<PY
import requests
import sys
import time

url = "${BASE_URL}/health"
deadline = time.time() + 900
last = None
while time.time() < deadline:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("server ready", flush=True)
            sys.exit(0)
        last = f"status={response.status_code} text={response.text[:200]}"
    except Exception as exc:
        last = repr(exc)
    time.sleep(5)
print("server did not become ready", last, flush=True)
sys.exit(1)
PY
}

run_client() {
  local label=$1
  CUDA_VISIBLE_DEVICES="" python3 /workspace/dflash-fresh-zlab-main/scripts/benchmark_sglang_concurrency.py \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset math500 \
    --num-prompts "$NUM_PROMPTS" \
    --concurrency "$CONCURRENCY" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --top-p 1.0 \
    --top-k 1 \
    --label "$label" \
    --output-json "$RUN_DIR/results/${label}.json" \
    2>&1 | tee "$RUN_DIR/logs/${label}_client.log"
}

start_server() {
  local label=$1
  local block=$2
  local mode=$3
  local alpha=${4:-}
  local dynamic_flags=()

  kill_server
  unset SGLANG_DFLASH_SURVIVAL_POLICY_CHECKPOINT
  unset SGLANG_DFLASH_SURVIVAL_ALPHA
  unset SGLANG_DFLASH_SURVIVAL_ARMS
  unset SGLANG_DFLASH_PROFILE_CYCLES
  unset SGLANG_DFLASH_PROFILE_LOG
  unset SGLANG_DFLASH_PROFILE_LOG_EVERY

  if [[ "$mode" == "survival" ]]; then
    export SGLANG_DFLASH_SURVIVAL_POLICY_CHECKPOINT="$POLICY"
    export SGLANG_DFLASH_SURVIVAL_ALPHA="$alpha"
    export SGLANG_DFLASH_SURVIVAL_ARMS="4,8,12,16"
    dynamic_flags=(--speculative-dflash-dynamic-block-size --speculative-dflash-dynamic-block-arms 4,8,12,16 --speculative-dflash-dynamic-warmup-batches 999999)
  fi

  echo "starting $label block=$block mode=$mode alpha=$alpha" | tee "$RUN_DIR/logs/${label}_server_start.txt"
  CUDA_VISIBLE_DEVICES="$SERVER_CUDA_VISIBLE_DEVICES" \
  SGLANG_DFLASH_PROFILE_CYCLES=1 \
  SGLANG_DFLASH_PROFILE_LOG="$RUN_DIR/profiles/${label}.jsonl" \
  SGLANG_DFLASH_PROFILE_LOG_EVERY=100 \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-dflash-block-size "$block" \
    --speculative-num-draft-tokens "$block" \
    "${dynamic_flags[@]}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp-size 1 \
    --attention-backend triton \
    --speculative-draft-attention-backend triton \
    --mem-fraction-static 0.75 \
    --max-running-requests 128 \
    --max-total-tokens 131072 \
    --cuda-graph-max-bs 128 \
    --trust-remote-code \
    > "$RUN_DIR/logs/${label}_server.log" 2>&1 &
  echo $! > "$RUN_DIR/logs/${label}_server.pid"
  wait_server
}

run_variant() {
  local label=$1
  case "$label" in
    fixed_b8) start_server "$label" 8 fixed ;;
    fixed_b12) start_server "$label" 12 fixed ;;
    fixed_b16) start_server "$label" 16 fixed ;;
    survival_alpha090) start_server "$label" 16 survival 0.90 ;;
    survival_alpha085) start_server "$label" 16 survival 0.85 ;;
    *) echo "unknown variant: $label" >&2; exit 2 ;;
  esac
  run_client "$label"
  kill_server
}

{
  echo "run_dir=$RUN_DIR"
  echo "started=$(date -Is)"
  echo "model=$MODEL"
  echo "draft=$DRAFT"
  echo "policy=$POLICY"
  echo "concurrency=$CONCURRENCY"
  echo "num_prompts=$NUM_PROMPTS"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "variants=$VARIANTS"
  echo "profile_sync=true"
} > "$RUN_DIR/MANIFEST.txt"

for variant in $VARIANTS; do
  run_variant "$variant"
done

python3 /workspace/dflash-fresh-zlab-main/scripts/summarize_dflash_cycle_profiles.py \
  --profile-dir "$RUN_DIR/profiles" \
  --output-json "$RUN_DIR/profile_summary.json" \
  2>&1 | tee "$RUN_DIR/logs/profile_summary.log"

ln -sfn "$RUN_DIR" /workspace/dflash-axis2-runs/math500_qwen3_8b_cycle_profile/latest
echo complete > "$RUN_DIR/STATUS"
echo "$RUN_DIR"
