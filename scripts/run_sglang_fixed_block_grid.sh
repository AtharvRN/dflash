#!/usr/bin/env bash
set -euo pipefail

RUN_BASE_DIR=${RUN_BASE_DIR:-/workspace/dflash-axis2-runs/math500_qwen3_8b_fixed_grid}
RUN_DIR=${RUN_DIR:-${RUN_BASE_DIR}/run_$(date +%Y%m%d_%H%M%S)}
PORT=${PORT:-30000}
BASE_URL=http://127.0.0.1:${PORT}
MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT=${DRAFT:-z-lab/Qwen3-8B-DFlash-b16}
BLOCKS=${BLOCKS:-"8 12 16"}
CONCURRENCIES=${CONCURRENCIES:-"1 8 16 32 64"}
NUM_PROMPTS=${NUM_PROMPTS:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
SERVER_CUDA_VISIBLE_DEVICES=${SERVER_CUDA_VISIBLE_DEVICES:-0}

export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTHONPATH=/root/dflash-src/sglang-dflash-pr23000/python:/sgl-workspace/sglang/python:/workspace/dflash-fresh-zlab-main:${PYTHONPATH:-}
export SGLANG_ENABLE_SPEC_V2=0
export SGLANG_ENABLE_DFLASH_SPEC_V2=0

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/profiles"

for concurrency in $CONCURRENCIES; do
  if (( NUM_PROMPTS < concurrency )); then
    echo "NUM_PROMPTS=${NUM_PROMPTS} must be >= requested concurrency=${concurrency}" >&2
    exit 2
  fi
done

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

start_server() {
  local block=$1
  local label="fixed_b${block}"

  kill_server
  unset SGLANG_DFLASH_SURVIVAL_POLICY_CHECKPOINT
  unset SGLANG_DFLASH_SURVIVAL_ALPHA
  unset SGLANG_DFLASH_SURVIVAL_ARMS

  echo "starting ${label}" | tee "$RUN_DIR/logs/${label}_server_start.txt"
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
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp-size 1 \
    --attention-backend triton \
    --speculative-draft-attention-backend triton \
    --mem-fraction-static 0.70 \
    --max-running-requests 128 \
    --max-total-tokens 65536 \
    --cuda-graph-max-bs 128 \
    --trust-remote-code \
    > "$RUN_DIR/logs/${label}_server.log" 2>&1 &
  echo $! > "$RUN_DIR/logs/${label}_server.pid"
  wait_server
}

run_client() {
  local block=$1
  local concurrency=$2
  local label="fixed_b${block}_c${concurrency}"
  CUDA_VISIBLE_DEVICES="" python3 /workspace/dflash-fresh-zlab-main/scripts/benchmark_sglang_concurrency.py \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset math500 \
    --num-prompts "$NUM_PROMPTS" \
    --concurrency "$concurrency" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --top-p 1.0 \
    --top-k 1 \
    --label "$label" \
    --output-json "$RUN_DIR/results/${label}.json" \
    2>&1 | tee "$RUN_DIR/logs/${label}_client.log"
}

{
  echo "run_dir=$RUN_DIR"
  echo "started=$(date -Is)"
  echo "model=$MODEL"
  echo "draft=$DRAFT"
  echo "blocks=$BLOCKS"
  echo "concurrencies=$CONCURRENCIES"
  echo "num_prompts=$NUM_PROMPTS"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
} > "$RUN_DIR/MANIFEST.txt"

for block in $BLOCKS; do
  start_server "$block"
  for concurrency in $CONCURRENCIES; do
    run_client "$block" "$concurrency"
  done
  kill_server
done

python3 /workspace/dflash-fresh-zlab-main/scripts/summarize_sglang_fixed_grid.py \
  --run-dir "$RUN_DIR" \
  --output-json "$RUN_DIR/summary.json" \
  --output-csv "$RUN_DIR/summary.csv" \
  2>&1 | tee "$RUN_DIR/logs/summary.log"

ln -sfn "$RUN_DIR" "$RUN_BASE_DIR/latest"
echo complete > "$RUN_DIR/STATUS"
echo "$RUN_DIR"
