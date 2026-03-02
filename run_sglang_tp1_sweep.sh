#!/usr/bin/env bash
set -euo pipefail

# TP is intentionally fixed to 1 for this sweep script.
TP_SIZE="${TP_SIZE:-1}"
if [[ "${TP_SIZE}" != "1" ]]; then
  echo "ERROR: run_sglang_tp1_sweep.sh enforces TP_SIZE=1. Got TP_SIZE=${TP_SIZE}." >&2
  exit 1
fi

DATASET_NAME="${DATASET_NAME:-gsm8k}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32}"
QUESTIONS_PER_CONCURRENCY_BASE="${QUESTIONS_PER_CONCURRENCY_BASE:-8}"
MAX_QUESTIONS_PER_CONFIG="${MAX_QUESTIONS_PER_CONFIG:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
ATTENTION_BACKENDS="${ATTENTION_BACKENDS:-flashinfer}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-DFLASH}"
SPECULATIVE_DFLASH_BLOCK_SIZE="${SPECULATIVE_DFLASH_BLOCK_SIZE:-16}"
# Optional explicit block-size list. Supports comma or space separators.
BLOCK_SIZES="${BLOCK_SIZES:-}"
BATCH_REQUESTS="${BATCH_REQUESTS:-1}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
ENABLE_SERVER_METRICS="${ENABLE_SERVER_METRICS:-1}"
SAVE_CALL_TRACE_PROMPT="${SAVE_CALL_TRACE_PROMPT:-0}"
SAVE_CALL_TRACE_RAW_META="${SAVE_CALL_TRACE_RAW_META:-0}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-0}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

RUN_TAG="${RUN_TAG:-sglang_tp1_sweep_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_DIR}/summary.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

conc_raw="${CONCURRENCIES//,/ }"
read -r -a CONC_LIST <<< "${conc_raw}"
if [[ "${#CONC_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: no valid concurrencies in CONCURRENCIES=${CONCURRENCIES}" >&2
  exit 1
fi

declare -a BS_LIST=()
if [[ -n "${BLOCK_SIZES}" ]]; then
  bs_raw="${BLOCK_SIZES//,/ }"
  read -r -a BS_LIST <<< "${bs_raw}"
elif [[ "${SPECULATIVE_ALGORITHM^^}" == "DFLASH" ]]; then
  BS_LIST=("${SPECULATIVE_DFLASH_BLOCK_SIZE}")
else
  BS_LIST=("NA")
fi

if [[ "${SPECULATIVE_ALGORITHM^^}" != "DFLASH" && "${#BS_LIST[@]}" -gt 1 ]]; then
  echo "WARN: BLOCK_SIZES ignored for speculative_algorithm=${SPECULATIVE_ALGORITHM}; using first entry only."
  BS_LIST=("${BS_LIST[0]}")
fi

if [[ "${#BS_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: no valid block sizes resolved." >&2
  exit 1
fi

for bs in "${BS_LIST[@]}"; do
  if [[ "${bs}" == "NA" ]]; then
    continue
  fi
  if ! [[ "${bs}" =~ ^[0-9]+$ ]] || [[ "${bs}" -lt 1 ]]; then
    echo "ERROR: invalid block size '${bs}' in BLOCK_SIZES." >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"

echo "block_size,concurrency,status,baseline_toks_per_s,spec_toks_per_s,speedup,tau,accept_rate,verify_calls_total,verify_per_s,draft_tok_per_s,accepted_tok_per_s,wall_per_verify_s,spec_e2e_avg_s,spec_e2e_p95_s,baseline_e2e_avg_s,draft_time_avg_s,verify_time_avg_s,call_rows,non_null_draft_time_rows,non_null_verify_time_rows,md_path,call_trace_jsonl,log_path" > "${SUMMARY_CSV}"

echo "Running SGLang TP=1 sweep (with block-size sweep support)"
echo "dataset=${DATASET_NAME} model=${TARGET_MODEL} draft=${DRAFT_MODEL}"
echo "concurrency_list=${CONC_LIST[*]}"
echo "block_sizes=${BS_LIST[*]}"
echo "attention_backends=${ATTENTION_BACKENDS} speculative_algorithm=${SPECULATIVE_ALGORITHM}"
echo "max_new_tokens=${MAX_NEW_TOKENS} qpc_base=${QUESTIONS_PER_CONCURRENCY_BASE} max_q_per_config=${MAX_QUESTIONS_PER_CONFIG}"
echo "batch_requests=${BATCH_REQUESTS} skip_baseline=${SKIP_BASELINE} enable_server_metrics=${ENABLE_SERVER_METRICS}"
echo "log_dir=${LOG_DIR}"

for bs in "${BS_LIST[@]}"; do
  for conc in "${CONC_LIST[@]}"; do
    if ! [[ "${conc}" =~ ^[0-9]+$ ]] || [[ "${conc}" -lt 1 ]]; then
      echo "Skipping invalid concurrency entry: ${conc}"
      continue
    fi

    bs_tag="$(echo "${bs}" | tr -c 'A-Za-z0-9._-' '_')"
    md_path="${LOG_DIR}/${RUN_TAG}_${DATASET_NAME}_bs${bs_tag}_c${conc}.md"
    trace_path="${LOG_DIR}/${RUN_TAG}_${DATASET_NAME}_bs${bs_tag}_c${conc}_calls.jsonl"
    log_path="${LOG_DIR}/${RUN_TAG}_${DATASET_NAME}_bs${bs_tag}_c${conc}.log"

    cmd=(
      "${PYTHON_BIN}" benchmark_sglang.py
      --dataset-name "${DATASET_NAME}"
      --target-model "${TARGET_MODEL}"
      --draft-model "${DRAFT_MODEL}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --tp-size 1
      --attention-backends "${ATTENTION_BACKENDS}"
      --concurrencies "${conc}"
      --questions-per-concurrency-base "${QUESTIONS_PER_CONCURRENCY_BASE}"
      --max-questions-per-config "${MAX_QUESTIONS_PER_CONFIG}"
      --max-running-requests "${MAX_RUNNING_REQUESTS}"
      --timeout-s "${TIMEOUT_S}"
      --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
      --output-md "${md_path}"
      --save-call-trace-path "${trace_path}"
    )

    if [[ "${SPECULATIVE_ALGORITHM^^}" == "DFLASH" && "${bs}" != "NA" ]]; then
      cmd+=(--speculative-dflash-block-size "${bs}")
    fi
    if [[ "${BATCH_REQUESTS}" == "1" ]]; then
      cmd+=(--batch-requests)
    fi
    if [[ "${SKIP_BASELINE}" == "1" ]]; then
      cmd+=(--skip-baseline)
    fi
    if [[ "${ENABLE_SERVER_METRICS}" == "1" ]]; then
      cmd+=(--enable-server-metrics)
    fi
    if [[ "${SAVE_CALL_TRACE_PROMPT}" == "1" ]]; then
      cmd+=(--save-call-trace-prompt)
    fi
    if [[ "${SAVE_CALL_TRACE_RAW_META}" == "1" ]]; then
      cmd+=(--save-call-trace-raw-meta)
    fi
    if [[ "${DISABLE_OVERLAP_SCHEDULE}" == "1" ]]; then
      cmd+=(--disable-overlap-schedule)
    fi
    if [[ -n "${SERVER_EXTRA_ARGS}" ]]; then
      cmd+=(--server-extra-args "${SERVER_EXTRA_ARGS}")
    fi

    {
      echo "--------------------------------------------------------"
      printf "Launch: "
      printf "%q " "${cmd[@]}"
      printf "\n"
      echo "--------------------------------------------------------"
    } | tee "${log_path}"

    set +e
    "${cmd[@]}" 2>&1 | tee -a "${log_path}"
    status=${PIPESTATUS[0]}
    set -e

    row="$("${PYTHON_BIN}" - "${md_path}" "${trace_path}" "${status}" "${bs}" "${conc}" "${log_path}" <<'PY'
import csv
import io
import json
import sys
from pathlib import Path

md_path = Path(sys.argv[1])
trace_path = Path(sys.argv[2])
status_code = int(sys.argv[3])
block_size = sys.argv[4]
concurrency = int(sys.argv[5])
log_path = sys.argv[6]

SECTION_TO_COL = {
    "Baseline output tok/s": "baseline_toks_per_s",
    "DFLASH output tok/s": "spec_toks_per_s",
    "Speedup (DFLASH / baseline)": "speedup",
    "DFLASH tau (accept length)": "tau",
    "DFLASH acceptance rate": "accept_rate",
    "DFLASH verify calls total": "verify_calls_total",
    "DFLASH verify calls per second": "verify_per_s",
    "DFLASH drafted tokens per second": "draft_tok_per_s",
    "DFLASH accepted draft tokens per second": "accepted_tok_per_s",
    "DFLASH wall time per verify call (s)": "wall_per_verify_s",
    "Request E2E latency avg (s)": "spec_e2e_avg_s",
    "Request E2E latency p95 (s)": "spec_e2e_p95_s",
    "Baseline request E2E latency avg (s)": "baseline_e2e_avg_s",
    "DFLASH reported draft time avg (s, if exposed by server)": "draft_time_avg_s",
    "DFLASH reported verify time avg (s, if exposed by server)": "verify_time_avg_s",
}

OUT_COLS = [
    "baseline_toks_per_s",
    "spec_toks_per_s",
    "speedup",
    "tau",
    "accept_rate",
    "verify_calls_total",
    "verify_per_s",
    "draft_tok_per_s",
    "accepted_tok_per_s",
    "wall_per_verify_s",
    "spec_e2e_avg_s",
    "spec_e2e_p95_s",
    "baseline_e2e_avg_s",
    "draft_time_avg_s",
    "verify_time_avg_s",
]

values = {k: "N/A" for k in OUT_COLS}

def split_row(line: str):
    return [x.strip() for x in line.strip().strip("|").split("|")]

def parse_section_table(lines: list[str], title: str):
    key = f"### {title}"
    for i, line in enumerate(lines):
        if line.strip() != key:
            continue
        header = None
        value = None
        for j in range(i + 1, min(i + 16, len(lines))):
            s = lines[j].strip()
            if s.startswith("### "):
                break
            if s.startswith("| conc |"):
                header = split_row(s)
            elif s.startswith("| value |"):
                value = split_row(s)
        if not header or not value:
            return {}
        out = {}
        for h, v in zip(header[1:], value[1:]):
            try:
                out[int(h)] = v
            except ValueError:
                continue
        return out
    return {}

if status_code == 0 and md_path.exists():
    lines = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for section, col in SECTION_TO_COL.items():
        table = parse_section_table(lines, section)
        if concurrency in table:
            values[col] = table[concurrency]

call_rows = 0
non_null_draft_time_rows = 0
non_null_verify_time_rows = 0
if trace_path.exists():
    with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            call_rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("draft_time_s") is not None:
                non_null_draft_time_rows += 1
            if obj.get("verify_time_s") is not None:
                non_null_verify_time_rows += 1

buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow([
    block_size,
    concurrency,
    "OK" if status_code == 0 else "ERROR",
    values["baseline_toks_per_s"],
    values["spec_toks_per_s"],
    values["speedup"],
    values["tau"],
    values["accept_rate"],
    values["verify_calls_total"],
    values["verify_per_s"],
    values["draft_tok_per_s"],
    values["accepted_tok_per_s"],
    values["wall_per_verify_s"],
    values["spec_e2e_avg_s"],
    values["spec_e2e_p95_s"],
    values["baseline_e2e_avg_s"],
    values["draft_time_avg_s"],
    values["verify_time_avg_s"],
    call_rows,
    non_null_draft_time_rows,
    non_null_verify_time_rows,
    str(md_path),
    str(trace_path),
    log_path,
])
print(buf.getvalue().strip())
PY
)"

    echo "${row}" >> "${SUMMARY_CSV}"
  done
done

echo "Sweep complete. Summary: ${SUMMARY_CSV}"
