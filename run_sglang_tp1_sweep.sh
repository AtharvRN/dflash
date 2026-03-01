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

mkdir -p "${LOG_DIR}"

echo "concurrency,status,baseline_toks_per_s,spec_toks_per_s,speedup,tau,accept_rate,verify_per_s,draft_tok_per_s,spec_verify_ct_sum,call_rows,md_path,call_trace_jsonl,log_path" > "${SUMMARY_CSV}"

echo "Running SGLang TP=1 sweep"
echo "dataset=${DATASET_NAME} model=${TARGET_MODEL} draft=${DRAFT_MODEL}"
echo "concurrency_list=${CONC_LIST[*]}"
echo "attention_backends=${ATTENTION_BACKENDS} speculative_algorithm=${SPECULATIVE_ALGORITHM}"
echo "max_new_tokens=${MAX_NEW_TOKENS} qpc_base=${QUESTIONS_PER_CONCURRENCY_BASE} max_q_per_config=${MAX_QUESTIONS_PER_CONFIG}"
echo "batch_requests=${BATCH_REQUESTS} skip_baseline=${SKIP_BASELINE} enable_server_metrics=${ENABLE_SERVER_METRICS}"
echo "log_dir=${LOG_DIR}"

for conc in "${CONC_LIST[@]}"; do
  if ! [[ "${conc}" =~ ^[0-9]+$ ]] || [[ "${conc}" -lt 1 ]]; then
    echo "Skipping invalid concurrency entry: ${conc}"
    continue
  fi

  md_path="${LOG_DIR}/${RUN_TAG}_${DATASET_NAME}_c${conc}.md"
  trace_path="${LOG_DIR}/${RUN_TAG}_${DATASET_NAME}_c${conc}_calls.jsonl"
  log_path="${LOG_DIR}/${RUN_TAG}_${DATASET_NAME}_c${conc}.log"

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

  if [[ "${SPECULATIVE_ALGORITHM^^}" == "DFLASH" ]] && [[ -n "${SPECULATIVE_DFLASH_BLOCK_SIZE}" ]]; then
    cmd+=(--speculative-dflash-block-size "${SPECULATIVE_DFLASH_BLOCK_SIZE}")
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

  call_rows=0
  if [[ -f "${trace_path}" ]]; then
    call_rows="$(wc -l < "${trace_path}" | tr -d ' ')"
  fi

  row="$("${PYTHON_BIN}" - "${log_path}" "${conc}" "${md_path}" "${trace_path}" "${log_path}" "${status}" "${call_rows}" <<'PY'
import csv
import io
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
conc = int(sys.argv[2])
md_path = sys.argv[3]
trace_path = sys.argv[4]
log_path_arg = sys.argv[5]
status_code = int(sys.argv[6])
call_rows = int(sys.argv[7])

def parse_num(tok: str):
    tok = tok.strip()
    if tok in {"", "N/A", "NA"}:
        return None
    tok = tok.replace(",", "")
    try:
        return float(tok)
    except ValueError:
        return None

baseline_toks = None
spec_toks = None
tau = None
accept_rate = None
verify_per_s = None
draft_tok_per_s = None
spec_verify_ct_sum = None

if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if line.startswith("[baseline]"):
            m = re.search(r"toks/s=([0-9,\\.]+)", line)
            if m:
                baseline_toks = parse_num(m.group(1))
        if line.startswith("[") and "toks/s=" in line and "tau=" in line and "spec_verify_ct_sum=" in line:
            m_tps = re.search(r"toks/s=([0-9,\\.]+)", line)
            m_tau = re.search(r"tau=([^\\s]+)", line)
            m_acc = re.search(r"accept_rate=([^\\s]+)", line)
            m_verify = re.search(r"verify/s=([^\\s]+)", line)
            m_draft = re.search(r"draft_tok/s=([^\\s]+)", line)
            m_vsum = re.search(r"spec_verify_ct_sum=([0-9,]+)", line)
            if m_tps:
                spec_toks = parse_num(m_tps.group(1))
            if m_tau:
                tau = parse_num(m_tau.group(1))
            if m_acc:
                accept_rate = parse_num(m_acc.group(1))
            if m_verify:
                verify_per_s = parse_num(m_verify.group(1))
            if m_draft:
                draft_tok_per_s = parse_num(m_draft.group(1))
            if m_vsum:
                spec_verify_ct_sum = parse_num(m_vsum.group(1))

speedup = None
if baseline_toks and spec_toks and baseline_toks > 0:
    speedup = spec_toks / baseline_toks

buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow([
    conc,
    "OK" if status_code == 0 else "ERROR",
    baseline_toks,
    spec_toks,
    speedup,
    tau,
    accept_rate,
    verify_per_s,
    draft_tok_per_s,
    spec_verify_ct_sum,
    call_rows,
    md_path,
    trace_path,
    log_path_arg,
])
print(buf.getvalue().strip())
PY
)"

  echo "${row}" >> "${SUMMARY_CSV}"
done

echo "Sweep complete. Summary: ${SUMMARY_CSV}"
