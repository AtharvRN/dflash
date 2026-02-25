#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-aime25}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3-4B-DFlash-b16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CANDIDATE_MODE="${CANDIDATE_MODE:-fixed_prefix_rank}"
SPARSE_MAX_POSITIONS="${SPARSE_MAX_POSITIONS:-4}"
RUN_TAG="${RUN_TAG:-fixed_prefix_sweep_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_DIR}/summary.csv}"
SAVE_OUTPUTS_DIR="${SAVE_OUTPUTS_DIR:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VANILLA_REF="${RUN_VANILLA_REF:-1}"
COLLECT_PROFILE="${COLLECT_PROFILE:-1}"
SAVE_CYCLE_TRACE="${SAVE_CYCLE_TRACE:-${COLLECT_PROFILE}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
CANDIDATE_EXTRA_ARGS="${CANDIDATE_EXTRA_ARGS:-}"

if [[ -n "${FIXED_PREFIX_LENS:-}" ]]; then
  read -r -a PREFIX_LIST <<< "${FIXED_PREFIX_LENS}"
else
  PREFIX_LIST=(4 5 6 7 8)
fi

if [[ -n "${TOP_K_LIST:-}" ]]; then
  read -r -a TOPK_LIST <<< "${TOP_K_LIST}"
else
  TOPK_LIST=(2 3 4)
fi

if [[ -n "${MAX_CANDIDATES_LIST:-}" ]]; then
  read -r -a MAXC_LIST <<< "${MAX_CANDIDATES_LIST}"
else
  MAXC_LIST=(2 3 4)
fi

mkdir -p "${LOG_DIR}"
if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
  mkdir -p "${SAVE_OUTPUTS_DIR}"
fi

cat > "${SUMMARY_CSV}" <<'EOF'
dataset,max_samples,block_size,candidate_mode,fixed_prefix_len,sparse_max_positions,branch_top_k,max_candidates,e2e_speedup_vs_bs1,tpot_speedup_vs_bs1,throughput_gain_vs_bs1,e2e_speedup_vs_vanilla_bs,tpot_speedup_vs_vanilla_bs,throughput_gain_vs_vanilla_bs,tau,speculative_total_wall_s,speculative_tokens_per_sec,speculative_tpot,speculative_ttft,spec_avg_target_decode_s,spec_avg_draft_decode_s,spec_target_share,spec_draft_share,spec_total_profiled_cycles,avg_candidates_per_cycle,avg_verify_calls_per_sample,baseline_total_wall_s,baseline_tokens_per_sec,baseline_tpot,vanilla_total_wall_s,vanilla_tokens_per_sec,vanilla_tpot,baseline_log_path,vanilla_log_path,log_path,output_jsonl_path,cycle_jsonl_path
EOF

is_number() {
  [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

ratio_or_na() {
  local num="${1:-}"
  local den="${2:-}"
  if is_number "${num}" && is_number "${den}"; then
    awk -v n="${num}" -v d="${den}" 'BEGIN { if (d > 0) printf "%.6f", n / d; else print "NA" }'
  else
    echo "NA"
  fi
}

extract_metric() {
  local pattern="$1"
  local log_path="$2"
  (grep -Eo "${pattern}" "${log_path}" || true) | tail -1 | awk '{print $NF}'
}

extract_after_colon() {
  local pattern="$1"
  local log_path="$2"
  (grep -E "^${pattern}" "${log_path}" || true) | tail -1 | sed "s/^${pattern}: //"
}

# Allow reusing an externally supplied shared baseline when RUN_BASELINE=0.
BASELINE_TOTAL_WALL_S="${BASELINE_TOTAL_WALL_S:-NA}"
BASELINE_TOKENS_PER_SEC="${BASELINE_TOKENS_PER_SEC:-NA}"
BASELINE_TPOT="${BASELINE_TPOT:-NA}"
BASELINE_LOG_PATH="${LOG_DIR}/${DATASET}_baseline_bs1.log"
VANILLA_TOTAL_WALL_S="${VANILLA_TOTAL_WALL_S:-NA}"
VANILLA_TOKENS_PER_SEC="${VANILLA_TOKENS_PER_SEC:-NA}"
VANILLA_TPOT="${VANILLA_TPOT:-NA}"
VANILLA_LOG_PATH="${LOG_DIR}/${DATASET}_vanilla_bs${BLOCK_SIZE}.log"

echo "Running fixed-prefix candidate sweep"
echo "dataset=${DATASET} max_samples=${MAX_SAMPLES} block_size=${BLOCK_SIZE} max_new_tokens=${MAX_NEW_TOKENS}"
echo "fixed_prefix_lens=${PREFIX_LIST[*]} top_k_list=${TOPK_LIST[*]} max_candidates_list=${MAXC_LIST[*]}"
echo "candidate_mode=${CANDIDATE_MODE} sparse_max_positions=${SPARSE_MAX_POSITIONS}"
if [[ -n "${CANDIDATE_EXTRA_ARGS}" ]]; then
  echo "candidate_extra_args=${CANDIDATE_EXTRA_ARGS}"
fi
echo "collect_profile=${COLLECT_PROFILE} run_baseline=${RUN_BASELINE} run_vanilla_ref=${RUN_VANILLA_REF} logs=${LOG_DIR}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  baseline_out_path=""
  if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
    baseline_out_path="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_baseline_bs1.jsonl"
  fi
  baseline_cmd=(
    "${PYTHON_BIN}"
    benchmark.py
    --dataset "${DATASET}"
    --max-samples "${MAX_SAMPLES}"
    --model-name-or-path "${MODEL}"
    --draft-name-or-path "${DRAFT}"
    --block-size 1
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
  )
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    baseline_cmd+=(--local-files-only)
  fi
  if [[ "${COLLECT_PROFILE}" == "1" ]]; then
    baseline_cmd+=(--collect-profile)
  fi
  if [[ -n "${baseline_out_path}" ]]; then
    baseline_cmd+=(--save-outputs-path "${baseline_out_path}")
  fi

  echo "--------------------------------------------------------" | tee "${BASELINE_LOG_PATH}"
  printf "Launch baseline: " | tee -a "${BASELINE_LOG_PATH}"
  printf "%q " "${baseline_cmd[@]}" | tee -a "${BASELINE_LOG_PATH}"
  printf "\n" | tee -a "${BASELINE_LOG_PATH}"
  echo "--------------------------------------------------------" | tee -a "${BASELINE_LOG_PATH}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    BASELINE_TOTAL_WALL_S="DRY_RUN"
    BASELINE_TOKENS_PER_SEC="DRY_RUN"
    BASELINE_TPOT="DRY_RUN"
  else
    set +e
    "${baseline_cmd[@]}" 2>&1 | tee -a "${BASELINE_LOG_PATH}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -ne 0 ]]; then
      echo "Baseline run failed (status=${status})." | tee -a "${BASELINE_LOG_PATH}"
      exit "${status}"
    fi
    BASELINE_TOTAL_WALL_S="$(extract_metric 'Baseline total_wall_s: [0-9.]+' "${BASELINE_LOG_PATH}")"
    BASELINE_TOKENS_PER_SEC="$(extract_metric 'Baseline tokens_per_sec: [0-9.]+' "${BASELINE_LOG_PATH}")"
    BASELINE_TPOT="$(extract_metric 'Baseline TPOT: [0-9.]+' "${BASELINE_LOG_PATH}")"
    BASELINE_TOTAL_WALL_S="${BASELINE_TOTAL_WALL_S:-NA}"
    BASELINE_TOKENS_PER_SEC="${BASELINE_TOKENS_PER_SEC:-NA}"
    BASELINE_TPOT="${BASELINE_TPOT:-NA}"
    echo "Shared baseline: total_wall_s=${BASELINE_TOTAL_WALL_S} tokens_per_sec=${BASELINE_TOKENS_PER_SEC} tpot=${BASELINE_TPOT}"
  fi
fi

if [[ "${RUN_VANILLA_REF}" == "1" ]]; then
  vanilla_out_path=""
  if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
    vanilla_out_path="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_vanilla_bs${BLOCK_SIZE}.jsonl"
  fi
  vanilla_cmd=(
    "${PYTHON_BIN}"
    benchmark.py
    --dataset "${DATASET}"
    --max-samples "${MAX_SAMPLES}"
    --model-name-or-path "${MODEL}"
    --draft-name-or-path "${DRAFT}"
    --block-size "${BLOCK_SIZE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --skip-baseline
  )
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    vanilla_cmd+=(--local-files-only)
  fi
  if [[ "${COLLECT_PROFILE}" == "1" ]]; then
    vanilla_cmd+=(--collect-profile)
  fi
  if [[ -n "${vanilla_out_path}" ]]; then
    vanilla_cmd+=(--save-outputs-path "${vanilla_out_path}")
  fi

  echo "--------------------------------------------------------" | tee "${VANILLA_LOG_PATH}"
  printf "Launch vanilla reference (bs=%s): " "${BLOCK_SIZE}" | tee -a "${VANILLA_LOG_PATH}"
  printf "%q " "${vanilla_cmd[@]}" | tee -a "${VANILLA_LOG_PATH}"
  printf "\n" | tee -a "${VANILLA_LOG_PATH}"
  echo "--------------------------------------------------------" | tee -a "${VANILLA_LOG_PATH}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    VANILLA_TOTAL_WALL_S="DRY_RUN"
    VANILLA_TOKENS_PER_SEC="DRY_RUN"
    VANILLA_TPOT="DRY_RUN"
  else
    set +e
    "${vanilla_cmd[@]}" 2>&1 | tee -a "${VANILLA_LOG_PATH}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -ne 0 ]]; then
      echo "Vanilla reference run failed (status=${status})." | tee -a "${VANILLA_LOG_PATH}"
      exit "${status}"
    fi
    VANILLA_TOTAL_WALL_S="$(extract_metric 'Speculative total_wall_s: [0-9.]+' "${VANILLA_LOG_PATH}")"
    VANILLA_TOKENS_PER_SEC="$(extract_metric 'Speculative tokens_per_sec: [0-9.]+' "${VANILLA_LOG_PATH}")"
    VANILLA_TPOT="$(extract_metric 'Speculative TPOT: [0-9.]+' "${VANILLA_LOG_PATH}")"
    VANILLA_TOTAL_WALL_S="${VANILLA_TOTAL_WALL_S:-NA}"
    VANILLA_TOKENS_PER_SEC="${VANILLA_TOKENS_PER_SEC:-NA}"
    VANILLA_TPOT="${VANILLA_TPOT:-NA}"
    echo "Vanilla reference: total_wall_s=${VANILLA_TOTAL_WALL_S} tokens_per_sec=${VANILLA_TOKENS_PER_SEC} tpot=${VANILLA_TPOT}"
  fi
fi

for prefix_len in "${PREFIX_LIST[@]}"; do
  for top_k in "${TOPK_LIST[@]}"; do
    for max_c in "${MAXC_LIST[@]}"; do
      if [[ "${CANDIDATE_MODE}" == "fixed_prefix_rank" && ${max_c} -gt ${top_k} ]]; then
        echo "Skipping config prefix=${prefix_len} top_k=${top_k} max_candidates=${max_c} (max_candidates > top_k)"
        continue
      fi

      cfg_tag="p${prefix_len}_k${top_k}_c${max_c}"
      log_path="${LOG_DIR}/${DATASET}_${cfg_tag}.log"
      out_path=""
      cycle_path=""
      if [[ -n "${SAVE_OUTPUTS_DIR}" ]]; then
        out_path="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_${cfg_tag}.jsonl"
        cycle_path="${SAVE_OUTPUTS_DIR}/${RUN_TAG}_${DATASET}_${cfg_tag}_cycle.jsonl"
      fi

      cmd=(
        "${PYTHON_BIN}"
        benchmark_candidate_solutions.py
        --dataset "${DATASET}"
        --max-samples "${MAX_SAMPLES}"
        --model-name-or-path "${MODEL}"
        --draft-name-or-path "${DRAFT}"
        --block-size "${BLOCK_SIZE}"
        --max-new-tokens "${MAX_NEW_TOKENS}"
        --temperature "${TEMPERATURE}"
        --candidate-mode "${CANDIDATE_MODE}"
        --fixed-prefix-len "${prefix_len}"
        --sparse-max-positions "${SPARSE_MAX_POSITIONS}"
        --branch-top-k "${top_k}"
        --max-candidates "${max_c}"
        --skip-baseline
      )
      if [[ -n "${CANDIDATE_EXTRA_ARGS}" ]]; then
        read -r -a EXTRA_ARGS_ARR <<< "${CANDIDATE_EXTRA_ARGS}"
        cmd+=("${EXTRA_ARGS_ARR[@]}")
      fi
      if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
        cmd+=(--local-files-only)
      fi
      if [[ "${COLLECT_PROFILE}" == "1" ]]; then
        cmd+=(--collect-profile)
      fi
      if [[ -n "${out_path}" ]]; then
        cmd+=(--save-outputs-path "${out_path}")
      fi
      if [[ "${SAVE_CYCLE_TRACE}" == "1" && -n "${cycle_path}" ]]; then
        cmd+=(--save-cycle-trace-path "${cycle_path}")
      else
        cycle_path=""
      fi

      echo "--------------------------------------------------------" | tee "${log_path}"
      printf "Launch: " | tee -a "${log_path}"
      printf "%q " "${cmd[@]}" | tee -a "${log_path}"
      printf "\n" | tee -a "${log_path}"
      echo "--------------------------------------------------------" | tee -a "${log_path}"

      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "${DATASET},${MAX_SAMPLES},${BLOCK_SIZE},${CANDIDATE_MODE},${prefix_len},${SPARSE_MAX_POSITIONS},${top_k},${max_c},DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,DRY_RUN,${BASELINE_TOTAL_WALL_S},${BASELINE_TOKENS_PER_SEC},${BASELINE_TPOT},${VANILLA_TOTAL_WALL_S},${VANILLA_TOKENS_PER_SEC},${VANILLA_TPOT},${BASELINE_LOG_PATH},${VANILLA_LOG_PATH},${log_path},${out_path:-NA},${cycle_path:-NA}" >> "${SUMMARY_CSV}"
        continue
      fi

      set +e
      "${cmd[@]}" 2>&1 | tee -a "${log_path}"
      status=${PIPESTATUS[0]}
      set -e
      if [[ "${status}" -ne 0 ]]; then
        echo "${DATASET},${MAX_SAMPLES},${BLOCK_SIZE},${CANDIDATE_MODE},${prefix_len},${SPARSE_MAX_POSITIONS},${top_k},${max_c},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,${BASELINE_TOTAL_WALL_S},${BASELINE_TOKENS_PER_SEC},${BASELINE_TPOT},${VANILLA_TOTAL_WALL_S},${VANILLA_TOKENS_PER_SEC},${VANILLA_TPOT},${BASELINE_LOG_PATH},${VANILLA_LOG_PATH},${log_path},${out_path:-NA},${cycle_path:-NA}" >> "${SUMMARY_CSV}"
        continue
      fi

      tau="$(extract_metric 'Average Acceptance length: [0-9.]+' "${log_path}")"
      spec_wall="$(extract_metric 'Speculative total_wall_s: [0-9.]+' "${log_path}")"
      spec_tps="$(extract_metric 'Speculative tokens_per_sec: [0-9.]+' "${log_path}")"
      spec_tpot="$(extract_metric 'Speculative TPOT: [0-9.]+' "${log_path}")"
      spec_ttft="$(extract_metric 'Speculative TTFT: [0-9.]+' "${log_path}")"
      spec_target_decode="$(extract_metric 'Speculative profile avg_target_decode_s: [0-9.]+' "${log_path}")"
      spec_draft_decode="$(extract_metric 'Speculative profile avg_draft_decode_s: [0-9.]+' "${log_path}")"
      spec_target_share="$(extract_metric 'Speculative profile target_share_decode: [0-9.]+' "${log_path}")"
      spec_draft_share="$(extract_metric 'Speculative profile draft_share_decode: [0-9.]+' "${log_path}")"
      spec_cycles="$(extract_metric 'Speculative profile total_profiled_cycles: [0-9.]+' "${log_path}")"
      avg_cands="$(extract_metric 'Candidate avg_candidates_per_cycle: [0-9.]+' "${log_path}")"
      avg_verify_calls="$(extract_metric 'Candidate avg_verify_calls_per_sample: [0-9.]+' "${log_path}")"

      tau="${tau:-NA}"
      spec_wall="${spec_wall:-NA}"
      spec_tps="${spec_tps:-NA}"
      spec_tpot="${spec_tpot:-NA}"
      spec_ttft="${spec_ttft:-NA}"
      spec_target_decode="${spec_target_decode:-NA}"
      spec_draft_decode="${spec_draft_decode:-NA}"
      spec_target_share="${spec_target_share:-NA}"
      spec_draft_share="${spec_draft_share:-NA}"
      spec_cycles="${spec_cycles:-NA}"
      avg_cands="${avg_cands:-NA}"
      avg_verify_calls="${avg_verify_calls:-NA}"

      e2e_speedup_bs1="$(ratio_or_na "${BASELINE_TOTAL_WALL_S}" "${spec_wall}")"
      tpot_speedup_bs1="$(ratio_or_na "${BASELINE_TPOT}" "${spec_tpot}")"
      throughput_gain_bs1="$(ratio_or_na "${spec_tps}" "${BASELINE_TOKENS_PER_SEC}")"
      e2e_speedup_vanilla="$(ratio_or_na "${VANILLA_TOTAL_WALL_S}" "${spec_wall}")"
      tpot_speedup_vanilla="$(ratio_or_na "${VANILLA_TPOT}" "${spec_tpot}")"
      throughput_gain_vanilla="$(ratio_or_na "${spec_tps}" "${VANILLA_TOKENS_PER_SEC}")"

      echo "${DATASET},${MAX_SAMPLES},${BLOCK_SIZE},${CANDIDATE_MODE},${prefix_len},${SPARSE_MAX_POSITIONS},${top_k},${max_c},${e2e_speedup_bs1},${tpot_speedup_bs1},${throughput_gain_bs1},${e2e_speedup_vanilla},${tpot_speedup_vanilla},${throughput_gain_vanilla},${tau},${spec_wall},${spec_tps},${spec_tpot},${spec_ttft},${spec_target_decode},${spec_draft_decode},${spec_target_share},${spec_draft_share},${spec_cycles},${avg_cands},${avg_verify_calls},${BASELINE_TOTAL_WALL_S},${BASELINE_TOKENS_PER_SEC},${BASELINE_TPOT},${VANILLA_TOTAL_WALL_S},${VANILLA_TOKENS_PER_SEC},${VANILLA_TPOT},${BASELINE_LOG_PATH},${VANILLA_LOG_PATH},${log_path},${out_path:-NA},${cycle_path:-NA}" >> "${SUMMARY_CSV}"
    done
  done
done

echo "Sweep complete. Summary: ${SUMMARY_CSV}"
