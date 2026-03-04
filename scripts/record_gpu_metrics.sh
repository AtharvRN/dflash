#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output_csv> [interval_s]" >&2
  exit 1
fi

OUT_CSV="$1"
INTERVAL_S="${2:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot record GPU metrics." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"
echo "timestamp,gpu_index,gpu_name,utilization_gpu_pct,utilization_memory_pct,memory_total_mib,memory_used_mib,memory_free_mib" > "${OUT_CSV}"

while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits >> "${OUT_CSV}" 2>/dev/null || true
  sleep "${INTERVAL_S}"
done
