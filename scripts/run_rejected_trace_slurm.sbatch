#!/usr/bin/env bash
# Submit from a detached, pinned checkout; log parent must already exist.
#SBATCH --job-name=dflash-rejected-2k
#SBATCH --partition=gpu
#SBATCH --account=users
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/data/scratch/zekaili/atharv/dflash/slurm_logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
: "${SLURM_JOB_ID:?This script must be submitted with sbatch}"
: "${SLURM_SUBMIT_DIR:?Submit from the pinned code checkout}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the full pinned git revision}"
cd "$SLURM_SUBMIT_DIR"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ||
      -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Pinned checkout changed; refusing to run a different experiment" >&2
  exit 1
fi
if [[ -n "${GPU:-}" || ! "${SLURM_JOB_GPUS:-}" =~ ^[0-7]$ ||
      -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Expected one Slurm GPU, inherited visibility, and no manual GPU override" >&2
  exit 1
fi
export RUN_ID="${RUN_ID:-rejected_trace_slurm_${SLURM_JOB_ID}}"
export MODE=pilot WORKERS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
printf 'Slurm job %s starting pinned revision %s; run %s\n' \
  "$SLURM_JOB_ID" "$EXPECTED_COMMIT" "$RUN_ID"
date -Is
exec bash scripts/run_rejected_trace_predictor.sh
