#!/usr/bin/env bash
# Submit from a detached pinned checkout; log directory must already exist.
#SBATCH --job-name=dflash-predraft-latency
#SBATCH --partition=gpu
#SBATCH --account=users
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --no-requeue
#SBATCH --output=/data/scratch/zekaili/atharv/dflash/slurm_logs/%x-%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
: "${SLURM_SUBMIT_DIR:?Submit from pinned checkout}"
: "${EXPECTED_COMMIT:?Full pinned git revision required}"
cd "$SLURM_SUBMIT_DIR"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" || -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Pinned checkout changed" >&2
  exit 1
fi
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1
task_python=/data/scratch/zekaili/atharv/dflash/envs/main/bin/python
task_root=/data/scratch/zekaili/atharv/dflash/runs/predraft_latency_20261005
task_output="${task_root}/slurm_${SLURM_JOB_ID}"
child_pid=
cleanup() {
  trap '' TERM INT
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" || true
  fi
  exit 143
}
trap cleanup TERM INT
run_child() {
  "$@" &
  child_pid=$!
  wait "$child_pid"
  child_pid=
}
printf 'Frozen commit %s, Slurm job %s\n' "$EXPECTED_COMMIT" "$SLURM_JOB_ID"
run_child "$task_python" scripts/run_midverify_latency.py --use-visible-gpu \
  --predraft-bundle "${task_root}/bundle" --output "${task_output}/smoke" --smoke --max-seconds 900
# Launcher's success requires COMPLETE; numerical smoke gates must also pass.
run_child "$task_python" scripts/run_midverify_latency.py --use-visible-gpu \
  --predraft-bundle "${task_root}/bundle" --output "${task_output}/full" --max-seconds 2400
run_child "$task_python" scripts/summarize_predraft_latency.py --run "${task_output}/full"
