#!/usr/bin/env bash
# Submit one SLURM job (run_local_arms.sbatch) per dataset for one local arm on the remote host
# (ssh alias HOST), as single jobs rather than a job array. At most MAX_CONCURRENT of the user's
# jobs are queued or running at once; the script waits for a free slot. If NODE names the GPU
# node, the script stops when that node is drained or down. --dry-run checks the node and lists
# the jobs without submitting them.
#
# Usage:
#   slurm/submit_shards.sh A3_scllm_percell human_pancreas
#   slurm/submit_shards.sh A3_scllm_percell            # all four datasets
#   MAX_CONCURRENT=2 slurm/submit_shards.sh A2_base_percell
#   PARTITION=<gpu partition> HF_DIR=/scratch/<user>/hf slurm/submit_shards.sh A3_scllm_percell
set -euo pipefail

HOST="${HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-context_ablation}"
SSH_OPTS="${SSH_OPTS:-}"            # e.g. "-J <jump host>", as in ship_to_gpu.sh
NODE="${NODE:-}"                    # the GPU node whose state is checked first; empty: no check
PARTITION="${PARTITION:-}"          # the GPU partition; empty: the one in run_local_arms.sbatch
HF_DIR="${HF_DIR:-}"                # the model weights on the remote host, as in ship_to_gpu.sh
STUDY="${STUDY:-ablation}"          # the study tree run_local.py reads and writes
# Jobs of the user queued or running at once; keep it below the cluster's per-user GPU limit.
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi

ARM="${1:?usage: submit_shards.sh [--dry-run] <arm> [dataset ...]}"
shift
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(HFRA HIHA Zeisel human_pancreas)

# Batch size: BATCH if set, else the arm's batch_size in config.yml, else 128.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${BATCH:-}" ]; then
  BATCH=$(python3 - "$CONFIG_DIR/config.yml" "$ARM" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
arm = cfg["arms"].get(sys.argv[2])
if arm is None:
    sys.exit(f"unknown arm: {sys.argv[2]}")
print(arm.get("batch_size", 128))
PY
) || { echo "could not resolve batch size for $ARM" >&2; exit 1; }
fi
echo "==> $ARM at batch $BATCH"

if [ -n "$NODE" ]; then
  echo "==> node $NODE must be usable before anything is submitted"
  state=$(ssh $SSH_OPTS "$HOST" "sinfo -N -h -o '%T' -n $NODE" | tr -d ' ')
  echo "  node state: $state"
  case "$state" in
    *drain*|*down*)
      echo "REFUSING: the node is $state, so nothing would start." >&2
      ssh $SSH_OPTS "$HOST" "sinfo -N -o '%.20N %.14T %.30E' -n $NODE" >&2
      echo "A drained node needs an admin to resume it." >&2
      exit 1 ;;
  esac
fi

for ds in "${DATASETS[@]}"; do
  # Wait for a free slot.
  while true; do
    n=$(ssh $SSH_OPTS "$HOST" "squeue -u \$USER -h -o '%i' | wc -l" | tr -d ' ')
    [ "${n:-0}" -lt "$MAX_CONCURRENT" ] && break
    echo "  $n job(s) in flight, waiting for a slot..."
    sleep 30
  done
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [dry-run] would submit $ARM / $ds"
    continue
  fi
  echo "==> submitting $ARM / $ds (batch $BATCH)"
  # sbatch runs in the remote shell, so local variables reach the job only through --export.
  ssh $SSH_OPTS "$HOST" "cd $REMOTE_DIR && mkdir -p logs && sbatch ${PARTITION:+--partition=$PARTITION }--export=ALL,ARM=$ARM,DATASET=$ds,BATCH=$BATCH,STUDY=$STUDY${HF_DIR:+,HF_DIR=$HF_DIR} slurm/run_local_arms.sbatch"
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> dry run, nothing submitted"
  exit 0
fi

echo "==> in flight"
ssh $SSH_OPTS "$HOST" "squeue -u \$USER -o '%.12i %.14j %.9T %.8M %R'"
