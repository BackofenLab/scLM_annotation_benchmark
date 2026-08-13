#!/usr/bin/env bash
# Submit (arm, dataset) shards as INDEPENDENT jobs, never as a job array.
#
# Why not an array: the site prolog cannot parse one and drains the node when it tries. The full
# explanation is at the top of run_local_arms.sbatch; the short version is that `scontrol show job`
# on an array's base id returns one record per task, the prolog's `declare -i` chokes on the
# resulting multi-line value, and its `set -e` turns that into a node-draining non-zero exit.
#
# Concurrency is enforced here instead of by %N: at most MAX_CONCURRENT jobs are in flight, so we
# stay within the documented 4-GPU per-user limit. The prolog itself also requeues anything over
# the limit with a 10-minute delay, but relying on that would mean provoking the failure path.
#
# Usage:
#   slurm/submit_shards.sh A3_scllm_percell human_pancreas
#   slurm/submit_shards.sh A3_scllm_percell            # all four datasets
#   MAX_CONCURRENT=2 slurm/submit_shards.sh A2_base_percell
set -euo pipefail

HOST="${HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-context_ablation}"
# 3, not the permitted 4, deliberately. The documented ceiling is 4 GPUs per user and the prolog
# approves at exactly "3 + 1 = 4 / 4", so 4 is legal -- but it is also the boundary, and the prolog
# evaluates that boundary by counting a user's RUNNING jobs at launch time. Sitting one below it
# means a miscount, a scheduling race, or another of our own jobs lingering can never push the
# check over the edge. Empirically 3 concurrent shards ran to completion cleanly (24966_4/5/6)
# while the 4th is where the trouble started -- unrelated to concurrency in that instance, but
# there is no reason to court the boundary for a run that takes ~10 minutes either way.
#
# Cost is one extra scheduling wave for a 4-dataset arm; wall clock goes from ~10 to ~20 minutes.
# Raise it with MAX_CONCURRENT=4 if that ever matters.
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
# --dry-run exists because "just check the node is usable" and "submit four GPU jobs" were the same
# command, and the difference only showed up after the jobs were already queued.
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi

ARM="${1:?usage: submit_shards.sh [--dry-run] <arm> [dataset ...]}"
shift
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(HFRA HIHA Zeisel human_pancreas)

# Batch size comes from the arm's own definition in config.yml, so it cannot be forgotten at submit
# time. The 27B arms declare 32; anything without a declaration takes the 2B-calibrated 128. An
# explicit BATCH=... in the environment still wins, for one-off tuning.
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

echo "==> node must be usable before anything is submitted"
state=$(ssh "$HOST" "sinfo -N -h -o '%T' -n <gpu-node>" | tr -d ' ')
echo "  node state: $state"
case "$state" in
  *drain*|*down*)
    echo "REFUSING: the node is $state, so nothing would start." >&2
    ssh "$HOST" "sinfo -N -o '%.20N %.14T %.30E' -n <gpu-node>" >&2
    echo "A drained node needs an admin to resume it (<email removed>)." >&2
    exit 1 ;;
esac

for ds in "${DATASETS[@]}"; do
  # Wait for a free slot rather than submitting everything at once. Queued jobs would be fine, but
  # holding submission also keeps the prolog's own limit path from ever being exercised.
  while true; do
    n=$(ssh "$HOST" "squeue -u \$USER -h -o '%i' | wc -l" | tr -d ' ')
    [ "${n:-0}" -lt "$MAX_CONCURRENT" ] && break
    echo "  $n job(s) in flight, waiting for a slot..."
    sleep 30
  done
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [dry-run] would submit $ARM / $ds"
    continue
  fi
  echo "==> submitting $ARM / $ds (batch $BATCH)"
  # BATCH must be named explicitly in --export. `--export=ALL` propagates the environment of the
  # process that runs sbatch, which is the REMOTE shell -- a variable set on the laptop never
  # crosses the ssh boundary. Before this, `BATCH=32 submit_shards.sh` looked like it worked and
  # silently ran at the 128 default, which for a 27B model overflows the card.
  ssh "$HOST" "cd $REMOTE_DIR && mkdir -p logs && sbatch --export=ALL,ARM=$ARM,DATASET=$ds,BATCH=$BATCH slurm/run_local_arms.sbatch"
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> dry run, nothing submitted"
  exit 0
fi

echo "==> in flight"
ssh "$HOST" "squeue -u \$USER -o '%.12i %.14j %.9T %.8M %R'"
