#!/usr/bin/env bash
# Remove what ship_to_gpu.sh and the jobs created on the remote host (ssh alias HOST):
#   ~/$REMOTE_DIR                  code, prompts, results and the model weights in hf/
#   $HF_DIR                        the model weights, if ship_to_gpu.sh had HF_DIR
#   conda env                      sc_lfm_context_ablation_local
#   ~/miniforge3                   only with --purge-conda
# Stops if the remote results/predictions/ holds more prediction files than there are local A2
# and A3 prediction files, or if a ctxabl job is still queued or running. Asks before deleting
# unless --yes is given.
#     [HF_DIR=/scratch/<user>/hf] slurm/cleanup_gpu.sh [--purge-conda] [--yes]
set -euo pipefail

HOST="${HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-context_ablation}"
# The folder of the model weights as given to ship_to_gpu.sh; it is deleted with everything in it.
HF_DIR="${HF_DIR:-}"
# Extra ssh options, such as a jump host (SSH_OPTS="-J <host>"), as in ship_to_gpu.sh.
SSH_OPTS="${SSH_OPTS:-}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PURGE_CONDA=0
ASSUME_YES=0
for a in "$@"; do
  case "$a" in
    --purge-conda) PURGE_CONDA=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

echo "==> safety check: are the remote results already here?"
remote_n=$(ssh $SSH_OPTS "$HOST" "find $REMOTE_DIR/results/predictions -name predictions.csv 2>/dev/null | wc -l" || echo 0)
remote_smoke=$(ssh $SSH_OPTS "$HOST" "find $REMOTE_DIR/results/smoke -name predictions.csv 2>/dev/null | wc -l" || echo 0)
local_n=$(find "$LOCAL_ROOT/results/predictions" -path '*A2_base*' -o -path '*A3_scllm*' 2>/dev/null | grep -c 'predictions.csv$' || true)
echo "  remote real predictions : $remote_n"
echo "  remote smoke predictions: $remote_smoke"
echo "  local A2/A3 predictions : $local_n"
if [ "$remote_n" -gt 0 ] && [ "$local_n" -lt "$remote_n" ]; then
  echo "REFUSING: the remote holds $remote_n prediction files but only $local_n are here." >&2
  echo "Run slurm/fetch_from_gpu.sh first." >&2
  exit 1
fi

echo "==> checking nothing of ours is still running"
running=$(ssh $SSH_OPTS "$HOST" "squeue -u \$USER -h -o '%i %j %T' | grep -c ctxabl" || true)
if [ "${running:-0}" -gt 0 ]; then
  echo "REFUSING: $running ctxabl job(s) still queued or running:" >&2
  ssh $SSH_OPTS "$HOST" "squeue -u \$USER -o '%.10i %.14j %.9T %.10M'" >&2
  exit 1
fi

echo "==> will delete on $HOST:"
echo "     ~/$REMOTE_DIR"
[ -n "$HF_DIR" ] && echo "     $HF_DIR"
echo "     conda env sc_lfm_context_ablation_local"
[ "$PURGE_CONDA" -eq 1 ] && echo "     ~/miniforge3  (--purge-conda given)"
if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "proceed? [y/N] " ans
  [[ "$ans" =~ ^[yY]$ ]] || { echo "cancelled"; exit 0; }
fi

ssh $SSH_OPTS "$HOST" PURGE_CONDA="$PURGE_CONDA" REMOTE_DIR="$REMOTE_DIR" HF_DIR="$HF_DIR" bash -s <<'REMOTE'
set -uo pipefail
echo "  removing ~/$REMOTE_DIR"
rm -rf "$HOME/${REMOTE_DIR:?}"
if [ -n "${HF_DIR:-}" ]; then
  echo "  removing $HF_DIR"
  rm -rf "${HF_DIR:?}"
fi
if [ -d "$HOME/miniforge3" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda env remove -n sc_lfm_context_ablation_local -y >/dev/null 2>&1 \
    && echo "  removed conda env" || echo "  conda env not present"
  if [ "${PURGE_CONDA:-0}" = "1" ]; then
    rm -rf "$HOME/miniforge3"; echo "  removed ~/miniforge3"
  fi
fi
echo "  remaining in home:"; ls "$HOME" | sed 's/^/    /'
REMOTE

echo "==> cleanup complete"
