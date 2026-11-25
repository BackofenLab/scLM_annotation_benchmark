#!/usr/bin/env bash
# Remove everything this study created on the GPU node, so the results live only in the repo.
#
# Deletes exactly four paths, all of which this study created:
#   ~/context_ablation             code, prompts, results
#   /scratch/global/$USER/hf       downloaded model weights
#   the conda env                  sc_lfm_context_ablation_local
#   ~/miniforge3                   ONLY with --purge-conda, and only if we installed it
#
# It does NOT touch ~/sc_language_foundation_models or anything else that predates this work.
#
# Refuses to run while results are still only on the remote: it verifies that predictions have
# been fetched first, because the whole point of the exercise is that they end up here, and a
# cleanup that runs one step early destroys hours of GPU time.
set -euo pipefail

HOST="${HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-context_ablation}"
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
remote_n=$(ssh "$HOST" "find $REMOTE_DIR/results/predictions -name predictions.csv 2>/dev/null | wc -l" || echo 0)
remote_smoke=$(ssh "$HOST" "find $REMOTE_DIR/results/smoke -name predictions.csv 2>/dev/null | wc -l" || echo 0)
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
running=$(ssh "$HOST" "squeue -u \$USER -h -o '%i %j %T' | grep -c ctxabl" || true)
if [ "${running:-0}" -gt 0 ]; then
  echo "REFUSING: $running ctxabl job(s) still queued or running:" >&2
  ssh "$HOST" "squeue -u \$USER -o '%.10i %.14j %.9T %.10M'" >&2
  exit 1
fi

echo "==> will delete on $HOST:"
echo "     ~/$REMOTE_DIR"
echo "     /scratch/global/\$USER/hf"
echo "     conda env sc_lfm_context_ablation_local"
[ "$PURGE_CONDA" -eq 1 ] && echo "     ~/miniforge3  (--purge-conda given)"
if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "proceed? [y/N] " ans
  [[ "$ans" =~ ^[yY]$ ]] || { echo "cancelled"; exit 0; }
fi

ssh "$HOST" PURGE_CONDA="$PURGE_CONDA" REMOTE_DIR="$REMOTE_DIR" bash -s <<'REMOTE'
set -uo pipefail
echo "  removing ~/$REMOTE_DIR"
rm -rf "$HOME/${REMOTE_DIR:?}"
echo "  removing /scratch/global/$USER/hf"
rm -rf "/scratch/global/${USER:?}/hf"
if [ -d "$HOME/miniforge3" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda env remove -n sc_lfm_context_ablation_local -y >/dev/null 2>&1 \
    && echo "  removed conda env" || echo "  conda env not present"
  if [ "${PURGE_CONDA:-0}" = "1" ]; then
    rm -rf "$HOME/miniforge3"; echo "  removed ~/miniforge3"
  fi
fi
echo "  remaining in home:"; ls "$HOME" | sed 's/^/    /'
echo "  scratch:"; ls "/scratch/global/$USER" 2>/dev/null | sed 's/^/    /' || echo "    (gone)"
REMOTE

echo "==> cleanup complete"
