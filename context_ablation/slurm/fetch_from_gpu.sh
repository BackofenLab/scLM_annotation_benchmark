#!/usr/bin/env bash
# Bring the GPU node's results back into the repo. Run this BEFORE cleanup_gpu.sh.
#
# Pulls results and logs only -- code and prompts already live here and are the source of truth, so
# copying them back could only ever overwrite the originals with a stale echo of themselves.
set -euo pipefail

HOST="${HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-context_ablation}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> what is on the remote"
ssh "$HOST" "cd $REMOTE_DIR 2>/dev/null && { \
  printf '  predictions: %s files\n' \"\$(find results/predictions -name predictions.csv 2>/dev/null | wc -l)\"; \
  printf '  smoke      : %s files\n' \"\$(find results/smoke -name predictions.csv 2>/dev/null | wc -l)\"; \
  printf '  caches     : %s files\n' \"\$(ls results/cache/*.jsonl 2>/dev/null | wc -l)\"; \
  printf '  logs       : %s files\n' \"\$(ls logs/ 2>/dev/null | wc -l)\"; \
  du -sh results 2>/dev/null | sed 's/^/  total: /'; }" || {
    echo "  nothing at $HOST:~/$REMOTE_DIR"; exit 1; }

echo "==> fetching predictions + caches + parse stats"
mkdir -p "$LOCAL_ROOT/results" "$LOCAL_ROOT/slurm/logs"
# No --delete: this merges the cluster's arms into a results tree that already holds the API arms.
rsync -az --stats \
  --include='predictions/***' --include='cache/***' --include='smoke/***' --exclude='*' \
  "$HOST:$REMOTE_DIR/results/" "$LOCAL_ROOT/results/"

echo "==> fetching SLURM logs (the record of what actually ran)"
rsync -az "$HOST:$REMOTE_DIR/logs/" "$LOCAL_ROOT/slurm/logs/" 2>/dev/null || echo "  (no logs yet)"

echo "==> fetching the environment lock, if it was produced"
rsync -az "$HOST:$REMOTE_DIR/env-local.lock.yml" "$LOCAL_ROOT/" 2>/dev/null \
  || echo "  (no env-local.lock.yml yet -- generate it before the real run)"

echo "==> local state now"
find "$LOCAL_ROOT/results/predictions" -name predictions.csv 2>/dev/null | wc -l \
  | xargs printf '  predictions.csv total (all arms): %s\n'
