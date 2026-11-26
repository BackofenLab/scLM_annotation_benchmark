#!/usr/bin/env bash
# Copy the predictions, caches and smoke-test output of a study tree, the SLURM logs and
# env-local.lock.yml from the remote host (ssh alias HOST) back to paper/experiments/. Code and
# prompts are not copied back. Run it before cleanup_gpu.sh.
#     [STUDY=lists] slurm/fetch_from_gpu.sh
set -euo pipefail

HOST="${HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-context_ablation}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Extra ssh options, such as a jump host (SSH_OPTS="-J <host>"); rsync gets them through RSYNC_RSH.
SSH_OPTS="${SSH_OPTS:-}"
export RSYNC_RSH="ssh $SSH_OPTS"
# The study tree (`studies` in config.yml): STUDY=lists ships, runs and fetches results_lists/ only.
STUDY="${STUDY:-ablation}"
TREE=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["studies"][sys.argv[2]])' \
  "$LOCAL_ROOT/config.yml" "$STUDY")

echo "==> what is on the remote"
ssh $SSH_OPTS "$HOST" "cd $REMOTE_DIR 2>/dev/null && { \
  printf '  predictions: %s files\n' \"\$(find $TREE/predictions -name predictions.csv 2>/dev/null | wc -l)\"; \
  printf '  smoke      : %s files\n' \"\$(find $TREE/smoke -name predictions.csv 2>/dev/null | wc -l)\"; \
  printf '  caches     : %s files\n' \"\$(ls $TREE/cache/*.jsonl 2>/dev/null | wc -l)\"; \
  printf '  logs       : %s files\n' \"\$(ls logs/ 2>/dev/null | wc -l)\"; \
  du -sh $TREE 2>/dev/null | sed 's/^/  total: /'; }" || {
    echo "  nothing at $HOST:~/$REMOTE_DIR"; exit 1; }

echo "==> fetching predictions + caches + parse stats"
mkdir -p "$LOCAL_ROOT/$TREE" "$LOCAL_ROOT/slurm/logs"
# No --delete: the local tree also holds the predictions of the OpenRouter arms.
rsync -az --stats \
  --include='predictions/***' --include='cache/***' --include='smoke/***' --exclude='*' \
  "$HOST:$REMOTE_DIR/$TREE/" "$LOCAL_ROOT/$TREE/"

echo "==> fetching SLURM logs (the record of what actually ran)"
rsync -az "$HOST:$REMOTE_DIR/logs/" "$LOCAL_ROOT/slurm/logs/" 2>/dev/null || echo "  (no logs yet)"

echo "==> fetching the environment lock, if it was produced"
rsync -az "$HOST:$REMOTE_DIR/env-local.lock.yml" "$LOCAL_ROOT/" 2>/dev/null \
  || echo "  (no env-local.lock.yml yet -- generate it before the real run)"

echo "==> local state now"
find "$LOCAL_ROOT/$TREE/predictions" -name predictions.csv 2>/dev/null | wc -l \
  | xargs printf '  predictions.csv total (all arms): %s\n'
