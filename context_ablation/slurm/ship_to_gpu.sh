#!/usr/bin/env bash
# Ship the local-arm materials to the IMBI GPU node and bootstrap the environment there.
#
# Split of storage, decided against the measured state of the machine on 2026-08-12:
#   ~/context_ablation        code, prompts, results. ~300 MB. No home quota, 22 TB free.
#   /scratch/global/$USER/hf  model weights, ~10 GB. Bulky and re-downloadable, which is what
#                             scratch is for. NOT /scratch/local: measured 100% full (885 MB free).
#
# Idempotent: rsync and `conda env create` both skip work already done, so re-running after a
# dropped connection is safe and cheap.
set -euo pipefail

HOST="${HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-context_ablation}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> shipping $LOCAL_ROOT -> $HOST:~/$REMOTE_DIR"

ssh "$HOST" "mkdir -p ~/$REMOTE_DIR/{src,slurm,logs,results/prompts}"

# Code and config. --delete is deliberately ABSENT: it would wipe results/ on a re-ship.
rsync -az --stats \
  "$LOCAL_ROOT/src/" "$HOST:$REMOTE_DIR/src/"
rsync -az \
  "$LOCAL_ROOT/config.yml" "$LOCAL_ROOT/env-local.yml" "$HOST:$REMOTE_DIR/"
rsync -az \
  "$LOCAL_ROOT/slurm/" "$HOST:$REMOTE_DIR/slurm/"

# Prompts for the two local arms only -- 168 files, 288 MB raw, ~86 MB on the wire compressed.
# The API arms' prompts are much larger and are not needed here.
echo "==> shipping prompts (A2 + A3 only)"
# The arm list is DERIVED from config.yml rather than hardcoded. It used to name A2 and A3
# literally, so adding the 27B arms silently shipped nothing for them and the jobs would have failed
# at the far end with "no prompt files" -- after the environment rebuild and a 152 GB download.
LOCAL_ARMS=$(python3 - "$LOCAL_ROOT/config.yml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(" ".join(f"--include=*{k}*.csv" for k, v in cfg["arms"].items() if v["kind"] == "local"))
PY
)
echo "  shipping prompts for: $(echo "$LOCAL_ARMS" | tr ' ' '\n' | sed 's/--include=\*//; s/\*\.csv//' | tr '\n' ' ')"
# shellcheck disable=SC2086
rsync -az --progress $LOCAL_ARMS --exclude='*' \
  "$LOCAL_ROOT/results/prompts/" "$HOST:$REMOTE_DIR/results/prompts/"

echo "==> verifying the shipment"
ssh "$HOST" "cd $REMOTE_DIR && ls results/prompts/*.csv | wc -l | xargs printf '  prompt files: %s (expect 168)\n' && du -sh results/prompts | xargs printf '  size: %s %s\n'"

echo "==> bootstrapping conda + environment (skips whatever already exists)"
ssh "$HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/context_ablation

if [ ! -d "$HOME/miniforge3" ]; then
  echo "  installing miniforge to ~/miniforge3"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  rm -f /tmp/miniforge.sh
else
  echo "  miniforge already present"
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if conda env list | grep -q "sc_lfm_context_ablation_local"; then
  echo "  env already exists"
else
  echo "  creating env from env-local.yml (torch + CUDA, this takes a while)"
  conda env create -f env-local.yml
fi

conda activate sc_lfm_context_ablation_local
# mamba/conda can exit 0 while pip failed underneath, so assert the stack imports rather than
# trusting the exit code -- that exact failure mode cost a full env rebuild on the laptop.
python - <<'PY'
import torch, transformers, pandas, yaml
print(f"  torch {torch.__version__} | cuda available: {torch.cuda.is_available()} "
      f"| devices: {torch.cuda.device_count()}")
print(f"  transformers {transformers.__version__}")
PY

mkdir -p "/scratch/global/$USER/hf"
echo "  HF_HOME will be /scratch/global/$USER/hf ($(df -h /scratch/global | tail -1 | awk '{print $4}') free)"
REMOTE

echo "==> done. Next: slurm/smoke_test_gpu.sh"
