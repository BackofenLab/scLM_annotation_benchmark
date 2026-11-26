#!/usr/bin/env bash
# Copy the code, config.yml, env-local.yml, the SLURM scripts and the prompt files of the local
# arms to the remote host (ssh alias HOST, folder ~/REMOTE_DIR), install Miniforge there if it is
# missing and create the environment of env-local.yml. Model weights go to
# /scratch/global/$USER/hf. Running it again skips what is already there.
#     [STUDY=lists] slurm/ship_to_gpu.sh
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

echo "==> shipping $LOCAL_ROOT -> $HOST:~/$REMOTE_DIR"

ssh $SSH_OPTS "$HOST" "mkdir -p ~/$REMOTE_DIR/{src,slurm,logs,$TREE/prompts}"

# Code and configuration.
rsync -az --stats \
  "$LOCAL_ROOT/src/" "$HOST:$REMOTE_DIR/src/"
rsync -az \
  "$LOCAL_ROOT/config.yml" "$LOCAL_ROOT/env-local.yml" "$HOST:$REMOTE_DIR/"
rsync -az \
  "$LOCAL_ROOT/slurm/" "$HOST:$REMOTE_DIR/slurm/"

# Only the prompt files of the arms with kind: local in config.yml.
echo "==> shipping prompts (A2 + A3 only)"
LOCAL_ARMS=$(python3 - "$LOCAL_ROOT/config.yml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(" ".join(f"--include=*{k}*.csv" for k, v in cfg["arms"].items() if v["kind"] == "local"))
PY
)
echo "  shipping prompts for: $(echo "$LOCAL_ARMS" | tr ' ' '\n' | sed 's/--include=\*//; s/\*\.csv//' | tr '\n' ' ')"
# shellcheck disable=SC2086
rsync -az --progress $LOCAL_ARMS --exclude='*' \
  "$LOCAL_ROOT/$TREE/prompts/" "$HOST:$REMOTE_DIR/$TREE/prompts/"

echo "==> verifying the shipment"
ssh $SSH_OPTS "$HOST" "cd $REMOTE_DIR && ls $TREE/prompts/*.csv | wc -l | xargs printf '  prompt files: %s (168 for the ablation)\n' && du -sh $TREE/prompts | xargs printf '  size: %s %s\n'"

echo "==> bootstrapping conda + environment (skips whatever already exists)"
ssh $SSH_OPTS "$HOST" bash -s <<'REMOTE'
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
# conda can exit with 0 although pip failed, so the imports are checked.
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
