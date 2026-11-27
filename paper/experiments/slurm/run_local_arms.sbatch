#!/bin/bash
# One SLURM job of the local arms: run_local.py for one arm and one dataset, given as ARM and
# DATASET (BATCH and STUDY are optional). submit_shards.sh submits these jobs one by one, not as a
# job array, from the experiment folder on the cluster, where the job runs. One GPU; CPUs and
# memory are the partition's defaults per GPU. A GPU partition with another name than gpu is
# given with sbatch -p (PARTITION in submit_shards.sh).
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=ctxabl_local
#SBATCH --output=logs/ctxabl_%j.out
#SBATCH --error=logs/ctxabl_%j.err
set -euo pipefail

: "${ARM:?set ARM, e.g. ARM=A3_scllm_percell}"
: "${DATASET:?set DATASET, e.g. DATASET=human_pancreas}"

cd "$SLURM_SUBMIT_DIR"

# Model weights go to HF_DIR (submit_shards.sh passes it on), else to hf/ in this folder.
export HF_HOME="${HF_DIR:-$SLURM_SUBMIT_DIR/hf}"
mkdir -p "$HF_HOME" logs

export PYTHONUNBUFFERED=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sc_lfm_context_ablation_local

echo "[$(date -Is)] $ARM / $DATASET on $(hostname), job $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Generation batch size: BATCH, default 128.
srun -u python src/run_local.py --arms "$ARM" --datasets "$DATASET" \
  --batch-size "${BATCH:-128}" --study "${STUDY:-ablation}"

echo "[$(date -Is)] $ARM / $DATASET done"
