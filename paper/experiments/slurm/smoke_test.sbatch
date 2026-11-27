#!/bin/bash
# Smoke test of the local arms: A3 on HFRA, the first 64 prompts of two prompt files. It runs the
# environment, the model download, the GPU and the output files once. With --limit, run_local.py
# writes predictions and cache to results/smoke/. Submit it from the experiment folder on the
# cluster (sbatch slurm/smoke_test.sbatch, with HF_DIR set if ship_to_gpu.sh had one), where the
# job runs. One GPU; CPUs and memory are the partition's defaults per GPU. A GPU partition with
# another name than gpu is given with sbatch -p.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ctxabl_smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
# Model weights go to HF_DIR if it is set, else to hf/ in this folder.
export HF_HOME="${HF_DIR:-$SLURM_SUBMIT_DIR/hf}"
mkdir -p "$HF_HOME" logs

export PYTHONUNBUFFERED=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sc_lfm_context_ablation_local

echo "[$(date -Is)] smoke test on $(hostname), job $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# The peak GPU memory in the log helps to choose the batch size of the full runs.
time srun -u python src/run_local.py \
  --arms A3_scllm_percell --datasets HFRA \
  --max-files 2 --limit 64 --batch-size 32

echo "[$(date -Is)] peak GPU memory:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "[$(date -Is)] smoke test finished"
