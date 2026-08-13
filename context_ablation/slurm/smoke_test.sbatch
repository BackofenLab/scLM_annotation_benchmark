#!/bin/bash
# Smoke test for the local arms: A3 (C2S-Scale-2B) on HFRA, a couple of hundred prompts.
#
# Purpose is to exercise the whole path -- env, weights download, GPU, tokenizer padding, output
# contract -- before committing eight shards. A3 rather than A2 because A2's weights are gated and
# need a token; A3 is open, so this runs today.
#
# Output is diverted to results/smoke/ by --limit (see run_local.py), so nothing it writes can be
# confused with a real result or make a later full run think it is already done.
#
# Resource flags follow "GPU Compute at the IMBI": 1 GPU (cap is 2/job), no --cpus-per-task (it is
# ignored; ~12 CPUs come automatically per GPU), no --mem (takes the documented 48 GB default;
# 60 GB is the ceiling for one GPU at 5 GB/CPU).
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ctxabl_smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
set -euo pipefail

cd "$HOME/context_ablation"
export HF_HOME="/scratch/global/$USER/hf"
mkdir -p "$HF_HOME" logs

export PYTHONUNBUFFERED=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sc_lfm_context_ablation_local

echo "[$(date -Is)] smoke test on $(hostname), job $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# batch 32 rather than the built-in 16: a 2B model in bf16 is ~5 GB against 96 GB of VRAM, so 16
# leaves the card idle. Kept modest here because prompt length varies a lot between modes and this
# run exists to MEASURE headroom, not to assume it -- read peak memory below before raising it.
time srun -u python src/run_local.py \
  --arms A3_scllm_percell --datasets HFRA \
  --max-files 2 --limit 64 --batch-size 32

echo "[$(date -Is)] peak GPU memory:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "[$(date -Is)] smoke test finished"
