#!/bin/bash
# One (arm, dataset) shard of the local arms on the IMBI H100 node.
#
# NOT a job array, deliberately, and this is the important part of the file.
#
# The IMBI documentation recommends arrays with a %N concurrency limit, but the site's own prolog
# (/etc/slurm/prolog.d/check-gpu-limits.sh) cannot parse them. It derives the GPU request with
#
#     requested_gpus=$(... | grep -oP "TresPerNode=gres:gpu:[^,\s]*" | grep -oP "[0-9]+$")
#     declare -i requested_gpus=$requested_gpus
#
# and for an array `scontrol show job <base_id>` returns ONE RECORD PER TASK, so the grep matches
# once per task and requested_gpus becomes "1\n1\n1\n1". `declare -i` cannot evaluate that, the
# script runs under `set -euo pipefail`, and the prolog exits non-zero -- which DRAINS THE NODE for
# every user. Observed 2026-08-12: the task holding the base array ID tripped it while its three
# siblings were approved cleanly at "3 + 1 = 4 / 4".
#
# Submitting independent jobs sidesteps it entirely: each has a unique job id whose `scontrol show
# job` output carries exactly one TresPerNode line. Concurrency is then controlled by how many are
# submitted, not by %N -- see submit_shards.sh, which keeps us at the 4-GPU per-user limit.
#
# Resource flags follow "GPU Compute at the IMBI": 1 GPU (cap is 2/job), no --cpus-per-task (it is
# ignored), no --mem.
#
# On --mem, TRUST THE SCHEDULER OVER THE DOCUMENTATION. The docs state that omitting it yields
# "48GB (4GB x 12 CPUs)" and that 60 GB is the ceiling. Neither matches the running config:
#
#     JobDefaults = DefCpuPerGPU=11,DefMemPerGPU=62000      (scontrol show config)
#
# so a bare --gres=gpu:1 actually gets 11 CPUs and 62000 MB, confirmed in sacct AllocTRES for jobs
# 24966_4/5/6. Following the docs and writing `--mem=60G` for "maximum memory" would therefore ask
# for LESS than doing nothing. (Note also that 62000/11 = 5636 MB per CPU exceeds the partition's
# own MaxMemPerCPU=5120 -- the site default breaches its own per-CPU cap, which is another reason
# not to hand-tune this.)
#
# Measured peak for this workload was MaxRSS 1.7 GB, so any of these is ample and the argument is
# academic -- but the discrepancy is recorded because the next person will otherwise "fix" this
# line to match the documentation and quietly reduce the allocation.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# 12h, not 8. A 2B shard takes ~8-18 min, but a 27B shard is ~13x the compute at a quarter of the
# batch, so budget ~2h with margin. The partition cap is 2 days, and an over-generous walltime costs
# nothing here (no backfill pressure on a single-node partition) whereas a shard killed at the limit
# loses only its in-flight batch but still has to be resubmitted.
#SBATCH --time=12:00:00
#SBATCH --job-name=ctxabl_local
#SBATCH --output=logs/ctxabl_%j.out
#SBATCH --error=logs/ctxabl_%j.err
set -euo pipefail

: "${ARM:?set ARM, e.g. ARM=A3_scllm_percell}"
: "${DATASET:?set DATASET, e.g. DATASET=human_pancreas}"

cd "$HOME/context_ablation"

# Scratch, never /tmp -- the docs warn that /tmp is cleared only on reboot and can fill the
# filesystem. /scratch/local is the documented first choice BUT was measured 100% full (885 MB
# free) on 2026-08-12, so the ~5 GB of model weights go to /scratch/global instead.
export HF_HOME="/scratch/global/$USER/hf"
mkdir -p "$HF_HOME" logs

export PYTHONUNBUFFERED=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sc_lfm_context_ablation_local

echo "[$(date -Is)] $ARM / $DATASET on $(hostname), job $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Batch size from measurement, not a guess: batch 32 used 9.0 of 93 GB, of which ~4.9 GB is the
# model, so activations cost ~0.13 GB per sequence. 128 measured at 21.9-28.9 GB across datasets,
# the spread tracking prompt length (HFRA 756 tokens, HIHA 1096). Override with BATCH=... .
srun -u python src/run_local.py --arms "$ARM" --datasets "$DATASET" \
  --batch-size "${BATCH:-128}"

echo "[$(date -Is)] $ARM / $DATASET done"
