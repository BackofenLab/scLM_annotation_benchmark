#!/bin/bash
#SBATCH --job-name=c2s_test
#SBATCH --output=c2s_test_%j.out
#SBATCH --error=c2s_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# =============================================================================
# C2S-Scale Single Cell Test Script
# =============================================================================
# Tests the C2S-Scale 27B model on a single cell for quick validation.
# Uses multi-GPU mode to distribute the model across available GPUs.
# =============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

# Activate conda environment
# Source conda directly (adjust path if your conda is installed elsewhere)
#source $(conda info --json | awk '/conda_prefix/ { gsub(/"|,/, "", $2); print $2 }')/bin/activate /scratch/local/menger/sc_language/work/conda/env-779f6dad20df12093abd6cd2cf2ec531

# Set paths (use absolute paths - SLURM copies scripts to temp locations)
SCRIPT_DIR="/Users/<user>/repos/sc_language_foundation_models/models/c2s_scale_27B/src"
INPUT_DATA='/Users/<user>/repos/sc_language_foundation_models/work/0d/7ffa8551b6ad99b324436c068a06fd/data/HIHA_L2_pp.h5ad'
CACHE_DIR="/Users/<user>/repos/sc_language_foundation_models/cache/models"
OUTPUT_DATA="/Users/<user>/repos/sc_language_foundation_models/test_predictions.csv"

# Check GPU availability
#echo ""
#echo "GPU Information:"
#nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
#echo ""

# Run the test script
echo "Running single cell test..."
python "${SCRIPT_DIR}/test_single_cell.py" \
    --input "${INPUT_DATA}" \
    --model "vandijklab/C2S-Scale-Gemma-2-27B" \
    --output "${OUTPUT_DATA}" \
    --annotation_key "AIFI_L2" \
    --n_genes 200 \
    --cache_dir "${CACHE_DIR}" \
    --multi_gpu

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

