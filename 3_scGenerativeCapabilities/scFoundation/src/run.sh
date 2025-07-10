#!/bin/bash
set -euo pipefail

INPUT_H5AD=$1
OUTPUT_EMBEDDING=$2

# Resolve absolute paths
INPUT_H5AD=$(realpath "$INPUT_H5AD")
OUTPUT_EMBEDDING=$(realpath "$OUTPUT_EMBEDDING")
WORKDIR=$(dirname "$OUTPUT_EMBEDDING")

ENV_NAME="scfoundation-env"

# Clone scFoundation repo if not present
if [ ! -d "scFoundation/src/scFoundation" ]; then
    echo "[INFO] Cloning scFoundation repository..."
    git clone https://github.com/biomap-research/scFoundation.git scFoundation/src/scFoundation
fi

cd scFoundation/src

if [ ! -f ".patched" ]; then
    echo "[INFO] Applying local patch to scFoundation"
    patch -p0 < fix_get_embedding.patch
    touch .patched  # prevent reapplying
fi

# load conda shell functions
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create conda environment
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "[INFO] Creating conda environment '$ENV_NAME' from requirements.txt..."
    conda create -y -n "$ENV_NAME" python=3.11
    conda activate "$ENV_NAME"
    pip install -r requirements.txt
else
    echo "[INFO] Using existing conda environment '$ENV_NAME'"
    conda activate "$ENV_NAME"
fi

cd scFoundation

# The model file is stored in a not programmatically accessible SharePoint
if [ ! -f "model/models/models.ckpt" ]; then
    echo "[ERROR] Model file not found"
    echo "[INFO] The model file is not programmatically accessible"
    echo "Download the files from https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/EmUQnvZMETlDvoCaBduCNeIBQArcOrd8T8iEpiGofFZ9CQ?e=3SpPZU"
    echo "Put the files in scFoundation/model/models/"
    exit 1
fi

echo "[INFO] Model file found"

cd model

# Run embedding extraction
echo "[INFO] Running embedding extraction for: $INPUT_H5AD"
python get_embedding.py \
    --task_name Baron \
    --input_type singlecell \
    --output_type cell \
    --pool_type all \
    --tgthighres a5 \
    --data_path "$INPUT_H5AD" \
    --save_path "$OUTPUT_EMBEDDING" \
    --pre_normalized F \
    --version rde

echo "[INFO] Embeddings saved to: $OUTPUT_EMBEDDING"

