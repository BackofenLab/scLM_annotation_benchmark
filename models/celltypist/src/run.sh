#!/bin/bash
set -euo pipefail

INPUT_H5AD=$1
OUTPUT_PREDICTIONS=$2

# Resolve absolute paths
INPUT_H5AD=$(realpath "$INPUT_H5AD")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUTPUT_PREDICTIONS")
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

echo "Running CellTypist prediction for: $INPUT_H5AD"
echo "Output predictions to: $OUTPUT_PREDICTIONS"

# Construct the full output path
OUTPUT_PREDICTIONS="$OUTPUT_DIR/$(basename "$OUTPUT_PREDICTIONS")"
WORKDIR="$OUTPUT_DIR"

# Setup conda environment
ENV_NAME="celltypist-env"
REQUIREMENTS_FILE="celltypist/src/requirements.txt"
source conda_env.sh
setup_conda_env "$ENV_NAME" "$REQUIREMENTS_FILE"

# Run CellTypist prediction
echo "[INFO] Running CellTypist prediction for: $INPUT_H5AD"
run_in_env "$ENV_NAME" python "celltypist/src/predict.py" \
    --input "$INPUT_H5AD" \
    --output "$OUTPUT_PREDICTIONS" \
    --model "Immune_All_High.pkl"

echo "[INFO] CellTypist predictions saved to: $OUTPUT_PREDICTIONS"
