#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# fix gmpy2 error
export SYMPY_USE_GMPY=0

# Run CellWhisperer prediction with all annotation keys
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --cell_types_file "all_cell_types.txt" \
  --cache_dir "$CACHE_DIR"
