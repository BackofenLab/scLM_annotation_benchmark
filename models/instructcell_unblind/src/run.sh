#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR="$4"

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Run InstructCell prediction
python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --config "$CFG_YAML" \
  --cache_dir "$CACHE_DIR" \
  --model_variant "instruct"

