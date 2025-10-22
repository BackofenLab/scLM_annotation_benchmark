#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
MODEL_PATH=$(realpath "$0")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Extract all annotation keys from config
ANNOTATION_KEYS=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_keys = config.get('annotation_keys', [])
if annotation_keys:
    print(' '.join(annotation_keys))  # Print all annotation keys space-separated
else:
    print('cell_type')  # Default fallback
PY
)

# Run Langcell prediction
python "$(dirname "$0")/predict.py" \
--input "$INPUT_H5AD" \
--output "$OUT_CSV" \
--model_path "$MODEL_PATH" \
--annotation_keys $ANNOTATION_KEYS