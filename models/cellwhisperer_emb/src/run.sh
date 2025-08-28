#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Extract true label key from config
TRUE_LABEL_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_keys = config.get('annotation_keys', [])
if annotation_keys:
    print(annotation_keys[0])  # Use first annotation key as true label key
else:
    print('cell_type')  # Default fallback
PY
)

# Run CellWhisperer prediction
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --true_label_key "$TRUE_LABEL_KEY"
