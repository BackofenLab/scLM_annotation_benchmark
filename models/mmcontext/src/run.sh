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

# Run MMContext prediction with all annotation keys
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --annotation_keys $ANNOTATION_KEYS \
  --cache_dir "$CACHE_DIR"

