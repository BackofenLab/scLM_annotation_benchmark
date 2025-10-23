#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

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


conda list
# Run CellTypist prediction
python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --model "Immune_All_High.pkl" \
  --annotation_keys $ANNOTATION_KEYS
