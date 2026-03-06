#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Extract annotation keys and celltypist model from config YAML (tab-delimited)
IFS=$'\t' read -r ANNOTATION_KEY CELLTYPIST_MODEL < <(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', ['cell_type'])
celltypist_model = config.get('celltypist_model', 'Immune_All_High.pkl')
print(' '.join(annotation_key) + '\t' + celltypist_model)
PY
)

# Run CellTypist prediction
python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --model "$CELLTYPIST_MODEL" \
  --annotation_key "$ANNOTATION_KEY"
