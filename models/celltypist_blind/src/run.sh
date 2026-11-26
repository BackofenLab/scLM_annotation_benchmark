#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")

OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# annotation_key and celltypist_model from the dataset YAML, separated by a tab
IFS=$'\t' read -r ANNOTATION_KEY CELLTYPIST_MODEL < <(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
celltypist_model = config.get('celltypist_model', 'Immune_All_High.pkl')
print(annotation_key + '\t' + celltypist_model)
PY
)

python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --model "$CELLTYPIST_MODEL" \
  --annotation_key "$ANNOTATION_KEY"
