#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

conda list
# Run CellTypist prediction
python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --model "Immune_All_High.pkl" \
  --annotation_keys $(python - <<'PY' "$CFG_YAML"
import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1])).get('annotation_keys',[])))
PY
)
