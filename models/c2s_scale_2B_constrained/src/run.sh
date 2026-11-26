#!/usr/bin/env bash
# Runs predict.py (C2S-Scale-Gemma-2-2B) on one dataset. The pipeline calls
# run.sh INPUT_H5AD OUTPUT_CSV DATASET_YAML MODEL_CACHE_DIR.
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR="$4"

OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

mkdir -p "$CACHE_DIR"

ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
print(annotation_key)
PY
)

# The pipeline sets METHOD_DIR; otherwise it is the folder above src.
METHOD_DIR="${METHOD_DIR:-$(dirname "$(dirname "$0")")}"
python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --cache_dir "$CACHE_DIR" \
  --annotation_key "$ANNOTATION_KEY" \
  --prompt "$METHOD_DIR/prompt.txt" \
  --n_genes 200
