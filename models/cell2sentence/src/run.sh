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

conda list
# Run Cell2Sentence prediction
python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --cache_dir "$CACHE_DIR" \
  --annotation_keys $(python - <<'PY' "$CFG_YAML"
import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1])).get('annotation_keys',[])))
PY
)
