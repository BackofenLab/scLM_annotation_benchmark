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

# Extract annotation key from config
ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
print(annotation_key)
PY
)

# CANDIDATE_TYPES_FILE (an absolute path) replaces the full Cell Ontology list with a list of the
# candidate-list study; unset, the wrapper runs as in the benchmark.
# Run MMContext prediction
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --cell_types_file "${CANDIDATE_TYPES_FILE:-all_cell_types.txt}" \
  --annotation_key "$ANNOTATION_KEY" \
  --cache_dir "$CACHE_DIR"

