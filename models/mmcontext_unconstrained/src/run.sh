#!/usr/bin/env bash
# Runs predict.py (mmcontext) on one dataset. The pipeline calls
# run.sh INPUT_H5AD OUTPUT_CSV DATASET_YAML MODEL_CACHE_DIR.
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")

OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
print(annotation_key)
PY
)

# Candidate cell types, one name per line: CANDIDATE_TYPES_FILE (an absolute path) if set,
# else all_cell_types.txt (all Cell Ontology names) in the parent of the model cache directory.
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --cell_types_file "${CANDIDATE_TYPES_FILE:-all_cell_types.txt}" \
  --annotation_key "$ANNOTATION_KEY" \
  --cache_dir "$CACHE_DIR"

