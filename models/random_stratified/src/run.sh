#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
# $4 (cache_dir) is not used.

OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
print(config.get('annotation_key', 'cell_type'))
PY
)

python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --annotation_key "$ANNOTATION_KEY" \
  --strategy stratified \
  --seed 42
