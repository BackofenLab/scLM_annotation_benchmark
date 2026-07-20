#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
OUTPUT_DIR=$(dirname "$OUT_CSV"); mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Per-dataset config: ground-truth column, ScTypeDB tissue, organism (human/mouse).
IFS=$'\t' read -r ANNOTATION_KEY SCTYPE_TISSUE ORGANISM < <(python - <<'PY' "$CFG_YAML"
import yaml, sys
c = yaml.safe_load(open(sys.argv[1]))
print('\t'.join([
    str(c.get('annotation_key', 'cell_type')),
    str(c.get('sctype_tissue', '')),
    str(c.get('organism', c.get('species', 'human'))),
]))
PY
)

MARKERS="$SCRIPT_DIR/ScTypeDB_full.xlsx"
if [ ! -f "$MARKERS" ]; then
  echo "ERROR: ScTypeDB not found at $MARKERS — run prepare.sh first." >&2
  exit 1
fi

python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --markers "$MARKERS" \
  --tissue "$SCTYPE_TISSUE" \
  --organism "$ORGANISM" \
  --annotation_key "$ANNOTATION_KEY"
