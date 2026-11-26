#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
CFG_YAML=$(realpath "$3")

# Absolute output path, since the script changes directory below
OUTPUT_DIR=$(dirname "$2")
mkdir -p "$OUTPUT_DIR"
OUT_CSV=$(realpath "$OUTPUT_DIR")/$(basename "$2")

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# annotation_key, species and scsa_tissue from the dataset YAML:
#  - species: SCSA.py compares it case-sensitively with "Mouse", so the species or organism of
#    the YAML is written as "Human" or "Mouse"; otherwise SCSA treats mouse data as human.
#  - scsa_tissue: one or more tissue names of SCSA's CellMarker database, separated by commas,
#    chosen per dataset; default "All" (the whole database).
IFS=$'\t' read -r ANNOTATION_KEY SPECIES SCSA_TISSUE < <(python - <<'PY' "$CFG_YAML"
import yaml, sys
c = yaml.safe_load(open(sys.argv[1]))
ann = c.get('annotation_key', 'cell_type')
raw = str(c.get('species') or c.get('organism') or 'human').strip().lower()
species = 'Mouse' if ('mouse' in raw or 'musculus' in raw) else 'Human'
tissue = str(c.get('scsa_tissue', 'All')).strip() or 'All'
print('\t'.join([ann, species, tissue]))
PY
)

# SCSA checkout made by prepare.sh
SCSA_DIR="$SCRIPT_DIR/SCSA"
if [ ! -d "$SCSA_DIR" ]; then
    echo "ERROR: SCSA directory not found at $SCSA_DIR"
    echo "Please run prepare.sh first to clone the SCSA repository"
    exit 1
fi

# SCSA's database with CellMarker version 2
SCSA_DB="$SCSA_DIR/whole_v2.db"

if [ ! -f "$SCSA_DB" ]; then
    echo "ERROR: SCSA database not found at $SCSA_DB"
    exit 1
fi

# predict.py calls SCSA.py by its relative path.
cd "$SCSA_DIR"

echo "SCSA: species=$SPECIES tissue=$SCSA_TISSUE"
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --db "$SCSA_DB" \
  --annotation_key "$ANNOTATION_KEY" \
  --species "$SPECIES" \
  --tissue "$SCSA_TISSUE"
