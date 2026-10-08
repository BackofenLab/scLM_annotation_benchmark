#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
CFG_YAML=$(realpath "$3")

# Create output directory if it doesn't exist, then resolve absolute path
OUTPUT_DIR=$(dirname "$2")
mkdir -p "$OUTPUT_DIR"
OUT_CSV=$(realpath "$OUTPUT_DIR")/$(basename "$2")

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Extract annotation key, species, and SCSA tissue from the dataset config.
#  - species: SCSA needs exactly "Human"/"Mouse" (SCSA.py:1146 is a case-sensitive == "Mouse"
#    check that gates mouse gene handling). Map the YAML species/organism to that form; without
#    it every dataset is scored as Human -> ~11x marker collapse on mouse data (e.g. Zeisel).
#  - scsa_tissue: exact tissueType string(s) from SCSA's CellMarker DB (comma-separated allowed),
#    hand-picked per dataset. Defaults to "All" (whole DB) when unset.
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

# Find SCSA directory and database
SCSA_DIR="$SCRIPT_DIR/SCSA"
if [ ! -d "$SCSA_DIR" ]; then
    echo "ERROR: SCSA directory not found at $SCSA_DIR"
    echo "Please run prepare.sh first to clone the SCSA repository"
    exit 1
fi

# Use whole_v2.db database
SCSA_DB="$SCSA_DIR/whole_v2.db"

if [ ! -f "$SCSA_DB" ]; then
    echo "ERROR: SCSA database not found at $SCSA_DB"
    exit 1
fi

# Change to SCSA directory so SCSA.py can be found
cd "$SCSA_DIR"

# Run SCSA prediction
echo "SCSA: species=$SPECIES tissue=$SCSA_TISSUE"
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --db "$SCSA_DB" \
  --annotation_key "$ANNOTATION_KEY" \
  --species "$SPECIES" \
  --tissue "$SCSA_TISSUE"
