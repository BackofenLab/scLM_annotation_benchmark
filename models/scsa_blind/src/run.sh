#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")

# Create output directory if it doesn't exist, then resolve absolute path
OUTPUT_DIR=$(dirname "$2")
mkdir -p "$OUTPUT_DIR"
OUT_CSV=$(realpath "$OUTPUT_DIR")/$(basename "$2")

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Extract all annotation keys from config
ANNOTATION_KEYS=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_keys = config.get('annotation_keys', [])
if annotation_keys:
    print(' '.join(annotation_keys))  # Print all annotation keys space-separated
else:
    print('cell_type')  # Default fallback
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

# Run SCSA prediction with all annotation keys
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --db "$SCSA_DB" \
  --annotation_keys $ANNOTATION_KEYS \
  --temp-dir "$CACHE_DIR"
