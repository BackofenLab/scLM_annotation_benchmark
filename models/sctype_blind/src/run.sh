#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
OUTPUT_DIR=$(dirname "$OUT_CSV"); mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR=$(dirname "$(realpath "$0")")

MARKERS="$SCRIPT_DIR/ScTypeDB_full.xlsx"
if [ ! -f "$MARKERS" ] || [ ! -f "$SCRIPT_DIR/sctype_score_.R" ]; then
  echo "ERROR: scType assets missing in $SCRIPT_DIR — run prepare.sh first." >&2
  exit 1
fi

# Use the conda env's own R and library only (avoid a system R / user library shadowing the env,
# a common gotcha on machines with a separate R install). Works under Nextflow conda activation.
export R_LIBS_USER="${CONDA_PREFIX:-}/lib/R/library"
RSCRIPT="${CONDA_PREFIX:-}/bin/Rscript"; [ -x "$RSCRIPT" ] || RSCRIPT="Rscript"

# predict.R reads annotation_key / sctype_tissue / organism from the dataset YAML directly.
"$RSCRIPT" "$SCRIPT_DIR/predict.R" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --markers "$MARKERS" \
  --sctype_dir "$SCRIPT_DIR" \
  --config "$CFG_YAML"
