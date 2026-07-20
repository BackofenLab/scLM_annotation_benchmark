#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")
OUTPUT_DIR=$(dirname "$OUT_CSV"); mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Use the conda env's own R and library only (avoid a system R / user library shadowing the env).
export R_LIBS_USER="${CONDA_PREFIX:-}/lib/R/library"
RSCRIPT="${CONDA_PREFIX:-}/bin/Rscript"; [ -x "$RSCRIPT" ] || RSCRIPT="Rscript"

# predict.R reads annotation_key + singler_reference from the dataset YAML; celldex reference
# is downloaded/cached under CACHE_DIR/ExperimentHub.
"$RSCRIPT" "$SCRIPT_DIR/predict.R" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --config "$CFG_YAML" \
  --cache_dir "$CACHE_DIR"
