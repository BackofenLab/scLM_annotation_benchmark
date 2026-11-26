#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")
OUTPUT_DIR=$(dirname "$OUT_CSV"); mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Use the R and R library of the conda environment, so that a system R or a user library
# does not shadow them.
export R_LIBS_USER="${CONDA_PREFIX:-}/lib/R/library"
RSCRIPT="${CONDA_PREFIX:-}/bin/Rscript"; [ -x "$RSCRIPT" ] || RSCRIPT="Rscript"

# predict.R reads annotation_key and the singler_reference settings from the dataset YAML and
# caches celldex references in CACHE_DIR/ExperimentHub.
"$RSCRIPT" "$SCRIPT_DIR/predict.R" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --config "$CFG_YAML" \
  --cache_dir "$CACHE_DIR"
