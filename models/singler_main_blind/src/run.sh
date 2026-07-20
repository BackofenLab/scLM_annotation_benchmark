#!/usr/bin/env bash
set -euo pipefail

# SingleR at celldex label tier 'main'. The three singler_<tier>_blind models share one
# implementation (src/predict.R is a symlink to models/singler_blind/src/predict.R); only
# --label_tier differs, so each tier gets its own cached predictions dir and results row.
#
# Tier granularity is reference-specific: for the references used here, MonacoImmuneData is
# 10/29/28, MouseRNAseqData 18/28/22 and HumanPrimaryCellAtlasData 36/157/66 for main/fine/ont.
# 'ont' emits Cell Ontology IDs, which resolve without any manual mapping review.

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")
OUTPUT_DIR=$(dirname "$OUT_CSV"); mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Use the conda env's own R and library only (avoid a system R / user library shadowing the env).
export R_LIBS_USER="${CONDA_PREFIX:-}/lib/R/library"
RSCRIPT="${CONDA_PREFIX:-}/bin/Rscript"; [ -x "$RSCRIPT" ] || RSCRIPT="Rscript"

"$RSCRIPT" "$SCRIPT_DIR/predict.R" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --config "$CFG_YAML" \
  --cache_dir "$CACHE_DIR" \
  --label_tier main
