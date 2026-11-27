#!/usr/bin/env bash
set -euo pipefail

# Download scType's helper functions (R/ of the sc-type repository) and its marker workbook
# into this folder unless they are present, at the sc-type commit used for the paper. They are
# under GPL-3.0 and not part of this repository. Written for bash 3.2 (no associative arrays).
SCRIPT_DIR=$(dirname "$(realpath "$0")")
SCTYPE_COMMIT=630e15cf1e51f2612eda4ad0406dfb17503fa8c9
BASE="https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/$SCTYPE_COMMIT"

fetch() {  # <local_name> <remote_path>
  if [ ! -f "$SCRIPT_DIR/$1" ]; then
    echo "Downloading $1"
    curl -fsSL -o "$SCRIPT_DIR/$1" "$BASE/$2"
  else
    echo "$1 already present"
  fi
}

fetch gene_sets_prepare.R R/gene_sets_prepare.R
fetch sctype_score_.R     R/sctype_score_.R
fetch ScTypeDB_full.xlsx  ScTypeDB_full.xlsx
echo "sctype_unconstrained preparation completed."
