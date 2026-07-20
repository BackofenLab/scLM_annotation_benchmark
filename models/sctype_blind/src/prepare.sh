#!/usr/bin/env bash
set -euo pipefail

# Fetch scType's helper functions (in the repo's R/ subfolder) + the marker workbook (repo root)
# into this model dir. Idempotent. bash-3.2 compatible (no associative arrays).
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE="https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master"

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
echo "sctype_blind preparation completed."
