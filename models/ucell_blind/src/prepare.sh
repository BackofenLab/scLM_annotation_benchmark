#!/usr/bin/env bash
set -euo pipefail

# Fetch the ScTypeDB marker workbook (same DB used by sctype_blind) into this model dir.
SCRIPT_DIR=$(dirname "$(realpath "$0")")
DB="$SCRIPT_DIR/ScTypeDB_full.xlsx"
URL="https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

if [ ! -f "$DB" ]; then
  echo "Downloading ScTypeDB -> $DB"
  curl -fsSL -o "$DB" "$URL"
else
  echo "ScTypeDB already present at $DB"
fi
echo "ucell_blind preparation completed."
