#!/usr/bin/env bash
set -euo pipefail

echo "Preparing SCSA model..."

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Clone SCSA into src/SCSA or update an existing checkout; no commit is pinned.
SCSA_DIR="$SCRIPT_DIR/SCSA"
if [ ! -d "$SCSA_DIR" ]; then
    cd "$SCRIPT_DIR"
    git clone https://github.com/bioinfo-ibms-pumc/SCSA.git
else
    cd "$SCSA_DIR"
    git pull
fi

echo "SCSA preparation completed."

