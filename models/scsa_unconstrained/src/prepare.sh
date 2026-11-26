#!/usr/bin/env bash
set -euo pipefail

echo "Preparing SCSA model..."

SCRIPT_DIR=$(dirname "$(realpath "$0")")
SCSA_COMMIT=dfa5ff64c46b66eae3c1c797c449be22cfe15ea6

# SCSA in src/SCSA, at the commit used for the paper.
SCSA_DIR="$SCRIPT_DIR/SCSA"
if [ ! -d "$SCSA_DIR" ]; then
    git clone https://github.com/bioinfo-ibms-pumc/SCSA.git "$SCSA_DIR"
fi
git -C "$SCSA_DIR" checkout --quiet "$SCSA_COMMIT"

echo "SCSA preparation completed."

