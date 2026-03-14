#!/usr/bin/env bash
set -euo pipefail

echo "Preparing SCSA model..."

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Check if SCSA repository exists, if not clone it
SCSA_DIR="$SCRIPT_DIR/SCSA"
if [ ! -d "$SCSA_DIR" ]; then
    cd "$SCRIPT_DIR"
    git clone https://github.com/bioinfo-ibms-pumc/SCSA.git
else
    cd "$SCSA_DIR"
    git pull
fi

echo "SCSA preparation completed successfully!"

