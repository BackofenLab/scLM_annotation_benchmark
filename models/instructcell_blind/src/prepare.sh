#!/usr/bin/env bash
# Clones the InstructCell code and gene vocabulary next to this script; the model weights
# come from the Hugging Face Hub at run time. The pipeline runs it before run.sh.
set -euo pipefail

echo "Preparing InstructCell model..."

SCRIPT_DIR=$(dirname "$(realpath "$0")")

INSTRUCTCELL_DIR="$SCRIPT_DIR/InstructCell"
if [ ! -d "$INSTRUCTCELL_DIR" ]; then
    echo "Cloning InstructCell repository..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/zjunlp/InstructCell.git
    echo "InstructCell repository cloned successfully"
else
    echo "InstructCell directory already exists, skipping clone."
fi

echo "InstructCell preparation completed successfully!"

