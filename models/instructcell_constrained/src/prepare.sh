#!/usr/bin/env bash
# Clones the InstructCell code and gene vocabulary next to this script, at the commit used for
# the paper; the model weights come from the Hugging Face Hub at run time. The pipeline runs it
# before run.sh.
set -euo pipefail

echo "Preparing InstructCell model..."

SCRIPT_DIR=$(dirname "$(realpath "$0")")
INSTRUCTCELL_COMMIT=b3ad68406600707c188b4122fb1efb37735a0289

INSTRUCTCELL_DIR="$SCRIPT_DIR/InstructCell"
if [ ! -d "$INSTRUCTCELL_DIR" ]; then
    echo "Cloning InstructCell repository..."
    git clone https://github.com/zjunlp/InstructCell.git "$INSTRUCTCELL_DIR"
fi
git -C "$INSTRUCTCELL_DIR" checkout --quiet "$INSTRUCTCELL_COMMIT"

echo "InstructCell preparation completed successfully!"

