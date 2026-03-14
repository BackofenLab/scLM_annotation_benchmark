#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Preparing InstructCell model..."

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Check if InstructCell repository exists, if not clone it
INSTRUCTCELL_DIR="$SCRIPT_DIR/InstructCell"
if [ ! -d "$INSTRUCTCELL_DIR" ]; then
    echo "📥 Cloning InstructCell repository..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/zjunlp/InstructCell.git
    echo "✅ InstructCell repository cloned successfully"
else
    echo "📁 InstructCell directory already exists — skipping clone."
fi

echo "✅ InstructCell preparation completed successfully!"

