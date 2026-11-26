#!/usr/bin/env bash
# Clones CellWhisperer with its Geneformer submodule next to this script, at the commit used for
# the paper, and fetches the Geneformer model files from Git LFS. The pipeline runs it before
# run.sh.
set -euo pipefail

echo "Preparing CellWhisperer model..."

SCRIPT_DIR=$(dirname "$(realpath "$0")")
CELLWHISPERER_COMMIT=08c41149c9ca82852ddd4005101656ba85b832bf

CELLWHISPERER_DIR="$SCRIPT_DIR/CellWhisperer"
if [ ! -d "$CELLWHISPERER_DIR" ]; then
    git clone https://github.com/mengerj/CellWhisperer.git "$CELLWHISPERER_DIR"
fi
cd "$CELLWHISPERER_DIR"
git checkout --quiet "$CELLWHISPERER_COMMIT"
# The submodules at the commits that this CellWhisperer commit records
git submodule update --init --recursive
# The Geneformer submodule keeps its model files in Git LFS.
GENEFORMER_DIR="$CELLWHISPERER_DIR/modules/Geneformer"

if [ -d "$GENEFORMER_DIR" ]; then
    cd "$GENEFORMER_DIR"
    if ! command -v git-lfs &> /dev/null; then
        echo "Installing git-lfs..."
        if command -v brew &> /dev/null; then
            brew install git-lfs
        elif command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        else
            echo "Please install git-lfs manually and rerun this script"
            exit 1
        fi
    fi
    echo "Installing git lfs..."
    git lfs install 

    # Without this .gitattributes entry, git lfs pull does not fetch the *.bin model weights.
    echo "Configuring git attributes for LFS..."
    echo "*.bin filter=lfs diff=lfs merge=lfs -text" > .gitattributes

    echo "Pulling git LFS files from Geneformer..."
    git lfs pull
    
    # CellWhisperer's config.yaml expects the Geneformer model in resources/geneformer-12L-30M.
    GENEFORMER_MODEL_DIR="$GENEFORMER_DIR/geneformer-12L-30M"
    TARGET_MODEL_DIR="$CELLWHISPERER_DIR/resources"
    mkdir -p "$TARGET_MODEL_DIR"
    
    if [ -d "$GENEFORMER_MODEL_DIR" ] && [ ! -d "$TARGET_MODEL_DIR/geneformer-12L-30M" ]; then
        echo "Copying geneformer-12L-30M to script directory..."
        cp -r "$GENEFORMER_MODEL_DIR" "$TARGET_MODEL_DIR"
    fi
else
    echo "Warning: Geneformer directory not found at $GENEFORMER_DIR"
fi

echo "CellWhisperer preparation completed successfully!"
