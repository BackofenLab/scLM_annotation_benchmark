#!/usr/bin/env bash
# Clones CellWhisperer with its Geneformer submodule next to this script and fetches the
# Geneformer model files from Git LFS. The pipeline runs it before run.sh.
set -euo pipefail

echo "Preparing CellWhisperer model..."

SCRIPT_DIR=$(dirname "$(realpath "$0")")

CELLWHISPERER_DIR="$SCRIPT_DIR/CellWhisperer"
if [ ! -d "$CELLWHISPERER_DIR" ]; then
    cd "$SCRIPT_DIR"
    git clone https://github.com/mengerj/CellWhisperer.git
    cd "$CELLWHISPERER_DIR"
    git submodule update --init --recursive
else
    cd "$CELLWHISPERER_DIR"
    git pull
fi
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
    # pip installs geneformer without its LFS-tracked .pkl data files; copy them into the package.
    GF_SITE_PKG=$(python -c "import geneformer, pathlib; print(pathlib.Path(geneformer.__file__).parent)" 2>/dev/null || true)
    if [ -n "$GF_SITE_PKG" ] && [ -d "$GF_SITE_PKG" ]; then
        echo "Copying Geneformer .pkl data files to $GF_SITE_PKG ..."
        cp -n "$GENEFORMER_DIR"/geneformer/*.pkl "$GF_SITE_PKG/" 2>/dev/null || true
    fi
else
    echo "Warning: Geneformer directory not found at $GENEFORMER_DIR"
fi

echo "CellWhisperer preparation completed successfully!"
