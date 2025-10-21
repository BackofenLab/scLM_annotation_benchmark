#!/usr/bin/env bash
set -euo pipefail

echo "Preparing LangCell model..."

git clone --depth 1 --filter=blob:none --sparse https://github.com/PharMolix/LangCell.git

cd LangCell
git sparse-checkout set geneformer_001

# Move into the package directory
cd geneformer_001

# Install the package in the current environment
echo "Installing geneformer_001 package..."

# Install dependencies for geneformer either via python3 setup.py install --user   # or: pip install .
python3 setup.py install --user

cd ../..

echo "Downloading model weights..."

# URL of the zipped file
ZIP_URL="https://zenodo.org/records/16994789/files/langcell_pretrained_model_datasets.zip"

# Output directories
DOWNLOAD_DIR="./downloads"
DATA_DIR="./data"

# Create directories if they don't exist
mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"

# File name
ZIP_FILE="$DOWNLOAD_DIR/$(basename "$ZIP_URL")"

echo "Downloading $ZIP_URL ..."
curl -L -o "$ZIP_FILE" "$ZIP_URL"

echo "Unzipping $ZIP_FILE ..."
unzip -o "$ZIP_FILE" -d "$DOWNLOAD_DIR/unzipped"

echo "Copying files to $DATA_DIR ..."
cp -r "$DOWNLOAD_DIR/unzipped/langcell_pretrained_model_datasets/"* "$DATA_DIR/"

echo "✅ LangCell-Zeroshot preparation completed successfully!. Files are in $DATA_DIR"

