#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Preparing LangCell model new..."

# --- Clone LangCell repository only if missing ---
if [ -d "LangCell" ]; then
    echo "📁 'LangCell' directory already exists — skipping clone."
else
    echo "📥 Cloning LangCell repository..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/PharMolix/LangCell.git
    cd LangCell
    git sparse-checkout set geneformer_001
    cd ..
fi

# --- Install geneformer_001 package ---
echo "📦 Installing geneformer_001 package..."
cd LangCell/geneformer_001
pip install .
cd ../..

# --- Prepare download directories ---
DOWNLOAD_DIR="./downloads"
DATA_DIR="./data"
ZIP_URL="https://zenodo.org/records/16994789/files/langcell_pretrained_model_datasets.zip"
ZIP_FILE="$DOWNLOAD_DIR/$(basename "$ZIP_URL")"

mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"

# --- Download model weights only if not already present ---
if [ -f "$ZIP_FILE" ]; then
    echo "✅ Zip file already downloaded: $ZIP_FILE"
else
    echo "⬇️ Downloading model weights from Zenodo..."
    curl -L -o "$ZIP_FILE" "$ZIP_URL"
fi

# --- Unzip if not already extracted ---
if [ -d "$DOWNLOAD_DIR/unzipped/langcell_pretrained_model_datasets" ]; then
    echo "📦 Model weights already unzipped — skipping extraction."
else
    echo "🧩 Unzipping model archive..."
    unzip -o "$ZIP_FILE" -d "$DOWNLOAD_DIR/unzipped"
fi

# --- Copy files if not already in data directory ---
if [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "📁 '$DATA_DIR' already contains files — skipping copy."
else
    echo "📂 Copying model datasets to $DATA_DIR ..."
    cp -r "$DOWNLOAD_DIR/unzipped/langcell_pretrained_model_datasets/"* "$DATA_DIR/"
fi

echo "✅ LangCell-Zeroshot preparation completed successfully!"
echo "📍 Model files available in: $DATA_DIR"