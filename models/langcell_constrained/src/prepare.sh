#!/usr/bin/env bash
# Fetches the LangCell code (its bundled geneformer_001 package and its model code in utils.py)
# and the model files into the current directory; the pipeline runs it from this script's folder
# before run.sh.
set -euo pipefail

echo "Preparing LangCell model..."

LANGCELL_COMMIT=69e41ef2fae485b67703294b50f3150bb1d5bb9b

# Sparse clone at the commit used for the paper: only geneformer_001 and the model code in
# LangCell-annotation-zeroshot/utils.py are needed from the repository.
if [ ! -d "LangCell" ]; then
    echo "Cloning LangCell repository..."
    git clone --filter=blob:none --sparse https://github.com/PharMolix/LangCell.git
fi
git -C LangCell sparse-checkout set geneformer_001 LangCell-annotation-zeroshot
git -C LangCell checkout --quiet "$LANGCELL_COMMIT"

DOWNLOAD_DIR="./downloads"
DATA_DIR="./data"
# The authors' checkpoint (the Google Drive release linked in the PharMolix/LangCell README),
# copied to Zenodo so that the download is scriptable.
ZIP_URL="https://zenodo.org/records/16994789/files/langcell_pretrained_model_datasets.zip"
ZIP_FILE="$DOWNLOAD_DIR/$(basename "$ZIP_URL")"

mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"

if [ -f "$ZIP_FILE" ]; then
    echo "Zip file already downloaded: $ZIP_FILE"
else
    echo "Downloading model weights from Zenodo..."
    curl -L -o "$ZIP_FILE" "$ZIP_URL"
fi

if [ -d "$DOWNLOAD_DIR/unzipped/langcell_pretrained_model_datasets" ]; then
    echo "Model weights already unzipped, skipping extraction."
else
    echo "Unzipping model archive..."
    unzip -o "$ZIP_FILE" -d "$DOWNLOAD_DIR/unzipped"
fi

# predict.py reads data/ckpt (model weights) and data/obo.json (Cell Ontology names,
# synonyms and definitions).
if [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "'$DATA_DIR' already contains files, skipping copy."
else
    echo "Copying model datasets to $DATA_DIR ..."
    cp -r "$DOWNLOAD_DIR/unzipped/langcell_pretrained_model_datasets/"* "$DATA_DIR/"
fi

echo "LangCell-Zeroshot preparation completed successfully!"
echo "Model files available in: $DATA_DIR"