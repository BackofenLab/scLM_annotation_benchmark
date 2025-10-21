#!/usr/bin/env bash
set -euo pipefail

#git clone --depth 1 --filter=blob:none --sparse https://github.com/PharMolix/LangCell.git

#cd LangCell

#git sparse-checkout set geneformer_001

# Move into the package directory
#cd geneformer_001

# Install the package in the current environment
#echo "Installing geneformer_001 package..."

#python3 setup.py install --user   # or: pip install .

#cd ../..

#echo "Downloading model weights..."

# URL of the zipped file
#ZIP_URL="https://zenodo.org/records/16994789/files/langcell_pretrained_model_datasets.zip"

# Output directories
#DOWNLOAD_DIR="./downloads"
#DATA_DIR="./data"

# Create directories if they don't exist
#mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"

# File name
#ZIP_FILE="$DOWNLOAD_DIR/$(basename "$ZIP_URL")"

#echo "Downloading $ZIP_URL ..."
#curl -L -o "$ZIP_FILE" "$ZIP_URL"

#echo "Unzipping $ZIP_FILE ..."
#unzip -o "$ZIP_FILE" -d "$DOWNLOAD_DIR/unzipped"

#echo "Copying files to $DATA_DIR ..."
#cp -r "$DOWNLOAD_DIR/unzipped/langcell_pretrained_model_datasets/"* "$DATA_DIR/"

#echo "✅ Done. Files are in $DATA_DIR"

# cd ../..

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
MODEL_PATH=$(realpath "$0")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Extract all annotation keys from config
ANNOTATION_KEYS=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_keys = config.get('annotation_keys', [])
if annotation_keys:
    print(' '.join(annotation_keys))  # Print all annotation keys space-separated
else:
    print('cell_type')  # Default fallback
PY
)

# Run Langcell prediction
python "$(dirname "$0")/predict.py" \
--input "$INPUT_H5AD" \
--output "$OUT_CSV" \
--model_path "$MODEL_PATH" \
--annotation_keys $ANNOTATION_KEYS