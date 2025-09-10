#!/usr/bin/env bash
set -euo pipefail

INPUT=$(realpath "$1")
OUTPUT="$2"
CONFIG=$(realpath "$3")
CACHE_DIR=$(realpath "$4")

GE_FILEID="17x9jfDorZYqoaG-MZGiduKFzP6663awO" # GPT 3.5 Gene embeddings
GE_FILENAME="$CACHE_DIR/gene_embeddings.pickle"
CT_FILEID="1AFD58eaF_V0Yt2OevQIOJLeu4Hg3NpGn" # GPT 3.5 Cell Type embeddings
CT_FILENAME="$CACHE_DIR/cell_type_embeddings.pickle"

if [[ ! -s "$CT_FILENAME" ]]; then
    gdown --fuzzy "https://drive.google.com/file/d/${CT_FILEID}/view?usp=sharing" -O ${CT_FILENAME}
fi
if [[ ! -s "$GE_FILENAME" ]]; then
    gdown --fuzzy "https://drive.google.com/file/d/${GE_FILEID}/view?usp=sharing" -O ${GE_FILENAME}
fi

OUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

ANNOTATION_KEYS=$(python - <<'PY' "$CONFIG"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_keys = config.get('annotation_keys', [])
if annotation_keys:
    print(' '.join(annotation_keys))  # Print all annotation keys space-separated
else:
    print('cell_type')  # Default fallback
PY
)

python "$SCRIPT_DIR/predict.py" \
    --input_h5ad "$INPUT" \
    --output_csv "$OUTPUT" \
    --gene-pickle "$GE_FILENAME" \
    --celltype-pickle "$CT_FILENAME" \
    --annotation-keys "$ANNOTATION_KEYS"
