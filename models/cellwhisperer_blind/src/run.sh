#!/usr/bin/env bash
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")

# Create output directory if it doesn't exist and resolve its absolute path
OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# fix gmpy2 error
export SYMPY_USE_GMPY=0

# Compatibility patch:
# Older/newer scanpy defaults may raise `IndexError: Positions outside range of features`
# when the filtered feature space is smaller than default percent_top values.
GENEFORMER_MODEL_PY="$SCRIPT_DIR/CellWhisperer/src/cellwhisperer/jointemb/geneformer_model.py"
if [ -f "$GENEFORMER_MODEL_PY" ] && grep -q "sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)" "$GENEFORMER_MODEL_PY"; then
  sed -i \
    's|sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)|sc.pp.calculate_qc_metrics(adata_w_id, inplace=True, percent_top=None)|g' \
    "$GENEFORMER_MODEL_PY"
fi

# Extract annotation key from config
ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
print(annotation_key)
PY
)

# CellWhisperer's config.py requires config.yaml in CWD at import time
CW_CONFIG="$SCRIPT_DIR/CellWhisperer/config.yaml"
if [ -f "$CW_CONFIG" ] && [ ! -f config.yaml ]; then
    ln -s "$CW_CONFIG" config.yaml
fi

# Use Geneformer from the cloned repo (has LFS .pkl data files) instead of
# the pip-installed copy in site-packages (which lacks LFS-tracked files and
# lives on a read-only container filesystem layer).
GF_SRC="$SCRIPT_DIR/CellWhisperer/modules/Geneformer"
if [ -d "$GF_SRC/geneformer" ]; then
    export PYTHONPATH="${GF_SRC}:${PYTHONPATH:-}"
fi

# Run CellWhisperer prediction
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --cell_types_file "all_cell_types.txt" \
  --annotation_key "$ANNOTATION_KEY" \
  --cache_dir "$CACHE_DIR"
