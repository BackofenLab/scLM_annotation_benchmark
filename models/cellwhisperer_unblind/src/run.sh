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

# Compatibility patch:
# Older/newer scanpy defaults may raise `IndexError: Positions outside range of features`
# when the filtered feature space is smaller than default percent_top values.
GENEFORMER_MODEL_PY="$SCRIPT_DIR/CellWhisperer/src/cellwhisperer/jointemb/geneformer_model.py"
if [ -f "$GENEFORMER_MODEL_PY" ] && grep -q "sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)" "$GENEFORMER_MODEL_PY"; then
  sed -i \
    's|sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)|sc.pp.calculate_qc_metrics(adata_w_id, inplace=True, percent_top=None)|g' \
    "$GENEFORMER_MODEL_PY"
fi

# Extract all annotation keys from config
ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
print(annotation_key)
PY
)

# fix gmpy2 error
export SYMPY_USE_GMPY=0

# Run CellWhisperer prediction
python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --annotation_key "$ANNOTATION_KEY" \
  --cache_dir "$CACHE_DIR"
