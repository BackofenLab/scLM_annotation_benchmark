#!/usr/bin/env bash
# Runs predict.py (CellWhisperer) on one dataset. After prepare.sh, the pipeline calls
# run.sh INPUT_H5AD OUTPUT_CSV DATASET_YAML MODEL_CACHE_DIR.
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
CACHE_DIR=$(realpath "$4")

OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Avoids a gmpy2 error raised through sympy.
export SYMPY_USE_GMPY=0

# Patch CellWhisperer's Geneformer wrapper to call sc.pp.calculate_qc_metrics with
# percent_top=None: the default percent_top raises `IndexError: Positions outside range of
# features` when the data have fewer genes than its largest value.
GENEFORMER_MODEL_PY="$SCRIPT_DIR/CellWhisperer/src/cellwhisperer/jointemb/geneformer_model.py"
if [ -f "$GENEFORMER_MODEL_PY" ] && grep -q "sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)" "$GENEFORMER_MODEL_PY"; then
  sed -i \
    's|sc.pp.calculate_qc_metrics(adata_w_id, inplace=True)|sc.pp.calculate_qc_metrics(adata_w_id, inplace=True, percent_top=None)|g' \
    "$GENEFORMER_MODEL_PY"
fi

ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
print(annotation_key)
PY
)

# CellWhisperer's config.py reads config.yaml from the working directory at import time.
CW_CONFIG="$SCRIPT_DIR/CellWhisperer/config.yaml"
if [ -f "$CW_CONFIG" ] && [ ! -f config.yaml ]; then
    ln -s "$CW_CONFIG" config.yaml
fi

# Put the Geneformer checkout first on PYTHONPATH: unlike the pip-installed copy, it has the
# LFS-tracked .pkl data files, and site-packages may be on a read-only container layer.
GF_SRC="$SCRIPT_DIR/CellWhisperer/modules/Geneformer"
if [ -d "$GF_SRC/geneformer" ]; then
    export PYTHONPATH="${GF_SRC}:${PYTHONPATH:-}"
fi

python "$SCRIPT_DIR/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --cell_types_file "${CANDIDATE_TYPES_FILE:-all_cell_types.txt}" \
  --annotation_key "$ANNOTATION_KEY" \
  --cache_dir "$CACHE_DIR"
