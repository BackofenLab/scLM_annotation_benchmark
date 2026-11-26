#!/usr/bin/env bash
# Runs predict.py (LangCell) on one dataset. After prepare.sh, the pipeline calls
# run.sh INPUT_H5AD OUTPUT_CSV DATASET_YAML MODEL_CACHE_DIR.
set -euo pipefail

INPUT_H5AD=$(realpath "$1")
OUT_CSV="$2"
CFG_YAML=$(realpath "$3")
# predict.py reads the model files from data/ in the folder of --model_path.
MODEL_PATH=$(realpath "$0")
CACHE_DIR=$(realpath "$4")

mkdir -p "$CACHE_DIR"
# Candidate cell types, one name per line: CANDIDATE_TYPES_FILE (an absolute path) if set,
# else all_cell_types.txt (all Cell Ontology names) in the parent of the model cache directory.
CELL_TYPES_FILE="${CANDIDATE_TYPES_FILE:-$(dirname $CACHE_DIR)/all_cell_types.txt}"

OUTPUT_DIR=$(dirname "$OUT_CSV")
mkdir -p "$OUTPUT_DIR"

ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
annotation_key = config.get('annotation_key', 'cell_type')
print(annotation_key)
PY
)

# LangCell uses its own Geneformer version (geneformer_001 from the LangCell repository), put first
# on PYTHONPATH. Nothing is installed into the environment, which may be shared or read-only.
SCRIPT_DIR=$(dirname "$MODEL_PATH")
GENEFORMER_PATH=$SCRIPT_DIR/LangCell/geneformer_001
if [ ! -d "$GENEFORMER_PATH" ]; then
  echo "ERROR: geneformer_001 not found at $GENEFORMER_PATH; run prepare.sh first" >&2
  exit 1
fi
export PYTHONPATH="$GENEFORMER_PATH${PYTHONPATH:+:$PYTHONPATH}"

python "$(dirname "$0")/predict.py" \
--input "$INPUT_H5AD" \
--output "$OUT_CSV" \
--model_path "$MODEL_PATH" \
--annotation_key "$ANNOTATION_KEY" \
--cell_types_file "$CELL_TYPES_FILE"
