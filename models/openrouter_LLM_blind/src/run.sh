#!/usr/bin/env bash
# Annotates a test set with the served models of src/openrouter.yml, using the prompt template
# prompt.txt of the method folder. Called by the pipeline as
#   run.sh <test_set.h5ad> <out/predictions.csv> <dataset.yml> <cache_dir>
# The cache folder is not used. Further arguments are passed to predict.py.
set -euo pipefail

if [[ -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is not set. Store the OpenRouter key as a Nextflow secret:" >&2
  echo "  nextflow secrets set OPENROUTER_API_KEY 'sk-or-v1-...'" >&2
  exit 64
fi

# The pipeline sets METHOD_DIR. Otherwise it is the folder that contains src in the path this
# script was called with: src of the setting with the list of cell types is a symlink, and
# resolving it would give the folder of the other setting.
METHOD_DIR="${METHOD_DIR:-$(dirname "$(dirname "$0")")}"

PYTHONIOENCODING=UTF-8 exec python "$METHOD_DIR/src/predict.py" \
  --input "$1" \
  --output "$2" \
  --dataset_config "$3" \
  --prompt "$METHOD_DIR/prompt.txt" \
  --config "$METHOD_DIR/src/openrouter.yml" \
  "${@:5}"
