#!/usr/bin/env bash
set -euo pipefail

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

# Install Geneformer package
echo "📦 Installing geneformer_001 package..."
SCRIPT_DIR=$(dirname "$0")
GENEFORMER_PATH=$SCRIPT_DIR/LangCell/geneformer_001
if [ ! -d "$GENEFORMER_PATH" ]; then
  echo "🔄 Cloning LangCell (sparse: geneformer_001)..."
  if [ ! -d "$SCRIPT_DIR/LangCell" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/PharMolix/LangCell.git "$SCRIPT_DIR/LangCell"
  fi
  git sparse-checkout set geneformer_001
fi

install_geneformer() {
  local pkg_dir="$1"
  local tmp_dir="$pkg_dir/.pip-tmp"
  mkdir -p "$tmp_dir"
  (
    cd "$pkg_dir"
    rm -rf build dist ./*.egg-info || true
    if ! TMPDIR="$tmp_dir" python -m pip install --ignore-installed --no-build-isolation --no-deps .; then
      echo "⚠️ geneformer install failed once, retrying after clean build artifacts..."
      rm -rf build dist ./*.egg-info || true
      TMPDIR="$tmp_dir" python -m pip install --ignore-installed --no-build-isolation --no-deps .
    fi
  )
}

LOCK_FILE="$SCRIPT_DIR/.geneformer_install.lock"
if command -v flock >/dev/null 2>&1; then
  (
    flock -x 200
    install_geneformer "$GENEFORMER_PATH"
  ) 200>"$LOCK_FILE"
else
  install_geneformer "$GENEFORMER_PATH"
fi

# Run Langcell prediction
python "$(dirname "$0")/predict.py" \
--input "$INPUT_H5AD" \
--output "$OUT_CSV" \
--model_path "$MODEL_PATH" \
--annotation_keys $ANNOTATION_KEYS
