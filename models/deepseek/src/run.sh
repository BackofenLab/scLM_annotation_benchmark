#!/usr/bin/env bash
set -Eeuo pipefail

# -------------------------------------------------------------------
# Usage & basic arg handling
# -------------------------------------------------------------------
if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 INPUT.h5ad OUT.csv config.yaml" >&2
  exit 1
fi

# Portable abspath using the python in the conda env
abspath() {
  python - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

INPUT_H5AD="$(abspath "$1")"
OUT_CSV="$2"          # make dir first, then resolve abs path
CFG_YAML="$(abspath "$3")"

# Where this script lives (so we can find .env & model_cfg.yml)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# -------------------------------------------------------------------
# Load API_KEY: env wins; else load from .env; else fail clearly
# -------------------------------------------------------------------
cleanup() { unset -v API_KEY 2>/dev/null || true; }
trap cleanup EXIT

if [[ -z "${API_KEY:-}" && -f "$SCRIPT_DIR/.env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

if [[ -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is not set. Provide it via environment or $SCRIPT_DIR/.env" >&2
  echo "Examples:" >&2
  echo "  export API_KEY=YOUR_KEY && nextflow run main.nf -profile local" >&2
  echo "  (or put 'API_KEY=YOUR_KEY' into $SCRIPT_DIR/.env)" >&2
  exit 64
fi

# -------------------------------------------------------------------
# Preflight: ensure PyYAML is present in this conda env
# -------------------------------------------------------------------
if ! python - <<'PY' 2>/dev/null
import yaml
PY
then
  echo "ERROR: PyYAML not available in the current environment (import yaml failed)." >&2
  echo "Make sure envs/models/deepseek.yml contains 'pyyaml' and rerun with -resume." >&2
  exit 70
fi

# -------------------------------------------------------------------
# Output dir and absolute paths
# -------------------------------------------------------------------
OUTPUT_DIR="$(dirname "$OUT_CSV")"
mkdir -p "$OUTPUT_DIR"
OUT_CSV="$(abspath "$OUT_CSV")"

# -------------------------------------------------------------------
# Model config file resolution (src/ first, then parent); allow override
#   Override with: export MODEL_CFG=/abs/path/to/model_cfg.yml
# -------------------------------------------------------------------
if [[ -n "${MODEL_CFG:-}" ]]; then
  : # respect user-provided env var
elif [[ -f "$SCRIPT_DIR/model_cfg.yml" ]]; then
  MODEL_CFG="$SCRIPT_DIR/model_cfg.yml"
elif [[ -f "$SCRIPT_DIR/../model_cfg.yml" ]]; then
  MODEL_CFG="$(cd "$SCRIPT_DIR/.." && pwd)/model_cfg.yml"
else
  MODEL_CFG=""  # not found; helper will use hard-coded defaults
fi

if [[ -z "$MODEL_CFG" ]]; then
  echo "WARNING: model config not found (looked in $SCRIPT_DIR and parent); using hard-coded defaults." >&2
else
  echo "Using model config: $MODEL_CFG" >&2
fi

# -------------------------------------------------------------------
# Helper: read YAML value or default (dot-path keys) from MODEL_CFG
# -------------------------------------------------------------------
yaml_get_or_default () {
python - "$MODEL_CFG" "$1" "$2" <<'PY'
import sys, yaml, os
cfg_path, key_path, default = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = {}
if cfg_path and os.path.isfile(cfg_path):
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
def dig(d, path):
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
val = dig(cfg, key_path)
print(val if val is not None else default)
PY
}

# -------------------------------------------------------------------
# Annotation keys come from the *dataset* YAML (3rd arg)
#   Expect a list under `annotation_keys:` or `annotation.keys:`
# -------------------------------------------------------------------
LABEL_KEYS="$(python - "$CFG_YAML" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
keys = cfg.get("annotation_keys") or (cfg.get("annotation") or {}).get("keys") or []
print(" ".join(keys))
PY
)"

if [[ -z "$LABEL_KEYS" ]]; then
  echo "ERROR: No 'annotation_keys' found in $CFG_YAML" >&2
  exit 2
fi

# -------------------------------------------------------------------
# Model parameters (from MODEL_CFG, with sane defaults)
# -------------------------------------------------------------------
SEED="$(yaml_get_or_default 'deepseek.seed' '42')"
MODEL_NAME="$(yaml_get_or_default 'deepseek.model_name' 'deepseek/deepseek-r1-0528:free')"
TASK_NAME="$(yaml_get_or_default 'deepseek.task_name' 'cell_type_prediction')"

TOP_K_GENES="$(yaml_get_or_default 'deepseek.top_k_genes' '200')"
MAX_SAMPLES="$(yaml_get_or_default 'deepseek.max_samples' '100')"   # 0 or '' => all cells
MAX_WORKERS="$(yaml_get_or_default 'deepseek.max_workers' '3')"
DELAY_BETWEEN_REQUESTS="$(yaml_get_or_default 'deepseek.delay_between_requests' '1.2')"

# Multiline prompt templates (defaults are used if not in MODEL_CFG)
INPUT_PROMPT_TEMPLATE_DEFAULT=$'You are an expert at classifying cell types based on gene expression.\nGiven below is a list of {num_genes} gene names ordered by descending expression level in a single cell.\nYour task is to identify the exact cell type this gene profile represents.\nCell sentence: {cell_sentence}\nImportant rules:\n1. Respond with only the name of the predicted cell type.\n2. Do not include explanations, reasoning, or extra words.\n3. Do not repeat the question or input.\nOutput:'
ANSWER_TEMPLATE_DEFAULT='{true_labels}'

INPUT_PROMPT_TEMPLATE="$(yaml_get_or_default 'deepseek.input_prompt_template' "$INPUT_PROMPT_TEMPLATE_DEFAULT")"
ANSWER_TEMPLATE="$(yaml_get_or_default 'deepseek.answer_template' "$ANSWER_TEMPLATE_DEFAULT")"

# Normalize "no limit" behavior for max_samples
MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES}" && "${MAX_SAMPLES}" != "0" ]]; then
  MAX_SAMPLES_ARG=(--max_samples "${MAX_SAMPLES}")
fi

# -------------------------------------------------------------------
# Invoke the Python driver (API key via env, never echoed)
# -------------------------------------------------------------------
PYTHONIOENCODING=UTF-8 \
API_KEY="${API_KEY}" \
python "${SCRIPT_DIR}/predict.py" \
  --input "${INPUT_H5AD}" \
  --output "${OUT_CSV}" \
  --seed "${SEED}" \
  --model "${MODEL_NAME}" \
  --task_name "${TASK_NAME}" \
  --input_prompt_template "${INPUT_PROMPT_TEMPLATE}" \
  --answer_template "${ANSWER_TEMPLATE}" \
  --top_k_genes "${TOP_K_GENES}" \
  "${MAX_SAMPLES_ARG[@]}" \
  --max_workers "${MAX_WORKERS}" \
  --delay_between_requests "${DELAY_BETWEEN_REQUESTS}" \
  --label_keys ${LABEL_KEYS}

# Extra belt-and-braces: clear in case trap was removed/overridden
unset -v API_KEY 2>/dev/null || true