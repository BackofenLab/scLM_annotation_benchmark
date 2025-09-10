#!/usr/bin/env bash
set -Eeuo pipefail

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
  echo "Make sure envs/models/openrouter_LLM.yml contains 'pyyaml' and rerun with -resume." >&2
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
# Helper: read YAML scalar or default (dot-path keys) from MODEL_CFG
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

# Helper: read YAML list -> print each item on its own line
yaml_get_list () {
python - "$MODEL_CFG" "$1" <<'PY'
import sys, yaml, os
cfg_path, key_path = sys.argv[1], sys.argv[2]
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
if isinstance(val, list):
    for item in val:
        if item is None: continue
        print(str(item))
elif isinstance(val, str) and val.strip():
    print(val.strip())
PY
}

# Helper: read YAML boolean -> prints "true"/"false" or nothing if null/missing
yaml_get_bool () {
python - "$MODEL_CFG" "$1" <<'PY'
import sys, yaml, os
cfg_path, key_path = sys.argv[1], sys.argv[2]
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
if isinstance(val, bool):
    print("true" if val else "false")
PY
}

# Helper: read YAML value -> JSON string (used for response_format)
yaml_get_json () {
python - "$MODEL_CFG" "$1" <<'PY'
import sys, yaml, json, os
cfg_path, key_path = sys.argv[1], sys.argv[2]
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
if val is not None:
    print(json.dumps(val, separators=(",", ":")))
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
SEED="$(yaml_get_or_default 'openrouter_LLM.seed' '42')"
MODEL_NAME="$(yaml_get_or_default 'openrouter_LLM.model_name' 'deepseek/deepseek-r1-0528:free')"
TASK_NAME="$(yaml_get_or_default 'openrouter_LLM.task_name' 'cell_type_prediction')"

OUTPUT_MODE="$(yaml_get_or_default 'openrouter_LLM.output_mode' 'label')"
TOP_K_GENES="$(yaml_get_or_default 'openrouter_LLM.top_k_genes' '200')"
MAX_SAMPLES="$(yaml_get_or_default 'openrouter_LLM.max_samples' '')"   # 0 or '' => all cells
MAX_WORKERS="$(yaml_get_or_default 'openrouter_LLM.max_workers' '3')"
DELAY_BETWEEN_REQUESTS="$(yaml_get_or_default 'openrouter_LLM.delay_between_requests' '1.2')"

# Multiline prompt templates (defaults are used if not in MODEL_CFG)
INPUT_PROMPT_TEMPLATE_DEFAULT=$'You are an expert at classifying cell types based on gene expression.\nGiven below is a list of {num_genes} gene names ordered by descending expression level in a single cell.\nYour task is to identify the exact cell type this gene profile represents.\nCell sentence: {cell_sentence}\nImportant rules:\n1. Respond with only the name of the predicted cell type.\n2. Do not include explanations, reasoning, or extra words.\n3. Do not repeat the question or input.\nOutput:'
ANSWER_TEMPLATE_DEFAULT='{true_labels}'

INPUT_PROMPT_TEMPLATE="$(yaml_get_or_default 'openrouter_LLM.input_prompt_template' "$INPUT_PROMPT_TEMPLATE_DEFAULT")"
ANSWER_TEMPLATE="$(yaml_get_or_default 'openrouter_LLM.answer_template' "$ANSWER_TEMPLATE_DEFAULT")"

# Normalize "no limit" behavior for max_samples
MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES}" && "${MAX_SAMPLES}" != "0" ]]; then
  MAX_SAMPLES_ARG=(--max_samples "${MAX_SAMPLES}")
fi

# -------------------------------------------------------------------
# Build predict.py argument list
# -------------------------------------------------------------------
declare -a ARGS=(
  "--input" "${INPUT_H5AD}"
  "--output" "${OUT_CSV}"
  "--seed" "${SEED}"
  "--model" "${MODEL_NAME}"
  "--output_mode" "${OUTPUT_MODE}"
  "--task_name" "${TASK_NAME}"
  "--input_prompt_template" "${INPUT_PROMPT_TEMPLATE}"
  "--answer_template" "${ANSWER_TEMPLATE}"
  "--top_k_genes" "${TOP_K_GENES}"
  "--max_workers" "${MAX_WORKERS}"
  "--delay_between_requests" "${DELAY_BETWEEN_REQUESTS}"
  "--label_keys" ${LABEL_KEYS}
)

# Add optional max_samples
if ((${#MAX_SAMPLES_ARG[@]})); then
  ARGS+=("${MAX_SAMPLES_ARG[@]}")
fi

# ---------------- Provider: lists (portable, no mapfile) ----------------
PROVIDER_ORDER=()
while IFS= read -r line; do [[ -n "$line" ]] && PROVIDER_ORDER+=("$line"); done < <(yaml_get_list 'openrouter_LLM.provider.order')
PROVIDER_ONLY=()
while IFS= read -r line; do [[ -n "$line" ]] && PROVIDER_ONLY+=("$line"); done < <(yaml_get_list 'openrouter_LLM.provider.only')
PROVIDER_IGNORE=()
while IFS= read -r line; do [[ -n "$line" ]] && PROVIDER_IGNORE+=("$line"); done < <(yaml_get_list 'openrouter_LLM.provider.ignore')
PROVIDER_QUANT=()
while IFS= read -r line; do [[ -n "$line" ]] && PROVIDER_QUANT+=("$line"); done < <(yaml_get_list 'openrouter_LLM.provider.quantizations')

((${#PROVIDER_ORDER[@]}))  && ARGS+=(--provider_order "${PROVIDER_ORDER[@]}")
((${#PROVIDER_ONLY[@]}))   && ARGS+=(--provider_only "${PROVIDER_ONLY[@]}")
((${#PROVIDER_IGNORE[@]})) && ARGS+=(--provider_ignore "${PROVIDER_IGNORE[@]}")
((${#PROVIDER_QUANT[@]}))  && ARGS+=(--provider_quantizations "${PROVIDER_QUANT[@]}")

# ---------------- Provider: booleans & scalars ----------------
ALLOW_FB="$(yaml_get_bool 'openrouter_LLM.provider.allow_fallbacks')"
[[ "${ALLOW_FB:-}" == "true"  ]] && ARGS+=(--provider_allow_fallbacks)
[[ "${ALLOW_FB:-}" == "false" ]] && ARGS+=(--provider_no_fallbacks)

REQ_PARAMS="$(yaml_get_bool 'openrouter_LLM.provider.require_parameters')"
[[ "${REQ_PARAMS:-}" == "true"  ]] && ARGS+=(--provider_require_parameters)
[[ "${REQ_PARAMS:-}" == "false" ]] && ARGS+=(--provider_no_require_parameters)

DATA_COLL="$(yaml_get_or_default 'openrouter_LLM.provider.data_collection' '')"
[[ -n "${DATA_COLL}" ]] && ARGS+=(--provider_data_collection "${DATA_COLL}")

P_SORT="$(yaml_get_or_default 'openrouter_LLM.provider.sort' '')"
[[ -n "${P_SORT}" ]] && ARGS+=(--provider_sort "${P_SORT}")

MAX_PRICE_PROMPT="$(yaml_get_or_default 'openrouter_LLM.provider.max_price.prompt' '')"
[[ -n "${MAX_PRICE_PROMPT}" ]] && ARGS+=(--provider_max_price_prompt "${MAX_PRICE_PROMPT}")
MAX_PRICE_COMPLETION="$(yaml_get_or_default 'openrouter_LLM.provider.max_price.completion' '')"
[[ -n "${MAX_PRICE_COMPLETION}" ]] && ARGS+=(--provider_max_price_completion "${MAX_PRICE_COMPLETION}")

# ---------------- Generation: scalars ----------------
for key in max_tokens temperature top_p top_k frequency_penalty presence_penalty repetition_penalty; do
  val="$(yaml_get_or_default "openrouter_LLM.generation.${key}" '')"
  if [[ -n "${val}" && "${val}" != "null" ]]; then
    ARGS+=("--${key}" "${val}")
  fi
done

# Generation: stop list (portable)
STOP_LIST=()
while IFS= read -r line; do [[ -n "$line" ]] && STOP_LIST+=("$line"); done < <(yaml_get_list 'openrouter_LLM.generation.stop')
((${#STOP_LIST[@]})) && ARGS+=(--stop "${STOP_LIST[@]}")

# Generation: booleans
STREAM="$(yaml_get_bool 'openrouter_LLM.generation.stream')"
[[ "${STREAM:-}" == "true" ]] && ARGS+=(--stream)

# Generation: response_format (JSON)
RESP_JSON="$(yaml_get_json 'openrouter_LLM.generation.response_format' 2>/dev/null || true)"
[[ -n "${RESP_JSON:-}" ]] && ARGS+=(--response_format_json "${RESP_JSON}")

# Reasoning
RE_EFFORT="$(yaml_get_or_default 'openrouter_LLM.generation.reasoning.effort' '')"
[[ -n "${RE_EFFORT}" ]] && ARGS+=(--reasoning_effort "${RE_EFFORT}")

RE_MAXTOK="$(yaml_get_or_default 'openrouter_LLM.generation.reasoning.max_tokens' '')"
[[ -n "${RE_MAXTOK}" ]] && ARGS+=(--reasoning_max_tokens "${RE_MAXTOK}")

RE_EXCLUDE="$(yaml_get_bool 'openrouter_LLM.generation.reasoning.exclude')"
[[ "${RE_EXCLUDE:-}" == "true"  ]] && ARGS+=(--reasoning_exclude)
[[ "${RE_EXCLUDE:-}" == "false" ]] && ARGS+=(--no_reasoning_exclude)

# Headers
HTTP_REF="$(yaml_get_or_default 'openrouter_LLM.generation.http_referer' '')"
[[ -n "${HTTP_REF}" ]] && ARGS+=(--http_referer "${HTTP_REF}")

X_TITLE="$(yaml_get_or_default 'openrouter_LLM.generation.x_title' '')"
[[ -n "${X_TITLE}" ]] && ARGS+=(--x_title "${X_TITLE}")

PREPEND_REASONING="$(yaml_get_bool 'openrouter_LLM.generation.prepend_reasoning')"
[[ "${PREPEND_REASONING:-}" == "true" ]] && ARGS+=(--prepend_reasoning)

# -------------------------------------------------------------------
# Invoke the Python driver (API key via env, never echoed)
# -------------------------------------------------------------------
PYTHONIOENCODING=UTF-8 \
API_KEY="${API_KEY}" \
python "${SCRIPT_DIR}/predict.py" "${ARGS[@]}"

# Extra belt-and-braces: clear in case trap was removed/overridden
unset -v API_KEY 2>/dev/null || true