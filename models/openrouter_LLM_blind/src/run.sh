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

# Where this script lives (so we can find model_cfg.yml)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# -------------------------------------------------------------------
# Require API_KEY from environment.
# In the Nextflow pipeline, run_model declares `secret 'OPENROUTER_API_KEY'`
# which Nextflow injects into this task's env; main.nf re-exports it as API_KEY.
# -------------------------------------------------------------------
cleanup() { unset -v API_KEY 2>/dev/null || true; }
trap cleanup EXIT

if [[ -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is not set. Provide the OpenRouter key via Nextflow secrets:" >&2
  echo "" >&2
  echo "       nextflow secrets set OPENROUTER_API_KEY 'sk-or-v1-...'" >&2
  echo "" >&2
  echo "  (prefix with a leading space if your shell honours HIST_IGNORE_SPACE)" >&2
  echo "  See README.md for details." >&2
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
# Annotation key comes from the *dataset* YAML (3rd arg)
#   Expect a single value under `annotation_key:`
# -------------------------------------------------------------------
LABEL_KEY="$(python - "$CFG_YAML" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
key = cfg.get("annotation_key") or cfg.get("annotation_key", "cell_type")
print(key)
PY
)"

if [[ -z "$LABEL_KEY" ]]; then
  echo "ERROR: No 'annotation_key' found in $CFG_YAML" >&2
  exit 2
fi

# -------------------------------------------------------------------
# Model parameters (from MODEL_CFG, with sane defaults)
# -------------------------------------------------------------------
SEED="$(yaml_get_or_default 'openrouter_LLM.seed' '42')"
# Read model_names as a list; fall back to model_name (singular) for backward compatibility
MODEL_NAMES=()
while IFS= read -r line; do [[ -n "$line" ]] && MODEL_NAMES+=("$line"); done < <(yaml_get_list 'openrouter_LLM.model_names')

if (( ${#MODEL_NAMES[@]} == 0 )); then
  # Backward compat: try singular model_name
  SINGLE_MODEL="$(yaml_get_or_default 'openrouter_LLM.model_name' 'deepseek/deepseek-r1-0528:free')"
  if [[ -n "$SINGLE_MODEL" && "$SINGLE_MODEL" != "None" ]]; then
    MODEL_NAMES=("$SINGLE_MODEL")
  fi
fi

if (( ${#MODEL_NAMES[@]} == 0 )); then
  echo "ERROR: No model_names (or model_name) found in model config." >&2
  exit 2
fi
TASK_NAME="$(yaml_get_or_default 'openrouter_LLM.task_name' 'cell_type_prediction')"
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
# Build JSON config strings from YAML (provider, generation, reasoning)
# -------------------------------------------------------------------
eval "$(python - "$MODEL_CFG" <<'PY'
import sys, yaml, json, os, shlex

cfg_path = sys.argv[1]
cfg = {}
if cfg_path and os.path.isfile(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

root = cfg.get("openrouter_LLM", {})

# Provider config — strip None values and empty lists
provider = root.get("provider", {}) or {}
provider_clean = {}
for k, v in provider.items():
    if k == "max_price" and isinstance(v, dict):
        mp = {mk: mv for mk, mv in v.items() if mv is not None}
        for mk, mv in mp.items():
            provider_clean["max_price_" + mk] = mv
    elif isinstance(v, list):
        if v:
            provider_clean[k] = v
    elif v is not None:
        provider_clean[k] = v

# Generation config — strip None values and empty lists
gen = root.get("generation", {}) or {}
gen_clean = {}
for k, v in gen.items():
    if k == "reasoning":
        continue
    elif isinstance(v, list):
        if v:
            gen_clean[k] = v
    elif v is not None:
        gen_clean[k] = v

# Reasoning config
reasoning = gen.get("reasoning", {}) or {}
reason_clean = {k: v for k, v in reasoning.items() if v is not None}

print("PROVIDER_JSON=" + shlex.quote(json.dumps(provider_clean, separators=(",", ":"))))
print("GENERATION_JSON=" + shlex.quote(json.dumps(gen_clean, separators=(",", ":"))))
print("REASONING_JSON=" + shlex.quote(json.dumps(reason_clean, separators=(",", ":"))))
PY
)"

# -------------------------------------------------------------------
# Common args for all model invocations
# -------------------------------------------------------------------
declare -a COMMON_ARGS=(
  "--input" "${INPUT_H5AD}"
  "--seed" "${SEED}"
  "--task_name" "${TASK_NAME}"
  "--input_prompt_template" "${INPUT_PROMPT_TEMPLATE}"
  "--answer_template" "${ANSWER_TEMPLATE}"
  "--top_k_genes" "${TOP_K_GENES}"
  "--max_workers" "${MAX_WORKERS}"
  "--label_key" "${LABEL_KEY}"
  "--provider_cfg_json" "${PROVIDER_JSON}"
  "--generation_cfg_json" "${GENERATION_JSON}"
  "--reasoning_cfg_json" "${REASONING_JSON}"
)

# Add optional max_samples
if ((${#MAX_SAMPLES_ARG[@]})); then
  COMMON_ARGS+=("${MAX_SAMPLES_ARG[@]}")
fi

# Add prepend_reasoning if set
PREPEND_REASONING="$(yaml_get_bool 'openrouter_LLM.generation.prepend_reasoning')"
[[ "${PREPEND_REASONING:-}" == "true" ]] && COMMON_ARGS+=(--prepend_reasoning)

# -------------------------------------------------------------------
# Pre-flight: validate ALL model names against OpenRouter before any work
# -------------------------------------------------------------------
echo "Validating model availability on OpenRouter..." >&2
API_KEY="${API_KEY}" python - "${MODEL_NAMES[@]}" <<'PY'
import os, sys, requests
api_key = os.environ['API_KEY']
models = sys.argv[1:]
try:
    r = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"WARNING: Could not validate models (HTTP {r.status_code}). Proceeding anyway.", file=sys.stderr)
        sys.exit(0)
    available = {m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)}
    missing = [m for m in models if m not in available]
    if missing:
        print(f"ERROR: The following models are NOT available on OpenRouter:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(f"Check model IDs at https://openrouter.ai/models", file=sys.stderr)
        sys.exit(1)
    print(f"All {len(models)} models validated OK.", file=sys.stderr)
except Exception as e:
    print(f"WARNING: Model validation failed ({e}). Proceeding anyway.", file=sys.stderr)
PY

# -------------------------------------------------------------------
# Launch predict.py once per model (in parallel)
# -------------------------------------------------------------------
PIDS=()
FAILED=0

for MODEL in "${MODEL_NAMES[@]}"; do
  # Filesystem-safe slug: replace / with --
  SLUG="${MODEL//\//--}"
  MODEL_OUT="${OUT_CSV%.csv}_${SLUG}.csv"

  # If main.nf pre-staged a cached prediction for this slug, skip the API call.
  if [[ -f "$MODEL_OUT" ]]; then
    echo "Skipping ${MODEL}: cached output present at ${MODEL_OUT}" >&2
    continue
  fi

  echo "Starting model: ${MODEL} (output: ${MODEL_OUT})" >&2

  PYTHONIOENCODING=UTF-8 \
  API_KEY="${API_KEY}" \
  python "${SCRIPT_DIR}/predict.py" \
    --model "${MODEL}" \
    --output "${MODEL_OUT}" \
    "${COMMON_ARGS[@]}" &

  PIDS+=($!)
done

# Wait for all models and track failures
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=$((FAILED + 1))
  fi
done

if ((FAILED > 0)); then
  echo "ERROR: ${FAILED} model(s) failed." >&2
  exit 1
fi

# Extra belt-and-braces: clear in case trap was removed/overridden
unset -v API_KEY 2>/dev/null || true