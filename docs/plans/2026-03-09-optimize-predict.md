# Optimize openrouter_LLM predict.py — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure predict.py for cleaner code (config dicts instead of 30+ params) and faster execution (parallel models via run.sh, adaptive rate limiting).

**Architecture:** predict.py handles one model per invocation. run.sh fans out across models in parallel. Nextflow handles dataset-level parallelism. Adaptive backoff replaces fixed delay.

**Tech Stack:** Python 3, requests, cell2sentence, Nextflow DSL2, bash

**Design doc:** `docs/plans/2026-03-09-optimize-predict-design.md`

---

### Task 1: Refactor query_openrouter_model to accept config dicts

**Files:**
- Modify: `models/openrouter_LLM/src/predict.py:125-282`

**Step 1: Replace the function signature and body**

Replace the current `query_openrouter_model` (lines 125-282) with this version that accepts config dicts:

```python
def query_openrouter_model(
    model: str,
    prompt: str,
    sysprompt: str,
    api_key: str,
    *,
    provider_cfg: Dict[str, Any] | None = None,
    generation_cfg: Dict[str, Any] | None = None,
    reasoning_cfg: Dict[str, Any] | None = None,
    max_retries: int = 5,
    backoff: float = 1.5,
    prepend_reasoning: bool = False,
) -> str:
    """Send a single chat completion request to OpenRouter with retry logic."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    provider_cfg = provider_cfg or {}
    generation_cfg = generation_cfg or {}
    reasoning_cfg = reasoning_cfg or {}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if generation_cfg.get("http_referer"):
        headers["HTTP-Referer"] = generation_cfg["http_referer"]
    if generation_cfg.get("x_title"):
        headers["X-Title"] = generation_cfg["x_title"]

    data: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": sysprompt or ""},
            {"role": "user", "content": prompt},
        ],
    }

    # Build provider block — filter out None/empty values
    provider = {}
    for key in ("order", "only", "ignore", "quantizations"):
        val = provider_cfg.get(key)
        if val:
            provider[key] = val
    for key in ("allow_fallbacks", "require_parameters", "data_collection", "sort"):
        val = provider_cfg.get(key)
        if val is not None:
            provider[key] = val
    # max_price sub-dict
    max_price = {}
    if provider_cfg.get("max_price_prompt") is not None:
        max_price["prompt"] = provider_cfg["max_price_prompt"]
    if provider_cfg.get("max_price_completion") is not None:
        max_price["completion"] = provider_cfg["max_price_completion"]
    if max_price:
        provider["max_price"] = max_price
    if provider:
        data["provider"] = provider

    # Generation params — add non-None values
    for key in ("max_tokens", "temperature", "top_p", "top_k", "seed",
                "frequency_penalty", "presence_penalty", "repetition_penalty", "stop"):
        val = generation_cfg.get(key)
        if val is not None:
            data[key] = val
    if generation_cfg.get("stream"):
        data["stream"] = True
    if generation_cfg.get("response_format") is not None:
        data["response_format"] = generation_cfg["response_format"]

    # Reasoning block
    reasoning = {}
    if reasoning_cfg.get("effort") is not None:
        reasoning["effort"] = reasoning_cfg["effort"]
    if reasoning_cfg.get("max_tokens") is not None:
        if "effort" in reasoning:
            logger.warning("Both reasoning effort and max_tokens supplied; using max_tokens.")
            reasoning.pop("effort", None)
        reasoning["max_tokens"] = reasoning_cfg["max_tokens"]
    if reasoning_cfg.get("exclude") is not None:
        reasoning["exclude"] = reasoning_cfg["exclude"]
    if reasoning:
        data["reasoning"] = reasoning

    # Send with retry + adaptive backoff
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=data, timeout=90)
            rid = r.headers.get("x-request-id") or r.headers.get("openrouter-request-id")
            rtext = r.text[:2000]

            if r.status_code == 200:
                js = r.json()
                msg = js["choices"][0]["message"]
                content = (msg.get("content") or "") if isinstance(msg, dict) else ""
                if prepend_reasoning and isinstance(msg, dict) and msg.get("reasoning"):
                    content = f"<think>{msg['reasoning']}</think>\n{content}"
                return content

            logger.error(
                f"OpenRouter HTTP {r.status_code} (attempt {attempt}/{max_retries}) "
                f"req_id={rid} model={model} body={rtext}"
            )

            if r.status_code == 429:
                ra = r.headers.get("retry-after")
                sleep_s = float(ra) if ra else backoff * (2 ** (attempt - 1))
                logger.info(f"Rate limited, sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
                last_err = rtext
                continue

            if 500 <= r.status_code < 600:
                time.sleep(backoff * (2 ** (attempt - 1)))
                last_err = rtext
                continue

            last_err = rtext
            break

        except requests.Timeout as e:
            logger.error(f"Timeout (attempt {attempt}/{max_retries}) model={model}: {e}")
            time.sleep(backoff * (2 ** (attempt - 1)))
            last_err = str(e)
        except Exception as e:
            logger.error(f"Request error (attempt {attempt}/{max_retries}) model={model}: {e}")
            time.sleep(backoff * (2 ** (attempt - 1)))
            last_err = str(e)

    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts. Last error: {last_err}")
```

Key changes from original:
- 30 individual params replaced by 3 dicts
- `max_retries` increased from 3 to 5
- Backoff is now true exponential: `backoff * 2^(attempt-1)` instead of `backoff * attempt`
- Removed commented-out debug logging

**Step 2: Verify the file is syntactically valid**

Run: `cd models/openrouter_LLM/src && python -c "import ast; ast.parse(open('predict.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add models/openrouter_LLM/src/predict.py
git commit -m "refactor: query_openrouter_model accepts config dicts instead of 30+ params"
```

---

### Task 2: Refactor run_OpenRouter_model_prediction to use config dicts

**Files:**
- Modify: `models/openrouter_LLM/src/predict.py:285-400`

**Step 1: Replace the function signature and body**

Replace `run_OpenRouter_model_prediction` (lines 285-400) with:

```python
def run_OpenRouter_model_prediction(
    model: str,
    api_key: str,
    csdata,
    task_name: str,
    input_prompt_template: str,
    answer_template: str,
    *,
    top_k_genes: int = 200,
    max_samples: int | None = None,
    max_workers: int = 10,
    provider_cfg: Dict[str, Any] | None = None,
    generation_cfg: Dict[str, Any] | None = None,
    reasoning_cfg: Dict[str, Any] | None = None,
    prepend_reasoning: bool = False,
):
    """Run cell type predictions for a single model across all cells."""
    hf_ds = load_from_disk(csdata.data_path)

    formatter = CustomPromptFormatter(
        task_name=task_name,
        input_prompt=input_prompt_template,
        answer_template=answer_template,
        top_k_genes=top_k_genes,
    )
    formatted = formatter.format_hf_ds(hf_ds)

    if max_samples is not None:
        max_samples = int(max_samples)
        max_samples = min(max_samples, formatted.num_rows)
        formatted = formatted.select(range(max_samples))

    logger.info(f"First formatted sample: {formatted[0]}")

    n = formatted.num_rows
    logger.info(f"Predicting {n} cells with model={model}, workers={max_workers}")

    def _work(idx, sample):
        for attempt in range(3):
            try:
                res = query_openrouter_model(
                    model=model,
                    prompt=sample["model_input"],
                    sysprompt="",
                    api_key=api_key,
                    provider_cfg=provider_cfg,
                    generation_cfg=generation_cfg,
                    reasoning_cfg=reasoning_cfg,
                    prepend_reasoning=prepend_reasoning,
                )
                return idx, res
            except Exception as e:
                logger.warning(f"[{idx}] retry {attempt+1} error: {e}")
                time.sleep(2 ** attempt)
        return idx, None

    preds = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, i, formatted[i]) for i in range(n)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
            i, res = fut.result()
            preds[i] = res

    return preds
```

Key changes:
- `delay_between_requests` removed — no more fixed sleep between requests
- `max_workers` default changed from 3 to 10
- All provider/generation/reasoning params replaced by 3 config dicts
- The massive parameter forwarding block (~40 lines) is gone

**Step 2: Verify syntax**

Run: `cd models/openrouter_LLM/src && python -c "import ast; ast.parse(open('predict.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add models/openrouter_LLM/src/predict.py
git commit -m "refactor: run_OpenRouter_model_prediction uses config dicts, remove fixed delay"
```

---

### Task 3: Refactor _parse_args and main() for single model

**Files:**
- Modify: `models/openrouter_LLM/src/predict.py:578-837`

**Step 1: Replace _parse_args**

Replace `_parse_args` (lines 578-653) with a streamlined version:

```python
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM prediction for single-cell data")

    # Required I/O
    p.add_argument("--input", "-i", required=True, help="Path to input .h5ad")
    p.add_argument("--output", "-o", required=True, help="Path to output predictions.csv")

    # Core config
    p.add_argument("--model", "-m", required=True, help="Single OpenRouter model ID")
    p.add_argument("--output_mode", "-om", default="label", choices=["cl_id", "label"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_name", default="cell_type_prediction")
    p.add_argument("--input_prompt_template", required=True)
    p.add_argument("--answer_template", required=True)
    p.add_argument("--top_k_genes", type=int, default=200)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=10)
    p.add_argument("--label_keys", "-l", nargs="+", required=True)
    p.add_argument("--api-key", dest="api_key", default=None)

    # Provider config (JSON string)
    p.add_argument("--provider_cfg_json", default="{}", help="JSON string of provider config")

    # Generation config (JSON string)
    p.add_argument("--generation_cfg_json", default="{}", help="JSON string of generation config")

    # Reasoning config (JSON string)
    p.add_argument("--reasoning_cfg_json", default="{}", help="JSON string of reasoning config")

    # Misc
    p.add_argument("--prepend_reasoning", action="store_true")

    args = p.parse_args()

    # Parse JSON config strings
    try:
        args.provider_cfg = json.loads(args.provider_cfg_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--provider_cfg_json is not valid JSON: {e}")
    try:
        args.generation_cfg = json.loads(args.generation_cfg_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--generation_cfg_json is not valid JSON: {e}")
    try:
        args.reasoning_cfg = json.loads(args.reasoning_cfg_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--reasoning_cfg_json is not valid JSON: {e}")

    return args
```

Key changes:
- `--models` (plural, nargs="+") replaced by `--model` (singular, required)
- 25+ individual provider/generation/reasoning CLI args replaced by 3 JSON strings
- run.sh will build these JSON strings from the YAML config

**Step 2: Replace main()**

Replace `main()` (lines 656-832) with:

```python
def main():
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = args.api_key or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: API key not provided. Set env API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(3)

    assert_model_available(args.model, api_key)

    # Inject seed into generation config if not already set
    if "seed" not in args.generation_cfg:
        args.generation_cfg["seed"] = args.seed

    adata = load_data(args.input)

    # Resolve gene symbols (unchanged logic)
    if adata.var_names.str.contains("ENS").all():
        for col in ("gene_names", "feature_name", "gene_symbols"):
            if col in adata.var.columns:
                adata.var_names = adata.var[col].astype(str)
                logger.info(f"Renamed var_names from Ensembl IDs using adata.var['{col}']")
                break
        else:
            print("ERROR: Gene names are not available in adata.var.", file=sys.stderr)
            sys.exit(4)
    elif not adata.var_names.str.contains("ENS").any():
        logger.info("var_names appears to be gene symbols already.")
    else:
        for col in ("gene_names", "feature_name", "gene_symbols"):
            if col in adata.var.columns:
                adata.var_names = adata.var[col].astype(str)
                logger.info(f"Overwrote mixed var_names with adata.var['{col}']")
                break
        else:
            print("ERROR: Mixed gene/ensembl IDs but no gene names in adata.var.", file=sys.stderr)
            sys.exit(4)

    adata.var_names = adata.var_names.str.upper()
    adata.var_names_make_unique()

    output_path = Path(args.output)
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    output_dir = output_path.parent

    for annotation_key in args.label_keys:
        logger.info(f"Processing annotation key: {annotation_key}")

        if annotation_key not in adata.obs.columns:
            print(f"ERROR: Label '{annotation_key}' not in adata.obs columns", file=sys.stderr)
            sys.exit(4)

        adata.obs["true_labels"] = adata.obs[annotation_key].astype("category")

        if pd.api.types.is_categorical_dtype(adata.obs["true_labels"]):
            labels = adata.obs["true_labels"].cat.categories.to_numpy()
        else:
            labels = pd.unique(adata.obs["true_labels"].values)

        rng = np.random.default_rng(args.seed)
        adata.obs["unique_cell_types"] = [rng.permutation(labels).tolist() for _ in range(adata.n_obs)]

        # Build C2S dataset
        arrow_ds, vocab = cs.CSData.adata_to_arrow(
            adata=adata,
            random_state=args.seed,
            sentence_delimiter=" ",
            label_col_names=["true_labels", "unique_cell_types"],
        )
        save_dir = Path(args.output).parent
        csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds,
            vocabulary=vocab,
            save_dir=str(save_dir),
            save_name=f"{Path(args.input).stem}_c2s",
            dataset_backend="arrow",
        )

        # Single model prediction
        preds = run_OpenRouter_model_prediction(
            model=args.model,
            api_key=api_key,
            csdata=csdata,
            task_name=args.task_name,
            input_prompt_template=args.input_prompt_template,
            answer_template=args.answer_template,
            top_k_genes=args.top_k_genes,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            provider_cfg=args.provider_cfg,
            generation_cfg=args.generation_cfg,
            reasoning_cfg=args.reasoning_cfg,
            prepend_reasoning=args.prepend_reasoning,
        )

        logger.info(f"First 10 predictions: {preds[:10]}")

        if args.output_mode == "cl_id":
            pred_cl_ids, pred_labels = cl_ids_and_labels_from_preds(preds, online=True, follow_replacements=True)
        elif args.output_mode == "label":
            pred_labels = preds
            pred_cl_ids = [None] * len(preds)
        else:
            raise ValueError(f"Invalid output mode: {args.output_mode}")

        df = create_results_dataframe(adata, pred_cl_ids, pred_labels, args.label_keys)

        output_csv = output_dir / f"{output_stem}_{annotation_key}{output_suffix}"
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=True)
        logger.info(f"Wrote predictions: {output_csv}")
```

Key changes:
- Single model, no model loop
- Config dicts passed directly to `run_OpenRouter_model_prediction`
- Gene symbol resolution refactored to avoid code duplication (DRY)
- `print()` replaced with `logger` where appropriate
- Output filename no longer includes model slug (run.sh handles this)
- Removed `model_to_slug` function (no longer needed)

**Step 3: Remove `model_to_slug` function (lines 62-68)**

Delete the `model_to_slug` function — run.sh handles slug generation now.

**Step 4: Verify syntax**

Run: `cd models/openrouter_LLM/src && python -c "import ast; ast.parse(open('predict.py').read()); print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add models/openrouter_LLM/src/predict.py
git commit -m "refactor: single-model predict.py with JSON config args, DRY gene symbol resolution"
```

---

### Task 4: Update run.sh for parallel model invocation with JSON configs

**Files:**
- Modify: `models/openrouter_LLM/src/run.sh`

**Step 1: Replace the YAML helper calls and predict.py invocation**

The run.sh needs these changes:

1. Replace the individual provider/generation/reasoning parameter extraction with a single Python call that builds JSON config strings from the YAML.
2. Loop over `MODEL_NAMES` and invoke predict.py per model in parallel.
3. Use `--model` (singular) instead of `--models`.

Replace everything from line 246 (`# Build predict.py argument list`) to line 348 (the python invocation) with:

```bash
# -------------------------------------------------------------------
# Build JSON config strings from YAML (provider, generation, reasoning)
# -------------------------------------------------------------------
read -r PROVIDER_JSON GENERATION_JSON REASONING_JSON < <(python - "$MODEL_CFG" <<'PY'
import sys, yaml, json, os

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
        if mp:
            # Flatten to max_price_prompt / max_price_completion
            for mk, mv in mp.items():
                provider_clean[f"max_price_{mk}"] = mv
    elif isinstance(v, list):
        if v:  # skip empty lists
            provider_clean[k] = v
    elif v is not None:
        provider_clean[k] = v

# Generation config — strip None values and empty lists
gen = root.get("generation", {}) or {}
gen_clean = {}
for k, v in gen.items():
    if k == "reasoning":
        continue  # handled separately
    elif isinstance(v, list):
        if v:
            gen_clean[k] = v
    elif v is not None:
        gen_clean[k] = v

# Reasoning config
reasoning = gen.get("reasoning", {}) or {}
reason_clean = {k: v for k, v in reasoning.items() if v is not None}

print(json.dumps(provider_clean, separators=(",", ":")), end=" ")
print(json.dumps(gen_clean, separators=(",", ":")), end=" ")
print(json.dumps(reason_clean, separators=(",", ":")))
PY
)

# -------------------------------------------------------------------
# Common args for all model invocations
# -------------------------------------------------------------------
declare -a COMMON_ARGS=(
  "--input" "${INPUT_H5AD}"
  "--seed" "${SEED}"
  "--output_mode" "${OUTPUT_MODE}"
  "--task_name" "${TASK_NAME}"
  "--input_prompt_template" "${INPUT_PROMPT_TEMPLATE}"
  "--answer_template" "${ANSWER_TEMPLATE}"
  "--top_k_genes" "${TOP_K_GENES}"
  "--max_workers" "${MAX_WORKERS}"
  "--label_keys" ${LABEL_KEYS}
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
# Launch predict.py once per model (in parallel)
# -------------------------------------------------------------------
PIDS=()
FAILED=0

for MODEL in "${MODEL_NAMES[@]}"; do
  # Filesystem-safe slug: replace / with --
  SLUG="${MODEL//\//-}"
  MODEL_OUT="${OUT_CSV%.csv}_${SLUG}.csv"

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
```

This replaces ~100 lines of individual parameter extraction (lines 266-341) with a single Python call that builds 3 JSON strings.

**Step 2: Remove the old individual parameter extraction**

The following sections are no longer needed and should be removed:
- Lines 266-279: `PROVIDER_ORDER`, `PROVIDER_ONLY`, `PROVIDER_IGNORE`, `PROVIDER_QUANT` list reads
- Lines 282-298: `ALLOW_FB`, `REQ_PARAMS`, `DATA_COLL`, `P_SORT`, `MAX_PRICE_*` scalar reads
- Lines 302-341: Generation scalars, stop list, stream, response_format, reasoning, headers

These are all replaced by the single JSON-building Python block above.

**Step 3: Remove the old unset at the bottom (line 351)**

The `unset -v API_KEY` at the end can stay as-is (the trap handles it).

**Step 4: Test that run.sh parses correctly**

Run: `bash -n models/openrouter_LLM/src/run.sh`
Expected: no output (no syntax errors)

**Step 5: Commit**

```bash
git add models/openrouter_LLM/src/run.sh
git commit -m "refactor: run.sh launches models in parallel with JSON config strings"
```

---

### Task 5: Update model_cfg.yml

**Files:**
- Modify: `models/openrouter_LLM/src/model_cfg.yml:59-65`

**Step 1: Update workload section**

Replace lines 59-65:

```yaml
  # Workload sizing / rate limiting
  max_samples: null            # 0 or null → use all cells
  max_workers: 3
  delay_between_requests: 1.2
```

With:

```yaml
  # Workload sizing
  max_samples: null            # 0 or null → use all cells
  max_workers: 10              # concurrent requests per model (adaptive backoff handles rate limits)
```

**Step 2: Commit**

```bash
git add models/openrouter_LLM/src/model_cfg.yml
git commit -m "config: increase max_workers to 10, remove delay_between_requests"
```

---

### Task 6: Update nextflow.config

**Files:**
- Modify: `nextflow.config:46-47`

**Step 1: Move openrouter_LLM to normal models**

Replace lines 46-47:

```
params.models = []
params.throttled_models = ['openrouter_LLM']
```

With:

```
params.models = ['openrouter_LLM']
params.throttled_models = []
```

**Step 2: Commit**

```bash
git add nextflow.config
git commit -m "config: move openrouter_LLM to normal models (no throttling needed for paid models)"
```

---

### Task 7: End-to-end smoke test

**Step 1: Dry-run syntax check**

Run:
```bash
cd models/openrouter_LLM/src
python -c "import ast; ast.parse(open('predict.py').read()); print('predict.py OK')"
bash -n run.sh && echo "run.sh OK"
```
Expected: both print OK

**Step 2: Test predict.py --help**

Run: `cd models/openrouter_LLM/src && python predict.py --help`
Expected: shows `--model` (singular), `--provider_cfg_json`, `--generation_cfg_json`, `--reasoning_cfg_json`

**Step 3: Run pipeline on small dataset**

Run:
```bash
nextflow run main.nf -profile local --datasets human_pancreas --models openrouter_LLM --cache_dir cache -resume
```
Expected: predictions CSV files generated per model in `model_cfg.yml`, named with model slug.

**Step 4: Verify output files**

Check that output CSVs exist for each model:
```bash
find cache/models/openrouter_LLM/ -name "predictions*.csv" | head -20
```
Expected: one CSV per model per annotation key.

**Step 5: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: address issues found in smoke test"
```
