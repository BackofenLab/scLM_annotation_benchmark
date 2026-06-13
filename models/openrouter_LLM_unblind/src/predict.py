#!/usr/bin/env python3
"""
OpenRouter LLM inference for single-cell data (e.g. DeepSeekR1 or GPT-5).

- Reads API key from env (API_KEY) or --api-key
- Builds prompts with cell2sentence custom prompt formatter
- Queries OpenRouter LLMs (e.g. DeepSeek R1, GPT-5, ...) with retries
- Writes predictions.csv in the standard schema shared with the other models:

    cell_id, true_labels, predicted_labels

For constructing cell sentences, make sure that gene symbols are available!

The model is expected to return a cell type label directly; the raw label is written as
predicted_labels (normalization + ontology mapping happen downstream in bin/map_predictions.py).

"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import logging
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Optional, Dict, List, Iterable, Tuple, Union, Mapping, Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import scanpy as sc
import cell2sentence as cs
from cell2sentence.prompt_formatter import get_cell_sentence_str, PromptFormatter
from datasets import Dataset, load_from_disk

# ---- logging ----
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("openrouter_LLM_predict")


# ---------------- I/O helpers ----------------
def load_data(data_path: str):
    adata = sc.read_h5ad(data_path)
    logger.info(f"AnnData loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata



# ---------------- Prompting ----------------
class CustomPromptFormatter(PromptFormatter):
    def __init__(self, task_name: str, input_prompt: str, answer_template: str, top_k_genes: int):
        super().__init__()
        assert top_k_genes > 0, "'top_k_genes' must be > 0"
        self.task_name = task_name
        self.input_prompt = input_prompt
        self.answer_template = answer_template
        self.top_k_genes = top_k_genes

    def format_hf_ds(self, hf_ds: Dataset) -> Dataset:
        model_inputs_list, responses_list = [], []

        for i in range(hf_ds.num_rows):
            sample = hf_ds[i]
            cell_sentence, num_genes_str = get_cell_sentence_str(sample, num_genes=self.top_k_genes)

            model_inputs_list.append(
                self.input_prompt.format(
                    cell_sentence=cell_sentence,
                    num_genes=num_genes_str,
                    unique_cell_types=sample["unique_cell_types"]
                )
            )
            responses_list.append(
                self.answer_template.format(
                    true_labels=sample.get("true_labels", "")
                )
            )

        return Dataset.from_dict(
            {
                "sample_type": [self.task_name] * hf_ds.num_rows,
                "model_input": model_inputs_list,
                "response": responses_list,
            }
        )


# ---------------- OpenRouter client ----------------
def assert_model_available(model: str, api_key: str) -> None:
    url = "https://openrouter.ai/api/v1/models"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        if r.status_code != 200:
            logger.warning(f"Could not list models (HTTP {r.status_code}): {r.text[:500]}")
            return
        models = {m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)}
        if model not in models:
            raise SystemExit(f"Model '{model}' is not available on your OpenRouter account. "
                             f"Choose one from: {sorted(list(models))[:20]} ...")
    except Exception as e:
        logger.warning(f"Model check failed: {e}")


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

    # Reasoning block — effort and max_tokens are mutually exclusive in the
    # OpenRouter API.  When both are configured globally, pick the right one
    # based on the model provider:
    #   • Anthropic / Google models → use max_tokens (effort not supported)
    #   • Non-reasoning models (meta-llama, mistralai) → skip reasoning entirely
    #   • Mandatory-reasoning models (deepseek-r1) → only send exclude if set
    #   • All other reasoning models → use effort
    _NON_REASONING_PREFIXES = ("meta-llama", "mistralai")
    _MANDATORY_REASONING_PATTERNS = ("deepseek-r1",)

    provider_prefix = model.split("/")[0] if "/" in model else ""
    model_suffix = model.split("/")[-1] if "/" in model else model

    # Skip reasoning entirely for non-reasoning models
    if provider_prefix in _NON_REASONING_PREFIXES:
        pass  # no reasoning block
    elif any(p in model_suffix for p in _MANDATORY_REASONING_PATTERNS):
        # Mandatory reasoning — only send exclude (effort/max_tokens ignored)
        reasoning = {}
        if reasoning_cfg.get("exclude") is not None:
            reasoning["exclude"] = reasoning_cfg["exclude"]
        if reasoning:
            data["reasoning"] = reasoning
    else:
        # Optional reasoning — pick effort vs max_tokens per provider
        reasoning = {}
        if reasoning_cfg.get("enabled") is not None:
            reasoning["enabled"] = reasoning_cfg["enabled"]

        has_effort = reasoning_cfg.get("effort") is not None
        has_max_tokens = reasoning_cfg.get("max_tokens") is not None

        if has_effort and has_max_tokens:
            if provider_prefix in ("anthropic", "google"):
                reasoning["max_tokens"] = reasoning_cfg["max_tokens"]
            else:
                reasoning["effort"] = reasoning_cfg["effort"]
        elif has_effort:
            reasoning["effort"] = reasoning_cfg["effort"]
        elif has_max_tokens:
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


def create_results_dataframe(
    adata,
    predicted_labels,
    label_key: str,
) -> pd.DataFrame:
    """Build the standard prediction frame shared with all other models:
    columns cell_id, true_labels, predicted_labels (written without an index)."""
    n = len(predicted_labels)
    cell_ids = [str(x) for x in list(adata.obs_names[:n])]

    if label_key in adata.obs.columns:
        true_labels = list(adata.obs[label_key].astype("string").values[:n])
    else:
        logger.warning(f"'{label_key}' not found in adata.obs; filling NA")
        true_labels = [None] * n

    return pd.DataFrame({
        "cell_id": cell_ids,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    })


# ---------------- Main ----------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM prediction for single-cell data")

    # Required I/O
    p.add_argument("--input", "-i", required=True, help="Path to input .h5ad")
    p.add_argument("--output", "-o", required=True, help="Path to output predictions.csv")

    # Core config
    p.add_argument("--model", "-m", required=True, help="Single OpenRouter model ID")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task_name", default="cell_type_prediction")
    p.add_argument("--input_prompt_template", required=True)
    p.add_argument("--answer_template", required=True)
    p.add_argument("--top_k_genes", type=int, default=200)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=10)
    p.add_argument("--label_key", "-l", required=True)
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


def main():
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = args.api_key or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: API key not provided. Set env API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(3)

    # Model availability is validated upfront in run.sh (pre-flight check).

    # Inject seed into generation config if not already set
    if "seed" not in args.generation_cfg:
        args.generation_cfg["seed"] = args.seed

    adata = load_data(args.input)

    # Resolve gene symbols
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
    output_dir = output_path.parent

    if args.label_key not in adata.obs.columns:
        print(f"ERROR: Label '{args.label_key}' not in adata.obs columns", file=sys.stderr)
        sys.exit(4)

    adata.obs["true_labels"] = adata.obs[args.label_key].astype(str)

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
        save_name=f"{Path(args.input).stem}_{args.model.replace('/', '--')}_c2s",
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

    # Label-only: the raw model output IS the predicted label (cleaned/normalized downstream).
    pred_labels = preds

    df = create_results_dataframe(adata, pred_labels, args.label_key)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote predictions: {output_path}")


if __name__ == "__main__":
    # Avoid accidental GPU grabs by default (harmless on CPU boxes)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    main()
