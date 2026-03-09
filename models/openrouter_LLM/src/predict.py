#!/usr/bin/env python3
"""
OpenRouter LLM inference for single-cell data (e.g. DeepSeekR1 or GPT-5).

- Reads API key from env (API_KEY) or --api-key
- Builds prompts with cell2sentence custom prompt formatter
- Queries OpenRouter LLMs (e.g. DeepSeek R1, GPT-5, ...) with retries
- Writes predictions.csv with predicted CL IDs + resolved cell-type labels + true labels

For constructing cell sentences, make sure that gene symbols are available!

This version assumes the model returns either cell type labels or CL terms (e.g., "CL:0000236"). CL terms,
get mapped to human-readable labels via EBI OLS, and writes:

    predicted_cl_id, predicted_cell_type, true_labels[...]

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

import re
from functools import lru_cache
from urllib.parse import quote
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
    logger.info(f"Loading data from: {data_path}")
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


# ---------------- CL mapping helpers ----------------
_CL_TERM_RE = re.compile(r'(CL[:_]\d{7})$', re.I)

def normalize_cl_id(term: str) -> Optional[str]:
    if term is None:
        return None
    s = str(term).strip()
    if not s:
        return None
    m = _CL_TERM_RE.search(s)
    if m:
        return m.group(1).upper().replace('_', ':')
    m = re.search(r'/CL[_:](\d{7})$', s, re.I)
    if m:
        return "CL:%s" % m.group(1)
    return None

@lru_cache(maxsize=4096)
def _ols_lookup(cl_id: str) -> Dict[str, Any]:
    iri = "http://purl.obolibrary.org/obo/%s" % cl_id.replace(':', '_')
    base = "https://www.ebi.ac.uk/ols4/api/ontologies/cl/terms"

    r = requests.get(f"{base}?iri={quote(iri, safe='')}", timeout=10)
    if r.ok and r.json().get("_embedded", {}).get("terms"):
        term = r.json()["_embedded"]["terms"][0]
    else:
        short_form = cl_id.replace(":", "_")
        r = requests.get(f"{base}?short_form={short_form}", timeout=10)
        if not (r.ok and r.json().get("_embedded", {}).get("terms")):
            raise LookupError(f"CL ID not found in OLS: {cl_id}")
        term = r.json()["_embedded"]["terms"][0]

    label = term.get("label")
    obsolete = bool(term.get("is_obsolete", False))

    replaced_by = None
    ann = term.get("annotation", {}) or {}
    for k in ("replacedBy", "oboInOwl:replaced_by", "IAO:0100001"):
        val = ann.get(k)
        if val:
            if isinstance(val, list) and val:
                replaced_by = val[0]
            elif isinstance(val, str):
                replaced_by = val
            break

    return {"label": label, "obsolete": obsolete, "replaced_by": replaced_by}

def map_cl_terms_to_labels(
    terms: Iterable[str],
    local_map: Optional[Mapping[str, str]] = None,
    *,
    online: bool = True,
    follow_replacements: bool = True,
    return_missing: bool = False,
) -> Union[Dict[Any, Dict[str, Any]], Tuple[Dict[Any, Dict[str, Any]], Dict[Any, str]]]:
    local_map = local_map or {}
    _local: Dict[str, str] = {}
    for k, v in local_map.items():
        nk = normalize_cl_id(k)
        if nk:
            _local[nk] = v

    out: Dict[Any, Dict[str, Any]] = {}
    missing: Dict[Any, str] = {}

    for t in terms:
        cl = normalize_cl_id(t)
        entry = {"cl_id": cl, "label": None, "source": None, "obsolete": False, "replaced_by": None}

        if not cl:
            missing[t] = "Not a recognizable CL ID/IRI"
            out[t] = entry
            continue

        if cl in _local:
            entry.update({"label": _local[cl], "source": "local"})
            out[t] = entry
            continue

        if not online:
            missing[t] = "Not in local_map and online=False"
            out[t] = entry
            continue

        try:
            res = _ols_lookup(cl)
            if res["obsolete"] and follow_replacements and res["replaced_by"]:
                new_cl = normalize_cl_id(res["replaced_by"])
                if new_cl:
                    res2 = _ols_lookup(new_cl)
                    entry.update({"cl_id": new_cl, "label": res2["label"], "source": "ols",
                                  "obsolete": res2["obsolete"], "replaced_by": None})
                else:
                    entry.update({"label": res["label"], "source": "ols",
                                  "obsolete": True, "replaced_by": res["replaced_by"]})
            else:
                entry.update({"label": res["label"], "source": "ols",
                              "obsolete": res["obsolete"], "replaced_by": res["replaced_by"]})
        except Exception as e:
            missing[t] = str(e)

        out[t] = entry

    if return_missing:
        return out, missing
    return out

def clean_model_output(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    # keep first line, strip 'Output:' prefix & quotes
    t = t.splitlines()[0]
    t = re.sub(r'(?i)^\s*output\s*:\s*', '', t).strip()
    t = t.strip('"').strip("'")
    return t or None

def cl_ids_and_labels_from_preds(
    preds: List[Optional[str]],
    *,
    online: bool = True,
    follow_replacements: bool = True,
) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    """
    Takes raw model predictions (expected to be CL terms), normalizes to CL:#######,
    and returns parallel lists: [cl_id], [label]. If lookup fails, label is None.
    """
    cleaned = [clean_model_output(p) for p in preds]
    mapping, _missing = map_cl_terms_to_labels(
        cleaned, online=online, follow_replacements=follow_replacements, return_missing=True
    )  # type: ignore
    cl_ids: List[Optional[str]] = []
    labels: List[Optional[str]] = []
    for p in cleaned:
        info = mapping.get(p, {})
        cl_ids.append(info.get("cl_id"))
        labels.append(info.get("label"))
    return cl_ids, labels

def create_results_dataframe(
    adata,
    predicted_cl_ids: List[Optional[str]],
    predicted_labels: List[Optional[str]],
    label_keys: List[str],
) -> pd.DataFrame:
    n = len(predicted_cl_ids)
    idx = list(adata.obs_names[:n])

    data: Dict[str, List[Optional[str]]] = {
        "predicted_cl_id": predicted_cl_ids,
        "predicted_cell_type": predicted_labels,  # human-readable label resolved via OLS
    }

    # Keep compatibility: if someone expects "predicted_labels", mirror the column
    data["predicted_labels"] = predicted_labels

    if not label_keys:
        label_keys = ["cell_type"]

    for j, key in enumerate(label_keys, start=1):
        col_name = f"true_labels_{j}" if len(label_keys) > 1 else "true_labels"
        if key in adata.obs.columns:
            data[col_name] = list(adata.obs[key].astype("string").values[:n])
        else:
            logger.warning(f"'{key}' not found in adata.obs; filling NA")
            data[col_name] = [None] * n

    df = pd.DataFrame(data, index=idx)
    return df


# ---------------- Main ----------------
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


if __name__ == "__main__":
    # Avoid accidental GPU grabs by default (harmless on CPU boxes)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    main()