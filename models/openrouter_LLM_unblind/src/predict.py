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
    provider_order: List[str] | None = None,
    provider_only: List[str] | None = None,
    provider_ignore: List[str] | None = None,
    provider_allow_fallbacks: bool | None = None,
    provider_require_parameters: bool | None = None,
    provider_data_collection: str | None = None,
    provider_sort: str | None = None,
    provider_quantizations: List[str] | None = None,
    provider_max_price_prompt: float | None = None,
    provider_max_price_completion: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    stop: Union[str, List[str], None] = None,
    stream: bool = False,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
    reasoning_exclude: bool | None = None,
    http_referer: str | None = None,
    x_title: str | None = None,
    max_retries: int = 3,
    backoff: float = 1.5,
    prepend_reasoning: bool = False,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    data: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": sysprompt or ""},
            {"role": "user", "content": prompt},
        ],
    }

    provider: dict = {}
    if provider_order:            provider["order"] = provider_order
    if provider_only:             provider["only"] = provider_only
    if provider_ignore:           provider["ignore"] = provider_ignore
    if provider_allow_fallbacks is not None: provider["allow_fallbacks"] = provider_allow_fallbacks
    if provider_require_parameters is not None: provider["require_parameters"] = provider_require_parameters
    if provider_data_collection is not None:    provider["data_collection"] = provider_data_collection
    if provider_sort:             provider["sort"] = provider_sort
    if provider_quantizations:    provider["quantizations"] = provider_quantizations
    if (provider_max_price_prompt is not None) or (provider_max_price_completion is not None):
        provider["max_price"] = {
            k: v for k, v in {
                "prompt": provider_max_price_prompt,
                "completion": provider_max_price_completion,
            }.items() if v is not None
        }
    if provider:
        data["provider"] = provider

    if max_tokens is not None:        data["max_tokens"] = max_tokens
    if temperature is not None:       data["temperature"] = temperature
    if top_p is not None:             data["top_p"] = top_p
    if top_k is not None:             data["top_k"] = top_k
    if seed is not None:              data["seed"] = seed
    if frequency_penalty is not None: data["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:  data["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:data["repetition_penalty"] = repetition_penalty
    if stop is not None:              data["stop"] = stop
    if stream:                        data["stream"] = True
    if response_format is not None:   data["response_format"] = response_format

    reasoning: Dict[str, Any] = {}
    if reasoning_effort is not None:
        reasoning["effort"] = reasoning_effort
    if reasoning_max_tokens is not None:
        if "effort" in reasoning:
            logger.warning("Both reasoning_effort and reasoning_max_tokens supplied; using max_tokens.")
            reasoning.pop("effort", None)
        reasoning["max_tokens"] = reasoning_max_tokens
    if reasoning_exclude is not None:
        reasoning["exclude"] = reasoning_exclude
    if reasoning:
        data["reasoning"] = reasoning

    # --- send with structured error logging and 429/5xx backoff ---
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=data, timeout=90)
            rid = r.headers.get("x-request-id") or r.headers.get("openrouter-request-id")
            rtext = r.text[:2000]  # cap for logs

            # # Information for debugging
            # logger.info("request")
            # logger.info(headers)
            # logger.info(data)
            # logger.info("response")
            # logger.info(str(r))
            # logger.info(str(r.headers))
            # logger.info(str(r.status_code))
            # logger.info(str(r.json()))

            if r.status_code == 200:
                js = r.json()
                msg = js["choices"][0]["message"]
                content = (msg.get("content") or "") if isinstance(msg, dict) else ""
                if prepend_reasoning and isinstance(msg, dict) and msg.get("reasoning"):
                    content = f"<think>{msg['reasoning']}</think>\n{content}"
                return content

            # Log the exact failure
            logger.error(
                f"OpenRouter HTTP {r.status_code} (attempt {attempt}/{max_retries}) "
                f"req_id={rid} model={model} provider={provider.get('order') if provider else None} body={rtext}"
            )

            # 429: respect Retry-After if present
            if r.status_code == 429:
                ra = r.headers.get("retry-after")
                sleep_s = float(ra) if ra else backoff * attempt
                time.sleep(sleep_s)
                last_err = rtext
                continue

            # 5xx: transient
            if 500 <= r.status_code < 600:
                time.sleep(backoff * attempt)
                last_err = rtext
                continue

            # 4xx other than 429 are not retriable
            last_err = rtext
            break

        except requests.Timeout as e:
            logger.error(f"Timeout (attempt {attempt}/{max_retries}) model={model}: {e}")
            time.sleep(backoff * attempt)
            last_err = str(e)
        except Exception as e:
            logger.error(f"Request error (attempt {attempt}/{max_retries}) model={model}: {e}")
            time.sleep(backoff * attempt)
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
    max_workers: int = 3,
    delay_between_requests: float = 1.2,
    # Provider
    provider_order: List[str] | None = None,
    provider_only: List[str] | None = None,
    provider_ignore: List[str] | None = None,
    provider_allow_fallbacks: bool | None = None,
    provider_require_parameters: bool | None = None,
    provider_data_collection: str | None = None,
    provider_sort: str | None = None,
    provider_quantizations: List[str] | None = None,
    provider_max_price_prompt: float | None = None,
    provider_max_price_completion: float | None = None,
    # Generation
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    stop: Union[str, List[str], None] = None,
    stream: bool = False,
    response_format: dict | None = None,
    # Reasoning
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
    reasoning_exclude: bool | None = None,
    # Headers
    http_referer: str | None = None,
    x_title: str | None = None,
    prepend_reasoning: bool = False,
):
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

    # Print the first formatted row for debugging
    print(formatted[0])

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
                    provider_order=provider_order,
                    provider_only=provider_only,
                    provider_ignore=provider_ignore,
                    provider_allow_fallbacks=provider_allow_fallbacks,
                    provider_require_parameters=provider_require_parameters,
                    provider_data_collection=provider_data_collection,
                    provider_sort=provider_sort,
                    provider_quantizations=provider_quantizations,
                    provider_max_price_prompt=provider_max_price_prompt,
                    provider_max_price_completion=provider_max_price_completion,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    stop=stop,
                    stream=stream,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    reasoning_max_tokens=reasoning_max_tokens,
                    reasoning_exclude=reasoning_exclude,
                    http_referer=http_referer,
                    x_title=x_title,
                    prepend_reasoning=prepend_reasoning,
                )
                time.sleep(delay_between_requests)
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

    # Process single annotation key
    key = label_keys[0] if label_keys else "cell_type"
    col_name = "true_labels"
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
    p.add_argument("--model", "-m", default="deepseek/deepseek-r1-0528:free")
    p.add_argument("--output_mode", "-om", default="label", choices=["cl_id", "label"])
    p.add_argument("--seed", type=int, default=42, help="Random seed (also passed to model where supported)")
    p.add_argument("--task_name", default="cell_type_prediction")
    p.add_argument("--input_prompt_template", required=True)
    p.add_argument("--answer_template", required=True)
    p.add_argument("--top_k_genes", type=int, default=200)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--delay_between_requests", type=float, default=1.2)
    p.add_argument("--label_key", "-l", required=True)
    p.add_argument("--api-key", dest="api_key", default=None, help="Override API key (else read env API_KEY)")

    # Provider routing
    p.add_argument("--provider_order", nargs="*", default=None, help="Preferred provider order")
    p.add_argument("--provider_only", nargs="*", default=None, help="Whitelist providers")
    p.add_argument("--provider_ignore", nargs="*", default=None, help="Blacklist providers")
    p.add_argument("--provider_allow_fallbacks", action="store_true", help="Allow provider fallbacks")
    p.add_argument("--provider_no_fallbacks", dest="provider_allow_fallbacks", action="store_false")
    p.set_defaults(provider_allow_fallbacks=None)
    p.add_argument("--provider_require_parameters", dest="provider_require_parameters", action="store_true")
    p.add_argument("--provider_no_require_parameters", dest="provider_require_parameters", action="store_false")
    p.set_defaults(provider_require_parameters=None)
    p.add_argument("--provider_data_collection", choices=["allow", "deny"], default=None)
    p.add_argument("--provider_sort", choices=["price", "throughput", "latency"], default=None)
    p.add_argument("--provider_quantizations", nargs="*", default=None)
    p.add_argument("--provider_max_price_prompt", type=float, default=None)
    p.add_argument("--provider_max_price_completion", type=float, default=None)

    # Generation params
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--frequency_penalty", type=float, default=None)
    p.add_argument("--presence_penalty", type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--stop", nargs="*", default=None, help="Stop sequences (space-separated, multiple allowed)")
    p.add_argument("--stream", action="store_true", help="Enable streaming (if supported)")
    p.add_argument("--response_format_json", default=None, help='JSON string for response_format, e.g. {"type":"json_object"}')

    # Reasoning params
    p.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default=None)
    p.add_argument("--reasoning_max_tokens", type=int, default=None)
    p.add_argument("--reasoning_exclude", action="store_true", help="Exclude reasoning tokens from response")
    p.add_argument("--no_reasoning_exclude", dest="reasoning_exclude", action="store_false")
    p.set_defaults(reasoning_exclude=None)
    p.add_argument("--prepend_reasoning", action="store_true", help="Prepend <think>...</think> when available")

    # Headers / attribution
    p.add_argument("--http_referer", default=None)
    p.add_argument("--x_title", default=None)

    args = p.parse_args()

    if args.response_format_json:
        try:
            args.response_format = json.loads(args.response_format_json)
        except Exception as e:
            raise SystemExit(f"--response_format_json is not valid JSON: {e}")
    else:
        args.response_format = None

    if isinstance(args.stop, list) and len(args.stop) == 0:
        args.stop = None

    return args


def main():
    args = _parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # API key
    api_key = args.api_key or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: API key not provided. Set env API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(3)

    # Sanity check the model up front
    assert_model_available(args.model, api_key)

    # Load adata
    adata = load_data(args.input)

    # Check if gene symbols are available somewhere in the anndata object:
    #adata.var_names_make_unique()
    if adata.var_names.str.contains("ENS").all():
        # All Ensembl IDs, try to swap to gene symbols if available
        if "gene_names" in adata.var.columns:
            adata.var_names = adata.var["gene_names"].astype(str)
            print("Info: Renamed adata.var_names from ensembl IDs to gene names in adata.var['gene_names'] for cell sentence construction.")
        elif "feature_name" in adata.var.columns:
            adata.var_names = adata.var["feature_name"].astype(str)
            print("Info: Renamed adata.var_names from ensembl IDs to feature names in adata.var['feature_name'] for cell sentence construction.")
        elif "gene_symbols" in adata.var.columns:
            adata.var_names = adata.var["gene_symbols"].astype(str)
            print("Info: Renamed adata.var_names from ensembl IDs to feature names in adata.var['feature_name'] for cell sentence construction.")
        else:
            print("ERROR: Gene names are not available in adata.var. Please add gene names to adata.var.", file=sys.stderr)
            sys.exit(4)
    elif not adata.var_names.str.contains("ENS").any():
        # No ensembl IDs at all, so likely gene symbols already present
        print("Info: adata.var_names appears to be gene symbols (no ensembl IDs found). Continuing with cell sentence construction.")
    else:
        # Mixture of gene symbols and ensembl IDs — ambiguous, warn user
        print("WARNING: adata.var_names contains a mixture of gene symbols and ensembl IDs.", file=sys.stderr)
        if "gene_names" in adata.var.columns:
            adata.var_names = adata.var["gene_names"].astype(str)
            print("Info: Overwrote adata.var_names with adata.var['gene_names'].")
        elif "feature_name" in adata.var.columns:
            adata.var_names = adata.var["feature_name"].astype(str)
            print("Info: Overwrote adata.var_names with adata.var['feature_name'].")
        elif "gene_symbols" in adata.var.columns:
            adata.var_names = adata.var["gene_symbols"].astype(str)
            print("Info: Overwrote adata.var_names with adata.var['gene_symbols'].")
        else:
            print("ERROR: Mixed gene/ensembl IDs but no gene names found in adata.var.", file=sys.stderr)
            sys.exit(4)

<<<<<<< HEAD:models/openrouter_LLM_unblind/src/predict.py
    annotation_key = args.label_key
    print(f"Processing annotation key: {annotation_key}")
=======
    # Uppercase before deduping to avoid collisions inside cell2sentence
    adata.var_names = adata.var_names.str.upper()
    adata.var_names_make_unique()

    # Create output filename for this annotation level
    output_path = Path(args.output)
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    output_dir = output_path.parent
>>>>>>> fix_for_helix:models/openrouter_LLM/src/predict.py

    if annotation_key not in adata.obs.columns:
        print(f"ERROR: Label '{annotation_key}' not in adata.obs columns", file=sys.stderr)
        print(f"Available columns: {list(adata.obs.columns)}", file=sys.stderr)
        sys.exit(4)

    adata.obs["true_labels"] = adata.obs[annotation_key].astype("category")

    # Get the set of labels once (robust to categorical dtype)
    if pd.api.types.is_categorical_dtype(adata.obs["true_labels"]):
        labels = adata.obs["true_labels"].cat.categories.to_numpy()
    else:
        labels = pd.unique(adata.obs["true_labels"].values)

    rng = np.random.default_rng(42)  # set a seed if you want reproducibility

    # One shuffled list per observation
    adata.obs["unique_cell_types"] = [rng.permutation(labels).tolist() for _ in range(adata.n_obs)]

    print(adata.var_names[:10])

    # Build C2S dataset (only needs true_labels)
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

    # Run predictions
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
        delay_between_requests=args.delay_between_requests,
        # Provider
        provider_order=args.provider_order,
        provider_only=args.provider_only,
        provider_ignore=args.provider_ignore,
        provider_allow_fallbacks=args.provider_allow_fallbacks,
        provider_require_parameters=args.provider_require_parameters,
        provider_data_collection=args.provider_data_collection,
        provider_sort=args.provider_sort,
        provider_quantizations=args.provider_quantizations,
        provider_max_price_prompt=args.provider_max_price_prompt,
        provider_max_price_completion=args.provider_max_price_completion,
        # Generation
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        stop=args.stop,
        stream=args.stream,
        response_format=args.response_format,
        # Reasoning
        reasoning_effort=args.reasoning_effort,
        reasoning_max_tokens=args.reasoning_max_tokens,
        reasoning_exclude=args.reasoning_exclude,
        # Headers
        http_referer=args.http_referer,
        x_title=args.x_title,
        prepend_reasoning=args.prepend_reasoning,
    )

    # Print the first 10 predictions for debugging
    print(preds[:10])

    # Convert predicted CL terms -> (normalized CL ID, cell type label)
    if args.output_mode == "cl_id":
        pred_cl_ids, pred_labels = cl_ids_and_labels_from_preds(preds, online=True, follow_replacements=True)
    elif args.output_mode == "label":
        pred_labels = preds
        pred_cl_ids = [None] * len(preds)
    else:
        raise ValueError(f"Invalid output mode: {args.output_mode}")

    # Build and save results
    df = create_results_dataframe(adata, pred_cl_ids, pred_labels, [annotation_key])
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=True)
    logger.info(f"Wrote predictions: {args.output}")


if __name__ == "__main__":
    # Avoid accidental GPU grabs by default (harmless on CPU boxes)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    main()