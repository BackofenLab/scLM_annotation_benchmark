#!/usr/bin/env python3
"""Cell type annotation with general-purpose LLMs through the OpenRouter API.

Each cell becomes one prompt: the template filled with the cell's most highly expressed genes (a
cell2sentence cell sentence) and, if the template contains {unique_cell_types}, the cell types of
the test set, shuffled for each cell. Every served model gets one request per cell, and its
answers are written unchanged; the pipeline maps them to the Cell Ontology afterwards.

Inputs: the test set (.h5ad), the dataset config (annotation_key), the prompt template and
openrouter.yml with the served models and the request settings. The key is read from API_KEY.

With --clusters and --markers, each cluster becomes one prompt instead: the template filled
with the cluster's marker genes ({marker_genes}, {num_genes}) and, for {unique_cell_types}, the
cell types of the test set in a new order for each cluster. Every cell gets the answer to its
cluster.

Outputs per served model, in the folder of --output (out/predictions.csv):
    predictions_<model>.csv  cell_id, true_labels, predicted_labels
    usage_<model>.csv        cell_id (or cluster and n_cells), provider, prompt_tokens,
                             completion_tokens, reasoning_tokens, cost, finish_reason
<model> is the model id with "/" replaced by "--". Models whose prediction file exists are
skipped. The models run in parallel, each with max_workers concurrent requests.

Called by run.sh as
    predict.py --input test_set.h5ad --output out/predictions.csv --dataset_config dataset.yml
               --prompt prompt.txt --config openrouter.yml [--clusters clusters.csv
               --markers markers.csv] [--skip_model_check]
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anndata
import cell2sentence as cs
import numpy as np
import pandas as pd
import requests
import yaml
from cell2sentence.prompt_formatter import get_cell_sentence_str

API_URL = "https://openrouter.ai/api/v1"
USAGE_COLUMNS = ["cell_id", "provider", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                 "cost", "finish_reason"]
CLUSTER_USAGE_COLUMNS = ["cluster", "n_cells"] + USAGE_COLUMNS[1:]

# A request is sent up to ATTEMPTS times per round, in up to ROUNDS rounds (see answer()).
ATTEMPTS = 5
ROUNDS = 3
BACKOFF = 1.5  # wait after the first failed attempt in seconds, doubled after each further one

# Vendors whose models get no reasoning field, and vendors whose models get reasoning.max_tokens
# rather than reasoning.effort when both are configured.
NO_REASONING = ("meta-llama", "mistralai")
TOKEN_BUDGET = ("anthropic", "google")
# Models that always reason and accept only reasoning.exclude.
ALWAYS_REASONING = ("deepseek-r1",)

log = logging.getLogger("openrouter")


def load_test_set(path, label_key):
    """The test set with upper-case gene symbols as var_names."""
    adata = anndata.read_h5ad(path)
    log.info("Loaded %s: %d cells, %d genes", path, adata.n_obs, adata.n_vars)
    if label_key not in adata.obs.columns:
        log.error("Column '%s' is not in adata.obs", label_key)
        sys.exit(4)
    if adata.var_names.str.contains("ENS").any():
        column = next((c for c in ("gene_names", "feature_name", "gene_symbols")
                       if c in adata.var.columns), None)
        if column is None:
            log.error("var_names contain Ensembl ids and adata.var has no gene_names, "
                      "feature_name or gene_symbols column")
            sys.exit(4)
        adata.var_names = adata.var[column].astype(str)
    adata.var_names = adata.var_names.str.upper()
    adata.var_names_make_unique()
    return adata


def build_prompts(adata, template, label_key, seed, top_k_genes):
    """One prompt per cell, in the order of the test set."""
    cell_types = [None] * adata.n_obs
    if "{unique_cell_types}" in template:
        rng = np.random.default_rng(seed)
        labels = pd.unique(adata.obs[label_key].astype(str).values)
        cell_types = [rng.permutation(labels).tolist() for _ in range(adata.n_obs)]
    cells, _ = cs.CSData.adata_to_arrow(adata=adata, random_state=seed, sentence_delimiter=" ")
    prompts = []
    for cell, types in zip(cells, cell_types):
        sentence, num_genes = get_cell_sentence_str(cell, num_genes=top_k_genes)
        prompts.append(template.format(cell_sentence=sentence, num_genes=num_genes,
                                       unique_cell_types=types))
    return prompts


def read_clusters(clusters_path, markers_path, cell_ids):
    """The cluster of each cell, in the order of cell_ids, and the marker genes of each cluster."""
    clusters = pd.read_csv(clusters_path, dtype=str).set_index("cell_id")["cluster"]
    missing = pd.Index(cell_ids).difference(clusters.index)
    if len(missing):
        log.error("%d cells of the test set have no cluster in %s", len(missing), clusters_path)
        sys.exit(4)
    markers = {cluster: genes.split(", ") for cluster, genes in
               pd.read_csv(markers_path, dtype=str).itertuples(index=False)}
    return clusters.loc[list(cell_ids)], markers


def build_cluster_prompts(template, markers, labels, seed):
    """One prompt per cluster, keyed by cluster id. For a template with {unique_cell_types}, the
    sorted labels are drawn in a new order for each cluster, the clusters taken in sorted order."""
    rng = np.random.default_rng(seed)
    labels = sorted(labels)
    prompts = {}
    for cluster in sorted(markers):
        types = None
        if "{unique_cell_types}" in template:
            types = ", ".join(rng.permutation(labels).tolist())
        prompts[cluster] = template.format(marker_genes=", ".join(markers[cluster]),
                                           num_genes=len(markers[cluster]),
                                           unique_cell_types=types)
    return prompts


def compact(mapping):
    """The mapping without null values, empty lists and empty mappings, at all levels."""
    result = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            value = compact(value)
        if value is not None and value != [] and value != {}:
            result[key] = value
    return result


def reasoning_field(model, reasoning):
    """The reasoning field of a request to the model, empty if none is sent."""
    vendor = model.split("/")[0] if "/" in model else ""
    if vendor in NO_REASONING:
        return {}
    if any(pattern in model.split("/")[-1] for pattern in ALWAYS_REASONING):
        return {"exclude": reasoning["exclude"]} if "exclude" in reasoning else {}
    field = {}
    if "enabled" in reasoning:
        field["enabled"] = reasoning["enabled"]
    if "max_tokens" in reasoning and (vendor in TOKEN_BUDGET or "effort" not in reasoning):
        field["max_tokens"] = reasoning["max_tokens"]
    elif "effort" in reasoning:
        field["effort"] = reasoning["effort"]
    if "exclude" in reasoning:
        field["exclude"] = reasoning["exclude"]
    return field


def request_body(model, prompt, settings):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
    }
    provider = compact(settings["provider"])
    if provider:
        body["provider"] = provider
    generation = compact(settings["generation"])
    reasoning = reasoning_field(model, generation.pop("reasoning", {}))
    body.update(generation)
    body.setdefault("seed", settings["seed"])
    if reasoning:
        body["reasoning"] = reasoning
    return body


def lookup(data, *keys):
    """data[key1][key2]..., or None if a key is missing or a level is not a mapping."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def send(body, headers, model):
    """Send one request, retrying rate limits, server errors and failed requests.

    Returns the answer text and the usage fields of the response.
    """
    error = None
    for attempt in range(1, ATTEMPTS + 1):
        backoff = BACKOFF * 2 ** (attempt - 1)
        try:
            response = requests.post(f"{API_URL}/chat/completions", headers=headers, json=body,
                                     timeout=90)
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                text = (message.get("content") or "") if isinstance(message, dict) else ""
                usage = {
                    "provider": lookup(data, "provider"),
                    "prompt_tokens": lookup(data, "usage", "prompt_tokens"),
                    "completion_tokens": lookup(data, "usage", "completion_tokens"),
                    "reasoning_tokens": lookup(data, "usage", "completion_tokens_details",
                                               "reasoning_tokens"),
                    "cost": lookup(data, "usage", "cost"),
                    "finish_reason": lookup(choice, "finish_reason"),
                }
                return text, usage
            error = f"HTTP {response.status_code}: {response.text[:2000]}"
            log.error("%s, attempt %d of %d: %s", model, attempt, ATTEMPTS, error)
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else backoff)
                continue
            if 500 <= response.status_code < 600:
                time.sleep(backoff)
                continue
            break
        except Exception as exc:
            error = str(exc)
            log.error("%s, attempt %d of %d: %s", model, attempt, ATTEMPTS, error)
            time.sleep(backoff)
    raise RuntimeError(f"no answer after {attempt} attempts: {error}")


def answer(body, headers, model):
    """The answer to one request and its usage fields, or (None, {}) if every round fails."""
    for round_ in range(ROUNDS):
        try:
            return send(body, headers, model)
        except Exception as exc:
            log.warning("%s, round %d of %d: %s", model, round_ + 1, ROUNDS, exc)
            time.sleep(2 ** round_)
    return None, {}


def run_model(model, prompts, settings, api_key):
    """Answers and usage fields of one model for all prompts."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    answers, usage = [None] * len(prompts), [{}] * len(prompts)
    log.info("%s: %d cells, %d concurrent requests", model, len(prompts), settings["max_workers"])
    with ThreadPoolExecutor(max_workers=settings["max_workers"]) as pool:
        futures = {pool.submit(answer, request_body(model, prompt, settings), headers, model): i
                   for i, prompt in enumerate(prompts)}
        for done, future in enumerate(as_completed(futures), 1):
            answers[futures[future]], usage[futures[future]] = future.result()
            if done % 100 == 0 or done == len(prompts):
                log.info("%s: %d of %d cells done", model, done, len(prompts))
    return answers, usage


def write_outputs(adata, label_key, answers, usage, predictions_path, usage_path):
    n = len(answers)
    cell_ids = [str(x) for x in adata.obs_names[:n]]
    rows = [{"cell_id": cell_id, **fields} for cell_id, fields in zip(cell_ids, usage)]
    pd.DataFrame(rows, columns=USAGE_COLUMNS, dtype=object).to_csv(usage_path, index=False)
    # The prediction file marks a model as done, so it is written last.
    pd.DataFrame({
        "cell_id": cell_ids,
        "true_labels": list(adata.obs[label_key].astype("string").values[:n]),
        "predicted_labels": answers,
    }).to_csv(predictions_path, index=False)
    log.info("Wrote %s and %s", predictions_path, usage_path)


def write_cluster_outputs(adata, label_key, cell_clusters, clusters, answers, usage,
                          predictions_path, usage_path):
    """Write the answer of each cluster to its cells and the usage fields of each cluster."""
    answer_of = dict(zip(clusters, answers))
    sizes = cell_clusters.value_counts()
    rows = [{"cluster": cluster, "n_cells": int(sizes[cluster]), **fields}
            for cluster, fields in zip(clusters, usage)]
    pd.DataFrame(rows, columns=CLUSTER_USAGE_COLUMNS, dtype=object).to_csv(usage_path,
                                                                          index=False)
    pd.DataFrame({
        "cell_id": [str(x) for x in adata.obs_names],
        "true_labels": list(adata.obs[label_key].astype("string").values),
        "predicted_labels": [answer_of[cluster] for cluster in cell_clusters],
    }).to_csv(predictions_path, index=False)
    log.info("Wrote %s and %s", predictions_path, usage_path)


def check_models(models, api_key):
    """Exit if a model id is not offered by OpenRouter; go on if the list cannot be fetched."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=30,
                                headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code != 200:
            log.warning("Model ids not checked: HTTP %d", response.status_code)
            return
        offered = {m.get("id") for m in response.json().get("data", []) if isinstance(m, dict)}
    except Exception as exc:
        log.warning("Model ids not checked: %s", exc)
        return
    missing = [model for model in models if model not in offered]
    if missing:
        log.error("Not offered by OpenRouter: %s", ", ".join(missing))
        sys.exit(1)
    log.info("All %d model ids are offered by OpenRouter", len(models))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cell type annotation with LLMs on OpenRouter.")
    parser.add_argument("--input", required=True, help="test set (.h5ad)")
    parser.add_argument("--output", required=True,
                        help="out/predictions.csv; each model writes predictions_<model>.csv")
    parser.add_argument("--dataset_config", required=True, help="dataset config (annotation_key)")
    parser.add_argument("--prompt", required=True, help="prompt template")
    parser.add_argument("--config", required=True, help="openrouter.yml")
    parser.add_argument("--clusters", help="cell_id and cluster of every cell (cluster mode)")
    parser.add_argument("--markers", help="cluster and its marker genes (cluster mode)")
    parser.add_argument("--skip_model_check", action="store_true",
                        help="do not check the model ids against the OpenRouter model list")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api_key = os.environ.get("API_KEY")
    if not api_key:
        log.error("API_KEY is not set")
        return 3

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8")) or {}
    label_key = dataset.get("annotation_key", "cell_type")
    # The template is the file content without its final line break.
    template = Path(args.prompt).read_text(encoding="utf-8").rstrip("\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for model in settings["models"]:
        slug = model.replace("/", "--")
        predictions_path = output.with_name(f"{output.stem}_{slug}.csv")
        if predictions_path.exists():
            log.info("Skipping %s: %s exists", model, predictions_path)
        else:
            paths[model] = (predictions_path, output.with_name(f"usage_{slug}.csv"))
    if not paths:
        return 0
    if not args.skip_model_check:
        check_models(list(paths), api_key)

    adata = load_test_set(args.input, label_key)
    if args.clusters:
        cell_clusters, markers = read_clusters(args.clusters, args.markers,
                                               [str(x) for x in adata.obs_names])
        by_cluster = build_cluster_prompts(template, markers,
                                           adata.obs[label_key].astype(str).unique(),
                                           settings["seed"])
        clusters = sorted(set(cell_clusters))
        prompts = [by_cluster[cluster] for cluster in clusters]
        log.info("%d cells in %d clusters", adata.n_obs, len(clusters))
    else:
        prompts = build_prompts(adata, template, label_key, settings["seed"],
                                settings["top_k_genes"])
        if settings.get("max_samples"):
            prompts = prompts[:settings["max_samples"]]

    failed = 0
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        futures = {pool.submit(run_model, model, prompts, settings, api_key): model
                   for model in paths}
        for future in as_completed(futures):
            model = futures[future]
            try:
                answers, usage = future.result()
                if args.clusters:
                    write_cluster_outputs(adata, label_key, cell_clusters, clusters, answers,
                                          usage, *paths[model])
                else:
                    write_outputs(adata, label_key, answers, usage, *paths[model])
            except Exception:
                log.exception("%s failed", model)
                failed += 1
    if failed:
        log.error("%d model(s) failed", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
