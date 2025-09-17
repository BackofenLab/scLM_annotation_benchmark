#!/usr/bin/env python3
"""
DeepSeekR1 inference for single-cell data.

- Reads API key from env (API_KEY) or --api-key
- Builds prompts with cell2sentence
- Queries OpenRouter (DeepSeek) with retries
- Writes predictions.csv with predicted + true label columns
"""

from __future__ import annotations

import os
import sys
import time
import json
import yaml
import argparse
import logging
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
logger = logging.getLogger("deepseek_predict")


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


# ---------------- Inference ----------------
def query_deepseek(model: str, prompt: str, sysprompt: str, api_key: str,
                   max_retries: int = 3, backoff: float = 1.5) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": sysprompt or ""},
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=data, timeout=90)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            logger.warning(f"DeepSeek http {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"DeepSeek exception (attempt {attempt+1}): {e}")
        time.sleep(backoff * (attempt + 1))

    raise RuntimeError(f"query_deepseek failed after {max_retries} retries")


def run_deepseek_prediction(
    model: str,
    api_key: str,
    csdata: cs.CSData,
    task_name: str,
    input_prompt_template: str,
    answer_template: str,
    top_k_genes: int = 200,
    max_samples: int | None = None,
    max_workers: int = 3,
    delay_between_requests: float = 1.2,
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

    n = formatted.num_rows
    logger.info(f"Predicting {n} cells with model={model}, workers={max_workers}")

    def _work(idx, sample):
        # simple retry wrapper
        for attempt in range(3):
            try:
                res = query_deepseek(model=model, prompt=sample["model_input"], sysprompt="", api_key=api_key)
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


# ---------------- Results ----------------
def create_results_dataframe(adata, predictions: list[str], label_keys: list[str]) -> pd.DataFrame:
    n = len(predictions)
    idx = list(adata.obs_names[:n])

    data = {"predicted_labels": predictions}
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
def main():
    parser = argparse.ArgumentParser(description="DeepSeek model prediction for single-cell data")
    parser.add_argument("--input", "-i", required=True, help="Path to input .h5ad")
    parser.add_argument("--output", "-o", required=True, help="Path to output predictions.csv")
    parser.add_argument("--model", "-m", default="deepseek/deepseek-r1-0528:free")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", default="cell_type_prediction")
    parser.add_argument("--input_prompt_template", required=True)
    parser.add_argument("--answer_template", required=True)
    parser.add_argument("--top_k_genes", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--delay_between_requests", type=float, default=1.2)
    parser.add_argument("--label_keys", "-l", nargs="+", required=True)
    parser.add_argument("--api-key", dest="api_key", default=None, help="Override API key (else read env API_KEY)")
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)

    # API key
    api_key = args.api_key or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: API key not provided. Set env API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(3)

    # Load adata
    adata = load_data(args.input)

    # Choose primary label column for prompt building
    primary_label = args.label_keys[0]
    if primary_label not in adata.obs.columns:
        print(f"ERROR: Primary label '{primary_label}' not in adata.obs columns", file=sys.stderr)
        print(f"Available columns: {list(adata.obs.columns)}", file=sys.stderr)
        sys.exit(4)

    adata.obs["true_labels"] = adata.obs[primary_label].astype("category")

    # Build C2S dataset (only needs true_labels)
    arrow_ds, vocab = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=args.seed,
        sentence_delimiter=" ",
        label_col_names=["true_labels"],
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
    preds = run_deepseek_prediction(
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
    )

    # Build and save results
    df = create_results_dataframe(adata, preds, args.label_keys)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=True)
    logger.info(f"Wrote predictions: {args.output}")


if __name__ == "__main__":
    # Avoid accidental GPU grabs by default (harmless on CPU boxes)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    main()
