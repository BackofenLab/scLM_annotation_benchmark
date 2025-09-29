#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from sklearn.metrics import balanced_accuracy_score, classification_report
import argparse
import json
import sys
import re
from typing import Optional, Dict, Any, Tuple, List


def combine_results(args):

    in_dir = Path(args.input)

    rows = []
    files = glob.glob(f"{in_dir}/*.json")
    print(files)
    if not files:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(2)

    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {fp}: cannot read/parse JSON ({e})", file=sys.stderr)
            continue

        dataset, model = parse_name(Path(fp))
        m = extract_metrics(data)
        row = {"file": str(fp), "dataset": dataset, "model": model, **m}
        rows.append(row)

    if not rows:
        print("No usable JSONs found.", file=sys.stderr)
        sys.exit(3)

    # combined metrics
    df_metrics = pd.DataFrame(rows)

    # Aggregations:
    # 1) by dataset + model
    agg_cols = [
        "balanced_accuracy","rada_avg_similarity","lin_avg_similarity","accuracy",
        "macro_precision","macro_recall","macro_f1",
        "weighted_precision","weighted_recall","weighted_f1"
    ]
    agg_funcs = {c: "mean" for c in agg_cols}

    by_ds_model = df.groupby(["dataset", "model"], dropna=False).agg(agg_funcs).reset_index()
    #by_ds       = df.groupby(["dataset"], dropna=False).agg(agg_funcs).reset_index()
    #by_model    = df.groupby(["model"], dropna=False).agg(agg_funcs).reset_index()

    by_ds_model_csv = in_dir / "aggregated_by_dataset_model.csv"

    by_ds_model.to_csv(by_ds_model_csv, index=False)

    pivot_balacc = by_ds_model.pivot(index="dataset", columns="model", values="balanced_accuracy")
    pivot_macro_f1 = by_ds_model.pivot(index="dataset", columns="model", values="macro_f1")

    print("Balanced accuracy \n")
    print(pivot_balacc)
    
    print("Macro F1 \n")
    print(pivot_macro_f1)

    print(f"Wrote:\n- {by_ds_model_csv}")


def parse_name(p: Path) -> Tuple[str, str]:
    REGEXES = [
        # dataset__model.json
        re.compile(r'(?P<dataset>.+?)__+(?P<model>.+?)\.json$', re.IGNORECASE),
        # dataset_model.json
        re.compile(r'(?P<dataset>.+?)_(?P<model>[^_/]+?)\.json$', re.IGNORECASE),
        # dataset.model.json
        re.compile(r'(?P<dataset>.+?)\.(?P<model>[^.]+?)\.json$', re.IGNORECASE),
    ]
    fname = p.name
    for rx in REGEXES:
        m = rx.search(fname)
        if m:
            return m.group("dataset"), m.group("model")
    # heuristic fallback: first two underscore chunks
    parts = p.stem.split("_")
    dataset = parts[0] if parts else "unknown_dataset"
    model = parts[1] if len(parts) > 1 else "unknown_model"
    return dataset, model


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = ["predictions_cell_ontology_class", "true_labels"]
    ba   = safe_get(payload, base + ["balanced_accuracy"])
    rada = safe_get(payload, base + ["rada_avg_similarity"])
    lin  = safe_get(payload, base + ["lin_avg_similarity"])

    # summary_dict aggregates
    acc  = safe_get(payload, base + ["summary_dict", "accuracy"])
    macro = safe_get(payload, base + ["summary_dict", "macro avg"], {}) or {}
    wavg  = safe_get(payload, base + ["summary_dict", "weighted avg"], {}) or {}

    return {
        "balanced_accuracy": ba,
        "rada_avg_similarity": rada,
        "lin_avg_similarity": lin,
        "accuracy": acc,
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "macro_f1": macro.get("f1-score"),
        "macro_support": macro.get("support"),
        "weighted_precision": wavg.get("precision"),
        "weighted_recall": wavg.get("recall"),
        "weighted_f1": wavg.get("f1-score"),
        "weighted_support": wavg.get("support"),
    }


if __name__ == "__main__":
    print("Inside aggregate")
    parser = argparse.ArgumentParser(description="Aggregate predictions")
    parser.add_argument("--input", required=True, type=str, help="Path to input JSONs")
    args = parser.parse_args()
    combine_results(args)