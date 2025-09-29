#!/usr/bin/env python3

from pathlib import Path
import glob
import argparse
import json
import sys
import re
from typing import Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

    by_ds_model = df_metrics.groupby(["dataset", "model"], dropna=False).agg(agg_funcs).reset_index()
    #by_ds       = df_metrics.groupby(["dataset"], dropna=False).agg(agg_funcs).reset_index()
    #by_model    = df_metrics.groupby(["model"], dropna=False).agg(agg_funcs).reset_index()

    by_ds_model_csv = in_dir / "aggregated_by_dataset_model.csv"

    by_ds_model.to_csv(by_ds_model_csv, index=False)

    # Create and export plot
    export_plot(by_ds_model, in_dir, group_by="model")

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

def export_plot(
    df_metrics: pd.DataFrame,
    out_dir: Path,
    group_by: str = "dataset",   # or "model"
    dpi: int = 300
) -> pd.DataFrame:
    """
    Aggregate metrics by (dataset, model), export CSV, and create ONE grouped bar plot
    showing ALL metrics. The x-axis groups are chosen via `group_by`:
      - "dataset" -> x = datasets; bars = all (model × metric) combos
      - "model"   -> x = models;   bars = all (dataset × metric) combos

    Returns the aggregated (dataset, model) dataframe.
    """
    # --- Checks & setup
    required_cols = {"dataset", "model"}
    if not required_cols.issubset(df_metrics.columns):
        missing = required_cols - set(df_metrics.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only useful numeric metric columns that exist
    metric_cols = [
        "balanced_accuracy", "rada_avg_similarity", "lin_avg_similarity", "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    ]
    present_metrics = [c for c in metric_cols if c in df_metrics.columns]
    if not present_metrics:
        raise ValueError("No metric columns found to aggregate.")

    # Aggregate by dataset+model (mean)
    by_ds_model = (
        df_metrics
        .groupby(["dataset", "model"], dropna=False)[present_metrics]
        .mean()
        .reset_index()
        .sort_values(["dataset", "model"])
    )

    # Write aggregated CSV
    #out_csv = out_dir / "aggregated_by_dataset_model.csv"
    #by_ds_model.to_csv(out_csv, index=False)

    # Helpful console pivots (if the metrics exist)
    if "balanced_accuracy" in present_metrics:
        print("Balanced accuracy\n")
        print(by_ds_model.pivot(index="dataset", columns="model", values="balanced_accuracy").round(4))

    if "macro_f1" in present_metrics:
        print("Macro F1\n")
        print(by_ds_model.pivot(index="dataset", columns="model", values="macro_f1").round(4))

    #print(f"Wrote:\n- {out_csv}")

    # --- Build ONE grouped bar plot with ALL metrics ---
    # Decide primary (x-axis) and secondary (bar series) keys
    if group_by not in {"dataset", "model"}:
        raise ValueError("group_by must be 'dataset' or 'model'.")

    x_key = group_by
    sec_key = "model" if group_by == "dataset" else "dataset"

    x_levels = sorted(by_ds_model[x_key].astype(str).unique().tolist())
    sec_levels = by_ds_model[sec_key].astype(str).unique().tolist()  # preserve original order

    # We will create one bar series per (secondary level × metric)
    series_labels = []
    series_values = []  # list of arrays in the x_levels order

    for sec in sec_levels:
        for metric in present_metrics:
            series_labels.append(f"{sec} • {metric}")
            vals = []
            for x in x_levels:
                row = by_ds_model[
                    (by_ds_model[x_key].astype(str) == x) &
                    (by_ds_model[sec_key].astype(str) == sec)
                ]
                if not row.empty and pd.notnull(row[metric].values[0]):
                    vals.append(float(row[metric].values[0]))
                else:
                    vals.append(0.0)
            series_values.append(np.array(vals, dtype=float))

    num_series = len(series_labels)
    x = np.arange(len(x_levels), dtype=float)
    bar_width = 0.8 / max(1, num_series)

    # Dynamic figure width to keep it somewhat readable for many series
    base_width = 6.0
    width_per_group = 0.6
    fig_w = base_width + width_per_group * max(4, len(x_levels))
    fig_h = 7.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    # Draw every series as a shifted bar
    for i, (label, vals) in enumerate(zip(series_labels, series_values)):
        positions = x + (i - (num_series - 1) / 2.0) * bar_width
        bars = ax.bar(positions, vals, width=bar_width, label=label)

    # Axes & styling (no explicit colors specified)
    ax.set_xticks(x)
    ax.set_xticklabels(x_levels, rotation=45, ha="right")
    ax.set_xlabel(x_key.title())
    ax.set_ylabel("Metric value")
    ax.set_title(f"All metrics grouped by {x_key.title()} (bars = {sec_key.title()} × metric)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    # Adjust columns heuristically to keep it compact
    ncol = 3 if num_series > 18 else 2 if num_series > 10 else 1
    ax.legend(title="Series (Secondary × Metric)", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=True, ncol=ncol)

    fig.tight_layout()
    out_png = out_dir / f"grouped_bars_{x_key}_ALL_METRICS.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote:\n- {out_png}")

    return by_ds_model


if __name__ == "__main__":
    print("Inside aggregate")
    parser = argparse.ArgumentParser(description="Aggregate predictions")
    parser.add_argument("--input", required=True, type=str, help="Path to input JSONs")
    args = parser.parse_args()
    combine_results(args)