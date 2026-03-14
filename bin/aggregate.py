#!/usr/bin/env python

from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def combine_results(args):
    out_dir = Path(args.output)
    rows = []
    input_files = args.input
    if not input_files:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(2)

    # Extract dataset names from YAML filenames (use filename stem as dataset name)
    known_dataset_names = []
    if args.datasets:
        # Handle both string (space-separated) and list inputs
        dataset_files = args.datasets if isinstance(args.datasets, list) else args.datasets.split()
        for dataset_config in dataset_files:
            dataset_path = Path(dataset_config)
            dataset_name = dataset_path.stem
            known_dataset_names.append(dataset_name)
            print(f"  Added dataset name '{dataset_name}' from file '{dataset_config}'", file=sys.stderr)
        print(f"Known dataset names from YAML files: {sorted(known_dataset_names, key=len, reverse=True)}", file=sys.stderr)

    for fp in input_files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {fp}: cannot read/parse JSON ({e})", file=sys.stderr)
            continue

        dataset, model = parse_name(Path(fp), known_dataset_names)
        # Validate dataset against known configs; if no match, try alternative splits
        # (handles model names with underscores like "openrouter_LLM")
        if not any(dataset in dc for dc in args.datasets):
            parts = Path(fp).stem.split("_")
            for i in range(len(parts) - 1, 0, -1):
                candidate_ds = "_".join(parts[:i])
                candidate_model = "_".join(parts[i:])
                if any(candidate_ds in dc for dc in args.datasets):
                    dataset, model = candidate_ds, candidate_model
                    break

        # Extract metrics from JSON keys produced by evaluate.py.
        # Keys are filename stems:
        #   Single-model: "predictions" (from predictions.csv)
        #   Multi-model:  "predictions_<slug>" (from predictions_<slug>.csv)
        matched = False
        for key in data:
            if not isinstance(data[key], dict) or "true_labels" not in data[key]:
                continue
            if key == "predictions":
                # Single-model case: use the Nextflow model name
                m = extract_metrics(data, key)
                row = {"file": str(fp), "dataset": dataset, "model": model, **m}
                rows.append(row)
                matched = True
            elif key.startswith("predictions_"):
                # Multi-model case: extract LLM model name from slug
                model_slug = key[len("predictions_"):]
                m = extract_metrics(data, key)
                row = {"file": str(fp), "dataset": dataset,
                       "model": model_slug.replace("--", "/"), **m}
                rows.append(row)
                matched = True
        if not matched:
            print(f"WARNING: No keys with 'true_labels' found in {fp}", file=sys.stderr)

    if not rows:
        print("No usable JSONs found.", file=sys.stderr)
        sys.exit(3)

    # combined metrics
    df_metrics = pd.DataFrame(rows)

    # Aggregations:
    # 1) by dataset + model
    agg_cols = [
        "balanced_accuracy","accuracy","rada_avg_similarity","lin_avg_similarity","id_accuracy","balanced_id_accuracy",
        "macro_precision","macro_recall","macro_f1",
        "weighted_precision","weighted_recall","weighted_f1"
    ]
    agg_funcs = {c: "mean" for c in agg_cols}

    by_ds_model = df_metrics.groupby(["dataset", "model"], dropna=False).agg(agg_funcs).reset_index()
    by_ds_model_csv = out_dir / "aggregated_by_dataset_model.csv"
    by_ds_model.to_csv(by_ds_model_csv, index=False)

    # Create and export barplots
    make_plot(by_ds_model, out_dir)

    # Create heatmaps (models × datasets)
    make_heatmaps(by_ds_model, out_dir)

    pivot_balacc = by_ds_model.pivot(index="dataset", columns="model", values="balanced_accuracy")
    pivot_macro_f1 = by_ds_model.pivot(index="dataset", columns="model", values="macro_f1")

    print("Balanced accuracy \n")
    print(pivot_balacc)

    print("Macro F1 \n")
    print(pivot_macro_f1)

    print(f"Export aggregated file to:\n- {by_ds_model_csv}")


def parse_name(p: Path, known_dataset_names: List[str] = None) -> Tuple[str, str]:
    """
    Parse dataset and model from filename.
    Expected format: dataset_model.json
    
    If known_dataset_names is provided, matches against those first for more reliable
    parsing (especially for dataset names with underscores like 'HIHA_L1').
    """
    fname = p.stem  # Remove .json extension
    
    # If we have known dataset names, try to match them first
    # This helps with datasets that have underscores in their names
    if known_dataset_names:
        # Sort by length (longest first) to match most specific names first
        # This prevents "HIHA" from matching when "HIHA_L1" should match
        sorted_datasets = sorted(known_dataset_names, key=len, reverse=True)
        
        for dataset_name in sorted_datasets:
            prefix = dataset_name + "_"
            if fname.startswith(prefix):
                model = fname[len(prefix):]
                print(f"  Matched dataset '{dataset_name}' for filename '{fname}' -> model='{model}'", file=sys.stderr)
                return dataset_name, model
        
        print(f"  No match found in known datasets for '{fname}'. Known datasets: {sorted_datasets}", file=sys.stderr)
    
    # Fallback: split on first underscore
    parts = fname.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    
    # If no underscore found, treat entire name as dataset with unknown model
    print(f"WARNING: Could not parse dataset and model from '{p.name}'. Expected format: dataset_model.json", file=sys.stderr)
    return fname, "unknown_model"


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_metrics(payload: Dict[str, Any], annotation_key) -> Dict[str, Any]:
    base = [annotation_key, "true_labels"]
    ba   = safe_get(payload, base + ["balanced_accuracy"])
    # Try top-level accuracy first (new format), fallback to summary_dict (old format)
    acc  = safe_get(payload, base + ["accuracy"]) or safe_get(payload, base + ["summary_dict", "accuracy"])
    rada = safe_get(payload, base + ["rada_avg_similarity"])
    lin  = safe_get(payload, base + ["lin_avg_similarity"])
    id_acc = safe_get(payload, base + ["id_accuracy"])
    bal_id_acc = safe_get(payload, base + ["balanced_id_accuracy"])

    # summary_dict aggregates
    macro = safe_get(payload, base + ["summary_dict", "macro avg"], {}) or {}
    wavg  = safe_get(payload, base + ["summary_dict", "weighted avg"], {}) or {}

    return {
        "balanced_accuracy": ba,
        "accuracy": acc,
        "rada_avg_similarity": rada,
        "lin_avg_similarity": lin,
        "id_accuracy": id_acc,
        "balanced_id_accuracy": bal_id_acc,
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "macro_f1": macro.get("f1-score"),
        "macro_support": macro.get("support"),
        "weighted_precision": wavg.get("precision"),
        "weighted_recall": wavg.get("recall"),
        "weighted_f1": wavg.get("f1-score"),
        "weighted_support": wavg.get("support"),
    }

def make_plot(dataframe, out_dir):
    """
    Create a grouped bar plot for a single metric at a time.
    - X-axis: models
    - Bars: datasets
    """
    # Metrics to plot
    metrics = [
        "balanced_accuracy",
        "accuracy",
        "rada_avg_similarity",
        "lin_avg_similarity",
        "id_accuracy",
        "balanced_id_accuracy",
        #"macro_precision",
        #"macro_recall",
        "macro_f1",
        #"weighted_precision",
        #"weighted_recall",
        "weighted_f1",
    ]

    # Set style
    sns.set(style="whitegrid")

    # Generate one plot per metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=dataframe,
            x="model",
            y=metric,
            hue="dataset",
            palette="tab10"
        )
        plt.title(f"Cell annotation performance of scFoundation models across datasets")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(out_dir / f"aggregated_plot_{metric}_barplot.png", dpi=300)
        plt.close()

    print(f"Plots saved to: {out_dir.resolve()}")


def make_heatmaps(dataframe, out_dir):
    """
    Create heatmaps with models on the x-axis and datasets on the y-axis.
    Values are annotated inside each cell.
    """
    metrics = [
        "balanced_accuracy",
        "accuracy",
        "rada_avg_similarity",
        "lin_avg_similarity",
        "id_accuracy",
        "balanced_id_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    sns.set(style="whitegrid")

    for metric in metrics:
        pivot = dataframe.pivot(index="dataset", columns="model", values=metric)
        if pivot.empty:
            continue

        # Sort columns by mean performance descending
        pivot = pivot[pivot.mean().sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.8)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"{metric.replace('_', ' ').title()} — Models × Datasets")
        ax.set_xlabel("Model")
        ax.set_ylabel("Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"aggregated_plot_{metric}_heatmap.png", dpi=300)
        plt.close()

    print(f"Heatmaps saved to: {out_dir.resolve()}")


def export_plot(
    df_metrics: pd.DataFrame,
    out_dir: Path,
    group_by: str = "dataset",  # or "model"
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
        "balanced_accuracy", "accuracy", "rada_avg_similarity", "lin_avg_similarity", 
        "id_accuracy", "balanced_id_accuracy",
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

    # Helpful console pivots (if the metrics exist)
    if "balanced_accuracy" in present_metrics:
        print("Balanced accuracy\n")
        print(by_ds_model.pivot(index="dataset", columns="model", values="balanced_accuracy").round(4))

    if "macro_f1" in present_metrics:
        print("Macro F1\n")
        print(by_ds_model.pivot(index="dataset", columns="model", values="macro_f1").round(4))

    # --- Build ONE grouped bar plot with ALL metrics ---
    if group_by not in {"dataset", "model"}:
        raise ValueError("group_by must be 'dataset' or 'model'.")

    x_key = group_by
    sec_key = "model" if group_by == "dataset" else "dataset"

    x_levels = sorted(by_ds_model[x_key].astype(str).unique().tolist())
    sec_levels = by_ds_model[sec_key].astype(str).unique().tolist()  # preserve original order

    # Create one bar series per (secondary level × metric)
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
    out_png = out_dir / f"aggregated_plot_{x_key}_ALL_METRICS.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote:\n- {out_png}")

    return by_ds_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate predictions")
    parser.add_argument("--input", nargs='+', required=True, type=str, help="Path to input JSONs")
    parser.add_argument("--output", required=True, type=str, help="Path to output CSV")
    parser.add_argument("--level", default=0, type=int, help="Annotation level to be compared (deprecated, kept for compatibility)")
    parser.add_argument("--datasets", nargs='+', required=False,
                   help="YAML files of the datasets (used to extract dataset names from filenames)")


    args = parser.parse_args()
    combine_results(args)
