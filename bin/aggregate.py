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

# The text-encoder result keys are defined with the metric. The import is safe without torch:
# text_embedding_similarity imports torch lazily, inside TextEncoder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.text_embedding_similarity import TEXT_EMB_KEYS  # noqa: E402

# Values read from the top level of a result's "true_labels" block.
SCALAR_KEYS = (
    "balanced_accuracy", "accuracy", "agreement_score", "lin_avg_similarity",
    "rada_avg_similarity", "share_broader", "share_finer",
    # Scoring coverage (direct CL term or reviewed=True), label-space diagnostics, subset size.
    "reviewed_fraction", "non_cl_ratio", "n_non_cl_labels", "n_unique_pred_raw",
    "n_unique_pred_mapped", "subset_size",
    *TEXT_EMB_KEYS,  # present only when the run used --use_text_encoder
)
# Averages taken from the sklearn classification report ("macro avg", "weighted avg").
REPORT_KEYS = tuple(f"{avg}_{stat}" for avg in ("macro", "weighted")
                    for stat in ("precision", "recall", "f1"))
# Metrics drawn as one bar plot and one heatmap each.
PLOT_METRICS = [
    "balanced_accuracy", "accuracy", "agreement_score", "lin_avg_similarity",
    "rada_avg_similarity", "text_emb_norm_balanced_raw", "text_emb_norm_balanced_mapped",
    "text_emb_norm_mapping_lift", "macro_f1", "weighted_f1",
    # Label-coverage / diversity diagnostics (lower non_cl_ratio = more native CL output).
    "non_cl_ratio", "n_non_cl_labels", "n_unique_pred_raw", "n_unique_pred_mapped",
]


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

        # The eval stage appends "__sub<size>" to the JSON filename so the same model scored at
        # multiple subset sizes does not overwrite its results. Strip it AFTER the dataset-validation
        # fallback (which re-derives model from the raw filename stem) so the model name is clean in
        # every path; the size itself is read from the JSON payload (subset_size field) below.
        if "__sub" in model:
            model = model.split("__sub", 1)[0]

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
                # Multi-model case: extract LLM model name and blind/unblind suffix
                model_slug = key[len("predictions_"):]
                llm_name = model_slug.replace("--", "/").split("/")[-1]  # e.g. "deepseek-v3.2"
                suffix = model.split("_")[-1] if "_" in model else ""    # e.g. "blind" or "unblind"
                display_name = f"{llm_name}_{suffix}" if suffix else llm_name
                m = extract_metrics(data, key)
                row = {"file": str(fp), "dataset": dataset,
                       "model": display_name, **m}
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
    agg_cols = [k for k in SCALAR_KEYS + REPORT_KEYS if k != "subset_size"]
    df_metrics[agg_cols] = df_metrics[agg_cols].apply(pd.to_numeric, errors="coerce")
    agg_funcs = {c: "mean" for c in agg_cols}
    # subset_size is part of the row identity, exactly like the blind/unblind suffix in the model
    # name: the same model scored at two subset sizes produces two rows, not an average. Group by it
    # whenever it is present (older JSONs without the field fall into a single NaN-size group).
    group_keys = ["dataset", "model"]
    if "subset_size" in df_metrics.columns:
        group_keys.append("subset_size")

    by_ds_model = df_metrics.groupby(group_keys, dropna=False).agg(agg_funcs).reset_index()

    # Annotate each row with the prediction mode, derived from the model name suffix.
    def _derive_mode(model):
        m = str(model).lower()
        if m.endswith("_unblind"):
            return "unblind"
        if m.endswith("_blind"):
            return "blind"
        return "baseline"   # cell_baseline, random_stratified, random_uniform
    by_ds_model.insert(2, "mode", by_ds_model["model"].map(_derive_mode))

    by_ds_model_csv = out_dir / "aggregated_by_dataset_model.csv"
    by_ds_model.to_csv(by_ds_model_csv, index=False)

    # Create and export barplots
    make_plot(by_ds_model, out_dir)

    # Create heatmaps (models × datasets)
    make_heatmaps(by_ds_model, out_dir)

    # Console previews: put subset_size in the index so each size is a separate row-block and the
    # numbers are never averaged across sizes (within one size, (dataset, model) is unique). With no
    # subset_size column this is just by dataset, as before. The CSV keeps the per-size rows, and
    # bin/plot_model_comparison.py renders the figures per size.
    _idx = ["subset_size", "dataset"] if "subset_size" in by_ds_model.columns else ["dataset"]
    pivot_balacc = by_ds_model.pivot_table(index=_idx, columns="model", values="balanced_accuracy", aggfunc="mean")
    pivot_macro_f1 = by_ds_model.pivot_table(index=_idx, columns="model", values="macro_f1", aggfunc="mean")

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
    metrics = {k: safe_get(payload, base + [k]) for k in SCALAR_KEYS}
    report = safe_get(payload, base + ["summary_dict"], {}) or {}
    for key in REPORT_KEYS:
        avg, stat = key.split("_")
        metrics[key] = (report.get(f"{avg} avg") or {}).get("f1-score" if stat == "f1" else stat)
    return metrics

def make_plot(dataframe, out_dir):
    """
    Create a grouped bar plot for a single metric at a time.
    - X-axis: models
    - Bars: datasets

    When the data carries a ``subset_size`` column with more than one distinct size, a separate set
    of barplots is produced *per subset size* (filenames suffixed ``_sub<size>``) so the two sizes
    are never merged into one bar. Splitting also makes each (model, dataset) cell unique, so
    seaborn draws no spurious CI/error bars. With a single size (or no ``subset_size`` column) one
    unsuffixed set is produced, as before.
    """
    # Set style
    sns.set(style="whitegrid")

    def _render(df_sub, file_suffix, title_suffix):
        """Render one barplot per metric for the given (already size-filtered) subframe."""
        for metric in PLOT_METRICS:
            # Skip metrics that were not computed in this run (e.g. text_emb_* when the text
            # encoder was disabled) to avoid emitting empty plots.
            if metric not in df_sub.columns or df_sub[metric].dropna().empty:
                continue
            # Sort models by mean of this metric across datasets, descending (NaN-skipping)
            model_order = (
                df_sub.groupby("model")[metric]
                .mean()
                .sort_values(ascending=False, na_position="last")
                .index.tolist()
            )
            plt.figure(figsize=(18, 10))
            sns.barplot(
                data=df_sub,
                x="model",
                y=metric,
                hue="dataset",
                palette="tab10",
                order=model_order,
            )
            plt.title(f"Cell annotation performance of scFoundation models across datasets{title_suffix}")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel(metric)
            plt.xlabel("Model")
            plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(out_dir / f"aggregated_plot_{metric}{file_suffix}_barplot.png", dpi=300)
            plt.close()

    # Split per subset size when the column is present with >1 distinct size; otherwise one set.
    if "subset_size" in dataframe.columns and not dataframe["subset_size"].dropna().empty:
        sizes = sorted(dataframe["subset_size"].dropna().unique())
    else:
        sizes = []

    if len(sizes) > 1:
        for s in sizes:
            s_int = int(s) if float(s).is_integer() else s
            _render(dataframe[dataframe["subset_size"] == s], f"_sub{s_int}", f"  (subset {s_int})")
    else:
        _render(dataframe, "", "")

    print(f"Plots saved to: {out_dir.resolve()}")


def make_heatmaps(dataframe, out_dir):
    """
    Create heatmaps with models on the x-axis and datasets on the y-axis. Values are annotated
    inside each cell.

    When the data carries a ``subset_size`` column with more than one distinct size, a separate
    set of heatmaps is produced *per subset size* (filenames suffixed ``_sub<size>``) so the two
    sizes are never averaged together. With a single size (or no ``subset_size`` column) one
    unsuffixed set is produced, exactly as before.
    """
    # Metrics whose natural range is [0, 1]; everything else (the label-count diagnostics) is
    # unbounded and must autoscale instead of clipping against a hardcoded vmax=1. The mapping
    # lifts are signed differences and belong OUTSIDE this set: a negative lift (mapping made a
    # model's label WORSE, e.g. llama-4-maverick) would otherwise be clipped away at vmin=0.
    zero_one_metrics = {
        "balanced_accuracy", "accuracy", "agreement_score", "rada_avg_similarity",
        "lin_avg_similarity", "text_emb_norm_balanced_raw", "text_emb_norm_balanced_mapped",
        "macro_f1", "weighted_f1", "non_cl_ratio", "reviewed_fraction",
    }

    sns.set(style="whitegrid")

    def _render(df_sub, file_suffix, title_suffix):
        """Render one heatmap per metric for the given (already size-filtered) subframe."""
        for metric in PLOT_METRICS:
            # Skip metrics not computed in this run (e.g. text_emb_* when disabled).
            if metric not in df_sub.columns or df_sub[metric].dropna().empty:
                continue
            # Within a single subset size (dataset, model) is unique; pivot_table stays robust if
            # any duplicates remain (e.g. when sizes are not split).
            pivot = df_sub.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
            if pivot.empty:
                continue

            # Sort columns by mean performance descending
            pivot = pivot[pivot.mean().sort_values(ascending=False).index]

            is_ratio = metric in zero_one_metrics
            fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.8)))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f" if is_ratio else ".0f",
                cmap="YlOrRd",
                vmin=0 if is_ratio else None,
                vmax=1 if is_ratio else None,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title(f"{metric.replace('_', ' ').title()} — Models × Datasets{title_suffix}")
            ax.set_xlabel("Model")
            ax.set_ylabel("Dataset")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(out_dir / f"aggregated_plot_{metric}{file_suffix}_heatmap.png", dpi=300)
            plt.close()

    # Split per subset size when the column is present with >1 distinct size; otherwise one set.
    if "subset_size" in dataframe.columns and not dataframe["subset_size"].dropna().empty:
        sizes = sorted(dataframe["subset_size"].dropna().unique())
    else:
        sizes = []

    if len(sizes) > 1:
        for s in sizes:
            s_int = int(s) if float(s).is_integer() else s
            _render(dataframe[dataframe["subset_size"] == s], f"_sub{s_int}", f"  (subset {s_int})")
    else:
        _render(dataframe, "", "")

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
        "balanced_accuracy", "accuracy", "agreement_score", "rada_avg_similarity",
        "lin_avg_similarity", "macro_precision", "macro_recall", "macro_f1",
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
        print(by_ds_model.pivot_table(index="dataset", columns="model", values="balanced_accuracy", aggfunc="mean").round(4))

    if "macro_f1" in present_metrics:
        print("Macro F1\n")
        print(by_ds_model.pivot_table(index="dataset", columns="model", values="macro_f1", aggfunc="mean").round(4))

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
