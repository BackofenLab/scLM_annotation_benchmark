#!/usr/bin/env python3
"""Standalone model/dataset comparison plots.

Reads the aggregated results CSV (one row per model x dataset, columns are
metrics) and produces several comparison views per metric, each written to its
own sub-folder under the output directory:

    results/plots/ranking/   mean-rank bar chart (rank models per dataset, sort
                             by average rank across datasets; the headline view)
    results/plots/heatmap/   model x dataset value heatmap (sorted by mean)
    results/plots/bump/      rank-trajectory ("bump") chart across datasets
    results/plots/barplot/   grouped barplot (x=model sorted by mean, hue=dataset)

This script is independent of the Nextflow pipeline. Run it in the lightweight
conda environment defined in ``envs/plotting.yml``:

    conda env create -f envs/plotting.yml
    conda activate sc_lfm_plotting

    # all plots, default metric set
    python bin/plot_model_comparison.py

    # only some models (everything else is dropped from this run)
    python bin/plot_model_comparison.py --models cellwhisperer_unblind gpt-5.5_unblind

    # every numeric metric, only the ranking + heatmap views
    python bin/plot_model_comparison.py --metrics all --plots ranking heatmap

If the input CSV gains a ``subset_size`` column (number of cells used per model;
expected to be constant per model), each model's axis label is annotated with a
marker (``*``, ``**`` ...) and a caption maps the markers to cell counts. The
script works unchanged when the column is absent.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed; write files only
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import seaborn as sns  # noqa: E402

# --------------------------------------------------------------------------- #
# Reuse the repo's publication style and colour definitions when importable.
# The script stays runnable (with sensible fallbacks) even outside the repo.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bin" / "util"))
try:
    from pub_style import set_pub_style  # type: ignore
    from color_definition import COLORMAPS, CATEGORICAL_PALETTES  # type: ignore

    _HAVE_REPO_STYLE = True
except Exception:  # pragma: no cover - styling is non-essential
    _HAVE_REPO_STYLE = False
    COLORMAPS, CATEGORICAL_PALETTES = {}, {}

    def set_pub_style() -> None:
        sns.set_context("paper", font_scale=1)
        sns.set_style("white")
        plt.rcParams.update({
            "figure.figsize": (6, 4),
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })


ID_COLS = ("dataset", "model")
DEFAULT_METRICS = [
    "balanced_accuracy",
    "accuracy",
    "rada_avg_similarity",
    "lin_avg_similarity",
    "id_accuracy",
    "balanced_id_accuracy",
    "macro_f1",
    "weighted_f1",
]
PLOT_TYPES = ["ranking", "heatmap", "bump", "barplot"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def fmt_size(value: float) -> str:
    """Render a subset size as a clean integer when it has no fractional part."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


def heatmap_cmap():
    return COLORMAPS.get("CM: Teal–White", "YlOrRd") if _HAVE_REPO_STYLE else "YlOrRd"


def build_dataset_colors(datasets) -> dict:
    """Stable colour per dataset, shared across every plot type."""
    palette = CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None
    if not palette:
        palette = sns.color_palette("tab10", n_colors=max(len(datasets), 3)).as_hex()
    return {ds: palette[i % len(palette)] for i, ds in enumerate(datasets)}


def build_subset_labels(df: pd.DataFrame, subset_col: str):
    """Map each model to a display label, optionally annotated with a marker.

    Returns ``(label_map, caption)``. When ``subset_col`` is absent the labels are
    the model names unchanged and the caption is empty. Subset size is expected to
    be constant per model; if a model has several sizes the dominant one is used
    and a warning is emitted.
    """
    models = list(df["model"].unique())
    if subset_col not in df.columns:
        return {m: m for m in models}, ""

    per_model: dict[str, float | None] = {}
    for model, grp in df.groupby("model"):
        sizes = grp[subset_col].dropna().unique()
        if len(sizes) == 0:
            per_model[model] = None
        elif len(sizes) == 1:
            per_model[model] = sizes[0]
        else:
            dominant = grp[subset_col].mode().iloc[0]
            warnings.warn(
                f"model '{model}' has multiple subset sizes {sorted(sizes)}; "
                f"using dominant value {dominant} for its axis marker."
            )
            per_model[model] = dominant

    distinct = sorted({s for s in per_model.values() if s is not None})
    marker_for = {size: "*" * (i + 1) for i, size in enumerate(distinct)}

    label_map = {}
    for model in models:
        size = per_model.get(model)
        marker = marker_for.get(size, "") if size is not None else ""
        label_map[model] = f"{model} {marker}".rstrip()

    caption = ""
    if distinct:
        caption = "Subset size:   " + "    ".join(
            f"{marker_for[s]} = {fmt_size(s)} cells" for s in distinct
        )
    return label_map, caption


def add_caption(fig, caption: str) -> None:
    if caption:
        fig.text(0.01, 0.005, caption, ha="left", va="bottom", fontsize=7, style="italic")


def make_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """models x datasets matrix of metric values (mean over any duplicate rows)."""
    return df.pivot_table(index="model", columns="dataset", values=metric, aggfunc="mean")


# --------------------------------------------------------------------------- #
# Plot builders (one figure per metric)
# --------------------------------------------------------------------------- #
def plot_ranking(df, metric, out_path, dataset_colors, label_map, caption, dpi):
    pivot = make_pivot(df, metric)
    if pivot.dropna(how="all").empty:
        return False
    # Rank within each dataset: highest metric value -> rank 1 (best).
    ranks = pivot.rank(axis=0, ascending=False, method="average")
    mean_rank = ranks.mean(axis=1, skipna=True)
    order = mean_rank.sort_values(ascending=True).index  # best first

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.34 * len(order))))
    ax.barh(y, mean_rank.loc[order].values, color="0.85", edgecolor="0.55",
            height=0.6, zorder=1)
    for ds in pivot.columns:
        ax.scatter(ranks.loc[order, ds].values, y, s=28, zorder=3,
                   color=dataset_colors[ds], edgecolor="white", linewidth=0.4,
                   label=ds)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(m, m) for m in order])
    ax.invert_yaxis()  # best model on top
    ax.set_xlabel("Rank across datasets  (1 = best;  bar = mean, dots = per dataset)")
    ax.set_xlim(left=0.5)
    ax.set_title(f"{pretty(metric)} — model ranking across datasets")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    add_caption(fig, caption)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_heatmap(df, metric, out_path, label_map, caption, dpi):
    pivot = make_pivot(df, metric)
    if pivot.dropna(how="all").empty:
        return False
    order = pivot.mean(axis=1, skipna=True).sort_values(ascending=False).index
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(
        figsize=(max(5.0, 1.15 * len(pivot.columns) + 3.0),
                 max(4.0, 0.32 * len(pivot))))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap=heatmap_cmap(), vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        yticklabels=[label_map.get(m, m) for m in pivot.index],
        cbar_kws={"label": pretty(metric)}, ax=ax,
    )
    ax.set_title(f"{pretty(metric)} — models × datasets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    add_caption(fig, caption)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_bump(df, metric, out_path, label_map, caption, dpi):
    pivot = make_pivot(df, metric)
    if pivot.dropna(how="all").empty:
        return False
    ranks = pivot.rank(axis=0, ascending=False, method="average")
    mean_rank = ranks.mean(axis=1, skipna=True)
    order = list(mean_rank.sort_values(ascending=True).index)  # best first
    datasets = list(ranks.columns)
    x = np.arange(len(datasets))
    colors = cm.turbo(np.linspace(0.05, 0.95, len(order)))

    fig, ax = plt.subplots(
        figsize=(max(6.0, 1.6 * len(datasets) + 3.0), max(5.0, 0.28 * len(order))))
    for color, model in zip(colors, order):
        yv = ranks.loc[model].values
        ax.plot(x, yv, marker="o", markersize=4, linewidth=1.2, color=color,
                alpha=0.85)
        last = ranks.loc[model].last_valid_index()
        if last is not None:
            xi = datasets.index(last)
            ax.text(xi + 0.08, ranks.loc[model, last], label_map.get(model, model),
                    fontsize=6, va="center", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlim(-0.3, len(datasets) - 1 + 1.6)
    ax.invert_yaxis()  # rank 1 at the top
    ax.set_ylabel("Rank  (1 = best)")
    ax.set_xlabel("Dataset")
    ax.set_title(f"{pretty(metric)} — rank trajectories across datasets")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    add_caption(fig, caption)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_barplot(df, metric, out_path, dataset_colors, label_map, caption, dpi):
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return False
    order = (sub.groupby("model")[metric].mean()
             .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(max(10.0, 0.5 * len(order)), 6.0))
    sns.barplot(data=sub, x="model", y=metric, hue="dataset", order=order,
                palette=dataset_colors, errorbar=None, ax=ax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label_map.get(m, m) for m in order], rotation=45, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel(pretty(metric))
    ax.set_title(f"{pretty(metric)} across datasets")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    add_caption(fig, caption)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Data loading / argument handling
# --------------------------------------------------------------------------- #
def resolve_metrics(df, requested, subset_col):
    numeric = [c for c in df.columns
               if c not in ID_COLS and c != subset_col
               and pd.api.types.is_numeric_dtype(df[c])]
    if requested is None:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
        if not metrics:
            metrics = numeric
        return metrics
    if len(requested) == 1 and requested[0].lower() == "all":
        return numeric
    unknown = [m for m in requested if m not in df.columns]
    if unknown:
        raise SystemExit(
            f"Unknown metric(s): {unknown}. Available: {numeric}")
    return requested


def apply_model_filter(df, models):
    if not models:
        return df
    available = set(df["model"].unique())
    keep = [m for m in models if m in available]
    unknown = [m for m in models if m not in available]
    if unknown:
        warnings.warn(
            f"Ignoring unknown model name(s): {unknown}.\n"
            f"Available models: {sorted(available)}")
    if not keep:
        raise SystemExit("None of the requested --models are present in the CSV.")
    return df[df["model"].isin(keep)].copy()


def apply_dataset_filter(df, datasets):
    if not datasets:
        return df
    available = set(df["dataset"].unique())
    keep = [d for d in datasets if d in available]
    unknown = [d for d in datasets if d not in available]
    if unknown:
        warnings.warn(f"Ignoring unknown dataset name(s): {unknown}. "
                      f"Available datasets: {sorted(available)}")
    if not keep:
        raise SystemExit("None of the requested --datasets are present in the CSV.")
    return df[df["dataset"].isin(keep)].copy()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Comparison plots for model x dataset metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=REPO_ROOT / "results" / "aggregated_by_dataset_model.csv",
                   help="Aggregated results CSV.")
    p.add_argument("-o", "--outdir", type=Path,
                   default=REPO_ROOT / "results" / "plots",
                   help="Base output directory; one sub-folder per plot type.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Only plot these models (others dropped from the run).")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Only plot these datasets.")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Metrics to plot, or 'all' for every numeric column. "
                        f"Default: {DEFAULT_METRICS}")
    p.add_argument("--plots", nargs="+", default=PLOT_TYPES, choices=PLOT_TYPES,
                   help="Which plot types to generate.")
    p.add_argument("--subset-size-col", default="subset_size",
                   help="Column holding the per-model subset size (used if present).")
    p.add_argument("--format", default="png", choices=["png", "pdf"],
                   help="Output file format.")
    p.add_argument("--dpi", type=int, default=300, help="Raster resolution.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)
    missing = set(ID_COLS) - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required column(s): {missing}")

    df = apply_dataset_filter(df, args.datasets)
    df = apply_model_filter(df, args.models)

    metrics = resolve_metrics(df, args.metrics, args.subset_size_col)
    if not metrics:
        raise SystemExit("No metrics to plot.")

    set_pub_style()
    datasets = sorted(df["dataset"].unique())
    dataset_colors = build_dataset_colors(datasets)
    label_map, caption = build_subset_labels(df, args.subset_size_col)

    builders = {
        "ranking": lambda m, path: plot_ranking(
            df, m, path, dataset_colors, label_map, caption, args.dpi),
        "heatmap": lambda m, path: plot_heatmap(
            df, m, path, label_map, caption, args.dpi),
        "bump": lambda m, path: plot_bump(
            df, m, path, label_map, caption, args.dpi),
        "barplot": lambda m, path: plot_barplot(
            df, m, path, dataset_colors, label_map, caption, args.dpi),
    }

    n_written = 0
    for plot_type in args.plots:
        sub_dir = args.outdir / plot_type
        sub_dir.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            out_path = sub_dir / f"{metric}.{args.format}"
            ok = builders[plot_type](metric, out_path)
            if ok:
                n_written += 1
            else:
                warnings.warn(f"Skipped {plot_type}/{metric}: no data after filtering.")

    print(f"Models:   {df['model'].nunique()}   Datasets: {len(datasets)} "
          f"({', '.join(datasets)})")
    if caption:
        print(f"Subset markers -> {caption}")
    print(f"Wrote {n_written} figure(s) under: {args.outdir.resolve()}")
    for plot_type in args.plots:
        print(f"  - {plot_type}/  ({len(metrics)} metric figures)")


if __name__ == "__main__":
    main()
