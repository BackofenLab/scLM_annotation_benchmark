#!/usr/bin/env python3
"""Standalone model/dataset comparison plots.

Reads the aggregated results CSV (one row per model x dataset, columns are
metrics) and produces several comparison views per metric, written under a
two-axis folder hierarchy ``<mode_variant>/<subset_size>/<plot_type>/``:

  Axis 1 - model-mode variant (derived from the model-name suffix; baselines
  are kept in every variant):

    unblind_and_baselines/   unblind models + baselines
    blind_and_baselines/     blind models + baselines
    all_models/              unblind + blind + baselines

  Axis 2 - subset size the model was run on (from the ``subset_size`` column):

    10000/   models run on 10,000 cells
    1000/    models run on 1,000 cells
    all/     every size together; model labels are annotated with their size
             (e.g. "celltypist_blind (10k)") so the size groups stay distinct

  Plot type (one sub-folder of figures per metric):

    .../ranking/   mean-rank bar chart (rank models per dataset, sort by average
                   rank across datasets; the headline view)
    .../heatmap/   model x dataset value heatmap (sorted by mean)
    .../bump/      rank-trajectory ("bump") chart across datasets
    .../barplot/   grouped barplot (x=model sorted by mean, hue=dataset)

Use ``--variants`` to render only some mode variants. The subset-size split is
automatic from the data; when the CSV has no ``subset_size`` column the script
falls back to a single ``all`` folder (and warns).

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
    # Label-coverage / diversity diagnostics (populated once eval+aggregate are re-run).
    "non_cl_ratio",          # fraction of cells whose raw output is NOT a CL term
    "n_non_cl_labels",       # distinct non-CL raw labels
    "n_unique_pred_raw",     # distinct predicted cell types before ontology mapping
    "n_unique_pred_mapped",  # distinct predicted cell types after ontology mapping
]
PLOT_TYPES = ["ranking", "heatmap", "bump", "barplot"]

# Metrics whose natural range is [0, 1]; the label-count diagnostics are unbounded counts and
# must autoscale in the heatmap rather than clipping against vmin=0/vmax=1.
ZERO_ONE_METRICS = {
    "balanced_accuracy", "accuracy", "rada_avg_similarity", "lin_avg_similarity",
    "id_accuracy", "balanced_id_accuracy", "macro_f1", "weighted_f1",
    "macro_precision", "macro_recall", "weighted_precision", "weighted_recall",
    "non_cl_ratio", "reviewed_fraction",
    "text_emb_norm_avg_raw", "text_emb_norm_avg_mapped", "text_emb_mapping_lift",
}


# Prediction "mode" derived from the model-name suffix (mirrors bin/aggregate.py). Used to split
# every figure set into blind / unblind variants while always keeping the baselines for reference.
def derive_mode(model: str) -> str:
    m = str(model).lower()
    if m.endswith("_unblind"):
        return "unblind"
    if m.endswith("_blind"):
        return "blind"
    return "baseline"  # cell_baseline, random_stratified, random_uniform


# Variant name -> set of modes kept. Baselines are included in every variant so each plot has a
# common reference. Each variant is rendered into its own sub-folder under the output directory.
VARIANTS = {
    "unblind_and_baselines": {"unblind", "baseline"},
    "blind_and_baselines": {"blind", "baseline"},
    "all_models": {"unblind", "blind", "baseline"},
}


def compact_size(n) -> str:
    """Short cell-count tag for inline labels: 1000 -> '1k', 10000 -> '10k'."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "?"
    n = int(n)
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def size_folder_name(n) -> str:
    """Folder name for a single subset size (plain integer, no thousands separator)."""
    return str(int(n)) if float(n).is_integer() else str(n)


def iter_size_variants(vdf: pd.DataFrame, size_col: str):
    """Yield ``(folder, df_subset, label_map, caption)`` for each subset-size view of ``vdf``:

    * one view per distinct subset size present (folder = the size, e.g. ``10000``), and
    * a combined ``all`` view across sizes when more than one size is present, in which each
      model label is annotated with its size (e.g. ``celltypist_blind (10k)``) so the two
      size groups stay distinguishable in a single figure.

    When the size column is absent (e.g. an older CSV not yet regenerated) a single ``all``
    view is produced with plain labels, so the script still runs.
    """
    models = list(vdf["model"].unique())
    if size_col not in vdf.columns or vdf[size_col].dropna().empty:
        warnings.warn(
            f"No '{size_col}' column in the input; subset-size split skipped "
            f"(emitting a single 'all' folder). Re-run eval+aggregate to populate it.")
        yield "all", vdf, {m: m for m in models}, ""
        return

    sizes = sorted(vdf[size_col].dropna().unique())
    # Per-size views: plain labels (the folder name already states the size).
    for s in sizes:
        sdf = vdf[vdf[size_col] == s].copy()
        yield (size_folder_name(s), sdf,
               {m: m for m in sdf["model"].unique()},
               f"All models run on {fmt_size(s)} cells.")

    # Combined cross-size view (only meaningful with >1 size). The size is folded into the model
    # identity (e.g. "c2s_blind (10k)") so a model run at two sizes appears as two distinct rows —
    # exactly like the blind/unblind suffix yields two rows. Plain `model` would collapse them.
    if len(sizes) > 1:
        sdf = vdf.copy()
        sdf["model"] = (sdf["model"].astype(str)
                        + " (" + sdf[size_col].map(compact_size).astype(str) + ")")
        label_map = {m: m for m in sdf["model"].unique()}
        caption = ("Each model appears once per subset size it was run on, tagged in parentheses; "
                   "per-size folders (" + ", ".join(size_folder_name(s) for s in sizes) +
                   ") hold each size on its own.")
        yield "all", sdf, label_map, caption


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

    is_ratio = metric in ZERO_ONE_METRICS
    fig, ax = plt.subplots(
        figsize=(max(5.0, 1.15 * len(pivot.columns) + 3.0),
                 max(4.0, 0.32 * len(pivot))))
    sns.heatmap(
        pivot, annot=True, fmt=".3f" if is_ratio else ".0f", cmap=heatmap_cmap(),
        vmin=0 if is_ratio else None, vmax=1 if is_ratio else None,
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
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                   help="Model-mode variants to render, each into its own sub-folder. "
                        "Baselines are always included. Default: all three.")
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
    df = df.assign(_mode=df["model"].map(derive_mode))
    # Stable dataset colours shared across every variant so the figures stay comparable.
    datasets = sorted(df["dataset"].unique())
    dataset_colors = build_dataset_colors(datasets)

    print(f"Datasets: {len(datasets)} ({', '.join(datasets)})")
    total_written = 0
    for variant in args.variants:
        keep_modes = VARIANTS[variant]
        vdf = df[df["_mode"].isin(keep_modes)].copy()
        if vdf.empty:
            warnings.warn(
                f"Variant '{variant}': no models match modes {sorted(keep_modes)}; skipped.")
            continue

        # Second axis: split each mode variant by subset size -> results/plots/<mode>/<size>/...
        for size_folder, sdf, label_map, caption in iter_size_variants(vdf, args.subset_size_col):
            if sdf.empty:
                continue
            builders = {
                "ranking": lambda m, path: plot_ranking(
                    sdf, m, path, dataset_colors, label_map, caption, args.dpi),
                "heatmap": lambda m, path: plot_heatmap(
                    sdf, m, path, label_map, caption, args.dpi),
                "bump": lambda m, path: plot_bump(
                    sdf, m, path, label_map, caption, args.dpi),
                "barplot": lambda m, path: plot_barplot(
                    sdf, m, path, dataset_colors, label_map, caption, args.dpi),
            }

            n_written = 0
            for plot_type in args.plots:
                sub_dir = args.outdir / variant / size_folder / plot_type
                sub_dir.mkdir(parents=True, exist_ok=True)
                for metric in metrics:
                    out_path = sub_dir / f"{metric}.{args.format}"
                    ok = builders[plot_type](metric, out_path)
                    if ok:
                        n_written += 1
                    else:
                        warnings.warn(
                            f"Skipped {variant}/{size_folder}/{plot_type}/{metric}: "
                            f"no data after filtering.")

            total_written += n_written
            print(f"[{variant}/{size_folder}]  models={sdf['model'].nunique()}  "
                  f"wrote {n_written} figure(s) under {args.outdir / variant / size_folder}")
            if caption:
                print(f"   note -> {caption}")

    print(f"Total: wrote {total_written} figure(s) under {args.outdir.resolve()}")
    print(f"  layout: <mode_variant>/<subset_size>/<plot_type>/<metric>.{args.format}")


if __name__ == "__main__":
    main()
