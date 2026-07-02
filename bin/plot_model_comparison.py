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
    .../correlation/  metric × metric Spearman correlation heatmap (one figure per
                   folder, covering every numeric metric; clustered ordering)

A ``composite_category/`` folder (a sibling of the mode-variant folders, since it combines
blind + unblind in one figure) holds the composite-score-by-category view: per subset size, a
two-panel (blind | unblind) figure where each model's composite score (mean of balanced
accuracy, lin/rada avg. similarity, and text-emb norm avg. raw) is plotted as a point within
its model family (LLMs / domain-adapted scLLMs / classical baselines), with a per-category mean
line and the simple baselines (random_*, cell_baseline) as dashed reference lines.

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


# Optional: SciPy gives us hierarchical-clustering ordering for the metric correlation
# heatmap so correlated metrics sit next to each other. The figure still renders (in input
# order) when SciPy is unavailable, so this stays an optional dependency.
try:
    from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
    from scipy.spatial.distance import squareform  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - clustering is a nicety, not essential
    _HAVE_SCIPY = False


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
    # Text-encoder (SapBERT) similarity between predicted and true labels, baseline-normalized.
    # Populated only when eval ran with --use_text_encoder true (NaN columns are skipped).
    "text_emb_norm_avg_raw",     # on raw predictions
    "text_emb_norm_avg_mapped",  # on ontology-mapped predictions
    "text_emb_mapping_lift",     # gain from ontology mapping (mapped - raw)
    # Label-coverage / diversity diagnostics (populated once eval+aggregate are re-run).
    "non_cl_ratio",          # fraction of cells whose raw output is NOT a CL term
    "n_non_cl_labels",       # distinct non-CL raw labels
    "n_unique_pred_raw",     # distinct predicted cell types before ontology mapping
    "n_unique_pred_mapped",  # distinct predicted cell types after ontology mapping
]
PLOT_TYPES = ["ranking", "heatmap", "bump", "barplot", "correlation", "composite_category"]

# --------------------------------------------------------------------------- #
# Composite-score-by-category figure (blind vs unblind).
# PROVISIONAL headline composite spanning the two axes we care about, ONE metric each:
#   - exact-match correctness : balanced_accuracy (class-balanced; == balanced_id_accuracy
#                               after the symmetric CL-canonicalization fix in evaluate.py)
#   - ontology closeness      : lin_avg_similarity (information-content CL-graph similarity;
#                               Lin chosen over the near-identical Rada — per-dataset ranking
#                               agreement Spearman ~0.95-0.98 — for its granularity-awareness)
# Deliberately one metric per axis: NO Rada (would double-weight closeness with Lin) and NO
# text-embedding similarity (redundant with the graph metric, encoder-dependent, high-floor).
# Each component is BASELINE-NORMALIZED to a skill score against the per-dataset chance model
# (random_stratified) before the mean — see _composite_per_model — because the raw floors differ
# sharply (balanced_accuracy chance ~1/#classes vs Lin chance ~0.4-0.66) and an unnormalized mean
# would be dominated by Lin. (Root-prediction / cell_baseline is NOT used as the Lin floor: "cell"
# is the ontology root, so its Lin is 0 and normalizing against it is a no-op.) OPEN decision:
# whether to also add text similarity. Every individual metric remains its own figure.
# --------------------------------------------------------------------------- #
COMPOSITE_METRICS = [
    "balanced_accuracy", "lin_avg_similarity",
]
# Per-dataset chance reference used to baseline-normalize each composite component.
COMPOSITE_BASELINE_MODEL = "random_stratified"
# Model-family membership keyed on the base name (model with any _blind/_unblind suffix removed).
SCLLM_BASES = {
    "c2s", "c2s_scale_2B", "c2s_scale_27B", "cellwhisperer", "instructcell", "langcell",
    "mmcontext",
}
CLASSICAL_BASES = {"celltypist", "scsa"}
SIMPLE_BASES = {"random_stratified", "random_uniform", "cell_baseline"}
# Subset of the simple baselines drawn as dashed reference lines in the composite figure.
REFERENCE_LINE_BASES = {"random_stratified"}
# Ordered (category key, x-axis display label) for the strip panels.
COMPOSITE_CATEGORIES = [
    ("LLM", "LLMs"),
    ("scLLM", "scLLMs"),
    ("classical", "baseline"),
]

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


def model_base(model: str) -> str:
    """Model name with any trailing prediction-mode suffix stripped."""
    m = str(model)
    for suffix in ("_blind", "_unblind"):
        if m.endswith(suffix):
            return m[: -len(suffix)]
    return m


def categorize_model(model: str) -> str:
    """Model-family category used by the composite-score figure.

    Returns one of ``"simple"`` (random / cell baselines, drawn as reference lines),
    ``"classical"`` (CellTypist, SCSA), ``"scLLM"`` (domain-adapted language models), or
    ``"LLM"``. Anything not explicitly a simple/classical/scLLM model falls through to
    ``"LLM"`` so newly added general-purpose chat models are picked up automatically.
    """
    base = model_base(model)
    if base in SIMPLE_BASES:
        return "simple"
    if base in CLASSICAL_BASES:
        return "classical"
    if base in SCLLM_BASES:
        return "scLLM"
    return "LLM"


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


def diverging_cmap():
    """Diverging colormap centered at 0 for the correlation heatmap.

    The repo's ``COLORMAPS`` are all sequential white-to-colour maps, which would hide the
    sign of a correlation; a correlation matrix needs a diverging map. Kept as a tiny helper
    so it is easy to swap for a repo map later if a diverging one is added.
    """
    return "RdBu_r"


def cluster_order(corr: pd.DataFrame) -> list:
    """Order metrics so correlated ones are adjacent (hierarchical clustering).

    Uses average linkage on the distance ``1 - |corr|`` (so strongly correlated *and* strongly
    anti-correlated metrics group together). Falls back to the input column order when SciPy is
    unavailable or there are too few metrics to cluster.
    """
    cols = list(corr.columns)
    if not _HAVE_SCIPY or len(cols) < 3:
        return cols
    dist = 1.0 - corr.abs().values
    # A pairwise correlation can be NaN even after all-NaN/constant columns are dropped, when two
    # surviving metrics share fewer than two non-null rows (empty overlap). squareform/linkage
    # reject non-finite values, so treat an undefined correlation as "maximally distant" (1.0).
    dist = np.where(np.isfinite(dist), dist, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce symmetry against tiny float asymmetries
    link = linkage(squareform(dist, checks=False), method="average")
    return [cols[i] for i in leaves_list(link)]


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
def _strip_trivial_baselines(df):
    """Drop the trivial chance baselines (random_stratified, random_uniform, cell_baseline) from
    the models drawn as bars / ranked entries. They are shown as dashed reference lines on the
    barplot instead (see ``_baseline_refs``). Classical annotators (CellTypist, SCSA) are kept —
    they are real methods, not chance floors."""
    return df[~df["model"].map(lambda m: model_base(m) in SIMPLE_BASES)]


def _baseline_refs(df, metric):
    """Dashed reference line(s) for ``metric``: ``[(label, value, linestyle, color), ...]``.

    ``random_stratified`` is the informed-chance floor for EVERY metric (value = its mean over the
    datasets in this view). ``cell_baseline`` (predict the ontology root "cell") is added ONLY for
    ``rada_avg_similarity`` — it is a non-zero floor there (~0.2, the distance-to-root level) but is
    ~0 for exact-match, Lin, and text similarity, so it is not shown on those. Computed from ``df``
    BEFORE the baselines are stripped."""
    def val(base):
        rows = df[df["model"].map(model_base) == base]
        if rows.empty:
            return None
        v = pd.to_numeric(rows[metric], errors="coerce").mean()
        return None if pd.isna(v) else float(v)
    refs = []
    v = val("random_stratified")
    if v is not None:
        refs.append(("random-stratified (chance)", v, "--", "0.35"))
    if metric == "rada_avg_similarity":
        vc = val("cell_baseline")
        if vc is not None:
            refs.append(("cell-baseline (root)", vc, ":", "0.55"))
    return refs


def plot_ranking(df, metric, out_path, dataset_colors, label_map, caption, dpi):
    df = _strip_trivial_baselines(df)  # rank real models only; chance floors are not competitors
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
    df = _strip_trivial_baselines(df)  # real models only (no chance-floor rows)
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
    df = _strip_trivial_baselines(df)  # real models only (no chance-floor rows)
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
    refs = _baseline_refs(df, metric)                     # compute BEFORE stripping baselines
    sub = _strip_trivial_baselines(df).dropna(subset=[metric])
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
    # Chance floor(s) as dashed reference lines (mean over datasets in this view).
    for _lbl, v, ls, c in refs:
        ax.axhline(v, linestyle=ls, color=c, linewidth=1.5, zorder=5)
    # Dataset legend (from the bars), plus a second legend for the reference lines.
    leg1 = ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    if refs:
        ax.add_artist(leg1)
        ref_handles = [plt.Line2D([0], [0], color=c, linestyle=ls, linewidth=1.5, label=lbl)
                       for lbl, v, ls, c in refs]
        ax.legend(handles=ref_handles, title="Baseline (dashed)",
                  bbox_to_anchor=(1.02, 0.45), loc="upper left", frameon=False)
    add_caption(fig, caption)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_metric_correlation(df, metrics, out_path, method, caption, dpi):
    """Metric × metric correlation heatmap (one figure per variant/size folder).

    Correlates every numeric metric column across the rows of ``df`` (model × dataset
    observations in the current variant/size view). Columns that are all-NaN or constant
    within the subset have an undefined correlation and are dropped (and noted in the caption)
    so they don't render as NaN stripes.
    """
    present = [m for m in metrics if m in df.columns]
    sub = df[present].apply(pd.to_numeric, errors="coerce")

    usable, dropped = [], []
    for m in present:
        col = sub[m]
        if col.notna().sum() < 2 or col.std(skipna=True) == 0 or np.isnan(col.std(skipna=True)):
            dropped.append(m)
        else:
            usable.append(m)
    if len(usable) < 2:
        return False

    corr = sub[usable].corr(method=method)
    order = cluster_order(corr)
    corr = corr.loc[order, order]

    n = len(order)
    side = max(6.0, 0.55 * n + 2.0)
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=diverging_cmap(),
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, linecolor="white", annot_kws={"size": 6},
        xticklabels=[pretty(m) for m in order],
        yticklabels=[pretty(m) for m in order],
        cbar_kws={"label": f"{method.title()} correlation", "shrink": 0.6}, ax=ax,
    )
    ax.set_title(f"Metric correlation ({method.title()})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    cap = caption
    if dropped:
        note = "Dropped (constant/all-NaN): " + ", ".join(pretty(m) for m in dropped)
        cap = f"{caption}   {note}" if caption else note
    add_caption(fig, cap)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _composite_per_model(df: pd.DataFrame) -> pd.Series:
    """Per-model composite score: mean of the BASELINE-NORMALIZED COMPOSITE_METRICS per row,
    then mean over the model's rows (datasets, and sizes when the view spans several).

    Each metric is turned into a skill score against the per-dataset chance model
    (``COMPOSITE_BASELINE_MODEL``, random_stratified): ``norm(x) = clip((x - b) / (1 - b), 0, 1)``
    where ``b`` is that model's score for the same (dataset, metric). So chance scores 0 and a
    perfect model scores 1, which removes each metric's floor (notably Lin's ~0.5) and lets the
    two axes contribute comparably to the mean. If the baseline is unavailable for a (dataset,
    metric) the metric is used unnormalized (b=0). Empty Series if the composite columns are
    absent."""
    if not all(c in df.columns for c in COMPOSITE_METRICS) or "dataset" not in df.columns:
        return pd.Series(dtype=float)
    num = df[list(COMPOSITE_METRICS)].apply(pd.to_numeric, errors="coerce")
    base = None
    if COMPOSITE_BASELINE_MODEL in set(df["model"].astype(str)):
        base = (df[df["model"].astype(str) == COMPOSITE_BASELINE_MODEL]
                .groupby("dataset")[list(COMPOSITE_METRICS)]
                .mean().apply(pd.to_numeric, errors="coerce"))
    normed = pd.DataFrame(index=df.index)
    for m in COMPOSITE_METRICS:
        if base is not None and m in base.columns:
            b = pd.to_numeric(df["dataset"].map(base[m]), errors="coerce").fillna(0.0).clip(upper=0.999)
        else:
            b = 0.0
        normed[m] = ((num[m] - b) / (1.0 - b)).clip(lower=0.0, upper=1.0)
    comp = normed.mean(axis=1)
    return comp.groupby(df["model"]).mean()


def plot_composite_category(df, out_path, caption, dpi, title=None):
    """Two-panel (blind | unblind) composite-score figure.

    In each panel the LLM / scLLM / classical-baseline categories show their individual models
    as points plus a horizontal mean line; the simple baselines (random_*, cell_baseline) are
    drawn as dashed reference lines (identical in both panels, since they are mode-agnostic).

    ``title`` adds a figure-level title (used by the per-dataset supplementary figures so they
    are self-identifying); the aggregated-over-datasets figures pass ``None`` (no title).
    """
    per_model = _composite_per_model(df)
    if per_model.empty:
        return False

    cat_palette = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    cat_color = {key: cat_palette[i % len(cat_palette)]
                 for i, (key, _) in enumerate(COMPOSITE_CATEGORIES)}

    modes = ["blind", "unblind"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    # Classical baselines (CellTypist, SCSA) are blind-only but act as a mode-agnostic reference,
    # so they are shown in BOTH panels with the same points (like the dashed simple baselines).
    classical_models = sorted(m for m in per_model.index if categorize_model(m) == "classical")

    def members_for(key, mode_models):
        if key == "classical":
            return classical_models
        return sorted(m for m in mode_models if categorize_model(m) == key)

    for ax, mode in zip(axes, modes):
        mode_models = [m for m in per_model.index if derive_mode(m) == mode]
        # Keep categories that have at least one model (classical is always present if any exist),
        # packed left-to-right so an empty mode-specific category leaves no gap.
        present = [(key, lbl) for key, lbl in COMPOSITE_CATEGORIES if members_for(key, mode_models)]
        pos = {key: i for i, (key, _) in enumerate(present)}
        for key, _lbl in present:
            members = members_for(key, mode_models)
            x0 = pos[key]
            vals = per_model.loc[members]                     # Series (index = model names)
            n = len(members)
            offsets = np.linspace(-0.09, 0.09, n) if n > 1 else np.array([0.0])
            ax.scatter(x0 + offsets, vals.values, s=42, color=cat_color[key],
                       edgecolor="white", linewidth=0.5, zorder=3)
            mean_v = float(np.nanmean(vals.values))
            ax.hlines(mean_v, x0 - 0.28, x0 + 0.28, color=cat_color[key],
                      linewidth=2.4, zorder=4)
            # Label only the TOP (best) model in each class next to its point — keeps the honest
            # all-points + class-mean view while identifying the champion.
            top_i = int(np.nanargmax(vals.values))
            ax.annotate(model_base(vals.index[top_i]),
                        (x0 + offsets[top_i], vals.values[top_i]),
                        xytext=(6, 3), textcoords="offset points", fontsize=8,
                        rotation=20, ha="left", va="bottom", zorder=6)

        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels([lbl for _, lbl in present], fontsize=14,
                           rotation=45, ha="right")
        ax.set_xlim(-0.6, max(len(present) - 0.4, 0.6))
        ax.set_title(mode.capitalize(), fontsize=17)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        sns.despine(ax=ax)  # frame off: drop the top and right spines

    axes[0].set_ylim(0, 1)
    # y=0 IS the random-stratified (chance) level by construction (the composite is normalized
    # against it), so no separate baseline reference line is drawn — it would sit on the axis.
    axes[0].set_ylabel("Composite score\n(bal. acc. + Lin. sim., chance-normalized)", fontsize=14)

    # Legend: only the category-mean marker (the individual points are self-explanatory).
    handles = [plt.Line2D([0], [0], color="0.2", linewidth=2.4, label="Category mean")]
    axes[1].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.02, 0.97),
                   frameon=False, fontsize=13)

    # Caption intentionally omitted; optional title only for the per-dataset supplementary figures.
    if title:
        fig.suptitle(title, y=0.99, fontsize=19)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def composite_size_views(df: pd.DataFrame, size_col: str):
    """Yield ``(folder, df_subset)`` for the composite figure: one view per subset size, plus a
    combined ``all`` view across sizes when more than one is present. Unlike
    ``iter_size_variants`` the size is NOT folded into the model name (we want one point per
    model, averaged over datasets and sizes in the combined view)."""
    if size_col not in df.columns or df[size_col].dropna().empty:
        yield "all", df
        return
    sizes = sorted(df[size_col].dropna().unique())
    for s in sizes:
        yield size_folder_name(s), df[df[size_col] == s]
    if len(sizes) > 1:
        yield "all", df


# --------------------------------------------------------------------------- #
# Data loading / argument handling
# --------------------------------------------------------------------------- #
def numeric_metrics(df, subset_col):
    """Every numeric metric column (excludes the id, mode, and subset-size columns)."""
    return [c for c in df.columns
            if c not in ID_COLS and c not in (subset_col, "mode")
            and pd.api.types.is_numeric_dtype(df[c])]


def resolve_metrics(df, requested, subset_col):
    numeric = numeric_metrics(df, subset_col)
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
    p.add_argument("--corr-method", default="spearman", choices=["spearman", "pearson"],
                   help="Correlation coefficient for the metric-correlation heatmap.")
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
    # The correlation heatmap always covers the full numeric metric set, independent of the
    # per-metric --metrics selection.
    all_numeric_metrics = numeric_metrics(df, args.subset_size_col)

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
            # Per-metric plot types: one figure per metric. (correlation and composite_category
            # are not per-metric and are handled separately.)
            for plot_type in (pt for pt in args.plots
                              if pt not in ("correlation", "composite_category")):
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

            # Correlation is a single metric × metric figure covering every numeric metric,
            # so it lives outside the per-metric loop.
            if "correlation" in args.plots:
                sub_dir = args.outdir / variant / size_folder / "correlation"
                sub_dir.mkdir(parents=True, exist_ok=True)
                out_path = sub_dir / f"metric_correlation_{args.corr_method}.{args.format}"
                if plot_metric_correlation(sdf, all_numeric_metrics, out_path,
                                           args.corr_method, caption, args.dpi):
                    n_written += 1
                else:
                    warnings.warn(
                        f"Skipped {variant}/{size_folder}/correlation: "
                        f"fewer than 2 usable metrics after filtering.")

            total_written += n_written
            print(f"[{variant}/{size_folder}]  models={sdf['model'].nunique()}  "
                  f"wrote {n_written} figure(s) under {args.outdir / variant / size_folder}")
            if caption:
                print(f"   note -> {caption}")

    # composite_category spans blind + unblind in a single two-panel figure, so it lives in its
    # own top-level folder rather than under a mode-variant.
    if "composite_category" in args.plots:
        cdir = args.outdir / "composite_category"

        def emit_composite(src_df, out_dir, log_tag, title=None):
            """Write one composite figure per subset-size view of ``src_df`` into ``out_dir``."""
            nonlocal total_written
            out_dir.mkdir(parents=True, exist_ok=True)
            for size_folder, sdf in composite_size_views(src_df, args.subset_size_col):
                out = out_dir / f"composite_category_{size_folder}.{args.format}"
                if plot_composite_category(sdf, out, "", args.dpi, title=title):
                    total_written += 1
                    print(f"[composite_category/{log_tag}/{size_folder}]  wrote {out.name}")
                else:
                    warnings.warn(
                        f"Skipped composite_category/{log_tag}/{size_folder}: composite metrics "
                        f"{COMPOSITE_METRICS} not all present.")

        # 1) Aggregated over all datasets (top-level folder), as before.
        emit_composite(df, cdir, "all_datasets")

        # 2) Per-dataset breakdown for the supplementary: one figure set per dataset, each titled
        #    with the dataset name so the supplementary panels are self-identifying.
        for dataset in sorted(df["dataset"].unique()):
            emit_composite(df[df["dataset"] == dataset], cdir / "by_dataset" / str(dataset),
                           f"by_dataset/{dataset}", title=str(dataset))

    print(f"Total: wrote {total_written} figure(s) under {args.outdir.resolve()}")
    print(f"  layout: <mode_variant>/<subset_size>/<plot_type>/<metric>.{args.format}")


if __name__ == "__main__":
    main()
