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
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.path as mpath  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
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
    "text_emb_mapping_lift",     # gain from mapping, on the RAW cosine scale (legacy; see below)
    # Class-balanced variants, and a lift that is a difference of two normalized scores. Prefer
    # these: text_emb_mapping_lift above is an unnormalized difference, so it is NOT on the same
    # scale as text_emb_norm_avg_* (they differ by 1/(1 - baseline), which is per-run and ranges
    # 1.5x-3.7x here). The *_balanced_* pair also matches every other headline metric in being
    # class-balanced rather than dominated by abundant cell types.
    "text_emb_norm_balanced_raw",
    "text_emb_norm_balanced_mapped",
    "text_emb_norm_mapping_lift",
    # Label-coverage / diversity diagnostics (populated once eval+aggregate are re-run).
    "non_cl_ratio",          # fraction of cells whose raw output is NOT a CL term
    "n_non_cl_labels",       # distinct non-CL raw labels
    "n_unique_pred_raw",     # distinct predicted cell types before ontology mapping
    "n_unique_pred_mapped",  # distinct predicted cell types after ontology mapping
]
PLOT_TYPES = ["ranking", "heatmap", "bump", "barplot", "correlation", "composite_category",
              # LLM-only figures built from the OpenRouter activity log (see --activity).
              # Both are top-level (not per-metric / per-mode-variant) and are skipped with a
              # warning when the activity CSV is absent.
              "llm_activity", "cost_performance"]
# Model publication/release dates for the date-vs-composite figure. Produced out-of-band (the
# dates are researched against primary sources, not derivable from the pipeline), so this is a
# soft dependency: the figure is skipped with a warning when the file is absent.
DEFAULT_DATES_CSV = REPO_ROOT / "cache" / "model_publication_dates.csv"

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
# The per-tier SingleR variants are listed explicitly: without them categorize_model() falls
# through to "LLM" and would color them as general-purpose LLMs if they were ever taken out of
# EXCLUDED_BASES.
CLASSICAL_BASES = {"celltypist", "scsa", "sctype", "ucell",
                   "singler", "singler_main", "singler_fine", "singler_ont"}
# Agentic "coding agent" blueprints: tool-using LLM-CLI agents (Claude Code / Codex) that
# annotate via the agentic-blueprint harness, as opposed to the one-shot OpenRouter LLMs.
AGENT_BASES = {"agentic_bp_claude", "agentic_bp_codex"}
SIMPLE_BASES = {"random_stratified", "random_uniform", "cell_baseline"}
# Subset of the simple baselines drawn as dashed reference lines in the composite figure.
REFERENCE_LINE_BASES = {"random_stratified"}
# Deprecated / superseded models whose result JSONs are kept on disk for the record but are NOT part
# of the benchmark, so they are dropped from every figure: UCell (its authors prescribe no argmax
# rule, so it's intentionally absent from params.models) and the old per-tier SingleR models
# (singler_main/fine/ont, replaced by the single `singler_blind`). Keyed by model_base (mode-stripped).
#
# `c2s` (Cell2Sentence, C2S-Pythia-410m) is excluded for a different reason: it is the only scLLM
# with NO unblind counterpart, so it orphaned one point in the paired blind/unblind composite
# figure and made the two scLLM category means average over different model sets (7 blind vs 6
# unblind). Because it scores above the blind scLLM mean, keeping it understated the blind->unblind
# gain (+0.140 with it, +0.154 on the matched six). An unblind variant is not simply missing: the
# model is driven through the cell2sentence library's own prompt formatter, whose templates expose
# only {num_genes}/{organism}/{cell_sentence} and have no slot for a candidate label list, so
# building one would require forking upstream. See models/c2s_blind/src/predict.py.
EXCLUDED_BASES = {"ucell", "singler_main", "singler_fine", "singler_ont",
                  "agentic_bp_claude", "agentic_bp_codex", "c2s"}
# Ordered (category key, x-axis display label) for the strip panels.
# Left-to-right: baselines, domain-adapted LLMs, one-shot LLMs, coding agents.
COMPOSITE_CATEGORIES = [
    ("classical", "baseline"),
    ("scLLM", "scLLMs"),
    ("LLM", "LLMs"),
    ("coding_agent", "coding agents"),
]

# --------------------------------------------------------------------------- #
# Sub-grouping WITHIN each category column of the composite strip figure.
#
# Every category is split into method tiers that get their own marker shape plus a thin black
# within-tier mean line drawn on top of the category-wide mean. The splits are:
#
#   LLM        closed-weight reasoning / open-weight reasoning / open-weight non-reasoning.
#              NOTE these two attributes are CONFOUNDED in the current model set -- every
#              closed-weight model is also a reasoning model, and there is no closed-weight
#              non-reasoning model -- so the tiers are a presentational grouping and cannot
#              separate "open weights" from "this vendor's models are better". Adding a
#              closed-weight non-reasoning model (e.g. mistral-large-2512) fills the empty cell.
#   scLLM      how the prediction is produced: generative models emit a free-text label
#              (c2s family via cell sentences, InstructCell via a prompt), whereas retrieval
#              models embed the cell and argmax a cosine similarity against a supplied set of
#              candidate label strings (CellWhisperer / mmContext / LangCell). The two are
#              given genuinely different tasks -- the retrieval models are closed-set by
#              construction, scoring against cache/all_cell_types.txt (~3.2k CL terms) in blind
#              mode and the dataset's own label list when unblind.
#   classical  marker-set scoring against a marker database (scType/UCell over ScTypeDB, SCSA
#              over CellMarker) vs. classifiers fitted to a labeled reference (CellTypist,
#              SingleR reference-profile correlation). This is the split already drawn in
#              benchmarking_scLLMs_story.md ("Note 2 -- Marker-based scoring / reference-
#              correlation approaches").
#
# Entries are (tier key, legend label, marker, marker size). Marker shapes are unique ACROSS
# categories so a legend entry is unambiguous on its own.
CATEGORY_TIERS = {
    "LLM": [
        ("closed_reasoning",  "closed-weight reasoning",   "*", 150),
        ("open_reasoning",    "open-weight reasoning",     "^", 52),
        ("open_nonreasoning", "open-weight non-reasoning", "s", 40),
    ],
    "scLLM": [
        ("generative", "generative (free-text label)",     "o", 46),
        ("retrieval",  "embedding retrieval (closed set)", "v", 52),
    ],
    "classical": [
        ("marker",    "marker-set scoring",         "D", 38),
        ("reference", "reference-based classifier", "P", 62),
    ],
}
# None of these attributes is recoverable from a model name, so membership is an explicit
# reviewed list keyed on model_base(). Models missing from it are drawn as "unclassified"
# rather than guessed at -- see model_tier(). LLM names must match the aggregated CSV
# (OpenRouter ids with the vendor prefix stripped), NOT the ids in model_cfg.yml.
CATEGORY_TIER_MEMBERS = {
    "LLM": {
        "closed_reasoning": {
            "claude-opus-4.7", "claude-sonnet-4.6", "gemini-3.1-pro-preview", "gpt-5.5",
            # configured in model_cfg.yml but not currently in params.models:
            "gpt-5.5-pro", "gemini-3-flash-preview",
        },
        "open_reasoning": {
            "deepseek-v3.2", "gpt-oss-120b", "gpt-oss-20b", "kimi-k2.6",
            # configured in model_cfg.yml but not currently in params.models:
            "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-r1-0528", "glm-5",
        },
        "open_nonreasoning": {
            "llama-3.1-70b-instruct", "llama-3.3-70b-instruct", "llama-4-maverick",
            # configured in model_cfg.yml but not currently in params.models:
            "llama-3.1-8b-instruct",
        },
        # Deliberately unlisted: mistral-large-2512 (non-reasoning but closed-weight, so it
        # needs a fourth tier) and gemma-4-31b-it (open-weight, reasoning behaviour
        # unverified). Both render as "unclassified" until someone confirms where they belong.
    },
    "scLLM": {
        "generative": {"c2s", "c2s_scale_2B", "c2s_scale_27B", "instructcell"},
        "retrieval": {"cellwhisperer", "cellwhisperer_emb", "langcell", "mmcontext"},
    },
    "classical": {
        "marker": {"scsa", "sctype", "ucell"},
        "reference": {"celltypist", "singler", "singler_main", "singler_fine", "singler_ont"},
    },
}
UNCLASSIFIED_TIER_STYLE = ("X", 40)
_WARNED_UNKNOWN_TIERS: set[str] = set()

# Metrics whose natural range is [0, 1]; the label-count diagnostics are unbounded counts and
# must autoscale in the heatmap rather than clipping against vmin=0/vmax=1. The mapping lifts are
# signed differences and belong OUTSIDE this set: a negative lift (ontology mapping made a model's
# label worse, e.g. llama-4-maverick) would otherwise be clipped away at vmin=0.
ZERO_ONE_METRICS = {
    "balanced_accuracy", "accuracy", "rada_avg_similarity", "lin_avg_similarity",
    "id_accuracy", "balanced_id_accuracy", "macro_f1", "weighted_f1",
    "macro_precision", "macro_recall", "weighted_precision", "weighted_recall",
    "non_cl_ratio", "reviewed_fraction",
    "text_emb_norm_avg_raw", "text_emb_norm_avg_mapped",
    "text_emb_norm_balanced_raw", "text_emb_norm_balanced_mapped",
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
    ``"classical"`` (CellTypist, SCSA), ``"scLLM"`` (domain-adapted language models),
    ``"coding_agent"`` (agentic-blueprint Claude/Codex CLI agents), or ``"LLM"``. Anything not
    explicitly matched falls through to ``"LLM"`` so newly added general-purpose chat models are
    picked up automatically.
    """
    base = model_base(model)
    if base in SIMPLE_BASES:
        return "simple"
    if base in CLASSICAL_BASES:
        return "classical"
    if base in AGENT_BASES or base.startswith("agentic"):
        return "coding_agent"
    if base in SCLLM_BASES:
        return "scLLM"
    return "LLM"


def model_tier(model: str):
    """Method tier within the model's category, or ``None`` if the category has no tiers.

    Returns one of the category's ``CATEGORY_TIERS`` keys for models listed in
    ``CATEGORY_TIER_MEMBERS``, or ``"unclassified"`` for a model whose tier has not been
    recorded yet -- warned once on stderr, drawn with a neutral marker, and kept out of the
    within-tier means so an unreviewed model can never silently shift a tier average.
    """
    cat = categorize_model(model)
    if cat not in CATEGORY_TIER_MEMBERS:
        return None
    base = model_base(model)
    for key, members in CATEGORY_TIER_MEMBERS[cat].items():
        if base in members:
            return key
    if base not in _WARNED_UNKNOWN_TIERS:
        _WARNED_UNKNOWN_TIERS.add(base)
        print(f"[composite_category]  {cat} model '{base}' is not in CATEGORY_TIER_MEMBERS; "
              f"drawing it as unclassified (excluded from the tier means).", file=sys.stderr)
    return "unclassified"


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
    # When blind AND unblind models share one figure (the all_models variant), mark the blind
    # models' bars with diagonal hatching so the two prediction modes are distinguishable at a
    # glance. seaborn draws one BarContainer per hue (dataset), each ordered like `order`.
    modes = [derive_mode(m) for m in order]
    mark_blind = ("blind" in modes) and ("unblind" in modes)
    if mark_blind:
        # Map each bar to its model by x-position (not container index): a dataset missing a model
        # yields fewer bars in that hue's container, so index-zipping would mis-align. Categorical x
        # places model i at integer i and dodges the hue bars within +/-0.5, so round(center)==i.
        # Black diagonals so the mark is visible on every dataset color (white vanished on light bars).
        for container in ax.containers:
            for bar in container:
                idx = int(round(bar.get_x() + bar.get_width() / 2.0))
                if 0 <= idx < len(modes) and modes[idx] == "blind":
                    bar.set_hatch("///")
                    bar.set_edgecolor("black")
                    bar.set_linewidth(0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label_map.get(m, m) for m in order], rotation=45, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel(pretty(metric))
    # Chance floor(s) as dashed reference lines (mean over datasets in this view).
    for _lbl, v, ls, c in refs:
        ax.axhline(v, linestyle=ls, color=c, linewidth=1.5, zorder=5)
    # Dataset legend (from the bars); plus reference-line and prediction-mode legends as needed.
    leg1 = ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.add_artist(leg1)
    if refs:
        ref_handles = [plt.Line2D([0], [0], color=c, linestyle=ls, linewidth=1.5, label=lbl)
                       for lbl, v, ls, c in refs]
        legr = ax.legend(handles=ref_handles, title="Baseline (dashed)",
                         bbox_to_anchor=(1.02, 0.45), loc="upper left", frameon=False)
        ax.add_artist(legr)
    if mark_blind:
        mode_handles = [mpatches.Patch(facecolor="0.7", edgecolor="black", label="unblind"),
                        mpatches.Patch(facecolor="0.7", edgecolor="black", hatch="///", label="blind")]
        ax.legend(handles=mode_handles, title="Prediction mode",
                  bbox_to_anchor=(1.02, 0.2), loc="upper left", frameon=False)
    # No in-plot title or subset-size caption on barplots: the metric is in the filename and the
    # subset size is in the folder path, so both are redundant here.
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


def _tiered_offsets(group_sizes, spacing=0.030, gap=0.080, max_half_width=0.24):
    """x-offsets for points laid out in contiguous groups separated by a visible gap.

    Keeps the within-group point density of the plain evenly-spaced layout, centres the whole
    block on the column, and shrinks it proportionally if the extra inter-group gaps would push
    it past the category mean line at +/-0.28.
    """
    pos, x = [], 0.0
    for i, k in enumerate(group_sizes):
        if i:
            x += gap
        for _ in range(k):
            pos.append(x)
            x += spacing
    arr = np.asarray(pos, dtype=float)
    if arr.size == 0:
        return arr
    arr -= (arr.min() + arr.max()) / 2.0
    half = float(arr.max())
    if half > max_half_width:
        arr *= max_half_width / half
    return arr


def _category_layout(key, members, use_tiers=True):
    """Point layout for one category column of the composite strip figure.

    Returns ``(members, offsets, segments)``: ``members`` possibly reordered so same-tier models
    are contiguous, ``offsets`` the per-point x-offset around the column centre, and
    ``segments`` a list of ``(tier_key, slice, marker, marker_size)`` -- one entry per scatter
    call, since matplotlib takes a single marker per call. Categories without a tier definition
    get one circle segment with ``tier_key=None`` and even spacing, so they are unchanged.

    ``use_tiers=False`` forces a flat single-circle layout for EVERY category, which is what the
    tier-free variant of the composite figure uses. There is no x-jitter in that variant: every
    point sits on its sub-column centre, so x carries no information at all and the column reads
    as a single distribution. (The tiered variant needs offsets because x there encodes which
    tier a point belongs to.)
    """
    tiers = CATEGORY_TIERS.get(key) if use_tiers else None
    if not tiers:
        members = sorted(members)
        n = len(members)
        return members, np.zeros(n), [(None, slice(0, n), "o", 44)]

    style = {k: (marker, size) for k, _lbl, marker, size in tiers}
    style["unclassified"] = UNCLASSIFIED_TIER_STYLE

    ordered, segments, start = [], [], 0
    for tkey in [k for k, _lbl, _m, _s in tiers] + ["unclassified"]:
        grp = sorted(m for m in members if model_tier(m) == tkey)
        if not grp:
            continue
        ordered.extend(grp)
        marker, size = style[tkey]
        segments.append((tkey, slice(start, start + len(grp)), marker, size))
        start += len(grp)
    offsets = _tiered_offsets([sl.stop - sl.start for _k, sl, _m, _s in segments])
    return ordered, offsets, segments


def plot_composite_category(df, out_path, caption, dpi, title=None, show_tiers=True, ax=None,
                            metric=None):
    """Single-panel composite-score figure with paired blind/unblind sub-columns.

    Pass ``ax`` to draw the panel into an existing axes instead of writing a file. In that mode
    nothing is saved and the function returns the legend handles, so a multi-panel composer
    (``manuscript_figures.py``) can place one shared legend. ``out_path``/``dpi`` are ignored.

    ``metric`` swaps the y axis from the baseline-normalized composite to a single raw metric
    (e.g. ``"balanced_accuracy"``), keeping the layout identical so the two variants are directly
    comparable. The composite needs no chance reference because zero IS chance by construction;
    a raw metric does not, so the chance model's mean is drawn as a dashed line instead. Which
    variant the manuscript uses is an open decision — raw balanced accuracy is defensible without
    argument, the composite has to be defended in Methods.

    Each category gets one slot on the x axis. Categories that were run in both prediction modes
    (scLLMs, LLMs, coding agents) are split into adjacent blind and unblind sub-columns, and every
    model that exists in both modes has its two points joined by a thin line, so the per-model
    effect of handing over the candidate label list is readable directly. Blind-only categories --
    the classical baselines, which cannot consume a label list — occupy a single sub-column and
    are drawn ONCE rather than duplicated.

    Within each sub-column the points are grouped into the method tiers defined by
    ``CATEGORY_TIERS`` (reasoning/weight availability for LLMs, generative vs. embedding retrieval
    for scLLMs, marker-set vs. reference-based for the classical baselines), each with its own
    marker shape and a thin black within-tier mean on top of the sub-column's category mean.

    A model keeps the SAME x-offset in both of its sub-columns, so the connecting lines stay
    parallel and never cross — that ordering is computed once per category from the union of the
    blind and unblind members.

    ``show_tiers=False`` renders the tier-free variant: every model is a plain circle, there is no
    within-tier grouping, offset gap or within-tier mean line, and the legend drops the marker
    vocabulary. Everything else -- categories, paired sub-columns, connectors, category means --
    is identical, so the two variants are read the same way and differ only in whether the method
    sub-structure is asserted.

    ``title`` adds a figure-level title (used by the per-dataset supplementary figures so they are
    self-identifying); the aggregated-over-datasets figures pass ``None`` (no title).
    """
    chance = None
    if metric is None:
        per_model = _composite_per_model(df)
        ylabel = "Composite score\n(bal. acc. + Lin. sim., chance-normalized)"
    else:
        if metric not in df.columns or "model" not in df.columns:
            return False
        per_model = pd.to_numeric(df[metric], errors="coerce").groupby(df["model"]).mean()
        ylabel = pretty(metric)
        # Read the chance level BEFORE the simple baselines are filtered out below: on a raw
        # metric the axis floor no longer marks chance, so it has to be drawn.
        if COMPOSITE_BASELINE_MODEL in per_model.index:
            value = float(per_model.loc[COMPOSITE_BASELINE_MODEL])
            chance = value if np.isfinite(value) else None
    if per_model.empty:
        return False

    cat_palette = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    cat_color = {key: cat_palette[i % len(cat_palette)]
                 for i, (key, _) in enumerate(COMPOSITE_CATEGORIES)}

    # (category, model_base) -> {mode: score}. Keyed on the mode-stripped base so the blind and
    # unblind variants of one model collapse to a single entry with two values.
    scores = {}
    for model, val in per_model.items():
        cat = categorize_model(model)
        if cat == "simple" or not np.isfinite(val):
            continue
        scores.setdefault((cat, model_base(model)), {})[derive_mode(model)] = float(val)

    present = []
    for key, lbl in COMPOSITE_CATEGORIES:
        bases = sorted(b for (c, b) in scores if c == key)
        if bases:
            present.append((key, lbl, bases))
    if not present:
        return False

    MODE_SHIFT = 0.25   # sub-column centre offset from the category centre
    SPLIT_SQUEEZE = 0.6  # point spread is tightened when a category splits into two sub-columns

    # Crop the empty band above the highest point instead of always drawing to 1.0. The floor
    # stays at 0 because that IS the random-stratified chance level by construction; only the
    # unused headroom is removed.
    ymax = min(1.0, max(v for md in scores.values() for v in md.values()) + 0.06)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(2.6 * len(present) + 1.6, 4.3))
    else:
        fig = ax.figure
    drawn_tiers, drew_link = set(), False
    tick_pos, tick_lab = [], []

    for i, (key, _lbl, bases) in enumerate(present):
        x0 = float(i)
        modes = [m for m in ("blind", "unblind") if any(m in scores[(key, b)] for b in bases)]
        split = len(modes) > 1
        # One shared ordering/offset per category, from the union of both modes.
        ordered, offsets, segments = _category_layout(key, bases, use_tiers=show_tiers)
        offs = offsets * (SPLIT_SQUEEZE if split else 1.0)
        xof = {m: x0 + (MODE_SHIFT if m == "unblind" else -MODE_SHIFT) if split else x0
               for m in modes}

        # Connectors first so points sit on top of them.
        for j, b in enumerate(ordered):
            sc = scores[(key, b)]
            if len(modes) > 1 and all(m in sc for m in modes):
                ax.plot([xof[modes[0]] + offs[j], xof[modes[1]] + offs[j]],
                        [sc[modes[0]], sc[modes[1]]],
                        color="0.62", linewidth=0.7, zorder=2)
                drew_link = True

        for mode in modes:
            xm = xof[mode]
            for tkey, sl, marker, msize in segments:
                idx = [j for j in range(sl.start, sl.stop) if mode in scores[(key, ordered[j])]]
                if not idx:
                    continue
                ax.scatter([xm + offs[j] for j in idx],
                           [scores[(key, ordered[j])][mode] for j in idx],
                           s=msize, marker=marker, color=cat_color[key],
                           edgecolor="white", linewidth=0.5, zorder=4)
                if tkey is not None:
                    drawn_tiers.add((key, tkey))

            # Mode ticks only where there is a contrast to label. A single-mode category (the
            # blind-only classical baselines) gets just its category name.
            if split:
                tick_pos.append(xm)
                tick_lab.append(mode)

    # Sub-column ticks carry the mode; the category name goes underneath, centred on its slot.
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=10)
    ax.set_xlim(-0.62, len(present) - 0.38)
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (_key, lbl, _bases) in enumerate(present):
        ax.text(float(i), -0.13, lbl, transform=blended, ha="center", va="top", fontsize=14)

    ax.set_ylim(0, ymax)
    # On the composite, y=0 IS the random-stratified (chance) level by construction, so no
    # reference line is drawn — it would sit on the axis. On a raw metric it has to be drawn.
    if chance is not None:
        ax.axhline(chance, linestyle="--", linewidth=0.9, color="0.55", zorder=1)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)

    # Legend outside right: the connector, then one entry per tier marker actually drawn, grouped
    # under a per-category header in that category's color.
    def _blank(label):
        return plt.Line2D([0], [0], linestyle="none", marker="", label=label)

    handles = []
    if chance is not None:
        handles.append(plt.Line2D([0], [0], color="0.55", linewidth=0.9, linestyle="--",
                                  label="chance (random, stratified)"))
    if drew_link:
        handles.append(plt.Line2D([0], [0], color="0.62", linewidth=0.7,
                                  label="same model, blind–unblind"))
    for key, clabel in COMPOSITE_CATEGORIES:
        entries = [(tk, tl, mk) for tk, tl, mk, _s in CATEGORY_TIERS.get(key, [])
                   if (key, tk) in drawn_tiers]
        unassigned = (key, "unclassified") in drawn_tiers
        if not entries and not unassigned:
            continue
        handles.append(_blank(""))
        handles.append(_blank(f"{clabel}:"))
        for _tk, tlabel, marker in entries:
            handles.append(plt.Line2D([0], [0], linestyle="none", marker=marker,
                                      color=cat_color[key],
                                      markersize=12 if marker == "*" else 7, label=tlabel))
        if unassigned:
            handles.append(plt.Line2D([0], [0], linestyle="none",
                                      marker=UNCLASSIFIED_TIER_STYLE[0], color=cat_color[key],
                                      markersize=6, label="tier unassigned"))
    if not own_fig:
        return handles   # composer owns the figure, the legend and the file
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.87, 0.5),
               frameon=False, fontsize=9, labelspacing=0.32, handletextpad=0.6)

    # Caption intentionally omitted; optional title only for the per-dataset supplementary figures.
    if title:
        fig.suptitle(title, y=1.0, fontsize=17)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# Category colors + legend labels for the per-method ranked bar figure. The first four keys keep
# the SAME palette indices as plot_composite_category (so the two composite figures are visually
# consistent); "simple" (random / cell baselines) is added as the next color since the ranked bar
# shows every method individually, baselines included.
_RANKED_CATEGORY_ORDER = [k for k, _ in COMPOSITE_CATEGORIES] + ["simple"]
# Labels identical to plot_composite_category's COMPOSITE_CATEGORIES display names so the two
# composite figures read the same; "simple" has no composite counterpart (drawn as reference lines
# there), so it keeps its own label here.
_RANKED_CATEGORY_LABELS = {
    "LLM": "LLMs", "coding_agent": "coding agents", "scLLM": "scLLMs",
    "classical": "baseline", "simple": "simple baseline",
}


def _ranked_category_colors():
    cat_palette = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    return {key: cat_palette[i % len(cat_palette)] for i, key in enumerate(_RANKED_CATEGORY_ORDER)}


def _prose_model_label(model):
    """Model name with the pipeline's regime suffix rewritten in the manuscript's vocabulary.

    The classical and chance baselines lose the suffix entirely: their ``_blind`` tag is a naming
    artifact and the regime axis does not apply to them.
    """
    base = model_base(model)
    if categorize_model(model) in ("classical", "simple"):
        return base
    return {"unblind": f"{base} (constrained)",
            "blind": f"{base} (unconstrained)"}.get(derive_mode(model), base)


def plot_composite_ranked_bars(df, out_path, dpi, title=None, metric=None,
                               include_simple=None, prose_regimes=False):
    """Per-method ranking: one bar per model, sorted best-first, each bar (and its x-axis name
    label) colored by the model's category.

    Instead of collapsing each category to points plus a mean line, this shows EVERY method
    individually, with the two prediction modes as separate bars.

    ``metric`` swaps the y axis from the baseline-normalized composite to a single raw metric
    (e.g. ``"balanced_accuracy"``). Two consequences, both handled here. On the composite the
    chance level is exactly 0 by construction, so it needs no marking; on a raw metric it does,
    and it is drawn as a dashed red reference line rather than as bars. And the y-axis label stops
    being able to say "0 = chance", so it names the metric plainly.

    The chance models occupy three bars (random-stratified, random-uniform, cell-baseline) that
    say the same thing as one line, and sorting them into the ranking implies they are methods
    competing with the rest. ``include_simple=True`` restores them as bars if that is wanted.

    ``prose_regimes`` writes the manuscript's ``constrained``/``unconstrained`` in the tick labels
    and the mode legend instead of the pipeline's ``unblind``/``blind``.

    Returns False if the requested metric is absent.
    """
    if metric is None:
        per_model = _composite_per_model(df)
        ylabel = "Composite score\n(bal. acc. + Lin. sim. — 0 = random-stratified chance)"
    else:
        if metric not in df.columns or "model" not in df.columns:
            return False
        per_model = pd.to_numeric(df[metric], errors="coerce").groupby(df["model"]).mean()
        ylabel = pretty(metric)
    if per_model.empty:
        return False
    if include_simple is None:
        include_simple = False

    # Read the chance level before the simple baselines are dropped below.
    chance = None
    if metric is not None and COMPOSITE_BASELINE_MODEL in per_model.index:
        value = float(per_model.loc[COMPOSITE_BASELINE_MODEL])
        chance = value if np.isfinite(value) else None

    # Drop models with no score (NaN under the coverage gate, e.g. not-yet-reviewed marker
    # baselines). The simple chance baselines are kept only when they carry information.
    ranked = per_model.dropna()
    if not include_simple:
        ranked = ranked[[categorize_model(m) != "simple" for m in ranked.index]]
    ranked = ranked.sort_values(ascending=False)
    if ranked.empty:
        return False
    models = list(ranked.index)
    cats = [categorize_model(m) for m in models]
    cat_color = _ranked_category_colors()
    colors = [cat_color.get(c, cat_color["LLM"]) for c in cats]

    n = len(models)
    fig_w = max(8.0, 0.34 * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    x = np.arange(n)
    bars = ax.bar(x, ranked.values, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    # This figure shows blind AND unblind models together; mark the blind bars with black diagonal
    # hatching so the two prediction modes are distinguishable on top of the category colors.
    for bar, m in zip(bars, models):
        if derive_mode(m) == "blind":
            bar.set_hatch("///")
            bar.set_edgecolor("black")
            bar.set_linewidth(0.5)

    # Chance reference. On the composite this is omitted: the metric is normalized AGAINST
    # random_stratified, so it is exactly 0 by construction and a line there would sit on the x
    # axis and read as a rendering artifact. The y-axis label states it instead. On a raw metric
    # the level is real and is drawn as a dashed red line across the ranking, which says what
    # three separate chance bars would while keeping them out of the sort order.
    if chance is not None:
        ax.axhline(chance, color="tab:red", linestyle="--", linewidth=1.2, zorder=5)

    # Value labels above each bar (small, rotated) so the rank order is readable at a glance.
    for xi, v in zip(x, ranked.values):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, rotation=90, color="0.3")

    ax.set_xticks(x)
    tick_names = [_prose_model_label(m) for m in models] if prose_regimes else models
    ax.set_xticklabels(tick_names, rotation=90, ha="center", fontsize=8)
    # Color each model-name tick label by its category (matches the bar).
    for tick, c in zip(ax.get_xticklabels(), cats):
        tick.set_color(cat_color.get(c, cat_color["LLM"]))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, min(1.02, float(np.nanmax(ranked.values)) + 0.10))
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)

    # Legend: one swatch per category actually present, in canonical order.
    present_cats = [k for k in _RANKED_CATEGORY_ORDER if k in set(cats)]
    handles = [mpatches.Patch(color=cat_color[k], label=_RANKED_CATEGORY_LABELS[k])
               for k in present_cats]
    if chance is not None:
        handles.append(plt.Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.2,
                                  label="chance (random, stratified)"))
    leg_cat = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=11,
                        title="model category", title_fontsize=11)
    ax.add_artist(leg_cat)
    # Second legend: prediction mode (solid = unblind, black diagonals = blind).
    _u, _c = ("constrained", "unconstrained") if prose_regimes else ("unblind", "blind")
    mode_handles = [mpatches.Patch(facecolor="0.7", edgecolor="black", label=_u),
                    mpatches.Patch(facecolor="0.7", edgecolor="black", hatch="///", label=_c)]
    ax.legend(handles=mode_handles, loc="upper right", bbox_to_anchor=(1.0, 0.60),
              frameon=False, fontsize=10, title="prediction mode", title_fontsize=10)

    if title:
        fig.suptitle(title, y=1.0, fontsize=17)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


DATE_COLUMNS = ("artifact_available_date", "first_public_date")


def load_model_dates(path, column=DATE_COLUMNS[0]):
    """Load ``model_base -> date`` from the publication-dates CSV.

    Defaults to ``artifact_available_date``: the date the RUNNABLE ARTIFACT first became
    obtainable. That is the only column whose meaning is the same for every model kind -- vendor
    release for hosted LLMs, weight-file publication for the domain-adapted models, first
    installable version for the classical tools -- which is what makes the x-axis a fair
    comparison between them.

    ``first_public_date`` (release for LLMs, first preprint for methods) is kept as an alternative
    but is NOT the default, because it silently mixes two different events: an LLM sits at the date
    you could run it while a method sits at the date its paper appeared. That mismatch is large and
    systematic, not noise -- the Cell2Sentence checkpoint postdates its origin preprint by ~2 years
    (2022-09-19 -> 2024-09-03), the C2S-Scale weights by ~6 months, and the scType R code by ~21
    months. It also runs the other way for four models whose code shipped before their preprint.
    Switching columns does not change the paper's conclusions (LLM rho is identical; scLLM stays
    non-significant either way), but the artifact axis is the defensible one.

    ``journal_date`` is deliberately not offered: four domain-adapted models have no journal
    version at all, so it would silently drop them.

    Returns an empty Series if the file or column is missing, so callers can skip the figure.
    """
    path = Path(path)
    if not path.exists():
        return pd.Series(dtype="datetime64[ns]")
    try:
        d = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - unreadable file should not kill the whole run
        warnings.warn(f"Could not read model dates from {path}: {exc}")
        return pd.Series(dtype="datetime64[ns]")
    if "model_base" not in d.columns:
        warnings.warn(f"{path} lacks a 'model_base' column; skipping date plot.")
        return pd.Series(dtype="datetime64[ns]")
    col = column if column in d.columns else next((c for c in DATE_COLUMNS if c in d.columns), None)
    if col is None:
        warnings.warn(f"{path} has none of {DATE_COLUMNS}; skipping date plot.")
        return pd.Series(dtype="datetime64[ns]")
    if col != column:
        warnings.warn(f"{path} lacks '{column}'; falling back to '{col}'.")
    s = pd.Series(pd.to_datetime(d[col], errors="coerce").values, index=d["model_base"].astype(str))
    return s.dropna()


def _tier_marker(model: str):
    """(marker, size) for a model, from the same vocabulary as ``plot_composite_category``.

    Falls back to the unclassified style for a model whose tier is not in
    ``CATEGORY_TIER_MEMBERS`` and to a plain circle for a category without tiers.
    """
    cat, tkey = categorize_model(model), model_tier(model)
    if tkey is None:
        return ("o", 46)
    if tkey == "unclassified":
        return UNCLASSIFIED_TIER_STYLE
    for key, _lbl, marker, size in CATEGORY_TIERS.get(cat, []):
        if key == tkey:
            return (marker, size)
    return UNCLASSIFIED_TIER_STYLE


def _place_point_labels(ax, labels, obstacles, fontsize, leader_color="0.7"):
    """Annotate scatter points without letting the text land on a marker or another label.

    ``labels`` is ``[(text, x_data, y_data, color)]`` and ``obstacles`` is
    ``[(x_data, y_data, radius_points)]`` for every drawn marker. Each label is tried at a ring
    of candidate offsets and keeps the first that overlaps nothing; if every candidate collides,
    the least-overlapping one is kept, so a label is never silently dropped. Placement uses real
    rendered bounding boxes, which is why the canvas is drawn once up front -- and why the
    obstacles arrive in DATA coordinates and are transformed after that draw, when the axes
    position is final.

    A faint leader line is attached whenever the text ends up far enough from its point that the
    association would otherwise be ambiguous.
    """
    if not labels:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    obstacle_boxes = []
    for x_data, y_data, r_pts in obstacles:
        xd, yd = ax.transData.transform((x_data, y_data))
        r = r_pts * fig.dpi / 72.0
        obstacle_boxes.append(mtransforms.Bbox.from_bounds(xd - r, yd - r, 2 * r, 2 * r))
    # Ring of offsets in points, nearest first, right/left before up/down so labels track the
    # reading direction before they start stacking vertically.
    candidates = [(9, 0), (-9, 0), (9, 8), (-9, 8), (9, -8), (-9, -8), (0, 11), (0, -11),
                  (18, 0), (-18, 0), (18, 14), (-18, 14), (18, -14), (-18, -14),
                  (0, 20), (0, -20), (30, 0), (-30, 0)]

    axes_box = ax.get_window_extent(renderer=renderer)

    def _cost(box, boxes):
        total = 0.0
        for other in boxes:
            inter = mtransforms.Bbox.intersection(box, other)
            if inter is not None and inter.width > 0 and inter.height > 0:
                total += inter.width * inter.height
        # Spilling outside the axes is a collision too: the label would sit on the tick labels or
        # be clipped away entirely. Charged by the area that falls outside.
        inside = mtransforms.Bbox.intersection(box, axes_box)
        area = box.width * box.height
        total += area - (inside.width * inside.height if inside is not None else 0.0)
        return total

    placed = []
    for text, px, py, color in sorted(labels, key=lambda t: -t[2]):
        best, best_cost, best_off = None, None, None
        for dx, dy in candidates:
            ann = ax.annotate(text, xy=(px, py), xytext=(dx, dy), textcoords="offset points",
                              ha="left" if dx >= 0 else "right",
                              va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                              fontsize=fontsize, color=color, zorder=6)
            box = ann.get_window_extent(renderer=renderer)
            cost = _cost(box, obstacle_boxes + placed)
            if cost == 0:
                # Clear any partial best already on the axes, or it stays as a duplicate label.
                if best is not None:
                    best.remove()
                best, best_off = ann, (dx, dy)
                break
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best.remove()
                best, best_cost, best_off = ann, cost, (dx, dy)
            else:
                ann.remove()
        if best is None:
            continue
        placed.append(best.get_window_extent(renderer=renderer))
        if abs(best_off[0]) > 12 or abs(best_off[1]) > 12:
            # Displaced far enough that the text-to-point association needs a leader.
            ax.annotate("", xy=(px, py), xytext=best_off, textcoords="offset points",
                        xycoords=best.xycoords,
                        arrowprops=dict(arrowstyle="-", linewidth=0.4, color=leader_color,
                                        shrinkA=0, shrinkB=2), zorder=3)


def plot_metric_disagreement(df, out_path, dpi, title=None,
                             x_metric="balanced_accuracy", y_metric="macro_f1",
                             ax=None, only_regime=None, annotate="tier_best",
                             baselines_in_both=True, cross_ref=True, regime_labels=None):
    """Companion panel to ``plot_composite_category``: do the two correctness metrics rank models
    the same way, and does that depend on the information regime?

    The composite on the ``plot_composite_category`` axis averages balanced accuracy and Lin
    similarity, so macro F1 is absent from that figure by construction. It is also the metric on
    which the family ordering reverses, which is why it needs its own panel rather than a sentence.

    One point per model per regime, split into two panels (unconstrained | constrained) sharing
    axes. EVERY model in ``df`` is drawn -- no selection -- with colors by category and markers by
    method tier, matching ``plot_composite_category`` so the panels read as one figure. The
    classical baselines have a single score and are drawn in the unconstrained panel only, since
    that is the regime they are compared under; the caption must say the regime axis does not
    apply to them.

    The identity line is the reference that carries the reading. A model sitting far below it
    predicts many more distinct labels than the ground truth has, which macro F1 charges for (it
    averages over predicted classes) and balanced accuracy does not (it is recall-based). Points
    near the line are models whose label vocabulary matches the target label set.

    Spearman rank correlation between the two metrics is annotated per panel over the models in
    that panel. Returns False if either metric is missing.

    ``ax`` + ``only_regime`` draw a SINGLE regime panel into an existing axes and return the
    legend handles instead of writing a file, so a multi-panel composer can place one shared
    legend. ``only_regime`` is ``"blind"`` or ``"unblind"``; the other regime's Spearman is then
    folded into the panel title so the contrast survives in the compact form.
    """
    if not all(c in df.columns for c in (x_metric, y_metric)) or "model" not in df.columns:
        return False

    per = (df.groupby("model")[[x_metric, y_metric]].mean().dropna())
    per = per[[categorize_model(m) != "simple" for m in per.index]]
    if per.empty:
        return False

    cat_palette = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    cat_color = {key: cat_palette[i % len(cat_palette)]
                 for i, (key, _) in enumerate(COMPOSITE_CATEGORIES)}

    # The classical baselines carry ONE score: the regime axis does not apply to them, because
    # their label space is fixed by a marker database or a reference rather than by anything we
    # supply. That is precisely what makes them a legitimate comparator under either setting, so
    # they are DRAWN in both panels (``baselines_in_both``) as the reference set.
    #
    # They are in NEITHER panel's rank correlation. The two coefficients exist to compare ONE
    # population across the two regimes, so both must cover the same models -- the language
    # models, which are the only ones that have both regimes. Putting the baselines in the
    # unconstrained coefficient alone drags it from 0.91 to 0.83 and manufactures a regime effect
    # out of a population change; they correlate at 1.00 among themselves (n = 4), so they are
    # not noise, they are a different population. Putting them in both would be worse: their
    # single score would then be read as a constrained result. Drawing set and statistics set are
    # therefore kept apart, and the caption must say which is which.
    panels = [("blind", "unconstrained", {"blind", "baseline"}),
              ("unblind", "constrained", {"unblind"})]
    members = {key: [m for m in per.index
                     if derive_mode(m) in modes and categorize_model(m) != "classical"]
               for key, _lbl, modes in panels}
    # Selected by CATEGORY, not by derive_mode: scsa_blind / sctype_blind / singler_blind /
    # celltypist_blind carry a `_blind` suffix that is a pipeline naming artifact, not a regime
    # assignment, so derive_mode reports them as "blind" and would never surface them here.
    baselines = [m for m in per.index if categorize_model(m) == "classical"]
    drawn_members = {key: (mods + [b for b in baselines if b not in mods]
                           if baselines_in_both else mods)
                     for key, mods in members.items()}
    if not any(members.values()):
        return False

    # One upper limit for both axes and both panels: the identity line only reads as a reference
    # if x and y are on the same scale, and the two regimes only compare if their panels are.
    # Padding is tight because the constrained panel reaches close to the maximum anyway.
    hi = max(0.02 + float(per[[x_metric, y_metric]].to_numpy().max()), 0.2)

    # Spearman per regime, computed up front so a single-panel render can quote the other one.
    rho_by_key = {}
    for key, _lbl, _modes in panels:
        mods = members[key]
        rho_by_key[key] = _spearman(per.loc[mods, x_metric].to_numpy(),
                                    per.loc[mods, y_metric].to_numpy()) \
            if len(mods) >= 3 else (float("nan"), float("nan"))

    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True)
        draw_panels = panels
    else:
        fig, axes = ax.figure, [ax]
        draw_panels = [p for p in panels if p[0] == (only_regime or "blind")]
    drawn_tiers = set()

    for ax, (key, lbl, _modes) in zip(axes, draw_panels):
        ax.plot([0, hi], [0, hi], color="0.75", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        mods = members[key]            # models entering this panel's rank correlation
        shown = drawn_members[key]     # models drawn, baselines included in both panels
        marker_px = []   # (x_data, y_data, radius_points) for label collision avoidance
        for mdl in shown:
            cat = categorize_model(mdl)
            marker, size = _tier_marker(mdl)
            mx, my = float(per.loc[mdl, x_metric]), float(per.loc[mdl, y_metric])
            ax.scatter(mx, my, s=size, marker=marker,
                       color=cat_color.get(cat, "0.5"), edgecolor="white", linewidth=0.5, zorder=4)
            # s is a marker AREA in points^2, so the half-width in points is sqrt(s)/2. Kept in
            # data coordinates and converted once the layout is final (see _place_point_labels).
            marker_px.append((mx, my, np.sqrt(size) / 2.0))
            tkey = model_tier(mdl)
            if tkey is not None:
                drawn_tiers.add((cat, tkey))
        # Label the best model per method tier, ranked on the x metric (balanced accuracy, the
        # primary metric). One name per tier is a symmetric rule that treats every family alike
        # and is stated in the caption; naming "the models the claim is about" would privilege
        # one family and read as selection. Every model is drawn either way -- only the naming
        # is restricted. `annotate="none"` turns labels off, `"all"` names every model.
        if annotate != "none":
            if annotate == "all":
                named = list(shown)
            else:
                best = {}
                for mdl in shown:
                    tkey = (categorize_model(mdl), model_tier(mdl))
                    if tkey not in best or per.loc[mdl, x_metric] > per.loc[best[tkey], x_metric]:
                        best[tkey] = mdl
                named = list(best.values())
            _place_point_labels(
                ax,
                [(model_base(mdl), float(per.loc[mdl, x_metric]), float(per.loc[mdl, y_metric]),
                  cat_color.get(categorize_model(mdl), "0.4")) for mdl in named],
                marker_px, fontsize=7.5)
        # ``regime_labels`` lets a composer title the panel with the same key its other panels
        # use on their axes, so the reader does not have to hold two vocabularies at once.
        lbl = (regime_labels or {}).get(key, lbl)
        rho, p = rho_by_key[key]
        if not np.isfinite(rho):
            head = f"{lbl}   (n = {len(mods)} models)"
        elif own_fig:
            head = f"{lbl}   (n = {len(mods)} models, Spearman $\\rho$ = {rho:.2f}, {_fmt_p(p)})"
        else:
            # Composed panels are half as wide, so the full title would run into its neighbour.
            # The p-value moves to the figure caption.
            head = f"{lbl}   $\\rho$ = {rho:.2f}  (n = {len(mods)})"
        if not own_fig and cross_ref:
            # Compact single-panel form: keep the contrast by quoting the other regime's rho.
            # Pointless when the composer draws both panels, hence `cross_ref`.
            other = next((k for k, _l, _m in panels if k != key), None)
            orho = rho_by_key.get(other, (float("nan"), float("nan")))[0]
            if other and np.isfinite(orho):
                olbl = next(l for k, l, _m in panels if k == other)
                head += f"\n{olbl}: $\\rho$ = {orho:.2f}"
        ax.set_title(head, fontsize=11)
        ax.set_xlabel(pretty(x_metric), fontsize=11)
        ax.grid(linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

    axes[0].set_ylabel(pretty(y_metric), fontsize=11)
    axes[0].set_xlim(0, hi)
    axes[0].set_ylim(0, hi)

    handles = [plt.Line2D([0], [0], color="0.75", linewidth=0.8, linestyle="--",
                          label="identity (equal on both metrics)")]

    def _blank(label):
        return plt.Line2D([0], [0], linestyle="none", marker="", label=label)

    for key, clabel in COMPOSITE_CATEGORIES:
        entries = [(tk, tl, mk) for tk, tl, mk, _s in CATEGORY_TIERS.get(key, [])
                   if (key, tk) in drawn_tiers]
        if not entries:
            continue
        handles.append(_blank(""))
        handles.append(_blank(f"{clabel}:"))
        for _tk, tlabel, marker in entries:
            handles.append(plt.Line2D([0], [0], linestyle="none", marker=marker,
                                      color=cat_color[key],
                                      markersize=12 if marker == "*" else 7, label=tlabel))
    if not own_fig:
        return handles   # composer owns the figure, the legend and the file
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.9, 0.5), frameon=False,
               fontsize=9, labelspacing=0.32, handletextpad=0.6)
    if title:
        fig.suptitle(title, y=1.02, fontsize=15)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# Per-class recall table, produced out-of-band because computing it needs the CL ontology (to
# canonicalize true and predicted labels the way evaluate.py does before scoring) and the cached
# per-cell mapped predictions. Soft dependency: the panels that use it are skipped with a warning
# when the file is absent, exactly like the model-dates CSV.
DEFAULT_PER_CLASS_CSV = REPO_ROOT / "cache" / "per_class_recall_1000_stratified.csv"

# Which (family, regime) series a family panel draws. The classical baselines carry ONE score, so
# they appear once with no regime; the language families appear twice.
FAMILY_SERIES = [("classical", None, "-", "D"),
                 ("scLLM", "unblind", "-", "o"), ("scLLM", "blind", "--", "o"),
                 ("LLM", "unblind", "-", "^"), ("LLM", "blind", "--", "^")]
_REGIME_STYLE_LABEL = {"unblind": "constrained", "blind": "unconstrained", None: "single score"}


def _annotate_families(df):
    """Add model_base / category / mode and drop excluded and chance models."""
    d = df.copy()
    d["model_base"] = d["model"].map(model_base)
    d["category"] = d["model"].map(categorize_model)
    d["mode"] = d["model"].map(derive_mode)
    d = d[~d["model_base"].isin(EXCLUDED_BASES)]
    return d[d["category"] != "simple"]


def load_per_class_recall(path=DEFAULT_PER_CLASS_CSV):
    """Per-class recall and prevalence per (model, dataset). Empty frame if the table is absent."""
    path = Path(path)
    if not path.exists():
        warnings.warn(f"No per-class recall table at {path}; skipping the prevalence panel.")
        return pd.DataFrame()
    try:
        d = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - unreadable file must not kill the run
        warnings.warn(f"Could not read {path}: {exc}")
        return pd.DataFrame()
    required = {"model", "dataset", "true_label", "prevalence", "recall"}
    if not required <= set(d.columns):
        warnings.warn(f"{path} lacks {sorted(required - set(d.columns))}; skipping the panel.")
        return pd.DataFrame()
    return _annotate_families(d)


def _family_panel(ax, draw_one, cat_color):
    """Run ``draw_one(category, mode, color, linestyle, marker)`` over the family series present."""
    drawn = []
    for key, mode, linestyle, marker in FAMILY_SERIES:
        if draw_one(key, mode, cat_color[key], linestyle, marker):
            drawn.append((key, mode, linestyle, marker))
    return drawn


def _family_handles(drawn, cat_color):
    """Legend entries for the family series a panel drew, split into two keys.

    A family key (color + marker) and a LINE-STYLE key (solid = constrained, dashed =
    unconstrained). Encoding the regime in the family labels instead would repeat every family
    twice and, worse, leave the dash pattern to be read off a short legend handle, where it is
    effectively invisible — the reason this was split.
    """
    labels = {k: lbl for k, lbl in COMPOSITE_CATEGORIES}
    seen, family = set(), []
    for key, _mode, _linestyle, marker in drawn:
        if key in seen:
            continue
        seen.add(key)
        family.append(plt.Line2D([0], [0], color=cat_color[key], linestyle="-", marker=marker,
                                 markersize=6, linewidth=1.4, label=labels.get(key, key)))
    modes = {mode for _k, mode, _ls, _m in drawn if mode is not None}
    style = []
    if len(modes) > 1:
        style = [plt.Line2D([0], [0], color="0.35", linestyle="-", linewidth=1.4,
                            label="constrained"),
                 plt.Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.4,
                            label="unconstrained")]
        if any(mode is None for _k, mode, _ls, _m in drawn):
            style.append(plt.Line2D([0], [0], linestyle="none", marker="",
                                    label="(baselines: one score)"))
    return family + style


def _tier_handles(drawn, cat_color):
    """One legend entry per method tier, regardless of how many regimes it was drawn at.

    The line-style key carries the regime, so repeating every tier twice would only pad the
    legend. ``drawn`` is ``[(category, tier, mode, linestyle, marker)]``.
    """
    tier_label = {(k, tk): tl for k, _l in COMPOSITE_CATEGORIES
                  for tk, tl, _m, _s in CATEGORY_TIERS.get(k, [])}
    seen, handles = set(), []
    for key, tier_key, _mode, _linestyle, marker in drawn:
        if (key, tier_key) in seen:
            continue
        seen.add((key, tier_key))
        handles.append(plt.Line2D([0], [0], color=cat_color[key], linestyle="-", marker=marker,
                                  markersize=_marker_size(marker), linewidth=1.4,
                                  label=tier_label[(key, tier_key)]))
    return handles


def _marker_size(marker, base=6.0):
    """Visually equal marker sizes across the tier vocabulary.

    Matplotlib sizes markers by their bounding box, so a five-pointed star at a given
    ``markersize`` reads noticeably smaller than a filled square or diamond. The star is scaled up
    to compensate; everything else keeps the base size.
    """
    return base * 1.6 if marker == "*" else base


def _tier_series(regimes):
    """(category, tier, mode, linestyle, marker) for every method tier, at the given regimes.

    Family pooling hides real structure in the prevalence panel: the embedding scLLMs are the
    second-strongest group on rare types when constrained and their curve is flat in prevalence,
    both of which vanish when they are averaged with the generative scLLMs. Tiers reuse the marker
    vocabulary already established in Figure 1B, so no new legend grammar is needed.
    """
    out = []
    for key, _lbl in COMPOSITE_CATEGORIES:
        for tier_key, _tlabel, marker, _size in CATEGORY_TIERS.get(key, []):
            # Baselines carry one score; the regime axis does not apply to them.
            modes = [None] if key == "classical" else list(regimes)
            for mode in modes:
                out.append((key, tier_key, mode, "-" if mode != "blind" else "--", marker))
    return out


def plot_recall_by_prevalence(rec, out_path, dpi, ax=None, band_edges=(0.03, 0.10), title=None,
                              level="family", regimes=("unblind", "blind")):
    """Mean per-class recall against how common the cell type is, by model family and regime.

    Balanced accuracy weights every class equally but hides WHERE a family's accuracy comes from.
    This panel splits it: if a family's advantage were only on abundant types, the curves would
    converge at the left.

    Binning rather than a scatter, because stratified subsetting with a 20-cell floor quantizes
    prevalence hard — at 1 000 cells the median class sits exactly at the 2 % floor, so a
    continuous x axis would show a spike and a tail rather than a trend. The band edges are stated
    in the tick labels together with the number of classes in each, since the bands are uneven by
    construction and a reader must be able to see that.

    ``rec`` is the frame from ``load_per_class_recall``. Returns False if it is empty.
    """
    if rec is None or len(rec) == 0:
        return False
    edges = [0.0, *band_edges, 1.01]
    labels = ([f"≤{band_edges[0] * 100:.0f}%"]
              + [f"{a * 100:.0f}–{b * 100:.0f}%"
                 for a, b in zip(band_edges[:-1], band_edges[1:])]
              + [f">{band_edges[-1] * 100:.0f}%"])
    rec = rec.copy()
    rec["band"] = pd.cut(rec["prevalence"], bins=edges, labels=labels)
    n_classes = (rec.drop_duplicates(["dataset", "true_label"])["band"]
                 .value_counts().reindex(labels).fillna(0).astype(int))

    cat_color = _ranked_category_colors()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
    x = np.arange(len(labels), dtype=float)

    def _line(sub, color, linestyle, marker):
        if sub.empty:
            return False
        means = sub.groupby("band", observed=False)["recall"].mean().reindex(labels)
        ax.plot(x, means.to_numpy(dtype=float), linestyle=linestyle, marker=marker,
                color=color, linewidth=1.4, markersize=_marker_size(marker),
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        return True

    if level == "tier":
        rec = rec.copy()
        rec["tier"] = rec["model"].map(model_tier)
        drawn = []
        for key, tier_key, mode, linestyle, marker in _tier_series(regimes):
            sub = rec[(rec["category"] == key) & (rec["tier"] == tier_key)]
            if mode is not None:
                sub = sub[sub["mode"] == mode]
            if _line(sub, cat_color[key], linestyle, marker):
                drawn.append((key, tier_key, mode, linestyle, marker))
    else:
        def draw_one(key, mode, color, linestyle, marker):
            sub = rec[rec["category"] == key]
            if mode is not None:
                sub = sub[sub["mode"] == mode]
            return _line(sub, color, linestyle, marker)

        drawn = _family_panel(ax, draw_one, cat_color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({n_classes[l]})" for l in labels])
    ax.set_xlim(-0.35, len(labels) - 0.65)
    ax.set_ylim(0, None)
    ax.set_xlabel("cell type prevalence (classes per band)")
    ax.set_ylabel("Mean per-class recall")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    if title:
        ax.set_title(title, fontsize=11)
    handles = _tier_handles(drawn, cat_color) if level == "tier" \
        else _family_handles(drawn, cat_color)
    if not own_fig:
        return handles
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False,
               fontsize=9)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_family_by_dataset(df, out_path, dpi, ax=None, metric="balanced_accuracy", title=None,
                           level="family", regimes=("unblind", "blind")):
    """Means per dataset, both regimes: does the ordering repeat on every dataset?

    The aggregated figures average datasets away, so a single reversal on one dataset would be
    invisible there. One point per (group, regime, dataset), joined across datasets so a crossing
    line is immediately visible as an ordering failure.

    ``level="tier"`` resolves the seven method tiers instead of the three families, at the cost of
    twelve lines on four x positions. Worth it when the figure's other panels are tier-resolved,
    since a family-level panel beside them raises the question of which sub-family is doing the
    work.
    """
    if metric not in df.columns:
        return False
    d = _annotate_families(df)
    if d.empty:
        return False
    datasets = sorted(d["dataset"].astype(str).unique())
    x = np.arange(len(datasets), dtype=float)
    cat_color = _ranked_category_colors()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.6, 4.4))

    def _line(sub, color, linestyle, marker):
        if sub.empty:
            return False
        means = (sub.groupby(sub["dataset"].astype(str))[metric].mean().reindex(datasets))
        ax.plot(x, means.to_numpy(dtype=float), linestyle=linestyle, marker=marker, color=color,
                linewidth=1.4, markersize=_marker_size(marker), markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)
        return True

    if level == "tier":
        d = d.copy()
        d["tier"] = d["model"].map(model_tier)
        drawn = []
        for key, tier_key, mode, linestyle, marker in _tier_series(regimes):
            sub = d[(d["category"] == key) & (d["tier"] == tier_key)]
            if mode is not None:
                sub = sub[sub["mode"] == mode]
            if _line(sub, cat_color[key], linestyle, marker):
                drawn.append((key, tier_key, mode, linestyle, marker))
    else:
        def draw_one(key, mode, color, linestyle, marker):
            sub = d[d["category"] == key]
            if mode is not None:
                sub = sub[sub["mode"] == mode]
            return _line(sub, color, linestyle, marker)

        drawn = _family_panel(ax, draw_one, cat_color)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.35, len(datasets) - 0.65)
    ax.set_ylim(0, None)
    ax.set_ylabel(pretty(metric))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    if title:
        ax.set_title(title, fontsize=11)
    handles = _tier_handles(drawn, cat_color) if level == "tier" \
        else _family_handles(drawn, cat_color)
    if not own_fig:
        return handles
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False,
               fontsize=9)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


DEFAULT_CONFUSION_CSV = REPO_ROOT / "cache" / "confusion_pairs_1000_stratified.csv"


def load_confusion_pairs(path=DEFAULT_CONFUSION_CSV):
    """(true label -> predicted label) cell counts per model and dataset, with class prevalence.

    Produced out-of-band for the same reason as the per-class recall table: building it needs the
    CL ontology to canonicalize both label columns the way scoring does. Soft dependency.
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(f"No confusion-pair table at {path}; skipping the rare-type flow figure.")
        return pd.DataFrame()
    d = pd.read_csv(path)
    required = {"model", "dataset", "true_label", "predicted_label", "n_cells", "prevalence"}
    if not required <= set(d.columns):
        warnings.warn(f"{path} lacks {sorted(required - set(d.columns))}; skipping.")
        return pd.DataFrame()
    return d


def _sankey(ax, flows, left_order, right_order, colors, gap_frac=0.02, label_size=7,
            show_left_labels=True):
    """Minimal two-column alluvial: ``flows`` is ``[(src, tgt, value)]``.

    Written here rather than pulled in as a dependency because the plotting environment is
    matplotlib + seaborn only, and the two-column case needs about thirty lines.

    Node heights are proportional to the cells passing through them, ribbons are cubic Beziers,
    and a ribbon whose target equals its source (a correct prediction) is drawn opaque with a dark
    edge so the recovered fraction is visible without reading the labels.
    """
    total = sum(v for _s, _t, v in flows)
    if not total:
        return
    gap = gap_frac * total

    def stack(order, weights):
        pos, y = {}, 0.0
        for key in order:
            h = weights.get(key, 0.0)
            pos[key] = (y, y + h)
            y += h + gap
        span = max(y - gap, 1e-9)
        return {k: (a / span, b / span) for k, (a, b) in pos.items()}

    lw = {k: sum(v for s, _t, v in flows if s == k) for k in left_order}
    rw = {k: sum(v for _s, t, v in flows if t == k) for k in right_order}
    lpos, rpos = stack(left_order, lw), stack(right_order, rw)
    lcur = {k: a for k, (a, _b) in lpos.items()}
    rcur = {k: a for k, (a, _b) in rpos.items()}
    scale = {k: (lpos[k][1] - lpos[k][0]) / max(lw[k], 1e-9) for k in left_order}
    rscale = {k: (rpos[k][1] - rpos[k][0]) / max(rw[k], 1e-9) for k in right_order}

    x0, x1 = 0.0, 1.0
    for src, tgt, val in sorted(flows, key=lambda f: (left_order.index(f[0]), -f[2])):
        h0, h1 = val * scale[src], val * rscale[tgt]
        y0, y1 = lcur[src], rcur[tgt]
        lcur[src] += h0
        rcur[tgt] += h1
        correct = src == tgt
        verts, codes = [], []
        for (ya, yb, forward) in ((y0, y1, True), (y1 + h1, y0 + h0, False)):
            xa, xb = (x0, x1) if forward else (x1, x0)
            if not verts:
                verts.append((xa, ya)); codes.append(mpath.Path.MOVETO)
            else:
                verts.append((xa, ya)); codes.append(mpath.Path.LINETO)
            verts += [((xa + xb) / 2, ya), ((xa + xb) / 2, yb), (xb, yb)]
            codes += [mpath.Path.CURVE4] * 3
        verts.append((x0, y0)); codes.append(mpath.Path.CLOSEPOLY)
        ax.add_patch(mpatches.PathPatch(
            mpath.Path(verts, codes), facecolor=colors.get(src, "0.6"),
            alpha=0.85 if correct else 0.38, linewidth=0.6 if correct else 0.0,
            edgecolor="0.15" if correct else "none", zorder=3 if correct else 2))

    for key in left_order:
        a, b = lpos[key]
        ax.add_patch(mpatches.Rectangle((-0.035, a), 0.03, b - a, color=colors.get(key, "0.6"),
                                        zorder=4))
        if show_left_labels:
            ax.text(-0.05, (a + b) / 2, key, ha="right", va="center", fontsize=label_size)
    for key in right_order:
        a, b = rpos[key]
        hit = key in left_order
        ax.add_patch(mpatches.Rectangle((1.005, a), 0.03, b - a, color="0.35", zorder=4))
        ax.text(1.05, (a + b) / 2, key, ha="left", va="center", fontsize=label_size,
                fontweight="bold" if hit else "normal", color="0.1" if hit else "0.35")
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")


def _flow_key(ax, rare, colors, counts, max_targets, label_size=8):
    """Encoding key plus the rare-type inventory, for the empty cell of a flow grid.

    Two things the panels cannot carry themselves. The ribbon encoding, so the figure is readable
    without the caption; and the cells per type, without which a ribbon's width is unreadable —
    stratified subsetting floors rare types at 20 cells, so a visually wide ribbon can rest on
    very few cells.
    """
    handles = [
        mpatches.Patch(facecolor="0.55", edgecolor="0.15", linewidth=0.6, alpha=0.85,
                       label="prediction matches the true type"),
        mpatches.Patch(facecolor="0.55", edgecolor="none", alpha=0.38,
                       label="prediction differs"),
        plt.Line2D([], [], linestyle="none", marker="", label="ribbon width $\\propto$ cells"),
        plt.Line2D([], [], linestyle="none", marker="",
                   label=f"top {max_targets} predictions per type;\nremainder pooled as \"other\""),
        plt.Line2D([], [], linestyle="none", marker="", label=""),
        plt.Line2D([], [], linestyle="none", marker="", label="cells per type:"),
    ]
    for t in rare:
        n = int(counts.get(t, 0))
        handles.append(mpatches.Patch(facecolor=colors.get(t, "0.6"), edgecolor="none",
                                      label=f"{t} — {n}"))
    # Centered in the empty cell, so the block sits under the panel above it rather than pinned to
    # the figure's left edge. The entries themselves stay left-aligned so the names line up.
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=label_size,
              labelspacing=0.6, handlelength=1.6, handletextpad=0.7, borderaxespad=0.0,
              alignment="left")


def plot_rare_type_flows(conf, out_path, dpi, dataset, models, n_types=5, max_targets=6,
                         prevalence_max=0.03, title=None):
    """Supplementary: for the rarest cell types, WHERE do the cells actually go?

    Panel D of Figure 2 says how often rare types are recovered; it cannot say what they are
    called instead, which is the difference between a near miss inside the right lineage and an
    unrelated call. One alluvial per model: rare true types on the left, predicted labels on the
    right, ribbon width proportional to cells. Ribbons that land on the correct label are opaque
    with a dark edge, so the recovered fraction reads without tracing labels.

    ``models`` is an explicit list — the selection rule belongs in the caption, not in a heuristic
    here. Pass a list of lists to get a grid, which is how the two regimes are shown: constrained
    on the top row, unconstrained beneath, with the same model in one column so the pair reads
    vertically. ``None`` leaves a cell empty, for the baselines, which carry a single score and so
    occupy the top row only.

    Per true type only the ``max_targets`` most frequent predictions are drawn; the rest are
    pooled into "other". Returns False if the table is empty or the dataset has no usable models.
    """
    if conf is None or len(conf) == 0:
        return False
    sub = conf[conf["dataset"].astype(str) == str(dataset)]
    rows = models if models and isinstance(models[0], (list, tuple)) else [models]
    available = set(sub["model"])
    rows = [[m if (m in available) else None for m in row] for row in rows]
    if sub.empty or not any(m for row in rows for m in row):
        warnings.warn(f"plot_rare_type_flows: nothing to draw for {dataset}.")
        return False

    prev = (sub.drop_duplicates(["true_label"]).set_index("true_label")["prevalence"])
    rare = list(prev[prev <= prevalence_max].nsmallest(n_types).index)
    if not rare:
        rare = list(prev.nsmallest(n_types).index)
    palette = sns.color_palette("tab10", n_colors=max(len(rare), 3)).as_hex()
    colors = {t: palette[i % len(palette)] for i, t in enumerate(rare)}
    # Cells per rare type. Identical for every model (they annotate the same cells), so it is read
    # off whichever model is available first.
    first = next(m for row in rows for m in row if m)
    counts = (sub[(sub["model"] == first) & (sub["true_label"].isin(rare))]
              .groupby("true_label")["n_cells"].sum())

    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(6.2 * n_cols, (0.62 * len(rare) + 2.4) * n_rows))
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            model = row[c] if c < len(row) else None
            if model is None:
                ax.axis("off")
                # The empty cell carries the encoding key and the per-type cell counts, so the
                # figure is readable without its caption and a ribbon's width can be judged
                # against the cells behind it.
                if r > 0 and any(rr[c] for rr in rows[:r] if c < len(rr)):
                    _flow_key(ax, rare, colors, counts, max_targets)
                continue
            m = sub[(sub["model"] == model) & (sub["true_label"].isin(rare))]
            flows = []
            for true_label, grp in m.groupby("true_label"):
                grp = grp.sort_values("n_cells", ascending=False)
                keep, rest = grp.head(max_targets), grp.tail(max(len(grp) - max_targets, 0))
                flows += [(true_label, t.predicted_label, float(t.n_cells))
                          for t in keep.itertuples()]
                if len(rest):
                    flows.append((true_label, "other", float(rest["n_cells"].sum())))
            if not flows:
                ax.axis("off")
                continue
            # Order targets by the mean position of their sources, to limit ribbon crossings.
            weight = {}
            for src, tgt, val in flows:
                a, b = weight.get(tgt, (0.0, 0.0))
                weight[tgt] = (a + rare.index(src) * val, b + val)
            right = sorted(weight, key=lambda t: (t == "other", weight[t][0] / weight[t][1]))
            # The left stack is identical in every panel, so only the first column of each row
            # names it; repeating the labels collides with the neighbour's right-hand column.
            _sankey(ax, flows, rare, right, colors, show_left_labels=(c == 0))
            # No regime tag for the classical baselines: their `_blind` suffix is a pipeline
            # naming artifact, and the regime axis does not apply to them at all.
            regime = None if categorize_model(model) == "classical" else \
                {"unblind": "constrained", "blind": "unconstrained"}.get(derive_mode(model))
            ax.set_title(f"{model_base(model)}" + (f"  ({regime})" if regime else ""), fontsize=10)

    if title:
        fig.suptitle(title, y=1.0, fontsize=12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _spearman(x, y):
    """Spearman rho and p-value; p is NaN when SciPy is unavailable (pandas gives rho only)."""
    if len(x) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr  # noqa: PLC0415 - optional dependency
        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float(pd.Series(x).corr(pd.Series(y), method="spearman")), float("nan")


def _fmt_p(p):
    if not np.isfinite(p):
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_date_vs_composite(df, dates, out_path, dpi, title=None):
    """Publication figure: first-public-availability date vs composite score, blind | unblind.

    One point per model shown in the composite figures (the simple chance baselines and
    ``EXCLUDED_BASES`` are already absent from ``df``/the composite), colored by category and
    shaped by method tier so the marker vocabulary matches ``plot_composite_category``.

    Statistics are reported BOTH overall and per category, deliberately. The overall correlation
    across all models is strongly confounded: the general-purpose LLMs are simultaneously the
    newest and the best-scoring entries, so a pooled positive rho largely restates "LLMs are
    recent" rather than showing that methods improve over time. The per-category rho values are
    the ones that speak to a within-family time trend, and they are computed on small n -- they
    are descriptive, not inferential.

    ``dates`` is a Series (model_base -> Timestamp) from ``load_model_dates``. Returns False when
    the composite metrics or the dates are unavailable.
    """
    per_model = _composite_per_model(df)
    if per_model.empty or dates.empty:
        return False

    cat_palette = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    cat_color = {key: cat_palette[i % len(cat_palette)]
                 for i, (key, _) in enumerate(COMPOSITE_CATEGORIES)}
    tier_style = {tk: (mk, sz) for tiers in CATEGORY_TIERS.values() for tk, _l, mk, sz in tiers}

    # Assemble the plotting frame: score + date + category/tier, dropping models we have no date
    # for (reported, never silently) and the simple baselines, which have no publication at all.
    recs, undated = [], []
    for model, score in per_model.items():
        cat = categorize_model(model)
        if cat == "simple" or not np.isfinite(score):
            continue
        base = model_base(model)
        if base not in dates.index:
            undated.append(base)
            continue
        recs.append(dict(model=model, base=base, cat=cat, tier=model_tier(model) or "",
                         mode=derive_mode(model), date=dates[base], score=float(score)))
    if undated:
        warnings.warn(f"date_vs_composite: no first_public_date for {sorted(set(undated))}; "
                      f"these models are omitted from the figure.")
    pts = pd.DataFrame(recs)
    if pts.empty:
        return False

    modes = [m for m in ("blind", "unblind") if (pts["mode"] == m).any()]
    if not modes:
        return False

    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 5.0),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes[0]

    for ax, mode in zip(axes, modes):
        # Classical baselines are blind-only but act as a mode-agnostic reference, so they appear
        # in BOTH panels — same convention as plot_composite_category. Without this the unblind
        # panel would lose every pre-2024 point and its correlation would not be comparable to the
        # blind panel's.
        sub = pts[(pts["mode"] == mode) | (pts["cat"] == "classical")].drop_duplicates("base")
        xnum = mdates.date2num(sub["date"])

        # Guide lines: OLS on (date, score), drawn thin because the reported statistic is the rank
        # correlation — these are for reading the trend, not fitted models.
        def _guide(x, y, color, dashes, lw, z):
            """Draw an OLS guide spanning only the x-range of the points it was fitted on."""
            if len(x) < 3 or np.ptp(x) <= 0:
                return
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(np.min(x), np.max(x), 100)
            line, = ax.plot(xs, slope * xs + intercept, color=color, linewidth=lw, zorder=z)
            line.set_dashes(dashes)

        # Pooled guide, long dashes, spanning the whole date range.
        _guide(xnum, sub["score"].values, "0.55", (6, 3), 1.2, 2)
        # Within-category guides, short dashes in the category color, each spanning only its own
        # category's dates — these are the trends the per-category rho values below refer to.
        for key, _lbl in COMPOSITE_CATEGORIES:
            g = sub[sub["cat"] == key]
            if len(g) >= 3:
                _guide(mdates.date2num(g["date"]), g["score"].values,
                       cat_color.get(key, "0.5"), (2.5, 2), 1.4, 3)

        for (cat, tier), grp in sub.groupby(["cat", "tier"], dropna=False):
            marker, size = tier_style.get(tier, ("o", 44))
            ax.scatter(mdates.date2num(grp["date"]), grp["score"].values,
                       s=size, marker=marker, color=cat_color.get(cat, "0.5"),
                       edgecolor="white", linewidth=0.5, zorder=4)

        # Stats box: pooled first, then per category, with n so small-sample rho is self-evident.
        rho, p = _spearman(xnum, sub["score"].values)
        lines = [f"all models: $\\rho$ = {rho:+.2f}, {_fmt_p(p)} (n = {len(sub)})"]
        for key, label in COMPOSITE_CATEGORIES:
            g = sub[sub["cat"] == key]
            if len(g) >= 3:
                r_c, p_c = _spearman(mdates.date2num(g["date"]), g["score"].values)
                lines.append(f"{label}: $\\rho$ = {r_c:+.2f}, {_fmt_p(p_c)} (n = {len(g)})")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, linespacing=1.5, zorder=6)

        ax.set_title(mode.capitalize(), fontsize=15)
        ax.set_xlabel("Artifact availability\n(release / weights published / first installable)",
                      fontsize=12)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(linestyle=":", alpha=0.5)
        sns.despine(ax=ax)

    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Composite score\n(bal. acc. + Lin. sim., chance-normalized)", fontsize=12)

    # Legend outside right: category color swatches, then the shared tier marker vocabulary.
    handles = [mpatches.Patch(color=cat_color[k], label=lbl)
               for k, lbl in COMPOSITE_CATEGORIES if (pts["cat"] == k).any()]
    handles.append(plt.Line2D([0], [0], linestyle="none", marker="", label=""))
    for key, clabel in COMPOSITE_CATEGORIES:
        for tk, tlabel, marker, _s in CATEGORY_TIERS.get(key, []):
            if (pts["tier"] == tk).any():
                handles.append(plt.Line2D([0], [0], linestyle="none", marker=marker,
                                          color=cat_color[key],
                                          markersize=12 if marker == "*" else 7, label=tlabel))
    guide_all = plt.Line2D([0], [0], color="0.55", linewidth=1.2, label="trend, all models")
    guide_all.set_dashes((6, 3))
    guide_cat = plt.Line2D([0], [0], color="0.35", linewidth=1.4, label="trend, within category")
    guide_cat.set_dashes((2.5, 2))
    handles.append(plt.Line2D([0], [0], linestyle="none", marker="", label=""))
    handles += [guide_all, guide_cat,
                plt.Line2D([0], [0], linestyle="none", marker="",
                           label="(OLS guides; stat is Spearman)")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=False,
               fontsize=9, labelspacing=0.32, handletextpad=0.6)

    if title:
        fig.suptitle(title, y=1.0, fontsize=16)
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
# OpenRouter activity diagnostics (LLM models only)
#
# Source: cache/openrouter_logs/openrouter_activity_merged.csv — one row per API call, with the
# `mode` (blind/unblind) and `in_benchmark` columns added when the campaigns were merged. Each
# (model, mode) holds exactly CELLS_PER_RUN calls: the four eval datasets at ~1000 cells each.
#
# These figures supersede bin/openrouter_summary/plot_openrouter_activity.py, which predates the
# mode column and therefore pools the blind and unblind campaigns into a single bar per model.
# --------------------------------------------------------------------------- #
DEFAULT_ACTIVITY_CSV = REPO_ROOT / "cache" / "openrouter_logs" / "openrouter_activity_merged.csv"

# Cells per (model, mode) run: HFRA 1000 + HIHA 999 + Zeisel 999 + human_pancreas 999.
DATASET_CELL_COUNTS = {"HFRA": 1000, "HIHA": 999, "Zeisel": 999, "human_pancreas": 999}
CELLS_PER_RUN = sum(DATASET_CELL_COUNTS.values())  # 3997

# OpenRouter permaslug -> the model name used in the aggregated results CSV (mode suffix aside).
# Explicit rather than regex-derived: the vendor's ordering differs from ours (`claude-4.7-opus`
# vs our `claude-opus-4.7`) and the maverick slug carries a size/config tail. An unmapped slug is
# warned about once and falls back to a best-effort strip, so a newly added model still plots.
ACTIVITY_MODEL_ALIASES = {
    "anthropic/claude-4.7-opus-20260416": "claude-opus-4.7",
    "anthropic/claude-4.6-sonnet-20260217": "claude-sonnet-4.6",
    "deepseek/deepseek-v3.2-20251201": "deepseek-v3.2",
    "google/gemini-3.1-pro-preview-20260219": "gemini-3.1-pro-preview",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "llama-4-maverick",
    "moonshotai/kimi-k2.6-20260420": "kimi-k2.6",
    "openai/gpt-5.5-20260423": "gpt-5.5",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
}
_WARNED_UNKNOWN_SLUGS: set[str] = set()

# Dataset attribution (see attribute_activity_datasets). The four eval datasets are run as four
# consecutive Nextflow tasks, so a (model, mode) group's rows form four contiguous time segments
# of ~1000 calls. Which segment is which dataset is read off the segments' median prompt length:
#
#   unblind — the prompt embeds the dataset's candidate-label list, which dominates:
#             HIHA (38 labels / 1126 chars) > Zeisel (29 / 612) > HFRA (9 / 156) > pancreas (11 / 156).
#             HFRA edges out pancreas on the gene sentence (1209 vs 1188 median chars).
#   blind   — no label list, so only the top-200 gene sentence varies:
#             Zeisel (1245 chars) > HFRA (1209) ~ HIHA (1207) > pancreas (1188).
#             HFRA/HIHA are within noise on characters but separate consistently by ~6-19 tokens
#             once tokenized, HIHA below HFRA in every model checked.
#
# Both orderings were validated against an independent ground truth: the per-dataset count of
# unanswered cells in cache/models/predictions/, which matches the per-segment count of non-`stop`
# calls (exactly for kimi/gpt-5.5/gpt-oss-20b/sonnet unblind, within retry noise elsewhere).
DATASET_ORDER_BY_PROMPT_LEN = {           # highest median prompt tokens first
    "unblind": ["HIHA", "Zeisel", "HFRA", "human_pancreas"],
    "blind": ["Zeisel", "HFRA", "HIHA", "human_pancreas"],
}
# Minimum gap (in median prompt tokens) between two adjacent segments for the assignment to be
# called confident. Below this the pair could be swapped, so the group is flagged.
DATASET_ATTRIBUTION_MIN_GAP = 3.0
# Max total |on-disk unanswered - per-segment failures|, as a fraction of a run, for the
# cross-check to count as corroboration rather than a mismatch.
ATTRIBUTION_FIT_TOL = 0.02


def activity_model_base(permaslug: str) -> str:
    """OpenRouter permaslug -> the model base name used in the aggregated results CSV."""
    slug = str(permaslug)
    if slug in ACTIVITY_MODEL_ALIASES:
        return ACTIVITY_MODEL_ALIASES[slug]
    if slug not in _WARNED_UNKNOWN_SLUGS:
        _WARNED_UNKNOWN_SLUGS.add(slug)
        print(f"[llm_activity]  permaslug '{slug}' is not in ACTIVITY_MODEL_ALIASES; falling back "
              f"to a stripped name (it may not join to the results CSV).", file=sys.stderr)
    import re
    return re.sub(r"-20\d{6}$", "", slug.split("/")[-1])


def load_openrouter_activity(path: Path, benchmarked_only: bool = True) -> pd.DataFrame:
    """Load the merged OpenRouter activity log into a per-request frame.

    Adds ``model_base`` (joins to the results CSV), ``model`` (``<base>_<mode>``, the exact key
    used in the aggregated CSV), ``answered`` (the call returned a usable completion) and the
    per-request token/latency/cost derivations the figures need. Returns an empty frame if the
    file is missing, so every activity figure degrades to a warning rather than an error.
    """
    if not path or not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("cost_total", "tokens_prompt", "tokens_completion", "tokens_reasoning",
                "tokens_cached", "generation_time_ms", "time_to_first_token_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "mode" not in df.columns:
        # Pre-merge layout: blind runs were tagged by infixing '-blind-' into the permaslug.
        warnings.warn(f"{path.name} has no 'mode' column; falling back to the '-blind-' permaslug "
                      f"convention. Re-build the merged CSV to split the campaigns properly.")
        df["mode"] = np.where(df["model_permaslug"].str.contains("-blind-"), "blind", "unblind")
        df["model_permaslug"] = df["model_permaslug"].str.replace("-blind-", "-", regex=False)
    if benchmarked_only and "in_benchmark" in df.columns:
        keep = df["in_benchmark"].astype(str).str.lower().isin(("true", "1", "yes"))
        df = df[keep]

    df = df.copy()
    df["model_base"] = df["model_permaslug"].map(activity_model_base)
    df["model"] = df["model_base"] + "_" + df["mode"].astype(str)
    # A call counts as answered only when it finished normally. `length` means the model burned
    # its budget (usually on reasoning) and returned no usable label; blank/error are failures.
    # predict.py retries three times and then writes an empty prediction, which the scorer maps
    # to the CL root term 'cell' -- i.e. a wrong answer -- so this fraction is a validity caveat,
    # not just an ops statistic.
    df["answered"] = df["finish_reason_normalized"].fillna("").str.lower().eq("stop")
    df["finish_reason"] = df["finish_reason_normalized"].fillna("").replace("", "no response")
    df["reasoning_tokens"] = df["tokens_reasoning"]
    df["visible_completion_tokens"] = (df["tokens_completion"] - df["tokens_reasoning"]).clip(lower=0)
    df["generation_time_s"] = df["generation_time_ms"] / 1000.0
    df["ttft_s"] = df["time_to_first_token_ms"] / 1000.0
    return df


def attribute_activity_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Infer which of the four eval datasets each API call belongs to.

    The log records no cell or dataset id, but the four datasets run as four consecutive tasks,
    so within a (model, mode) group the rows sort into four contiguous ~1000-call segments whose
    median prompt length identifies the dataset (see DATASET_ORDER_BY_PROMPT_LEN).

    Boundaries are placed in two passes: a provisional even split is enough to estimate each
    segment's median (a one-row offset cannot move a median over ~1000 values), the labels are
    read off the ranking, and the exact boundaries are then re-cut with the 1000-row segment at
    whichever position came out as HFRA. A group is low-confidence when two adjacent segment
    medians sit closer than DATASET_ATTRIBUTION_MIN_GAP tokens, i.e. the pair could be swapped --
    in practice this catches exactly the blind-mode HFRA/HIHA pair on models that do not separate
    them. (Segment SIZE cannot be used as a cross-check: 999 vs 1000 rows is a one-row shift that
    leaves within-segment prompt spread unchanged, so it carries no signal.)

    Adds ``dataset`` and ``dataset_confident`` (bool). Groups whose row count differs from
    CELLS_PER_RUN -- i.e. a run that was retried and trimmed, where the contiguous-segment
    assumption breaks -- get ``dataset = NA`` and ``dataset_confident = False``.
    """
    if df.empty or "tokens_prompt" not in df.columns:
        return df
    out = df.copy()
    out["dataset"] = pd.NA
    out["dataset_confident"] = False
    sizes = [DATASET_CELL_COUNTS[d] for d in sorted(DATASET_CELL_COUNTS)]
    big = max(sizes)                       # 1000 (HFRA); the rest are 999
    small = min(sizes)

    for (base, mode), g in out.groupby(["model_base", "mode"], sort=False):
        if len(g) != CELLS_PER_RUN or mode not in DATASET_ORDER_BY_PROMPT_LEN:
            continue
        idx = g.sort_values("created_at").index
        prompt = out.loc[idx, "tokens_prompt"].to_numpy(dtype=float)
        order = DATASET_ORDER_BY_PROMPT_LEN[mode]

        # Pass 1 - provisional even split, just to rank the segments by prompt length.
        prov = np.cumsum([0] + [small, small, small, CELLS_PER_RUN - 3 * small])
        medians = [float(np.median(prompt[prov[i]:prov[i + 1]])) for i in range(4)]
        rank = np.argsort(np.argsort(-np.asarray(medians)))    # 0 = longest prompt
        labels = [order[r] for r in rank]

        # Pass 2 - re-cut with the 1000-row segment where HFRA landed.
        big_pos = labels.index("HFRA")
        bounds = np.cumsum([0] + [big if i == big_pos else small for i in range(4)])
        medians = [float(np.median(prompt[bounds[i]:bounds[i + 1]])) for i in range(4)]

        sorted_med = sorted(medians, reverse=True)
        gaps = [sorted_med[i] - sorted_med[i + 1] for i in range(3)]
        confident = bool(min(gaps) >= DATASET_ATTRIBUTION_MIN_GAP)
        if not confident:
            closest = order[int(np.argmin(gaps))], order[int(np.argmin(gaps)) + 1]
            print(f"[llm_activity]  dataset attribution for {base} ({mode}) is low-confidence: "
                  f"segment medians {[round(m) for m in medians]} put {closest[0]} and "
                  f"{closest[1]} within {min(gaps):.0f} token(s), so the two could be swapped.",
                  file=sys.stderr)

        for i in range(4):
            out.loc[idx[bounds[i]:bounds[i + 1]], "dataset"] = labels[i]
        out.loc[idx, "dataset_confident"] = confident
    return out


def validate_dataset_attribution(df: pd.DataFrame, predictions_root: Path) -> pd.DataFrame:
    """Check the inferred dataset labels against an independent ground truth.

    The prediction CSVs under ``cache/models/predictions/openrouter_LLM_<mode>/<dataset>/`` record,
    per dataset, how many cells came back empty. Those failures must line up with the count of
    non-`stop` calls in the correspondingly-labelled log segment. Nothing in the attribution uses
    this signal, so agreement is genuine corroboration.

    Returns a per (model, mode) frame with the two count vectors and whether they agree, or an
    empty frame when the predictions tree is unavailable. Agreement is not exact equality -- a
    cell that failed and was retried leaves two log rows but one empty prediction -- so the test
    is whether the inferred labelling is the best of all 24 dataset permutations by total
    absolute difference. That is the question that matters: could relabelling fit truth better?
    """
    import itertools
    root = Path(predictions_root)
    if df.empty or "dataset" not in df.columns or not root.exists():
        return pd.DataFrame()
    # Prediction files are named from the model_cfg id (vendor + our base name), NOT from the
    # dated permaslug: predictions_openai--gpt-5.5.csv, not ...--gpt-5.5-20260423.csv.
    vendor = {b: p.split("/")[0] for p, b in ACTIVITY_MODEL_ALIASES.items()}
    rows = []
    for (base, mode), g in df[df["dataset"].notna()].groupby(["model_base", "mode"], sort=False):
        slug = f"{vendor[base]}--{base}" if base in vendor else base
        truth, inferred = {}, {}
        for ds in DATASET_CELL_COUNTS:
            p = root / f"openrouter_LLM_{mode}" / ds / "1000_stratified" / f"predictions_{slug}.csv"
            if not p.exists():
                truth = {}
                break
            pred = pd.read_csv(p)
            col = "predicted_labels"
            truth[ds] = int(pred[col].isna().sum() + (pred[col].astype(str).str.strip() == "").sum())
            inferred[ds] = int((~g[g["dataset"] == ds]["answered"]).sum())
        if not truth:
            continue
        keys = list(DATASET_CELL_COUNTS)
        t = np.array([truth[k] for k in keys], dtype=float)
        i = np.array([inferred[k] for k in keys], dtype=float)
        # With no failures anywhere the vectors are trivially equal and corroborate nothing.
        informative = bool(t.sum() > 0)
        identity_cost = float(np.abs(t - i).sum())
        best_cost = min(float(np.abs(t - i[list(p)]).sum())
                        for p in itertools.permutations(range(4)))
        rows.append({
            "model_base": base, "mode": mode,
            "unanswered_on_disk": {k: int(v) for k, v in zip(keys, t)},
            "non_stop_in_segment": {k: int(v) for k, v in zip(keys, i)},
            # Agrees when no relabelling fits better AND the fit is actually close. The second
            # condition matters: for a run that was retried, the log was trimmed (failures first)
            # so its counts are simply too low everywhere, and no permutation fixes that -- the
            # identity would win by default and a badly wrong split would pass unnoticed.
            "agrees": bool(identity_cost <= best_cost
                           and identity_cost <= ATTRIBUTION_FIT_TOL * CELLS_PER_RUN)
                      if informative else None,
            "informative": informative,
        })
    return pd.DataFrame(rows)


def activity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, mode) roll-up: spend, tokens, latency and answered fraction."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["model_base", "mode"], sort=False)
    out = pd.DataFrame({
        "n_calls": g.size(),
        "n_answered": g["answered"].sum(),
        "cost_total": g["cost_total"].sum(),
        "prompt_tokens": g["tokens_prompt"].sum(),
        "reasoning_tokens": g["reasoning_tokens"].sum(),
        "visible_tokens": g["visible_completion_tokens"].sum(),
        "median_latency_s": g["generation_time_s"].median(),
        "median_ttft_s": g["ttft_s"].median(),
    }).reset_index()
    out["answered_frac"] = out["n_answered"] / out["n_calls"]
    # Spend normalised to a fixed workload so models are comparable regardless of retries.
    out["cost_per_1k_cells"] = out["cost_total"] / out["n_calls"] * 1000.0
    out["model"] = out["model_base"] + "_" + out["mode"]
    return out


def _mode_colors() -> dict:
    """Two fixed hues for blind/unblind, taken from the repo's categorical palette."""
    pal = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    return {"blind": pal[0], "unblind": pal[2 % len(pal)]}


def _llm_tier_style(model_base_name: str):
    """(marker, size) for an LLM's method tier, reusing the composite figure's shapes."""
    tiers = {k: (mk, sz) for k, _lbl, mk, sz in CATEGORY_TIERS.get("LLM", [])}
    tier = model_tier(f"{model_base_name}_blind")
    return tiers.get(tier, UNCLASSIFIED_TIER_STYLE), tier


def _order_by(summary: pd.DataFrame, column: str) -> list:
    """Model bases ordered by the max of ``column`` across modes (descending)."""
    return (summary.groupby("model_base")[column].max()
            .sort_values(ascending=False).index.tolist())


def _paired_barh(ax, summary, column, order, colors, log=False):
    """Horizontal paired bars: one row per model, blind and unblind side by side.

    A 2-point gap between the paired bars keeps them readable as two marks rather than one
    stacked block, and the bars are anchored to the axis baseline.
    """
    h = 0.38
    for i, base in enumerate(order):
        for j, mode in enumerate(("blind", "unblind")):
            row = summary[(summary["model_base"] == base) & (summary["mode"] == mode)]
            if row.empty:
                continue
            val = float(row[column].iloc[0])
            ax.barh(i + (j - 0.5) * h, val, height=h * 0.92, color=colors[mode],
                    edgecolor="white", linewidth=0.6, zorder=3,
                    label=mode if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_llm_spend_and_tokens(df, out_path, dpi):
    """A: spend per 1000 cells (log). B: mean token composition per request.

    Replaces figure_1 of the old summary script, which pooled blind and unblind into one bar.
    Spend is normalised per 1000 cells so it reads as a unit price for the annotation task.
    """
    summary = activity_summary(df)
    if summary.empty:
        return False
    colors = _mode_colors()
    order = _order_by(summary, "cost_per_1k_cells")

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 2.2))
    _paired_barh(axes[0], summary, "cost_per_1k_cells", order, colors, log=True)
    axes[0].set_xlabel("Cost per 1,000 cells annotated (USD, log scale)")
    axes[0].set_title("A. Spend", loc="left")
    axes[0].legend(title="Mode", frameon=False, loc="lower right")

    # Token composition: stacked means per request, blind and unblind as separate rows.
    parts = [("prompt_tokens", "Prompt"), ("reasoning_tokens", "Reasoning"),
             ("visible_tokens", "Visible completion")]
    pal = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    part_colors = {"Prompt": pal[4 % len(pal)], "Reasoning": pal[6 % len(pal)],
                   "Visible completion": pal[8 % len(pal)]}
    ax = axes[1]
    h = 0.38
    for i, base in enumerate(order):
        for j, mode in enumerate(("blind", "unblind")):
            row = summary[(summary["model_base"] == base) & (summary["mode"] == mode)]
            if row.empty:
                continue
            left = 0.0
            for col, lbl in parts:
                w = float(row[col].iloc[0]) / float(row["n_calls"].iloc[0])
                ax.barh(i + (j - 0.5) * h, w, left=left, height=h * 0.92,
                        color=part_colors[lbl], edgecolor="white", linewidth=0.6, zorder=3,
                        label=lbl if i == 0 and j == 0 else None)
                left += w
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Mean tokens per request  (upper bar = blind, lower = unblind)")
    ax.set_title("B. Token composition", loc="left")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_llm_answer_rate(df, out_path, dpi, disk_counts=None):
    """Fraction of cells that came back with a usable label, and why the rest did not.

    This is a benchmark-validity figure, not an ops statistic: an unanswered cell is written as
    an empty prediction, normalised to "no prediction" and ontology-mapped to the CL root term
    ``cell``, then scored as a wrong answer. A model with a low answered rate is penalised for an
    infrastructure failure. Replaces figure_6, which dropped blank finish reasons entirely (they
    read as NaN) and so showed the largest failure mode as absent.
    """
    if df.empty:
        return False
    summary = activity_summary(df)
    order = summary.groupby("model_base")["answered_frac"].min().sort_values().index.tolist()
    reasons = ["stop", "length", "no response", "error"]
    pal = (CATEGORICAL_PALETTES.get("tab20_new") if _HAVE_REPO_STYLE else None) \
        or sns.color_palette("tab10", n_colors=10).as_hex()
    # Sequential-by-severity: answered is the neutral/base hue, failures escalate.
    reason_colors = {"stop": pal[0], "length": pal[5 % len(pal)],
                     "no response": pal[7 % len(pal)], "error": pal[9 % len(pal)]}

    counts = (df.groupby(["model_base", "mode", "finish_reason"]).size()
              .rename("n").reset_index())
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(order) + 2.2))
    h = 0.38
    seen = set()
    for i, base in enumerate(order):
        for j, mode in enumerate(("blind", "unblind")):
            sub = counts[(counts["model_base"] == base) & (counts["mode"] == mode)]
            if sub.empty:
                continue
            total = sub["n"].sum()
            left = 0.0
            for r in reasons:
                n = float(sub.loc[sub["finish_reason"] == r, "n"].sum())
                if n <= 0:
                    continue
                frac = n / total
                ax.barh(i + (j - 0.5) * h, frac, left=left, height=h * 0.92,
                        color=reason_colors[r], edgecolor="white", linewidth=0.6, zorder=3,
                        label=r if r not in seen else None)
                seen.add(r)
                left += frac
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of calls  (upper bar = blind, lower = unblind)")
    ax.set_title("Answered rate — unanswered cells are scored as wrong, not dropped", loc="left")
    ax.legend(title="Finish reason", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    # Where a run was retried, the merged log was trimmed to one run's worth of calls and the trim
    # drops failed calls first -- so for those groups this figure UNDERSTATES the failure rate.
    # Name them by comparing against the prediction files, which are the ground truth.
    if disk_counts is not None and not disk_counts.empty:
        disk = disk_counts.groupby(["model_base", "mode"])["unanswered"].sum()
        log_fail = (summary.set_index(["model_base", "mode"])["n_calls"]
                    - summary.set_index(["model_base", "mode"])["n_answered"])
        off = [f"{b} ({m}): {int(log_fail.get((b, m), 0))} in log vs {int(v)} on disk"
               for (b, m), v in disk.items()
               if abs(int(v) - int(log_fail.get((b, m), 0))) > 0.01 * CELLS_PER_RUN]
        if off:
            add_caption(fig, "Lower bound for retried runs, whose log was trimmed to one run's "
                             "worth of calls (failures dropped first) — " + "; ".join(off)
                             + ". See unanswered_by_dataset for the on-disk counts.")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_llm_latency(df, out_path, dpi):
    """Per-request generation time by model and mode (log), medians direct-labelled."""
    if df.empty:
        return False
    summary = activity_summary(df)
    order = _order_by(summary, "median_latency_s")
    colors = _mode_colors()
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(order) + 2.2))
    h = 0.38
    for i, base in enumerate(order):
        for j, mode in enumerate(("blind", "unblind")):
            vals = df[(df["model_base"] == base) & (df["mode"] == mode)]["generation_time_s"]
            vals = vals[vals > 0]
            if vals.empty:
                continue
            bp = ax.boxplot([vals.to_numpy()], positions=[i + (j - 0.5) * h], vert=False,
                            widths=h * 0.85, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.2))
            for box in bp["boxes"]:
                box.set(facecolor=colors[mode], edgecolor="white", linewidth=0.6, alpha=0.95)
            for whisk in bp["whiskers"] + bp["caps"]:
                whisk.set(color=colors[mode], linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Generation time per request (s, log scale)")
    ax.set_title("Latency — box = IQR, whiskers 1.5x IQR, outliers hidden", loc="left")
    handles = [mpatches.Patch(facecolor=colors[m], label=m) for m in ("blind", "unblind")]
    ax.legend(handles=handles, title="Mode", frameon=False, loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_llm_cost_by_dataset(df, out_path, dpi):
    """Spend per model split across the four eval datasets (inferred — see
    attribute_activity_datasets). Low-confidence groups are hatched so an inferred split is never
    mistaken for a recorded one."""
    if df.empty or "dataset" not in df.columns:
        return False
    sub = df[df["dataset"].notna()]
    if sub.empty:
        return False
    datasets = [d for d in DATASET_CELL_COUNTS if d in set(sub["dataset"])]
    colors = build_dataset_colors(datasets)
    cost = (sub.groupby(["model_base", "mode", "dataset"])["cost_total"].sum()
            .rename("cost").reset_index())
    conf = sub.groupby(["model_base", "mode"])["dataset_confident"].first()
    order = (cost.groupby("model_base")["cost"].sum().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(order) + 2.4))
    h = 0.38
    seen = set()
    for i, base in enumerate(order):
        for j, mode in enumerate(("blind", "unblind")):
            rows = cost[(cost["model_base"] == base) & (cost["mode"] == mode)]
            if rows.empty:
                continue
            ok = bool(conf.get((base, mode), False))
            # Shares, not absolutes: spend spans four orders of magnitude across models, so a
            # linear stacked axis would collapse every cheap model into the spine. The absolute
            # total is direct-labelled at the end of each bar instead.
            total = float(rows["cost"].sum())
            if total <= 0:
                continue
            left = 0.0
            for ds in datasets:
                v = float(rows.loc[rows["dataset"] == ds, "cost"].sum()) / total
                if v <= 0:
                    continue
                ax.barh(i + (j - 0.5) * h, v, left=left, height=h * 0.92, color=colors[ds],
                        edgecolor="white", linewidth=0.6, zorder=3,
                        hatch=None if ok else "//",
                        label=ds if ds not in seen else None)
                seen.add(ds)
                left += v
            ax.annotate(f"${total:,.2f}", (1.0, i + (j - 0.5) * h), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color="0.35")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the model's spend  (upper bar = blind, lower = unblind;  "
                  "hatched = low-confidence split)")
    ax.set_title("Where the spend went — dataset inferred from run order and prompt length",
                 loc="left")
    ax.legend(title="Dataset", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def unanswered_by_dataset(predictions_root: Path, model_bases) -> pd.DataFrame:
    """Per (model, mode, dataset) count of cells with no usable prediction, read from disk.

    Deliberately NOT derived from the activity log. Predictions are already stored one directory
    per dataset, so this needs no inference at all -- and the log is the weaker source here: for a
    group whose run was retried, the merged log was trimmed to one run's worth of calls and the
    trim drops failed calls first, so its failure count is a lower bound (kimi-k2.6 blind: 703
    non-stop calls in the log vs 996 empty predictions on disk).
    """
    root = Path(predictions_root)
    if not root.exists():
        return pd.DataFrame()
    vendor = {b: p.split("/")[0] for p, b in ACTIVITY_MODEL_ALIASES.items()}
    rows = []
    for base in model_bases:
        slug = f"{vendor[base]}--{base}" if base in vendor else base
        for mode in ("blind", "unblind"):
            for ds in DATASET_CELL_COUNTS:
                p = root / f"openrouter_LLM_{mode}" / ds / "1000_stratified" / f"predictions_{slug}.csv"
                if not p.exists():
                    continue
                pred = pd.read_csv(p)
                col = "predicted_labels"
                empty = int(pred[col].isna().sum()
                            + (pred[col].astype(str).str.strip() == "").sum())
                rows.append({"model_base": base, "mode": mode, "dataset": ds,
                             "n_cells": len(pred), "unanswered": empty,
                             "unanswered_pct": 100.0 * empty / max(len(pred), 1)})
    return pd.DataFrame(rows)


def plot_llm_unanswered_by_dataset(counts, out_path, dpi):
    """Unanswered cells per model x mode x dataset (counts from ``unanswered_by_dataset``).

    The pooled answered rate hides where a model actually fails: kimi-k2.6 blind reads as one
    number overall, but its failures concentrate on the datasets with the hardest label spaces.
    Since unanswered cells are scored as wrong -- normalised to "no prediction" and mapped to the
    CL root term ``cell`` -- that concentration is what determines how much each dataset's score
    is depressed.

    Magnitude question, so a single-hue sequential ramp, pale -> dark as the failure rate rises.
    Cells at or above 0.5% are direct-labelled; the rest would be noise.
    """
    if counts is None or counts.empty:
        return False
    datasets = [d for d in DATASET_CELL_COUNTS if d in set(counts["dataset"])]
    cols = [(ds, m) for ds in datasets for m in ("blind", "unblind")]
    models = (counts.groupby("model_base")["unanswered_pct"].max()
              .sort_values(ascending=False).index.tolist())
    grid = np.full((len(models), len(cols)), np.nan)
    for r, base in enumerate(models):
        for c, (ds, m) in enumerate(cols):
            v = counts[(counts["model_base"] == base) & (counts["mode"] == m)
                       & (counts["dataset"] == ds)]
            if not v.empty:
                grid[r, c] = float(v["unanswered_pct"].iloc[0])

    # White-anchored ramp built from the repo's soft_reds: a fully-answered cell must read as
    # blank, not as pale red. (COLORMAPS["Red"] runs dark -> light and would paint 0% darkest;
    # "Red_r" is the right direction but still starts at a visible pink.)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#eb9d9d", "#a63737"])
    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 4.2, 0.36 * len(models) + 2.6))
    im = ax.imshow(grid, aspect="auto", cmap=cmap or "Reds", vmin=0,
                   vmax=max(1.0, float(np.nanmax(grid))))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([m for _ds, m in cols], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    # Dataset name spanning its blind/unblind pair, plus a divider between datasets.
    for i, ds in enumerate(datasets):
        ax.text(2 * i + 0.5, -0.85, ds, ha="center", va="bottom", fontsize=8.5)
        if i:
            ax.axvline(2 * i - 0.5, color="white", linewidth=2.5)
    for r in range(len(models)):
        for c in range(len(cols)):
            v = grid[r, c]
            if np.isnan(v) or v < 0.5:
                continue
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 * np.nanmax(grid) else "0.2")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Cells with no usable answer (%)")
    ax.set_title("Where answering fails — unanswered cells are scored as wrong", loc="left",
                 pad=22)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    add_caption(fig, "Counted from the prediction files on disk, so the dataset split is exact "
                     "(no inference from the activity log).")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _pareto_front(points):
    """Indices of the cost/score points not dominated by a cheaper-and-better point."""
    front, best = [], -np.inf
    for i in sorted(range(len(points)), key=lambda k: points[k][0]):
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def plot_cost_vs_performance(agg_df, act_df, out_path, dpi, metric="balanced_accuracy",
                             dataset=None):
    """Annotation quality against what it costs, one point per (LLM, mode).

    ``metric`` is either a column of the aggregated CSV (averaged over datasets) or the literal
    ``"composite"``, which uses the same baseline-normalized composite as the category figure.
    x is spend per 1,000 cells on a log axis (it spans four orders of magnitude); the step line
    marks the cost-efficient frontier — the models no other model beats while also being cheaper.

    With ``dataset`` set, both sides are restricted to that dataset: the score comes straight
    from the aggregate (which is per-dataset already) while the cost side uses the INFERRED
    dataset attribution, so any low-confidence group is named in the caption.
    """
    if agg_df.empty or act_df.empty:
        return False
    caption = ""
    if dataset is not None:
        agg_df = agg_df[agg_df["dataset"].astype(str) == str(dataset)]
        if "dataset" not in act_df.columns:
            return False
        act_df = act_df[act_df["dataset"].astype(str) == str(dataset)]
        if agg_df.empty or act_df.empty:
            return False
        low = sorted({f"{b} ({m})" for (b, m), ok
                      in act_df.groupby(["model_base", "mode"])["dataset_confident"].first().items()
                      if not ok})
        caption = ("Cost split across datasets is inferred from run order and prompt length"
                   + (f"; low-confidence for {', '.join(low)}." if low else "."))
    summary = activity_summary(act_df).set_index("model")

    if metric == "composite":
        scores = _composite_per_model(agg_df)
        ylab = "Composite score (baseline-normalized)"
    else:
        if metric not in agg_df.columns:
            return False
        scores = (agg_df.assign(_v=pd.to_numeric(agg_df[metric], errors="coerce"))
                  .groupby("model")["_v"].mean())
        ylab = pretty(metric) if dataset else f"{pretty(metric)} (mean over datasets)"
    if scores.empty:
        return False

    rows = []
    for model, score in scores.items():
        if categorize_model(model) != "LLM" or model not in summary.index:
            continue
        if pd.isna(score):
            continue
        rows.append((model_base(model), derive_mode(model),
                     float(summary.loc[model, "cost_per_1k_cells"]), float(score),
                     float(summary.loc[model, "answered_frac"])))
    if len(rows) < 2:
        return False

    colors = _mode_colors()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    tiers_seen = {}
    for base, mode, cost, score, answered in rows:
        (marker, size), tier = _llm_tier_style(base)
        ax.scatter(cost, score, marker=marker, s=size * 1.6, color=colors.get(mode, "gray"),
                   edgecolor="white", linewidth=0.8, zorder=4)
        tiers_seen[tier] = marker
        # Direct-label only the models that define the frontier or answered poorly; labelling
        # every point would collide at this density.
        if answered < 0.98:
            ax.annotate(f"{base} ({answered:.0%} answered)", (cost, score),
                        textcoords="offset points", xytext=(6, -9), fontsize=6.5,
                        color="0.35", zorder=5)

    for mode in ("blind", "unblind"):
        pts = [(c, s) for _b, m, c, s, _a in rows if m == mode]
        if len(pts) < 2:
            continue
        front = _pareto_front(pts)
        fx = [pts[i][0] for i in front]
        fy = [pts[i][1] for i in front]
        ax.step(fx, fy, where="post", color=colors[mode], linewidth=1.4, alpha=0.55, zorder=2)
        for i in front:
            base = [b for b, m, c, s, _a in rows if m == mode and (c, s) == pts[i]][0]
            ax.annotate(base, pts[i], textcoords="offset points", xytext=(6, 5),
                        fontsize=7, color=colors[mode], zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 cells annotated (USD, log scale)")
    ax.set_ylabel(ylab)
    ax.set_title(f"Cost vs. annotation quality{f' — {dataset}' if dataset else ''} "
                 f"— step line = cost-efficient frontier", loc="left")
    handles = [mpatches.Patch(facecolor=colors[m], label=m) for m in ("blind", "unblind")]
    handles += [plt.Line2D([], [], linestyle="none", marker=mk, color="0.4",
                           label=(tier or "unclassified").replace("_", " "))
                for tier, mk in sorted(tiers_seen.items(), key=lambda kv: str(kv[0]))]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    add_caption(fig, caption)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


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
    p.add_argument("--dates-csv", type=Path, default=DEFAULT_DATES_CSV,
                   help="CSV of model dates (needs 'model_base' plus the chosen date column). "
                        "Drives the date-vs-composite figure; that figure is skipped with a "
                        "warning if the file is absent.")
    p.add_argument("--date-column", default=DATE_COLUMNS[0], choices=list(DATE_COLUMNS),
                   help="Which date to plot. 'artifact_available_date' (default) is when the "
                        "runnable artifact appeared and means the same thing for every model "
                        "kind; 'first_public_date' mixes LLM release dates with method preprint "
                        "dates and is offered only for sensitivity checks.")
    p.add_argument("--activity", type=Path, default=DEFAULT_ACTIVITY_CSV,
                   help="Merged OpenRouter activity CSV (one row per API call, with 'mode' and "
                        "'in_benchmark' columns). Drives the llm_activity and cost_performance "
                        "figures; both are skipped with a warning if the file is absent.")
    p.add_argument("--activity-include-non-benchmarked", action="store_true",
                   help="Keep activity rows flagged in_benchmark=False (e.g. llama-3.1-8b, which "
                        "was run but is not part of the benchmark). Off by default.")
    p.add_argument("--predictions-root", type=Path,
                   default=REPO_ROOT / "cache" / "models" / "predictions",
                   help="Prediction cache root. Used only to cross-check the inferred dataset "
                        "attribution against the per-dataset count of unanswered cells; skipped "
                        "silently when absent.")
    p.add_argument("--cost-metrics", nargs="+", default=["balanced_accuracy", "composite"],
                   help="y-axes for the cost-performance figures: any metric column, and/or the "
                        "literal 'composite' for the baseline-normalized composite score.")
    p.add_argument("--corr-method", default="spearman", choices=["spearman", "pearson"],
                   help="Correlation coefficient for the metric-correlation heatmap.")
    p.add_argument("--format", default="png", choices=["png", "pdf"],
                   help="Output file format.")
    p.add_argument("--dpi", type=int, default=300, help="Raster resolution.")
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "cache",
                   help="Cache root; a version-tagged copy of --input is saved under "
                        "<cache-dir>/result/ on every run.")
    p.add_argument("--no-cache-snapshot", action="store_true",
                   help="Skip writing the version-tagged aggregate snapshot to <cache-dir>/result/.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    # Retain a version-tagged snapshot of the exact aggregate these figures are built from.
    if not args.no_cache_snapshot:
        try:
            from snapshot_aggregate import snapshot_aggregate
            dest, wrote = snapshot_aggregate(args.input, args.cache_dir)
            print(f"{'Wrote' if wrote else 'Up-to-date'} aggregate snapshot: {dest}")
        except Exception as exc:  # a snapshot failure must never block plotting
            warnings.warn(f"Could not write aggregate snapshot: {exc}")

    df = pd.read_csv(args.input)
    missing = set(ID_COLS) - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required column(s): {missing}")

    # Drop deprecated / superseded models (see EXCLUDED_BASES) from every figure.
    n_before = df["model"].nunique()
    df = df[~df["model"].map(lambda m: model_base(m) in EXCLUDED_BASES)].copy()
    dropped = n_before - df["model"].nunique()
    if dropped:
        print(f"Excluded {dropped} deprecated model(s) (bases: {', '.join(sorted(EXCLUDED_BASES))}).")

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
            # Per-metric plot types: one figure per metric. (correlation, composite_category and
            # the LLM activity figures are not per-metric and are handled separately.)
            for plot_type in (pt for pt in args.plots if pt in builders):
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
        model_dates = load_model_dates(args.dates_csv, args.date_column)
        if model_dates.empty:
            warnings.warn(f"No usable model dates at {args.dates_csv}; skipping the "
                          f"date-vs-composite figures.")

        def emit_composite(src_df, out_dir, log_tag, title=None):
            """Write one composite figure per subset-size view of ``src_df`` into ``out_dir``."""
            nonlocal total_written
            out_dir.mkdir(parents=True, exist_ok=True)
            for size_folder, sdf in composite_size_views(src_df, args.subset_size_col):
                # Two variants of the same figure: the default asserts the method tiers via marker
                # shapes, the "_flat" one drops them and draws every model as a plain circle.
                for suffix, tiers in (("", True), ("_flat", False)):
                    out = out_dir / f"composite_category{suffix}_{size_folder}.{args.format}"
                    if plot_composite_category(sdf, out, "", args.dpi, title=title,
                                               show_tiers=tiers):
                        total_written += 1
                        print(f"[composite_category/{log_tag}/{size_folder}]  wrote {out.name}")
                    elif not suffix:
                        warnings.warn(
                            f"Skipped composite_category/{log_tag}/{size_folder}: composite "
                            f"metrics {COMPOSITE_METRICS} not all present.")
                # Companion per-method ranked bar figure (every model individually, sorted best-first,
                # name colored by category) — same size views + per-dataset breakdown.
                outr = out_dir / f"composite_ranked_{size_folder}.{args.format}"
                if plot_composite_ranked_bars(sdf, outr, args.dpi, title=title):
                    total_written += 1
                    print(f"[composite_category/{log_tag}/{size_folder}]  wrote {outr.name}")
                # Companion date-vs-composite figure — same category colors and tier markers.
                outd = out_dir / f"composite_vs_date_{size_folder}.{args.format}"
                if plot_date_vs_composite(sdf, model_dates, outd, args.dpi, title=title):
                    total_written += 1
                    print(f"[composite_category/{log_tag}/{size_folder}]  wrote {outd.name}")
                # Companion metric-disagreement panel. The composite axis is balanced accuracy +
                # Lin, so macro F1 -- the metric on which the family ordering reverses -- cannot
                # appear in the composite figure at all. Same colors and tier markers.
                outm = out_dir / f"metric_disagreement_{size_folder}.{args.format}"
                if plot_metric_disagreement(sdf, outm, args.dpi, title=title):
                    total_written += 1
                    print(f"[composite_category/{log_tag}/{size_folder}]  wrote {outm.name}")

        # 1) Aggregated over all datasets (top-level folder), as before.
        emit_composite(df, cdir, "all_datasets")

        # 2) Per-dataset breakdown for the supplementary: one figure set per dataset, each titled
        #    with the dataset name so the supplementary panels are self-identifying.
        for dataset in sorted(df["dataset"].unique()):
            emit_composite(df[df["dataset"] == dataset], cdir / "by_dataset" / str(dataset),
                           f"by_dataset/{dataset}", title=str(dataset))

    # LLM-only figures from the OpenRouter activity log. Top-level (they span blind + unblind and
    # are not per-metric), and a missing/unusable log downgrades to a warning, never an error.
    wants_activity = [p for p in args.plots if p in ("llm_activity", "cost_performance")]
    if wants_activity:
        activity = load_openrouter_activity(
            args.activity, benchmarked_only=not args.activity_include_non_benchmarked)
        if activity.empty:
            warnings.warn(f"No OpenRouter activity at {args.activity}; skipping {wants_activity}.")
        else:
            activity = attribute_activity_datasets(activity)
            n_conf = (activity.groupby(["model_base", "mode"])["dataset_confident"].first().sum())
            n_grp = activity.groupby(["model_base", "mode"]).ngroups
            print(f"OpenRouter activity: {len(activity):,} calls, "
                  f"{activity['model_base'].nunique()} models, "
                  f"dataset attributed with confidence for {int(n_conf)}/{n_grp} model x mode groups")

            check = validate_dataset_attribution(activity, args.predictions_root)
            if not check.empty:
                inf = check[check["informative"]]
                if not inf.empty:
                    print(f"  attribution cross-check vs. per-dataset unanswered counts on disk: "
                          f"{int(inf['agrees'].sum())}/{len(inf)} groups agree "
                          f"({len(check) - len(inf)} groups had no failures, so carry no signal)")
                    for _, r in inf[~inf["agrees"].astype(bool)].iterrows():
                        print(f"    DISAGREES {r['model_base']} ({r['mode']}): "
                              f"on disk {r['unanswered_on_disk']} vs "
                              f"segments {r['non_stop_in_segment']}", file=sys.stderr)
                    # A group whose segments cannot be reconciled with the on-disk counts is not
                    # trustworthy for the per-dataset cost split either -- mark it low-confidence
                    # so spend_by_dataset hatches it.
                    bad = {(r["model_base"], r["mode"])
                           for _, r in inf[~inf["agrees"].astype(bool)].iterrows()}
                    if bad:
                        key = list(zip(activity["model_base"], activity["mode"]))
                        activity.loc[[k in bad for k in key], "dataset_confident"] = False

            if "llm_activity" in args.plots:
                adir = args.outdir / "llm_activity"
                adir.mkdir(parents=True, exist_ok=True)
                # Per-dataset unanswered counts come from the prediction files, not the log:
                # the split is exact there, and the log understates failures for retried runs.
                disk_counts = unanswered_by_dataset(
                    args.predictions_root, sorted(activity["model_base"].unique()))
                builders_llm = {
                    "spend_and_tokens": plot_llm_spend_and_tokens,
                    "answer_rate": lambda d, o, dpi: plot_llm_answer_rate(
                        d, o, dpi, disk_counts=disk_counts),
                    "unanswered_by_dataset": lambda _d, o, dpi: plot_llm_unanswered_by_dataset(
                        disk_counts, o, dpi),
                    "latency": plot_llm_latency,
                    "spend_by_dataset": plot_llm_cost_by_dataset,
                }
                for name, fn in builders_llm.items():
                    out = adir / f"{name}.{args.format}"
                    if fn(activity, out, args.dpi):
                        total_written += 1
                        print(f"[llm_activity]  wrote {out.name}")
                    else:
                        warnings.warn(f"Skipped llm_activity/{name}: no usable data.")

            if "cost_performance" in args.plots:
                cdir = args.outdir / "cost_performance"
                cdir.mkdir(parents=True, exist_ok=True)
                for metric in args.cost_metrics:
                    out = cdir / f"cost_vs_{metric}.{args.format}"
                    if plot_cost_vs_performance(df, activity, out, args.dpi, metric=metric):
                        total_written += 1
                        print(f"[cost_performance]  wrote {out.name}")
                    else:
                        warnings.warn(
                            f"Skipped cost_performance/{metric}: metric absent, or fewer than two "
                            f"LLMs joined the activity log.")
                    # Per-dataset breakdown: score is per-dataset in the aggregate already, the
                    # cost side uses the inferred split (flagged in each figure's caption).
                    for dataset in sorted(df["dataset"].astype(str).unique()):
                        ddir = cdir / "by_dataset" / dataset
                        ddir.mkdir(parents=True, exist_ok=True)
                        outd = ddir / f"cost_vs_{metric}.{args.format}"
                        if plot_cost_vs_performance(df, activity, outd, args.dpi, metric=metric,
                                                    dataset=dataset):
                            total_written += 1
                            print(f"[cost_performance/by_dataset/{dataset}]  wrote {outd.name}")

    print(f"Total: wrote {total_written} figure(s) under {args.outdir.resolve()}")
    print(f"  layout: <mode_variant>/<subset_size>/<plot_type>/<metric>.{args.format}")


if __name__ == "__main__":
    main()
