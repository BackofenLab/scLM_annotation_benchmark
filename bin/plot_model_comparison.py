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
    "text_emb_mapping_lift",     # gain from ontology mapping (mapped - raw)
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


def plot_composite_category(df, out_path, caption, dpi, title=None, show_tiers=True):
    """Single-panel composite-score figure with paired blind/unblind sub-columns.

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
    per_model = _composite_per_model(df)
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

    fig, ax = plt.subplots(figsize=(2.6 * len(present) + 1.6, 4.3))
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
    # y=0 IS the random-stratified (chance) level by construction (the composite is normalized
    # against it), so no separate baseline reference line is drawn — it would sit on the axis.
    ax.set_ylabel("Composite score\n(bal. acc. + Lin. sim., chance-normalized)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)

    # Legend outside right: the connector, then one entry per tier marker actually drawn, grouped
    # under a per-category header in that category's color.
    def _blank(label):
        return plt.Line2D([0], [0], linestyle="none", marker="", label=label)

    handles = []
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


def plot_composite_ranked_bars(df, out_path, dpi, title=None):
    """Per-method composite ranking: one bar per model, sorted best-first (highest composite on the
    left), each bar (and its x-axis name label) colored by the model's category.

    Uses the same baseline-normalized composite score as ``plot_composite_category``
    (``_composite_per_model``), but instead of collapsing each category to points + a mean line it
    shows EVERY method individually — blind and unblind models are separate bars, and the simple
    baselines are included. Returns False if the composite metrics are absent.
    """
    per_model = _composite_per_model(df)
    if per_model.empty:
        return False

    # Drop models with no composite score (NaN under the coverage gate, e.g. not-yet-reviewed marker
    # baselines) and the simple chance baselines so they don't occupy slots on the ranked axis.
    ranked = per_model.dropna()
    ranked = ranked[[categorize_model(m) != "simple" for m in ranked.index]].sort_values(ascending=False)
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

    # Chance reference line (random_stratified composite), red and in front of the bars.
    # No chance reference line: the composite is normalized AGAINST random_stratified, so that
    # model scores exactly 0 by construction and the line would lie on top of the x axis (it read
    # as a rendering bug). The chance level is stated in the y-axis label instead, matching what
    # plot_composite_category already does.

    # Value labels above each bar (small, rotated) so the rank order is readable at a glance.
    for xi, v in zip(x, ranked.values):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, rotation=90, color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90, ha="center", fontsize=8)
    # Color each model-name tick label by its category (matches the bar).
    for tick, c in zip(ax.get_xticklabels(), cats):
        tick.set_color(cat_color.get(c, cat_color["LLM"]))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Composite score\n(bal. acc. + Lin. sim. — 0 = random-stratified chance)",
                  fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)

    # Legend: one swatch per category actually present, in canonical order.
    present_cats = [k for k in _RANKED_CATEGORY_ORDER if k in set(cats)]
    handles = [mpatches.Patch(color=cat_color[k], label=_RANKED_CATEGORY_LABELS[k])
               for k in present_cats]
    leg_cat = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=11,
                        title="model category", title_fontsize=11)
    ax.add_artist(leg_cat)
    # Second legend: prediction mode (solid = unblind, black diagonals = blind).
    mode_handles = [mpatches.Patch(facecolor="0.7", edgecolor="black", label="unblind"),
                    mpatches.Patch(facecolor="0.7", edgecolor="black", hatch="///", label="blind")]
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
            "unanswered_on_disk": dict(zip(keys, t.astype(int))),
            "non_stop_in_segment": dict(zip(keys, i.astype(int))),
            # Agrees when no relabelling of the four segments fits the on-disk counts better.
            "agrees": bool(identity_cost <= best_cost) if informative else None,
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


def plot_llm_answer_rate(df, out_path, dpi):
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


def _pareto_front(points):
    """Indices of the cost/score points not dominated by a cheaper-and-better point."""
    front, best = [], -np.inf
    for i in sorted(range(len(points)), key=lambda k: points[k][0]):
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def plot_cost_vs_performance(agg_df, act_df, out_path, dpi, metric="balanced_accuracy"):
    """Annotation quality against what it costs, one point per (LLM, mode).

    ``metric`` is either a column of the aggregated CSV (averaged over datasets) or the literal
    ``"composite"``, which uses the same baseline-normalized composite as the category figure.
    x is spend per 1,000 cells on a log axis (it spans four orders of magnitude); the step line
    marks the cost-efficient frontier — the models no other model beats while also being cheaper.
    """
    if agg_df.empty or act_df.empty:
        return False
    summary = activity_summary(act_df).set_index("model")

    if metric == "composite":
        scores = _composite_per_model(agg_df)
        ylab = "Composite score (baseline-normalized)"
    else:
        if metric not in agg_df.columns:
            return False
        scores = (agg_df.assign(_v=pd.to_numeric(agg_df[metric], errors="coerce"))
                  .groupby("model")["_v"].mean())
        ylab = f"{pretty(metric)} (mean over datasets)"
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
    ax.set_title("Cost vs. annotation quality — step line = cost-efficient frontier", loc="left")
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

            if "llm_activity" in args.plots:
                adir = args.outdir / "llm_activity"
                adir.mkdir(parents=True, exist_ok=True)
                for name, fn in (("spend_and_tokens", plot_llm_spend_and_tokens),
                                 ("answer_rate", plot_llm_answer_rate),
                                 ("latency", plot_llm_latency),
                                 ("spend_by_dataset", plot_llm_cost_by_dataset)):
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

    print(f"Total: wrote {total_written} figure(s) under {args.outdir.resolve()}")
    print(f"  layout: <mode_variant>/<subset_size>/<plot_type>/<metric>.{args.format}")


if __name__ == "__main__":
    main()
