#!/usr/bin/env python
"""Compose the manuscript display items from the panels in ``plot_model_comparison.py``.

This module does NOT re-implement any panel. It imports the panel functions and asks them to
draw into axes of one shared figure, so every panel inherits the same font family, font sizes and
tick style. Stitching the standalone PNGs together would not: each is rendered at its own figure
size, so identical point sizes come out at different physical sizes once the images are scaled.

One function per manuscript figure, named ``figure_N``. Each returns the output path. The panel
letters, the shared legend and the regime vocabulary are handled here rather than in the panel
functions, because they are properties of the composed figure, not of the panel.

Regime vocabulary: the pipeline, the result filenames and the aggregate ``mode`` column use
``blind``/``unblind``. The manuscript uses ``unconstrained``/``constrained`` (see
``WRITING_STYLE.md`` Rule 5). The relabeling happens here, at the last step before the file is
written, so no code or cached artifact has to be renamed.

Usage:
    python bin/manuscript_figures.py --input Results/aggregated_by_dataset_model.csv \
        --outdir scLLMs_benchmark_paper/bioRxiv_template/figures
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_model_comparison import (  # noqa: E402
    CATEGORY_TIERS,
    COMPOSITE_CATEGORIES,
    DEFAULT_ACTIVITY_CSV,
    EXCLUDED_BASES,
    REGIME,
    REGIME_SHORT,
    _place_point_labels,
    _ranked_category_colors,
    _tier_marker as _tier_style,
    _tiered_offsets,
    categorize_model,
    derive_mode,
    group_name,
    _flow_key,
    _sankey,
    load_confusion_pairs,
    load_openrouter_activity,
    load_per_class_recall,
    model_base,
    model_tier,
    pretty,
    plot_composite_category,
    plot_cost_vs_performance,
    plot_family_by_dataset,
    plot_metric_disagreement,
    plot_rare_type_flows,
    plot_recall_by_prevalence,
    run_label,
    spread,
    unsupported_mark,
    with_unsupported_as_zero,
)

# --------------------------------------------------------------------------- #
# One typographic scale for every manuscript figure. Panel functions set their own sizes for
# standalone use; `_harmonize` overrides them here so panels drawn side by side match. Sizes are
# in points at the figure's native size -- do not scale the saved file afterwards.
# --------------------------------------------------------------------------- #
# Helvetica, as in Figure 1, with Arial for a glyph Helvetica lacks.
FONT_FAMILY = ["Helvetica", "Arial"]
FS_PANEL_LETTER = 15   # A, B, C
FS_TITLE = 10          # panel titles
FS_LABEL = 11          # axis labels
FS_TICK = 10           # tick labels
FS_GROUP = 12          # in-axes group headings (the family names under the strip panel)
FS_POINT = 7.5         # in-axes point annotations (model names in the scatter)
FS_LEGEND = 9
FS_ROSTER = 7.5        # model-name roster beside panel D

SUBSET_SIZE_COL = "subset_size"


def _rc():
    """rcParams shared by every manuscript figure."""
    return {
        "font.family": FONT_FAMILY,
        # Embed TrueType rather than Type 3 outlines, so the text stays text in the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Math text (the bold U and C of the setting keys) in Helvetica too.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.linewidth": 0.8,
        "figure.dpi": 100,
    }


def _harmonize(ax, text_size=FS_POINT):
    """Force one panel's typography onto the shared scale.

    The panel functions hardcode sizes that suit them standalone. Axis labels, ticks and titles
    must match across panels, so they are reset unconditionally. In-axes text is semantically
    different per panel (group headings vs point names), so its size is passed in.
    """
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    ax.title.set_size(FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    for t in ax.texts:
        t.set_fontsize(text_size)


def _to_prose_regime(text):
    """Rewrite the code's regime names as the manuscript's.

    Longest key first: replacing ``blind`` before ``unblind`` would turn "unblind" into
    "ununconstrained", because ``blind`` is a substring of ``unblind``.
    """
    for code in sorted(REGIME, key=len, reverse=True):
        text = text.replace(code, REGIME[code])
    return text


def _relabel_regimes(ax):
    """Replace blind/unblind tick labels with the bold single-letter regime keys.

    Spelling the regimes out under each sub-column costs more width than the panel has, and
    rotating them pushes the labels into the family headings underneath. Single bold letters keep
    the ticks horizontal and the headings where the panel function put them; the legend carries
    the key. The manuscript prose still uses the full terms everywhere (style Rule 5).
    """
    labels = [t.get_text() for t in ax.get_xticklabels()]
    if not any(l in REGIME_SHORT for l in labels):
        return
    ax.set_xticklabels([REGIME_SHORT.get(l, l) for l in labels],
                       fontweight="bold", fontsize=FS_LABEL)


def _regime_key_handles():
    """Legend entries spelling out the bold letters used on the strip panel's ticks."""
    return [plt.Line2D([], [], linestyle="none", marker="",
                       label=rf"$\bf{{{short}}}$   {REGIME[code]}")
            for code, short in REGIME_SHORT.items()]


def _blank(label=""):
    return plt.Line2D([], [], linestyle="none", marker="", label=label)


def _panel_letters(fig, items, pad_x=0.012, pad_y=0.008):
    """Place bold panel letters outside the top-left corner of each axes in ``items``.

    ``items`` is ``[(ax, letter)]``. Two rules, both of which need the whole set rather than one
    panel at a time:

    * **Vertical** anchors on each panel's own TIGHT bounding box, so a letter clears whatever its
      panel carries — above the title where there is one, just above the frame where there is not.
    * **Horizontal** aligns every panel that starts in the same figure column to a single x, taken
      from the leftmost tight bbox in that column. Anchoring each letter to its own bbox would
      indent the letters of panels without y-axis decorations relative
      to panels with a y-label, which reads as a mistake.

    Draws the canvas first, since tight bounding boxes need a renderer and a final layout.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    measured = [(letter, ax.get_tightbbox(renderer).transformed(inv),
                 round(ax.get_position().x0, 2)) for ax, letter in items]

    column_x = {}
    for _letter, tight, column in measured:
        column_x[column] = min(column_x.get(column, 1.0), tight.x0)

    for letter, tight, column in measured:
        fig.text(max(column_x[column] - pad_x, 0.004), min(tight.y1 + pad_y, 0.997), letter,
                 fontsize=FS_PANEL_LETTER, fontweight="bold", va="bottom", ha="left")


def _align_left(fig, gridspec, ax, reference):
    """Shift ``gridspec`` sideways so that ``ax``, labels included, starts where ``reference``
    does: two stacked rows of panels then share one left edge, and _panel_letters puts their
    letters above each other."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    shift = (reference.get_tightbbox(renderer).transformed(inv).x0
             - ax.get_tightbbox(renderer).transformed(inv).x0)
    gridspec.update(left=gridspec.left + shift)


def load_aggregate(path, subset_size=1000):
    """Read the aggregate snapshot with the same exclusions the comparison figures apply."""
    df = pd.read_csv(path)
    df = df[~df["model"].map(lambda m: model_base(m) in EXCLUDED_BASES)].copy()
    if subset_size is not None and SUBSET_SIZE_COL in df.columns:
        df = df[pd.to_numeric(df[SUBSET_SIZE_COL], errors="coerce") == subset_size].copy()
    if df.empty:
        raise SystemExit(f"No rows left in {path} at subset_size={subset_size}.")
    return df


# --------------------------------------------------------------------------- #
# Figure 1
# --------------------------------------------------------------------------- #
def _save(fig, out_path, dpi):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_1(df, out_path, dpi=600, metric=None):
    """The family plot of Figure 2B on its own, by default on the composite score.

    Score by model family, unconstrained vs constrained, one line per model. Figure 1 is the
    pipeline and benchmark overview, drawn outside this script
    (scLLMs_benchmark_paper/bioRxiv_template/figures/model_overview_figure.*).
    """
    df = with_unsupported_as_zero(df, model_cols=("model", "mode"))  # means over datasets
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 4.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.34], wspace=0.04)
        ax_b = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        handles = plot_composite_category(df, None, "", dpi, ax=ax_b, metric=metric)
        if not handles:
            raise SystemExit(f"Figure 2: {metric or 'composite'} missing from the aggregate.")
        _harmonize(ax_b, text_size=FS_GROUP)
        _relabel_regimes(ax_b)

        handles = _regime_key_handles() + [_blank()] + list(handles)
        for h in handles:
            h.set_label(_to_prose_regime(h.get_label() or ""))
        ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=FS_LEGEND,
                      labelspacing=0.32, handletextpad=0.6, borderaxespad=0.0)
        return _save(fig, out_path, dpi)


# Display names for figure axes. `pretty()` title-cases the aggregate's model keys, which yields
# "Celltypist" and "Gpt-5.5". The prose spells these differently, and a figure that disagrees with
# the text about a method's NAME reads as a different method. Only the keys where pretty() and the
# manuscript diverge are listed; everything else falls through to pretty().
DISPLAY_NAMES = {
    "celltypist": "CellTypist", "scsa": "SCSA", "sctype": "scType", "singler": "SingleR",
    "c2s_scale_2B": "C2S-Scale 2B", "c2s_scale_27B": "C2S-Scale 27B",
    "cellwhisperer": "CellWhisperer", "langcell": "LangCell", "mmcontext": "mmContext",
    "instructcell": "InstructCell", "gpt-5.5": "GPT-5.5", "gpt-oss-20b": "GPT-OSS-20B",
    "gpt-oss-120b": "GPT-OSS-120B", "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "claude-opus-4.7": "Claude Opus 4.7", "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "deepseek-v3.2": "DeepSeek-V3.2", "kimi-k2.6": "Kimi-K2.6",
    "llama-3.1-70b-instruct": "Llama-3.1-70B", "llama-3.3-70b-instruct": "Llama-3.3-70B",
    "llama-4-maverick": "Llama-4-Maverick",
}
DATASET_ORDER = ["HFRA", "HIHA", "Zeisel", "human_pancreas"]

#: Dataset names for anything a READER sees, on a figure or in a caption. The manuscript uses the
#: pipeline's keys only in Methods, where they let someone move between the paper and the
#: repository. Only human_pancreas differs -- the other three keys are already the names, and the
#: underscore is what makes this one read as a variable rather than as a tissue.
DATASET_DISPLAY = {"human_pancreas": "human pancreas"}


def dataset_label(name):
    """Readable dataset name for a tick label, panel title or legend entry."""
    return DATASET_DISPLAY.get(str(name), str(name))


def _display(base):
    return DISPLAY_NAMES.get(base, pretty(base))


def _tier_marker(model):
    """Marker for a model's method tier, matching Figure 2's key."""
    for key, _label, marker, _size in CATEGORY_TIERS.get(categorize_model(model), []):
        if key == model_tier(model):
            return marker
    return "o"


def figure_harmonization(df, out_path, dpi=600, pairs=None):
    """Supplementary Figure S1 — what the curated mapping contributes, per method.

    The per-method detail behind Figure 2G.
    (A) share of cells whose answer needed the curated mapping: neither a Cell Ontology term nor
        one of the dataset's own labels, as written.
    (B) balanced accuracy counting only the answers that needed no curated mapping, and with the
        curated mapping of the other answers. The gap is what the mapping buys.

    A and B share one row order (by the gap in B, descending) so the reader can read across: how
    much of a method's output needs the mapping, and what the mapping then buys it. The two are
    related but not monotonic, because an answer that needs the mapping can still be wrong.
    """
    d = with_unsupported_as_zero(df, model_cols=("model", "mode"))  # means over datasets
    pairs = _benchmark_pairs(d, pairs)
    scores = _mapping_scores(pairs).rename(columns={"without": "ba_direct", "with": "ba_mapped"})
    scores["needs_mapping"] = _needs_mapping(pairs)
    d = d.merge(scores.reset_index(), on=["model", "dataset"], how="left")
    d["base"] = d["model"].map(model_base)
    d["cat"] = d["model"].map(categorize_model)
    d["regime"] = d["model"].map(derive_mode)
    d = d[d["cat"] != "simple"]          # chance floors have no label vocabulary to characterize

    # One row block per setting, so every method appears under both.
    groups = [(mode, d[d["regime"] == mode]) for mode in ("blind", "unblind")]

    def _agg(frame):
        out = (frame.groupby(["base", "cat"], as_index=False)
                    .agg(model=("model", "first"), needs=("needs_mapping", "mean"),
                         raw=("ba_direct", "mean"), mapped=("ba_mapped", "mean")))
        out["lift"] = out["mapped"] - out["raw"]
        return out.sort_values("lift", ascending=False).reset_index(drop=True)

    GAP = 1.9                      # blank rows between blocks, leaving room for the heading
    blocks, rows, pos = [], [], 0.0
    for mode, frame in groups:
        if frame.empty:
            continue
        pos += GAP
        block = _agg(frame)
        blocks.append((REGIME[mode], pos - 1.0, block))
        for _, r in block.iterrows():
            rows.append({**r.to_dict(), "ypos": pos, "regime": mode})
            pos += 1
    agg = pd.DataFrame(rows)
    cat_color = _ranked_category_colors()

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 11.6))
        # The panels, then Figure 2's key across the full width under them.
        outer = fig.add_gridspec(2, 1, height_ratios=[10.0, 0.9], hspace=0.25, left=0.15,
                                 right=0.985)
        # A is narrow: it carries one bounded fraction. B gets the width because the LENGTH of
        # each segment is the quantity being read off it. The gap keeps A's last tick label
        # clear of B's first.
        top = outer[0].subgridspec(1, 2, width_ratios=[0.52, 1.0], wspace=0.08)
        ax_a = fig.add_subplot(top[0, 0])
        ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
        ax_key = fig.add_subplot(outer[1])
        ax_key.axis("off")

        y = agg["ypos"].to_numpy()
        colors = [cat_color[c] for c in agg["cat"]]

        def _cells(row, column):
            """The four per-dataset values behind one drawn row, matched on method AND setting."""
            return d.loc[(d["base"] == row["base"]) & (d["regime"] == row["regime"]),
                         column].dropna()

        def _dots(ax, values, yy, color):
            ax.scatter(values, np.full(len(values), yy), s=DOT_SIZE, color=color, linewidths=0,
                       zorder=3)

        # --- A: how much of each method's output needs the curated mapping -----------------
        ax_a.barh(y, agg["needs"], height=0.62, color=colors, alpha=0.85, zorder=2)
        for (_, row), yy in zip(agg.iterrows(), y):
            _dots(ax_a, _cells(row, "needs_mapping"), yy, DOT_COLOR)
        ax_a.set_xlim(0, 1.04)
        ax_a.set_xticks(np.arange(0, 1.01, 0.2))
        ax_a.set_xlabel("Share of cells whose answer needs\nthe curated mapping")

        # --- B: what the curated mapping then buys -----------------------------------------
        # Per-dataset values are dots, as in A: light without the curated mapping and dark with
        # it, the dark ones drawn second so that they hide an equal light one. Lines would not
        # do: 84 near-horizontal lines read as stray rules across the panel.
        for (_, r), yy in zip(agg.iterrows(), y):
            col = cat_color[r["cat"]]
            _dots(ax_b, _cells(r, "ba_direct"), yy, DOT_COLOR_NO_MAPPING)
            _dots(ax_b, _cells(r, "ba_mapped"), yy, DOT_COLOR)
            ax_b.plot([r["raw"], r["mapped"]], [yy, yy], color=col, lw=1.8, alpha=0.95, zorder=2)
            ax_b.scatter(r["raw"], yy, s=34, facecolor="white", edgecolor=col, linewidths=1.3,
                         zorder=4)
            ax_b.scatter(r["mapped"], yy, s=46, marker=_tier_marker(r["model"]), color=col,
                         zorder=4)
        ax_b.set_xlabel("Balanced accuracy (averaged over datasets)")
        # Bound on the per-dataset points, not on the means: several datasets sit well outside
        # their model's mean and would otherwise be clipped at the axis.
        ax_b.set_xlim(-0.02, float(np.nanmax([d["ba_direct"].max(), d["ba_mapped"].max()]))
                      + 0.04)

        ax_a.set_yticks(y)
        ax_a.set_yticklabels([_display(b) + unsupported_mark(b) for b in agg["base"]])
        ax_a.set_ylim(agg["ypos"].max() + 1.0, 0.0)   # inverted: first block at the top
        plt.setp(ax_b.get_yticklabels(), visible=False)
        for ax in (ax_a, ax_b):
            ax.grid(axis="x", color="0.9", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        # Block headings sit in the blank rows, anchored to the left spine so they clear both the
        # method names and the bars. Sorting is WITHIN a block, so each heading also marks where
        # the ordering restarts.
        trans = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
        for title, head_y, _block in blocks:
            ax_a.text(0.0, head_y, title, transform=trans, ha="left", va="center",
                      fontsize=FS_LEGEND, fontweight="bold", color="0.25")
        for ax in (ax_a, ax_b):
            _harmonize(ax, text_size=FS_POINT)

        # Figure 2's key: B's markers and the dots, then the families with their subcategories.
        dot = dict(linestyle="none", marker="o", markersize=np.sqrt(DOT_SIZE))
        _family_key(ax_key, [
            plt.Line2D([], [], linestyle="none", marker="o", markersize=7,
                       markerfacecolor="white", markeredgecolor="0.3", markeredgewidth=1.3,
                       label="no curated mapping"),
            plt.Line2D([], [], linestyle="none", marker="o", markersize=7, color="0.3",
                       label="with curated mapping"),
            plt.Line2D([], [], color=DOT_COLOR, label="one dataset", **dot),
            plt.Line2D([], [], color=DOT_COLOR_NO_MAPPING,
                       label="one dataset, no curated mapping", **dot)])

        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B")])
        return _save(fig, out_path, dpi)


DEFAULT_RARE_CSV = "cache/rare_type_outcomes_1000_stratified.csv"
DEFAULT_PER_CLASS_CSV = "cache/per_class_scores_1000_stratified.csv"

# --------------------------------------------------------------------------- #
# Figure 4 and Supplementary Figures S7-S11 -- rare cell types (R3).
#
# Rare means below RARE_PREVALENCE_MAX of the FULL dataset, not of the evaluation subset: the
# subsets hold at least 20 cells of every type, so a subset's own prevalence is mostly the sampler
# (bin/rare_type_table.py). One definition for every panel, the supplement and the flows.
# --------------------------------------------------------------------------- #
RARE_PREVALENCE_MAX = 0.01
# The flows draw the rarest types below it, at most this many per dataset: HFRA has two, human
# pancreas three, HIHA and Zeisel more than five.
RARE_FLOW_MAX_TYPES = 5
# The dataset of the worked example in Figure 4C. The other three keep their full grids in the
# supplement; this one is a window onto them, drawn with the same primitive and the same model
# representatives.
RARE_FLOW_EXAMPLE = "human_pancreas"
# The four closed-weight LLMs, a group within the LLM family -- NOT a fourth family. Membership is
# the intrinsic property recorded in CATEGORY_TIER_MEMBERS["LLM"]["closed_reasoning"], never a
# rank on our own benchmark, so the comparison stays non-circular.
CLOSED_WEIGHT_TIER = "closed_reasoning"
# The method groups of S7A, named as Figure 4's rows (RARITY_ROWS) and in their order.
BLOCK_LABEL = {"unconstrained baselines": "baselines, U", "constrained baselines": "baselines, C",
               "unconstrained scLLM": "scLLMs, U", "constrained scLLM": "scLLMs, C",
               "unconstrained LLM": "LLMs, U", "constrained LLM": "LLMs, C",
               "unconstrained closed-weight LLM": "closed-weight LLMs, U",
               "constrained closed-weight LLM": "closed-weight LLMs, C"}
BLOCK_ORDER = list(BLOCK_LABEL)
# Figure 4B's outcomes for a rare cell type, best first as Figure 2H puts the exact type first:
# (key, legend text, grey). A type the constrained baselines never named is split by whether the
# method has a label for it at all (`reachable`, bin/rare_type_table.py); that part is hatched.
RARE_OUTCOMES = [
    ("most", "\u2265 50% of cells correct", "0.94"),
    ("some", "< 50% of cells correct", "0.78"),
    ("wrong", "named, never correct", "0.56"),
    ("never", "never named", "0.30"),
    ("unreachable", "never named, no label", "0.30"),
]


def load_per_class_scores(path=DEFAULT_PER_CLASS_CSV):
    """Per (model, dataset, true cell type) recall and Cell Ontology Lin similarity.

    Built by bin/per_class_scores.py. Its recall column reproduces the rare-type table exactly,
    so the two are joined rather than reconciled: prevalence and `emitted` come from
    load_rare_outcomes(), the graded score from here.
    """
    if not Path(path).exists():
        raise SystemExit(
            f"{path} not found. Build it with:\n"
            f"    conda run -n evaluate python bin/per_class_scores.py")
    d = pd.read_csv(path)
    d = d[d["cell_type"] != "__OVERALL__"].copy()
    # Every panel built from this table averages over a family's models.
    d = with_unsupported_as_zero(d, zero=dict.fromkeys(("recall", "lin", "rada", "agreement"), 0.0),
                                 keep=("cell_type", "n_cells"))
    d["model_base"] = d["model"].map(model_base)
    d = d[~d["model_base"].isin(EXCLUDED_BASES)]
    d["category"] = d["model"].map(categorize_model)
    d["mode"] = d["model"].map(derive_mode)
    # Lowercased as the key the rare-type table joins on; the Cell Ontology spelling is kept for
    # display.
    d["cell_type_name"] = d["cell_type"].astype(str).str.strip()
    d["cell_type"] = d["cell_type_name"].str.lower()
    d = d[d["category"].isin({"classical", "LLM", "scLLM"})]
    return d.assign(block=d["model"].map(group_name))


def _with_closed_weight(d):
    """``d`` plus a copy of the closed-weight LLMs' rows under their own ``block``, for the panels
    that group by it. The run is ``model`` in the per-class scores and ``method`` in the rare-type
    table."""
    col = "model" if "model" in d.columns else "method"
    closed = d[(d[col].map(categorize_model) == "LLM")
               & (d[col].map(model_tier) == CLOSED_WEIGHT_TIER)]
    return pd.concat([d, closed.assign(block=closed["block"].str.replace(
        " LLM", " closed-weight LLM", regex=False))], ignore_index=True)


def load_rare_outcomes(path=DEFAULT_RARE_CSV):
    """Per-class recall, emission and reachability by true dataset prevalence. Built by
    bin/rare_type_table.py."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"No rare-type table at {path}. Build it with:\n"
            f"    conda run -n evaluate python bin/rare_type_table.py")
    # Every panel built from this table averages over a family's models.
    return with_unsupported_as_zero(
        pd.read_csv(path), zero={"recall": 0.0, "emitted": False},
        keep=("k", "n_cells", "orig_prevalence", "orig_count"),
        model_cols=("method", "mode", "family", "block"))


def _prevalence(rare):
    """``{(dataset, cell type): share of the full dataset}``, cell types lowercased."""
    return (rare.drop_duplicates(["dataset", "k"])
                .set_index(["dataset", "k"])["orig_prevalence"].to_dict())


def _rarity_rates(rare):
    """Per run (index ``<method>_<mode>``), Figure 4's two panels.

    ``common`` and ``rare`` are the mean per-class recall of the common and of the rare cell types,
    each averaged over a dataset's types and then over the datasets, as balanced accuracy is: the
    comparison stays within a dataset, since the rarest types sit almost all in HIHA and Zeisel.
    The RARE_OUTCOMES keys are shares of all rare types of the four datasets.
    """
    rare = rare.assign(run=rare["method"] + "_" + rare["mode"],
                       is_rare=rare["orig_prevalence"] < RARE_PREVALENCE_MAX)
    recall = (rare.groupby(["run", "dataset", "is_rare"])["recall"].mean()
                  .groupby(["run", "is_rare"]).mean().unstack("is_rare")
                  .rename(columns={False: "common", True: "rare"}))
    types = rare[rare["is_rare"]]
    emitted = types["emitted"].astype(bool)
    unreachable = ~emitted & types["reachable"].eq(False)
    outcomes = pd.DataFrame({
        "run": types["run"],
        "most": types["recall"] >= 0.5,
        "some": (types["recall"] > 0) & (types["recall"] < 0.5),
        "wrong": emitted & (types["recall"] == 0),
        "never": ~emitted & ~unreachable,
        "unreachable": unreachable}).groupby("run").mean()
    assert np.allclose(outcomes.sum(axis=1), 1.0), "the outcomes must partition the rare types"
    return recall.join(outcomes)


def _row_marker(row):
    """Marker and area for a row of RARITY_ROWS: a circle for a family, as in Figure 2G, and for a
    row of one subcategory its marker of Figure 1B, at its area in Figure 2's family plots."""
    if len(row) < 4:
        return "o", 46
    return next((marker, area) for tier, _label, marker, area in CATEGORY_TIERS[row[0]]
                if tier == row[3])


def _paired_rows(ax, means, rows, open_col, filled_col):
    """One row per entry of ``rows``, first on top: an open marker at ``open_col`` and a filled one
    at ``filled_col`` of ``means``, joined by a line in the family's colour (Figure 2G, 4A)."""
    colors = _ranked_category_colors()
    y = np.arange(len(rows))[::-1]
    for yy, row, (_label, mean) in zip(y, rows, means.iterrows()):
        color = colors[row[0]]
        marker, size = _row_marker(row)
        ax.plot([mean[open_col], mean[filled_col]], [yy, yy], color=color, linewidth=2.2,
                zorder=2)
        ax.scatter(mean[open_col], yy, s=size, marker=marker, facecolor="white", edgecolor=color,
                   linewidth=1.6, zorder=3)
        ax.scatter(mean[filled_col], yy, s=size, marker=marker, color=color, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(means.index)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    return y


def _outcome_bars(ax, shares, y):
    """Figure 4B: one stacked bar of RARE_OUTCOMES per row of ``shares``, at heights ``y``."""
    left = np.zeros(len(shares))
    for key, label, grey in RARE_OUTCOMES:
        hatched = key == "unreachable"
        ax.barh(y, shares[key].to_numpy(), left=left, height=0.7, color=grey,
                edgecolor="0.55", linewidth=0.4, hatch="////" if hatched else None,
                hatchcolor="white" if hatched else None, label=label)
        left += shares[key].to_numpy()
    ax.set_xlim(0, 1)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)


def _rare_dotplot(ax, rows, rare, per_class, blocks, s_min=22.0, s_span=170.0,
                  label_fs=None, cmap_name="viridis"):
    """Draw the per-cell-type dot grid of Supplementary Figure S7A.

    ``rows`` is ``[("__header__", label, None) | (dataset, cell_type, prevalence)]`` in draw order.
    Marker area is per-class recall, color is per-class Lin similarity, and a ringed marker with a
    cross marks a cell type no method in that group ever emitted.

    Returns ``(norm, s_min, s_span)`` so the caller can build a matching key.
    """
    label_fs = FS_POINT if label_fs is None else label_fs
    never = {}
    for r in rare.itertuples():
        k = (r.dataset, r.k, r.block)
        never[k] = never.get(k, False) or bool(r.emitted)
    grouped = per_class.groupby(["dataset", "cell_type", "block"])[["recall", "lin"]].mean()
    names = (per_class.drop_duplicates(["dataset", "cell_type"])
             .set_index(["dataset", "cell_type"])["cell_type_name"])
    cmap = plt.get_cmap(cmap_name); norm = mcolors.Normalize(0, 1)

    ypos, y, y_first = {}, 0.0, None
    for kind, name, p in rows:
        if kind == "__header__":
            y -= 0.78
            if y_first is None:
                y_first = y
            ax.text(-0.012, y, name, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=label_fs, fontweight="bold", clip_on=False)
            y -= 0.80
            continue
        ypos[(kind, name)] = y
        pv = f"{p * 100:.3f}%" if p < 0.1 else f"{p * 100:.1f}%"
        # The Cell Ontology spelling with the prevalence, wrapped rather than cut short; at 52
        # characters only the longest name takes two lines.
        label = textwrap.fill(f"{names.get((kind, name), name)} ({pv})", width=52)
        ax.text(-0.012, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=label_fs, linespacing=1.2, clip_on=False)
        y -= 1.0
    for (ds, t), yy in ypos.items():
        for xi, block in enumerate(blocks):
            if (ds, t, block) not in grouped.index:
                continue
            rec, lin = grouped.loc[(ds, t, block), ["recall", "lin"]]
            size = s_min + s_span * float(rec)
            if not never.get((ds, t, block), True):
                ax.scatter(xi, yy, s=size, facecolors="none", edgecolors=cmap(norm(lin)),
                           linewidths=1.4, zorder=3)
                ax.scatter(xi, yy, marker="x", s=15, c="0.35", linewidths=0.85, zorder=4)
            else:
                ax.scatter(xi, yy, s=size, color=cmap(norm(lin)), edgecolors="0.25",
                           linewidths=0.4, zorder=3)
    for xi in range(len(blocks)):
        ax.axvline(xi, color="0.93", lw=0.6, zorder=0)
    ax.set_xlim(-0.6, len(blocks) - 0.4)
    ax.set_ylim(y + 0.5, (y_first if y_first is not None else 0.0) + 0.45)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([BLOCK_LABEL[b] for b in blocks], fontsize=label_fs, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks([]); ax.tick_params(axis="x", length=0, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return norm, s_min, s_span


def figure_s7_rare_full(df, out_path, dpi=600, rare=None, per_class=None):
    """Supplementary -- every rare cell type per method group: all cell types below
    RARE_PREVALENCE_MAX, the set Figure 4B counts, in Figure 4's groups.

    Sized to print at the full text width: the float holds the figure and its caption within the
    text height (S7 continues on the next float with the answer types).
    """
    rare = load_rare_outcomes() if rare is None else rare
    per_class = load_per_class_scores() if per_class is None else per_class
    prevalence = _prevalence(rare)
    rows = []
    for dataset in DATASET_ORDER:
        types = sorted(((k, p) for (d, k), p in prevalence.items()
                        if d == dataset and p < RARE_PREVALENCE_MAX), key=lambda kp: kp[1])
        rows.append(("__header__", f"{dataset_label(dataset)}  ({len(types)})", None))
        rows += [(dataset, k, p) for k, p in types]
    n_types = sum(1 for r in rows if r[0] != "__header__")
    # _rare_dotplot steps 1.0 per cell type and 1.58 per dataset heading.
    units = n_types + 1.58 * (len(rows) - n_types)

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(11.5, 0.26 * units + 1.1))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_key = fig.add_subplot(gs[0, 1]); ax_key.axis("off")
        norm, s_min, s_span = _rare_dotplot(ax, rows, _with_closed_weight(rare),
                                            _with_closed_weight(per_class), BLOCK_ORDER)
        # Inside the key axes, left-aligned with the marker column.
        cax = ax_key.inset_axes([0.185, 0.30, 0.065, 0.32])
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("viridis")), cax=cax)
        cb.set_label("Lin similarity to the true label", fontsize=FS_POINT)
        cb.ax.tick_params(labelsize=FS_LEGEND)
        handles = [plt.Line2D([0], [0], linestyle="none", marker="o", mfc="0.62", mec="0.25",
                              ms=np.sqrt(s_min + s_span * v) / 1.7, label=f"recall {v:g}")
                   for v in (0.0, 0.5, 1.0)]
        # The never-emitted marker is a hollow ring with a cross inside it, so the key overlays
        # the same pair of handles.
        handles.append((plt.Line2D([0], [0], linestyle="none", marker="o", mfc="none",
                                   mec="0.4", ms=7.5, mew=1.4),
                        plt.Line2D([0], [0], linestyle="none", marker="x", color="0.35",
                                   ms=4.2, mew=0.9)))
        labels = [h.get_label() for h in handles[:3]] + ["label never emitted"]
        ax_key.legend(handles=handles, labels=labels, loc="upper left", frameon=False,
                      fontsize=FS_LEGEND, labelspacing=1.0, handletextpad=0.6,
                      borderaxespad=0.0, alignment="left",
                      handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)})
        _panel_letters(fig, [(ax, "A")])
        print(f"    [s7] {n_types} cell types below 1%")
        return _save(fig, out_path, dpi)


def figure_s7b_rare_answer_types(df, out_path, dpi=600, rare=None, pairs=None):
    """Supplementary -- S7 continued: where the cells of the common and of the rare cell types go.

    Backs R3's statement that the missed cells of rare types mostly go to another cell type
    rather than to a coarser or finer one: Figure 2H's answer types, for the common and the rare
    types separately, in Figure 4's rows and averaged like them within each dataset.
    """
    rare = load_rare_outcomes() if rare is None else rare
    pairs = _benchmark_pairs(with_unsupported_as_zero(df, model_cols=("model", "mode")), pairs)
    prevalence = _prevalence(rare)
    is_rare = np.array([prevalence[(d, t.strip().lower())] < RARE_PREVALENCE_MAX
                        for d, t in zip(pairs["dataset"], pairs["true_label"])])

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(11.5, 3.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.36], wspace=0.14)
        axes = [fig.add_subplot(gs[0, 0])]
        axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
        ax_key = fig.add_subplot(gs[0, 2]); ax_key.axis("off")
        y = np.arange(len(RARITY_ROWS))[::-1]
        for ax, keep, title in ((axes[0], ~is_rare, "Common cell types (\u2265 1%)"),
                                (axes[1], is_rare, "Rare cell types (< 1%)")):
            shares = _group_means(_answer_shares(pairs[keep]).groupby("model").mean(),
                                  RARITY_ROWS)
            _stacked_shares(ax, shares, y)
            ax.set_yticks(y)
            ax.set_yticklabels(shares.index)
            ax.set_ylim(-0.6, len(RARITY_ROWS) - 0.4)
            ax.set_title(title, fontsize=FS_TITLE)
            ax.set_xlabel("Share of cells (averaged over cell types,\nmethods and datasets)")
            _harmonize(ax, text_size=FS_POINT)
        axes[1].tick_params(labelleft=False)
        ax_key.legend(handles=axes[0].get_legend_handles_labels()[0], loc="center left",
                      frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0,
                      handlelength=1.6)
        _panel_letters(fig, [(axes[0], "B")])
        return _save(fig, out_path, dpi)


def figure_3(df, out_path, dpi=600, rare=None):
    """Figure 4 -- rare cell types, in Figure 2G/H's rows: every family in both settings, then the
    closed-weight LLMs.

    (A) Mean per-class recall of the common and of the rare cell types, compared within each
        dataset (_rarity_rates).
    (B) For the rare types, how far each group got: correct for most of the type's cells, for
        some, named but never on its cells, or never named, which for the constrained baselines is
        split by whether the method has a label for the type at all.
    (C) One worked example of where the cells of the rare types go, for human pancreas. The
        methods are the representatives of the supplementary flow figures (select_flow_models),
        so the example is a window onto them rather than a separate selection.
    """
    rare = load_rare_outcomes() if rare is None else rare
    conf = load_confusion_pairs()
    flow_models = select_flow_models(df)
    rates = _group_means(_rarity_rates(rare), RARITY_ROWS)

    with plt.rc_context(_rc()):
        # Drawn at the output dpi: _align_left measures text, and hinted text widths depend on it.
        fig = plt.figure(figsize=(13.4, 9.4), dpi=dpi)
        # A and B share their rows, so B carries no row names. The flows below get the full width:
        # only the first of them draws left-hand labels, and _sankey draws the right-hand ones
        # outside its axes, so `wspace` has to clear them (see _sankey's right_margin). A's left
        # edge is aligned with C's at the end (_align_left), so `left` here is only a start.
        top = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.30], wspace=0.10,
                               left=0.16, right=0.985, top=0.985, bottom=0.64)
        ax_a = fig.add_subplot(top[0, 0])
        ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
        ax_bkey = fig.add_subplot(top[0, 2]); ax_bkey.axis("off")
        bottom = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.11],
                                  width_ratios=[1.16, 1.0, 1.0, 1.0], wspace=0.30, hspace=0.12,
                                  left=0.05, right=0.985, top=0.53, bottom=0.0)
        ax_c = [fig.add_subplot(bottom[0, i]) for i in range(4)]
        ax_ckey = fig.add_subplot(bottom[1, :]); ax_ckey.axis("off")

        # --- A: common against rare cell types --------------------------------------------
        _paired_rows(ax_a, rates, RARITY_ROWS, "common", "rare")
        ax_a.set_xlim(0, 0.8)
        ax_a.set_xticks(np.arange(0, 0.81, 0.2))
        ax_a.set_xlabel("Mean per-class recall")
        ax_a.legend(handles=[
            plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                       markeredgecolor="0.3", markersize=7, label="common (\u2265 1%)"),
            plt.Line2D([], [], linestyle="none", marker="o", color="0.3", markersize=7,
                       label="rare (< 1%)")],
            loc="upper right", frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0)

        # --- B: what became of each rare cell type -----------------------------------------
        _outcome_bars(ax_b, rates, np.arange(len(RARITY_ROWS))[::-1])
        ax_b.tick_params(labelleft=False)
        ax_b.set_xlabel("Share of rare cell types")
        ax_bkey.legend(handles=ax_b.get_legend_handles_labels()[0], loc="center left",
                       frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0,
                       handlelength=1.6, labelspacing=0.9)
        for ax in (ax_a, ax_b):
            _harmonize(ax, text_size=FS_POINT)

        # --- C: one worked example of where the rare cells go -------------------------------
        # The top representative of each family, constrained, plus the unconstrained run of the
        # LLM, placed before the constrained one so its two panels read U then C.
        c_models = [flow_models[0][0], flow_models[0][1], flow_models[1][2], flow_models[0][2]]
        csub = conf[conf["dataset"].astype(str) == RARE_FLOW_EXAMPLE]
        # Full-dataset prevalence, as in the supplementary flow figures, so the rare types are
        # chosen, ordered and coloured the same way in both.
        prevalence = _prevalence(rare)
        prev_s = pd.Series({t: prevalence[(RARE_FLOW_EXAMPLE, t.strip().lower())]
                            for t in csub["true_label"].unique()})
        rare_t = list(prev_s[prev_s < RARE_PREVALENCE_MAX].nsmallest(RARE_FLOW_MAX_TYPES).index)
        pal = sns.color_palette("tab10", n_colors=max(len(rare_t), 3)).as_hex()
        fcolors = {t: pal[i % len(pal)] for i, t in enumerate(rare_t)}
        fcounts = (csub[(csub["model"] == c_models[0]) & (csub["true_label"].isin(rare_t))]
                   .groupby("true_label")["n_cells"].sum())
        for i, (ax, model) in enumerate(zip(ax_c, c_models)):
            m = csub[(csub["model"] == model) & (csub["true_label"].isin(rare_t))]
            if m.empty:
                raise SystemExit(f"Figure 4C: {model} has no predictions on {RARE_FLOW_EXAMPLE}.")
            flows = []
            for true_label, grp in m.groupby("true_label"):
                grp = grp.sort_values("n_cells", ascending=False)
                keep, rest = grp.head(6), grp.tail(max(len(grp) - 6, 0))
                flows += [(true_label, t.predicted_label, float(t.n_cells))
                          for t in keep.itertuples()]
                if len(rest):
                    flows.append((true_label, "other", float(rest["n_cells"].sum())))
            w = {}
            for src, tgt, val in flows:
                a0, b0 = w.get(tgt, (0.0, 0.0))
                w[tgt] = (a0 + rare_t.index(src) * val, b0 + val)
            right = sorted(w, key=lambda t: (t == "other", w[t][0] / w[t][1]))
            # right_margin=2.9 rather than the 1.9 default: the longest target label would
            # otherwise overrun the next panel.
            _sankey(ax, flows, rare_t, right, fcolors, show_left_labels=(i == 0),
                    label_pad=0.11, right_margin=2.9)
            ax.set_title(_run_label(model), fontsize=FS_POINT)
        # The ribbon-width and top-6 rules are sentences, not encodings: they go in the caption.
        _flow_key(ax_ckey, rare_t, fcolors, fcounts, 6, label_size=FS_LEGEND, ncol=5,
                  notes=False)
        _align_left(fig, top, ax_a, ax_c[0])
        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c[0], "C")])
        return _save(fig, out_path, dpi)


def select_flow_models(df, metric="balanced_accuracy"):
    """One representative per family for the rare-type flow figures, by a stated rule.

    Rule: the highest mean ``metric`` of a constrained run within its family, among models that
    ran on EVERY dataset. Two rows come back — constrained on top, unconstrained beneath, same
    model per column.

    Why not the best model per dataset. It changes the scLLM across the four figures (at present
    LangCell, mmContext, mmContext and InstructCell) and in doing so crosses the
    generative/retrieval split, whose members behave oppositely when unconstrained. The four
    supplementary figures would stop being comparable, and a change of sub-family would read as a
    change of failure mode.

    Why "ran on every dataset" is part of the rule rather than a caveat. A model the pipeline does
    not run on one organism (params.model_blacklist_by_organism) would leave that dataset's figure
    without its family, and a mean over fewer datasets is not comparable with four-dataset means.
    The blacklist is empty at present, so the clause excludes nothing.
    """
    d = df.copy()
    d["model_base"] = d["model"].map(model_base)
    d["category"] = d["model"].map(categorize_model)
    d["mode"] = d["model"].map(derive_mode)
    d = d[(~d["model_base"].isin(EXCLUDED_BASES)) & (d["category"] != "simple")]
    n_datasets = d["dataset"].nunique()

    picks = {}
    for category in ("classical", "scLLM", "LLM"):
        sub = d[(d["category"] == category) & (d["mode"] == "unblind")]
        if sub.empty:
            picks[category] = None
            continue
        by_model = sub.groupby("model_base").agg(score=(metric, "mean"),
                                                 n=("dataset", "nunique"))
        complete = by_model[by_model["n"] == n_datasets]
        by_model = complete if not complete.empty else by_model
        picks[category] = by_model["score"].idxmax()

    def named(base, mode):
        if base is None:
            return None
        hit = d[(d["model_base"] == base) & (d["mode"] == mode)]["model"]
        return hit.iloc[0] if len(hit) else None

    return [[named(picks[category], mode) for category in ("classical", "scLLM", "LLM")]
            for mode in ("unblind", "blind")]
# Order the per-dataset flow figures appear in the supplement, and therefore their S-numbers.
# Alphabetical, the dataset order of every per-dataset panel (DATASET_ORDER).
RARE_FLOW_ORDER = ["HFRA", "HIHA", "Zeisel", "human_pancreas"]


def figure_rare_flows(df, out_path, dpi=600, dataset="HIHA", models=None):
    """Supplementary — where the cells of the rarest types actually go.

    Figure 4A and 4B say how often rare types are labeled correctly; neither says what they are
    called instead. One alluvial per model and regime: the rarest types below RARE_PREVALENCE_MAX, at
    most RARE_FLOW_MAX_TYPES, on the left, and the predicted labels on the right.

    ``df`` is unused — kept so every figure builder has the same signature.
    """
    conf = load_confusion_pairs()
    if conf is None or len(conf) == 0:
        raise SystemExit("Rare-type flows: no confusion-pair table; see cache/confusion_pairs_*.")
    models = models or select_flow_models(df)
    with plt.rc_context(_rc()):
        # No in-figure title: the caption states the dataset and what is shown, and a title
        # duplicating it is redundant ink in a published figure.
        # Full-dataset prevalence and the manuscript's display names, so the supplementary flows
        # select the same cell types the rare-type analysis reports and spell the methods the way
        # the main figures do.
        rare_tbl = load_rare_outcomes()
        prev_map = (rare_tbl[rare_tbl["dataset"] == dataset]
                    .drop_duplicates(["k"]).set_index("k")["orig_prevalence"].to_dict())
        ok = plot_rare_type_flows(conf, out_path, dpi, dataset=dataset, models=models,
                                  prevalence_map=prev_map, name_fn=_display,
                                  n_types=RARE_FLOW_MAX_TYPES,
                                  prevalence_max=RARE_PREVALENCE_MAX, title=None)
    if not ok:
        raise SystemExit(f"Rare-type flows: nothing to draw for {dataset}.")
    return Path(out_path)


# --------------------------------------------------------------------------- #
# Figure 2 — the benchmark results (R1).
#
# A ranks every run on balanced accuracy averaged over the datasets, each dataset's value a dot.
# B is the family plot averaged over the datasets and C-F the same plot per dataset, so the effect
# of the candidate list can be read on every dataset. G shows how much each family's score rests
# on the curated mapping (Figure 1D), H where the LLMs' answers sit in the ontology relative to the
# true type (Figure 1E). G and H read the confusion pairs of bin/per_class_scores.py.
# --------------------------------------------------------------------------- #
# Panel H's answer types in stacking order: key, legend text, colour. They are those of
# CellTaxonomy.lineage (bin/util/cell_ontology_similarity.py), which sets the cut between a
# specific and a broad coarser type.
LINEAGE_KEY = [
    ("exact", "exact type", "#1b7837"),
    ("coarser", "coarser, specific\n(e.g. T cell)", "#7fbf7b"),
    ("broad", "coarser, broad\n(e.g. neuron)", "#d9f0d3"),
    ("finer", "finer", "#c2a5cf"),
    ("other", "another cell type", "#bababa"),
    ("none", "no usable answer", "#ffffff"),
]
# Where B-F try a method's name, in points from its marker: above it, higher up over its own
# column, and only then beside it, so a name never drifts to another column. Far enough out that
# the thin line linking every name to its marker stays visible.
LABEL_CANDIDATES = [(0, 12, "center", "bottom"), (0, 20, "center", "bottom"),
                    (0, 28, "center", "bottom"), (-12, 0, "right", "center"),
                    (12, 0, "left", "center"), (-10, 8, "right", "bottom"),
                    (10, 8, "left", "bottom"), (0, -12, "center", "top"),
                    (-20, 0, "right", "center"), (20, 0, "left", "center")]
# The chance level in A to F and in the key: black and heavier than the grid, so it reads over bars.
CHANCE_LINE = dict(linestyle="--", linewidth=1.2, color="black")
# One dot per dataset, in Figure 2A and Supplementary Figure S1; light for a score without the
# curated mapping.
DOT_SIZE, DOT_COLOR, DOT_COLOR_NO_MAPPING = 6, "0.15", "0.6"
# The two settings in the keys of Figure 2 and Supplementary Figure S6, worded as in Figure 1B.
SETTING_KEY = {"unblind": r"$\bf{C}$  constrained: expected cell types supplied",
               "blind": r"$\bf{U}$  unconstrained: no expected cell types"}
# Figure 1B's names for the families and the subcategories that differ from CATEGORY_TIERS.
FAMILY_HEADER = {"classical": "Marker- and reference-based baselines",
                 "scLLM": "Single-cell language models (scLLMs)",
                 "LLM": "General-purpose LLMs"}
TIER_NAME = {"generative": "generative (free text)", "retrieval": "embedding (closed set)"}
# Panel G's rows, top to bottom: (family, mode, label), every family in both settings.
MAPPING_ROWS = [(family, mode, f"{name}, {REGIME_SHORT[mode]}")
                for family, name in (("classical", "baselines"), ("scLLM", "scLLMs"),
                                     ("LLM", "LLMs"))
                for mode in ("blind", "unblind")]
# Figure 4's rows and S7B's: G's, then the closed-weight LLMs, who are also among the LLMs.
RARITY_ROWS = MAPPING_ROWS + [("LLM", mode, f"closed-weight LLMs, {REGIME_SHORT[mode]}",
                               CLOSED_WEIGHT_TIER) for mode in ("blind", "unblind")]


def _run_label(model):
    """Display name with the run's setting, U or C."""
    return run_label(model, _display(model_base(model)))


def _is_unconstrained(model):
    return derive_mode(model) == "blind"


def _ranked_panel(ax, df):
    """Panel A: every run, best first; dots are the datasets in DATASET_ORDER, left to right."""
    runs = df[df["model"].map(categorize_model) != "simple"]
    wide = runs.pivot_table(index="model", columns="dataset", values="balanced_accuracy")
    wide = wide[DATASET_ORDER].loc[wide.mean(axis=1).sort_values(ascending=False).index]
    colors = _ranked_category_colors()
    x = np.arange(len(wide))
    for xi, model in enumerate(wide.index):
        unconstrained = _is_unconstrained(model)
        ax.bar(xi, wide.loc[model].mean(), width=0.8, color=colors[categorize_model(model)],
               hatch="///" if unconstrained else None, linewidth=0.5,
               edgecolor="black" if unconstrained else "white", zorder=3)
    for offset, dataset in zip(np.linspace(-0.27, 0.27, len(DATASET_ORDER)), DATASET_ORDER):
        ax.scatter(x + offset, wide[dataset], s=DOT_SIZE, color=DOT_COLOR, linewidth=0, zorder=4)
    chance = df.loc[df["model"] == "random_stratified", "balanced_accuracy"].mean()
    ax.axhline(chance, zorder=5, **CHANCE_LINE)  # over the bars
    ax.set_xticks(x)
    # The hatching gives the setting, so the names carry no U or C.
    ax.set_xticklabels([_display(model_base(m)) for m in wide.index], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(wide) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)


def _list_effect_panel(ax, frame, title):
    """The family plot on ``frame``, with the best constrained run of each family named. Returns
    the legend handles."""
    handles, points = plot_composite_category(frame, None, "", 300, ax=ax,
                                              metric="balanced_accuracy", return_points=True)
    _harmonize(ax, text_size=FS_POINT + 1)  # family names under the axis, five panels abreast
    _relabel_regimes(ax)
    for line in ax.lines:
        if line.get_linestyle() == "--":  # the chance level, styled as in A
            line.set(zorder=3, **CHANCE_LINE)
    ax.set_ylim(0, 1)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=FS_LABEL)
    best = {}
    for (base, mode), (x, y, _r) in points.items():
        family = categorize_model(base)
        if mode == "unblind" and y > best.get(family, ("", 0, -1))[2]:
            best[family] = (base, x, y)
    # Centred above the point first, then beside it, so each name sits on its own marker. The
    # LLM's name is lifted and centred over both LLM columns, clear of the scLLMs' names, unless
    # the best LLM is too close to the top for that. When the space right above the scLLM's
    # marker is taken, its name goes centred over both scLLM columns, above the family's highest
    # marker, before it tries the sides. Every name is linked to its marker.
    def family_points(family):
        return [(px, py) for (b, _m), (px, py, _r) in points.items() if categorize_model(b) == family]
    llm_centre = np.mean([px for px, _py in family_points("LLM")])
    ax.figure.canvas.draw()  # final axes position, to turn that data offset into points
    to_points = 72.0 / ax.figure.dpi
    labels = []
    for family, (base, x, y) in best.items():
        label = (_display(base), x, y, "0.15")
        if family == "LLM" and y < 0.85:
            (x0, y0), (x1, y1) = ax.transData.transform([(x, y), (llm_centre, y + 0.05)])
            dx, dy = (x1 - x0) * to_points, (y1 - y0) * to_points
            label += ([(dx, dy + lift, "center", "bottom") for lift in (0, 8, 16)]
                      + LABEL_CANDIDATES,)
        elif family == "scLLM":
            scllm = family_points("scLLM")
            top = (np.mean([px for px, _py in scllm]), max(py for _px, py in scllm))
            (x0, y0), (x1, y1) = ax.transData.transform([(x, y), top])
            dx, dy = (x1 - x0) * to_points, (y1 - y0) * to_points
            label += (LABEL_CANDIDATES[:3]
                      + [(dx, dy + lift, "center", "bottom") for lift in (12, 20, 28)]
                      + LABEL_CANDIDATES[3:],)
        labels.append(label)
    _place_point_labels(ax, labels, list(points.values()), FS_POINT,
                        candidates=LABEL_CANDIDATES, leader_min=0)
    return handles


def _class_balanced(pairs, counts):
    """Per run and dataset, the mean over true cell types of each count's share of the type's
    cells, the way balanced accuracy averages recall. ``counts`` maps a name to a count per pair."""
    keys = [pairs["model"], pairs["dataset"], pairs["true_label"]]
    per_type = pd.DataFrame(counts).groupby(keys).sum()
    return per_type.div(pairs["n_cells"].groupby(keys).sum(), axis=0) \
                   .groupby(["model", "dataset"]).mean()


def _benchmark_pairs(df, pairs=None, chance=False):
    """The confusion pairs of the runs in ``df``, with the columns G and H need; those of the
    chance models only with ``chance``."""
    pairs = load_confusion_pairs() if pairs is None else pairs
    if pairs.empty or not {"n_direct", "lineage"} <= set(pairs.columns):
        raise SystemExit("Figure 2G/H need the confusion pairs with n_direct and lineage: "
                         "run bin/per_class_scores.py.")
    runs = df["model"] if chance else df.loc[df["model"].map(categorize_model) != "simple", "model"]
    return pairs[pairs["model"].isin(set(runs))]


def _group_means(per_model, rows):
    """Mean of ``per_model`` (one row per run) over the runs of each ``(family, mode, label)`` of
    ``rows``, optionally ``(family, mode, label, tier)``."""
    means = []
    for family, mode, _label, *tier in rows:
        keep = [categorize_model(m) == family and derive_mode(m) == mode
                and (not tier or model_tier(m) == tier[0]) for m in per_model.index]
        means.append(per_model[keep].mean())
    return pd.DataFrame(means, index=[row[2] for row in rows])


def _mapping_scores(pairs):
    """Balanced accuracy per run and dataset with the curated mapping, and counting only the
    answers that needed none: Cell Ontology terms or the dataset's own labels, as written."""
    hit = pairs["true_label"] == pairs["predicted_label"]
    return _class_balanced(pairs, {"with": pairs["n_cells"].where(hit, 0),
                                   "without": pairs["n_direct"].where(hit, 0)})


def _needs_mapping(pairs):
    """Share of cells per run and dataset whose answer needed the curated mapping, over all cells
    rather than per cell type."""
    cells = pairs.groupby(["model", "dataset"])[["n_cells", "n_direct"]].sum()
    return 1.0 - cells["n_direct"] / cells["n_cells"]


def _answer_shares(pairs):
    """Class-balanced share of each answer type of LINEAGE_KEY per run and dataset."""
    return _class_balanced(pairs, {k: pairs["n_cells"].where(pairs["lineage"] == k, 0)
                                   for k, *_ in LINEAGE_KEY})


def _stacked_shares(ax, shares, y):
    """One stacked bar of answer-type shares per row of ``shares``, at heights ``y``."""
    left = np.zeros(len(shares))
    for key, label, color in LINEAGE_KEY:
        ax.barh(y, shares[key].to_numpy(), left=left, height=0.7, color=color,
                edgecolor="0.55", linewidth=0.4, label=label)
        left += shares[key].to_numpy()
    ax.set_xlim(0, 1)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)


def _mapping_panel(ax, pairs):
    """Panel G: balanced accuracy with the curated mapping (filled) and counting only the answers
    that needed none (open), per family and setting, averaged over the family's methods and the
    datasets."""
    means = _group_means(_mapping_scores(pairs).groupby("model").mean(), MAPPING_ROWS)
    _paired_rows(ax, means, MAPPING_ROWS, "without", "with")  # first row on top, level with H's
    ax.set_xlim(0, 0.7)  # room on the right for the key
    ax.set_xlabel("Balanced accuracy (averaged over methods and datasets)")
    ax.legend(handles=[
        plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                   markeredgecolor="0.3", markersize=7, label="no curated mapping"),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.3", markersize=7,
                   label="with curated mapping")],
        loc="upper right", frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0)


def _lineage_panel(ax, pairs):
    """Panel H: where each family's answers sit in the ontology relative to the true type, in
    G's rows and averaged like G over the family's methods and the datasets. The per-dataset and
    per-subcategory views are a supplementary figure."""
    shares = _group_means(_answer_shares(pairs).groupby("model").mean(), MAPPING_ROWS)
    y = np.arange(len(MAPPING_ROWS))[::-1]
    _stacked_shares(ax, shares, y)
    ax.set_yticks(y)
    ax.set_yticklabels(shares.index)
    ax.set_ylim(-0.6, len(MAPPING_ROWS) - 0.4)
    ax.set_xlabel("Share of cells (averaged over cell types, methods and datasets)")


def figure_benchmark(df, out_path, dpi=600, pairs=None):
    """Figure 2 — the benchmark results: ranking, effect of the list per dataset, dependence on
    the curated mapping, and where the answers sit in the ontology."""
    df = with_unsupported_as_zero(df, model_cols=("model", "mode"))
    pairs = _benchmark_pairs(df, pairs)

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 14.56))
        # A and B-F take the full width. Between them sit A's slanted run names and the key that
        # serves both rows; the spacer after B-F holds their family names. The two spacers are
        # sized so the white gaps above B-F and above G/H are equal (0.3 in).
        outer = fig.add_gridspec(7, 1, height_ratios=[3.4, 1.65, 1.5, 0.73, 2.8, 1.27, 2.9],
                                 hspace=0, left=0.075, right=0.985)
        ax_a = fig.add_subplot(outer[0])
        _ranked_panel(ax_a, df)

        middle = outer[4].subgridspec(1, 5, wspace=0.08)
        panels = [fig.add_subplot(middle[0, 0])]
        panels += [fig.add_subplot(middle[0, i], sharey=panels[0]) for i in range(1, 5)]
        _list_effect_panel(panels[0], df, "All four datasets")
        panels[0].set_ylabel("Balanced accuracy")
        for ax, dataset in zip(panels[1:], DATASET_ORDER):
            _list_effect_panel(ax, df[df["dataset"] == dataset], dataset_label(dataset))
            ax.tick_params(labelleft=False)

        # G, then H with its key close beside it, so the key stays inside the page width.
        bottom = outer[6].subgridspec(1, 2, width_ratios=[1.0, 1.75], wspace=0.25)
        right = bottom[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.36], wspace=0.04)
        ax_g, ax_h = fig.add_subplot(bottom[0, 0]), fig.add_subplot(right[0, 0])
        _mapping_panel(ax_g, pairs)
        _lineage_panel(ax_h, pairs)
        ax_h_key = fig.add_subplot(right[0, 1])
        ax_h_key.axis("off")
        ax_h_key.legend(handles=ax_h.get_legend_handles_labels()[0], loc="center left",
                        frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0,
                        handlelength=1.6)

        # The key for A to F.
        ax_key = fig.add_subplot(outer[2])
        ax_key.axis("off")
        _family_key(ax_key, [
            mpatches.Patch(facecolor="0.75", edgecolor="white", label=SETTING_KEY["unblind"]),
            mpatches.Patch(facecolor="0.75", edgecolor="black", hatch="///",
                           label=SETTING_KEY["blind"]),
            plt.Line2D([], [], label="chance (random, stratified)", **CHANCE_LINE),
            plt.Line2D([], [], color="0.62", linewidth=0.7, label="same method")])
        _panel_letters(fig, [(ax_a, "A"), *zip(panels, "BCDEF"), (ax_g, "G"), (ax_h, "H")])
        return _save(fig, out_path, dpi)


def _lighter(color, white=0.5):
    """``color`` mixed with white, ``white`` being the share of white."""
    return mcolors.to_hex([c + (1 - c) * white for c in mcolors.to_rgb(color)])


def _family_key(ax, encodings, drawn=None, extra=(), spacing=0.8):
    """The key of Figure 2, worded as Figure 1B: one column of ``encodings``, then one column per
    family with its subcategories under the family's name in bold, then one column per ``extra``
    ``(heading, [lines])``. ``drawn`` keeps only those ``(family, subcategory)`` pairs. Columns are
    padded to equal length, so the legend fills them column by column."""
    colors = _ranked_category_colors()
    columns = [encodings] + [
        [_blank(FAMILY_HEADER[family])] + [
            plt.Line2D([], [], linestyle="none", marker=marker, color=colors[family],
                       markersize=11 if marker == "*" else 7, label=TIER_NAME.get(tier, label))
            for tier, label, marker, _size in CATEGORY_TIERS[family]
            if drawn is None or (family, tier) in drawn]
        for family, _ in COMPOSITE_CATEGORIES]
    columns += [[_blank(heading)] + [_blank(line) for line in lines] for heading, lines in extra]
    headings = {*FAMILY_HEADER.values(), *(heading for heading, _ in extra)}
    rows = max(len(column) for column in columns)
    key = ax.legend(handles=[h for column in columns
                             for h in column + [_blank()] * (rows - len(column))],
                    ncol=len(columns), loc="center", frameon=False, fontsize=FS_LEGEND,
                    columnspacing=spacing, handlelength=1.4, handletextpad=0.5, borderaxespad=0)
    for text in key.get_texts():
        if text.get_text() in headings:
            text.set_fontweight("bold")


def _subcategory_rows():
    """Rows ``(family, mode, label, tier)`` for every method subcategory in both settings, named
    as in Figure 1B."""
    family_name = {"classical": "baseline", "scLLM": "scLLM", "LLM": "LLM"}
    return [(family, mode, f"{family_name[family]}, {TIER_NAME.get(tier, label)}, "
                           f"{REGIME_SHORT[mode]}", tier)
            for family, _ in COMPOSITE_CATEGORIES
            for tier, label, *_ in CATEGORY_TIERS[family]
            for mode in ("blind", "unblind")]


def figure_answer_types(df, out_path, dpi=600, pairs=None):
    """Supplementary — Figure 2H resolved per dataset (A) and per method subcategory (B).

    The main panel averages over the datasets and over each family's methods; the dataset view
    shows where the coarse answers come from (T cell on HIHA, neuron on HFRA and Zeisel), the
    subcategory view which methods give them.
    """
    df = with_unsupported_as_zero(df, model_cols=("model", "mode"))
    per_run = _answer_shares(_benchmark_pairs(df, pairs))
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 7.6))
        # The gap holds B's subcategory names; the key sits under both panels.
        grid = fig.add_gridspec(1, 2, left=0.11, right=0.985, wspace=0.78)
        ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

        # A: G's six rows for each dataset, the dataset's name above its block.
        blocks = [_group_means(per_run.xs(dataset, level="dataset"), MAPPING_ROWS)
                  for dataset in DATASET_ORDER]
        step = len(MAPPING_ROWS) + 1.4
        y = np.concatenate([-(i * step + np.arange(len(MAPPING_ROWS)))
                            for i in range(len(blocks))])
        _stacked_shares(ax_a, pd.concat(blocks), y)
        ax_a.set_yticks(y)
        ax_a.set_yticklabels([row[2] for _ in blocks for row in MAPPING_ROWS])
        heads = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
        for i, dataset in enumerate(DATASET_ORDER):
            ax_a.text(0.0, -i * step + 0.75, dataset_label(dataset), transform=heads, ha="left",
                      va="bottom", fontsize=FS_LEGEND, fontweight="bold")
        ax_a.set_ylim(y.min() - 0.6, 1.4)
        ax_a.set_xlabel("Share of cells (averaged over cell types and methods)")

        # B: every subcategory, averaged over the datasets.
        rows = _subcategory_rows()
        shares = _group_means(per_run.groupby("model").mean(), rows)
        y_b = np.arange(len(rows))[::-1]
        _stacked_shares(ax_b, shares, y_b)
        ax_b.set_yticks(y_b)
        ax_b.set_yticklabels(shares.index)
        ax_b.set_ylim(-0.6, len(rows) - 0.4)
        ax_b.set_xlabel("Share of cells (averaged over cell types, methods and datasets)")

        for ax in (ax_a, ax_b):
            _harmonize(ax, text_size=FS_LEGEND)
        fig.legend(handles=ax_b.get_legend_handles_labels()[0], loc="upper center",
                   bbox_to_anchor=(0.55, 0.0), ncol=len(LINEAGE_KEY), frameon=False,
                   fontsize=FS_LEGEND, handlelength=1.6, columnspacing=1.4)
        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B")])
        return _save(fig, out_path, dpi)


# The supplementary heatmap's metrics, in Figure 1E's groups.
METRIC_GROUPS = [
    ("exact match", [("balanced_accuracy", "balanced accuracy"), ("accuracy", "accuracy"),
                     ("macro_f1", "macro F1"), ("weighted_f1", "weighted F1")]),
    ("ontology-aware", [("agreement_score", "agreement score"), ("rada_avg_similarity", "Rada"),
                        ("lin_avg_similarity", "Lin")]),
    ("text similarity", [("text_emb_norm_balanced_mapped", "text, mapped"),
                         ("text_emb_norm_balanced_raw", "text, unmapped")]),
]


def figure_metric_agreement(df, out_path, dpi=600):
    """Supplementary — do the metrics rank the methods alike?

    Spearman correlation over the runs within each dataset, averaged over the datasets. Pooling
    the datasets instead would mix in how hard each dataset is, which moves every metric at once
    and Lin with a floor of its own per dataset, and so would understate how alike Lin ranks.
    """
    runs = df[df["model"].map(categorize_model) != "simple"]
    keys = [key for _, group in METRIC_GROUPS for key, _ in group]
    names = [name for _, group in METRIC_GROUPS for _, name in group]
    per_dataset = [g[keys].corr(method="spearman") for _, g in runs.groupby("dataset")]
    corr = (sum(per_dataset) / len(per_dataset)).to_numpy()
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(8.0, 6.8))
        image = ax.imshow(np.ma.array(corr, mask=np.triu(np.ones_like(corr, dtype=bool), 1)),
                          cmap="viridis", vmin=0.5, vmax=1.0)
        for i in range(len(keys)):
            for j in range(i + 1):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=FS_POINT + 1, color="white" if corr[i, j] < 0.8 else "black")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(names)
        edge = -0.5
        for _, group in METRIC_GROUPS[:-1]:  # white rules between Figure 1E's groups
            edge += len(group)
            ax.axhline(edge, color="white", linewidth=2.5)
            ax.axvline(edge, color="white", linewidth=2.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        bar.set_label("Spearman correlation (mean over the four datasets)")
        return _save(fig, out_path, dpi)


# --------------------------------------------------------------------------- #
# Supplementary — cost against annotation quality (serves R3's arm choice).
#
# Soft dependency on the OpenRouter activity log, handled like the rare-type table and the
# ablation: a missing file is reported as an instruction, so building every other figure still
# works. Only the general-purpose LLMs appear -- the panel's x axis is metered API spend, and the
# local methods have none.
# --------------------------------------------------------------------------- #
def load_activity(path=DEFAULT_ACTIVITY_CSV):
    """Per-request OpenRouter log, restricted to the benchmarked models."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return load_openrouter_activity(path, benchmarked_only=True)


def figure_cost_performance(df, out_path, dpi=600, activity=None, ablation=None):
    """Supplementary Figure S6 — annotation quality against cost.

    (A) Every general-purpose LLM of the benchmark against its measured API spend per 1,000 cells,
        per setting. The step line is the cost-efficient frontier: GPT-OSS-120B is the cheapest
        model in the roster and no cheaper model scores higher, which is why it carries the
        ablation.
    (B) The prompt-context ablation at genes + list: balanced accuracy against the model calls
        that annotate 1,000 cells, per cell or per cluster (``_cost_panel``).

    The panels meter different runs. The benchmark and the ablation measured GPT-OSS-120B at ~$0.04
    and $0.23 per 1,000 cells, because the benchmark sorted providers by price while the ablation
    left routing open and `gpt-oss-120b` is served by many providers at different rates. Both
    numbers are correct measurements of different runs; the caption says so.

    The in-panel title is dropped, the names are the display names and the settings are worded as
    in Figure 2's key; the LaTeX caption does the describing (WRITING_STYLE.md, Scientific
    reporting Rule 1).
    """
    activity = load_activity() if activity is None else activity
    if activity.empty:
        raise SystemExit(
            f"Cost-performance figure needs the OpenRouter activity log at "
            f"{DEFAULT_ACTIVITY_CSV}, which does not exist. It is not derivable from the "
            f"aggregate: spend is per request, and the aggregate is per model and dataset.")

    ablation = load_ablation() if ablation is None else ablation
    with plt.rc_context(_rc()):
        fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"wspace": 0.3})
        # All points are general-purpose LLMs, so they take the family's orange: C in full, U
        # lighter, as the hatching lightens the U bars of Figure 2A. Names as in Figure 2B-F.
        orange = _ranked_category_colors()["LLM"]
        ok = plot_cost_vs_performance(df, activity, out_path, dpi, ax=ax,
                                      metric="balanced_accuracy", name=_display,
                                      label_size=FS_POINT, text_color="0.15",
                                      colors={"unblind": orange, "blind": _lighter(orange)},
                                      step_alpha=0.9)
        if not ok:
            raise SystemExit(
                "Cost-performance figure: fewer than two models carry both a balanced accuracy "
                "and a spend figure. Check that the aggregate and the activity log overlap.")
        # The panel function sets its title with loc="left", which is a separate Text object from
        # the centered one -- clearing the default would leave it on the page.
        ax.set_title("", loc="left")
        ax.set_ylabel("Balanced accuracy (averaged over datasets)")
        legend = ax.get_legend()
        if legend is not None:  # the settings as in Figure 2's key
            for text in legend.get_texts():
                text.set_text(SETTING_KEY.get(text.get_text(), text.get_text()))
        _harmonize(ax)
        # B: what annotating clusters instead of cells saves, from the prompt-context ablation.
        _cost_panel(ax_b, ablation, _ranked_category_colors(), bench=df)
        _harmonize(ax_b)
        _panel_letters(fig, [(ax, "A"), (ax_b, "B")])
        return _save(fig, out_path, dpi)


def _rare_flow_builder(dataset):
    def build(df, out_path, dpi=600):
        return figure_rare_flows(df, out_path, dpi, dataset=dataset)
    build.__doc__ = f"Rare-type prediction flows for {dataset}."
    return build


# --------------------------------------------------------------------------- #
# Figure 3 — the context ablation (Results R2).
#
# Soft dependency, like the model dates and the per-class recall table: the study lives in
# ../context_ablation and has to be RUN before this can draw anything. Absent input is reported as
# a clear instruction rather than an exception, so `python bin/manuscript_figures.py` with no
# arguments still builds every other figure.
# --------------------------------------------------------------------------- #
DEFAULT_ABLATION_CSV = (Path(__file__).resolve().parents[1] / "context_ablation" / "results"
                        / "context_ablation.csv")

# Arm styling, as in Figure 3: colour the family and marker the subcategory of Figure 2's key, so
# a model looks the same in every figure; line style the unit, solid per cell and dashed per
# cluster. The base model C2S-Scale was fine-tuned from, an open-weight non-reasoning LLM, takes
# that subcategory's square, dotted.
ABLATION_ARMS = {
    "A1_llm_percell":    ("LLM",   "^", "-",  "GPT-OSS-120B, per cell"),
    "A5_llm20_percell":  ("LLM",   "^", "-",  "GPT-OSS-20B, per cell"),
    "A2_base_percell":   ("LLM",   "s", ":",  "Gemma-2-2B (base), per cell"),
    "A3_scllm_percell":  ("scLLM", "o", "-",  "C2S-Scale 2B, per cell"),
    "A4_llm_percluster": ("LLM",   "^", "--", "GPT-OSS-120B, per cluster"),
    "A8_gemini_percluster": ("LLM", "*", "--", "Gemini 3.1 Pro, per cluster"),
}
#: Display order, NOT the mode numbering. Ordered by how much the model is given, so the x axis
#: reads as a ladder: no transcriptome at all, then the transcriptome alone, then the transcriptome
#: plus each kind of context, then both. That makes two things legible that M1..M5 order hides --
#: how far above the no-genes floor the gene list alone gets you, and that the last three steps add
#: much less than the first.
#: M6 sits at the LEFT of the ladder, below "genes only": it supplies the same 200 genes but with
#: the expression ranking destroyed, so it isolates what the ORDER is worth. It is the study's
#: floor now that M5 (no genes at all) has been dropped. Only A1 and A3 were run at M6 -- the
#: cluster arms are deliberately absent, since the ordering claim is about per-cell sentences.
MODE_ORDER = ["M6", "M1", "M3", "M2", "M4"]

#: "none" was ambiguous: M1 has no CONTEXT but does have the gene list, and reading it as "nothing
#: supplied" would put the floor in the wrong place. Every label now names what the model receives.
MODE_LABELS = {"M6": "genes,\nshuffled\norder", "M1": "genes\nonly", "M3": "genes\n+ tissue",
               "M2": "genes\n+ list", "M4": "genes\n+ tissue\n+ list"}


def load_ablation(path=DEFAULT_ABLATION_CSV):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


DEFAULT_LIST_RUNS_CSV = (Path(__file__).resolve().parents[1] / "context_ablation" / "results_lists"
                         / "runs.csv")
# The arms of the ablation and the list study as the benchmark's models, which give the family,
# marker and name of Figure 2.
STUDY_ARM_MODEL = {"A1_llm_percell": "gpt-oss-120b", "A4_llm_percluster": "gpt-oss-120b",
                   "A8_gemini_percluster": "gemini-3.1-pro-preview",
                   "A3_scllm_percell": "c2s_scale_2B", "S1_mmcontext": "mmcontext",
                   "S2_langcell": "langcell", "B1_singler": "singler",
                   "B2_celltypist": "celltypist", "B3_sctype": "sctype", "B4_scsa": "scsa"}
# The arms that label clusters: the per-cluster language models, and scType and SCSA, which score
# their marker sets per cluster.
PER_CLUSTER_ARMS = {"A4_llm_percluster", "A8_gemini_percluster", "B3_sctype", "B4_scsa"}
# A: what the prompt carries, in the order the model is given more.
PROMPT_ARMS = ["A1_llm_percell", "A4_llm_percluster", "A8_gemini_percluster", "A3_scllm_percell"]
PROMPT_MODES = [("M1", "genes\nonly"), ("M3", "genes\n+ tissue"), ("M2", "genes\n+ list"),
                ("M4", "genes\n+ tissue\n+ list")]
# B: the exact list, the three imperfections and their union; the key row spells out A1-A3.
LIST_CONDITIONS = [("E", "exact list"), ("L1", "A1"), ("L2-5", "A2"), ("L3", "A3"),
                   ("L4", "A1 + A2 + A3")]
LIST_KEY = ("Imperfect lists", ["A1  mixed levels (parents, children)",
                                "A2  five absent types added",
                                "A3  a quarter of the types removed"])
# C: Figure 2H's classes, worded for the cells of a type the list left out.
MISSING_TYPE_CONDITIONS = [("L3", "¼ of the types missing (A3)"),
                           ("L4", "all combined (A1 + A2 + A3)")]
MISSING_TYPE_KEY = {"exact": "the missing type itself",
                    "coarser": "coarser, specific\n(e.g. a listed parent)",
                    "broad": "coarser, broad", "finer": "finer", "other": "another cell type",
                    "none": "no usable answer\n(incl. uncertain, Unknown)"}
# B's x positions: the no-list reference and the union stand apart; the gap between families
# holds the methods' names.
REF_GAP, UNION_GAP, FAMILY_GAP = 1.3, 1.3, 2.6
# The score of A and B, as (column in the ablation and the benchmark, column in the list study,
# axis label): balanced accuracy in Figure 3, the agreement score in its ontology-aware twin.
FIG3_METRICS = {"balanced_accuracy": ("balanced_accuracy", "balanced_accuracy",
                                      "Balanced accuracy"),
                "agreement": ("agreement_score", "agreement", "Agreement score")}
# Where B places a method's name: beside its last point first.
NAME_BESIDE = [(9, 0, "left", "center"), (9, 7, "left", "bottom"), (9, -7, "left", "top"),
               (16, 0, "left", "center"), (16, 10, "left", "bottom"), (16, -10, "left", "top"),
               (0, 9, "center", "bottom"), (0, -9, "center", "top")]


def load_list_runs(path=DEFAULT_LIST_RUNS_CSV):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _arm_style(arm):
    """(colour, marker, scatter size) of an arm, from its model's family and subcategory."""
    model = STUDY_ARM_MODEL[arm]
    marker, size = _tier_style(model)
    return _ranked_category_colors()[categorize_model(model)], marker, size


def _list_study_arms(runs):
    """The list study's arms in Figure 2's order: family, subcategory, then name."""
    order = {(family, tier): (i, j) for i, (family, _) in enumerate(COMPOSITE_CATEGORIES)
             for j, (tier, *_rest) in enumerate(CATEGORY_TIERS[family])}
    model = {arm: STUDY_ARM_MODEL[arm] for arm in runs["arm"].unique()}
    return sorted(model, key=lambda a: (*order[(categorize_model(model[a]),
                                                model_tier(model[a]))], model[a]))


def _prompt_content_panel(ax, ablation, bench, metric="balanced_accuracy"):
    """A: ``metric`` by what the prompt carries, one line per arm. Gemini per cell exists only in
    the benchmark, whose two prompts are the ablation's M1 and M2 verbatim: hollow stars."""
    column, _list_column, label = FIG3_METRICS[metric]
    x = np.arange(len(PROMPT_MODES))
    handles = []
    for arm in PROMPT_ARMS:
        rows = _arm_rows(ablation, arm)
        colour, marker, _size = _arm_style(arm)
        unit = "cluster" if arm in PER_CLUSTER_ARMS else "cell"
        line, = ax.plot(x, [rows.loc[rows["mode"] == m, column].mean() for m, _ in PROMPT_MODES],
                        color=colour, marker=marker, markersize=11 if marker == "*" else 7,
                        markeredgecolor="white", markeredgewidth=0.5, linewidth=1.4, zorder=3,
                        linestyle="--" if unit == "cluster" else "-",
                        label=f"{_display(STUDY_ARM_MODEL[arm])}, per {unit}")
        handles.append(line)
    colour = _ranked_category_colors()["LLM"]
    for mode, setting in (("M1", "blind"), ("M2", "unblind")):
        score = bench.loc[bench["model"] == f"gemini-3.1-pro-preview_{setting}", column]
        ax.scatter([m for m, _ in PROMPT_MODES].index(mode), score.mean(), marker="*", s=150,
                   facecolor="white", edgecolor=colour, linewidth=1.3, zorder=4)
    handles.append(plt.Line2D([], [], linestyle="none", marker="*", markersize=11,
                              markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.3,
                              label="Gemini 3.1 Pro, per cell (benchmark)"))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in PROMPT_MODES])
    ax.set_xlim(-0.35, len(PROMPT_MODES) - 0.65)
    ax.set_ylabel(label)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=FS_LEGEND - 0.5,
              handlelength=2.6, borderaxespad=0.1)


def _no_list_reference(ablation, bench, arms, dataset=None, metric="balanced_accuracy"):
    """Each list-study method without a list, the list being the only difference: the ablation's
    genes-only runs for the prompted arms, the unconstrained benchmark runs on the same cells for
    the others. Over the four datasets, or for ``dataset``."""
    column = FIG3_METRICS[metric][0]
    if dataset is not None:
        ablation = ablation[ablation["dataset"] == dataset]
        bench = bench[bench["dataset"] == dataset]
    ref = {}
    for arm in arms:
        rows = _arm_rows(ablation, arm)
        ref[arm] = (rows.loc[rows["mode"] == "M1", column].mean() if not rows.empty
                    else bench.loc[bench["model"] == f"{STUDY_ARM_MODEL[arm]}_blind",
                                   column].mean())
    return ref


def _imperfect_lists_panel(ax, runs, reference, metric="balanced_accuracy"):
    """B: ``metric`` over all true types, a missing type counting as missed, per family: no list,
    the exact list, A1-A3 and their union. A thin line joins the points of one method."""
    _column, list_column, label = FIG3_METRICS[metric]
    means = runs.groupby(["arm", "condition"])[list_column].mean()
    steps = [0.0, 1.0, 2.0, 3.0, 3.0 + UNION_GAP]
    span = REF_GAP + steps[-1]
    arms = _list_study_arms(runs)
    ticks, tick_labels, points, labels = [], [], [], []
    for g, (family, group_label) in enumerate(COMPOSITE_CATEGORIES):
        members = [a for a in arms if categorize_model(STUDY_ARM_MODEL[a]) == family]
        sizes = [sum(model_tier(STUDY_ARM_MODEL[a]) == t for a in members)
                 for t, *_rest in CATEGORY_TIERS[family]]
        offsets = _tiered_offsets([s for s in sizes if s], spacing=0.09, gap=0.12,
                                  max_half_width=0.26)
        ref_x = g * (span + FAMILY_GAP)
        cond_x = [ref_x + REF_GAP + s for s in steps]
        for j, arm in enumerate(members):
            colour, marker, size = _arm_style(arm)
            xs = [ref_x + offsets[j]] + [x + offsets[j] for x in cond_x]
            ys = [reference[arm]] + [means[(arm, c)] for c, _ in LIST_CONDITIONS]
            ax.plot(xs, ys, color="0.62", linewidth=0.7, zorder=2,
                    linestyle=(0, (3, 2)) if arm in PER_CLUSTER_ARMS else "-")
            ax.scatter(xs, ys, s=size, marker=marker, color=colour, edgecolor="white",
                       linewidth=0.5, zorder=4)
            points += [(xx, yy, np.sqrt(size) / 2) for xx, yy in zip(xs, ys)]
            labels.append((_display(STUDY_ARM_MODEL[arm]), xs[-1], ys[-1], "0.15"))
        ticks += [ref_x] + cond_x
        tick_labels += ["no list"] + [label for _, label in LIST_CONDITIONS]
        ax.text(ref_x + span / 2, 1.02, group_label, ha="center", va="bottom", fontsize=FS_GROUP,
                transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(ticks[0] - 0.7, ticks[-1] + FAMILY_GAP + 0.9)
    ax.set_ylabel(label)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    _place_point_labels(ax, labels, points, FS_POINT, candidates=NAME_BESIDE, leader_min=10)


def _missing_types_panel(axes, runs):
    """C: where the cells of the types left off the list go, one bar per method and facet, in
    Figure 2H's classes and averaged over the missing types, as balanced accuracy averages."""
    arms = _list_study_arms(runs)[::-1]  # the first method on top
    family = [categorize_model(STUDY_ARM_MODEL[a]) for a in arms]
    y = np.cumsum([0.0] + [1.0 + 0.5 * (b != a) for a, b in zip(family, family[1:])])
    classes = [key for key, *_rest in LINEAGE_KEY]
    for ax, (condition, title) in zip(axes, MISSING_TYPE_CONDITIONS):
        shares = (runs[runs["condition"] == condition].groupby("arm")
                  [[f"removed_{k}" for k in classes]].mean().loc[arms].set_axis(classes, axis=1))
        _stacked_shares(ax, shares, y)
        ax.set_ylim(y.min() - 0.55, y.max() + 0.55)
        ax.set_title(title, fontsize=FS_LABEL, pad=4)
        ax.set_yticks(y)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axes[0].set_yticklabels([_display(STUDY_ARM_MODEL[a]) for a in arms])
    axes[0].tick_params(axis="y", pad=13)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    # Each name carries its marker, so the bars connect to B's points.
    beside = mtransforms.blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for yy, arm in zip(y, arms):
        colour, marker, size = _arm_style(arm)
        axes[0].scatter(-0.022, yy, s=size * 0.8, marker=marker, color=colour, edgecolor="white",
                        linewidth=0.4, transform=beside, clip_on=False, zorder=5)


def _figure3_top(fig, outer, ablation, bench, runs, metric="balanced_accuracy"):
    """Figure 3's A | B on ``metric`` in ``outer``'s first row and their key in its third, for
    Figure 3 and its ontology-aware twin; returns the two axes."""
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 2.55], wspace=0.13)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
    _prompt_content_panel(ax_a, ablation, bench, metric)
    _imperfect_lists_panel(ax_b, runs, _no_list_reference(ablation, bench, _list_study_arms(runs),
                                                          metric=metric), metric)
    ax_b.tick_params(labelleft=True)
    ax_a.set_ylim(0, 0.72)
    # An invisible heading as tall as B's family names, so the letters of A and B align.
    ax_a.text(0.5, 1.02, "baseline", transform=ax_a.transAxes, ha="center", va="bottom",
              fontsize=FS_GROUP, alpha=0)

    drawn = {(categorize_model(m), model_tier(m)) for m in STUDY_ARM_MODEL.values()}
    ax_key = fig.add_subplot(outer[2])
    ax_key.axis("off")
    _family_key(ax_key, [
        plt.Line2D([], [], color="0.35", linewidth=1.4, label="per cell"),
        plt.Line2D([], [], color="0.35", linewidth=1.4, linestyle="--", label="per cluster"),
        plt.Line2D([], [], color="0.62", linewidth=0.7, label="same method")],
        drawn=drawn, extra=[LIST_KEY], spacing=1.5)
    return ax_a, ax_b


def figure_2(df, out_path, dpi=600, ablation=None, runs=None):
    """Figure 3 — what the list of expected cell types adds, and what an imperfect list costs.

    (A) The prompt-context ablation: balanced accuracy by what the prompt carries.
    (B) The imperfect-list study: balanced accuracy over all true types, per family, without a list,
        with the exact list, with each of three imperfections and with all three.
    (C) Where the cells of the types missing from the list go.

    ``df`` (the benchmark aggregate) supplies Gemini per cell in A and the no-list runs of the
    methods that take no prompt in B.
    """
    ablation = load_ablation() if ablation is None else ablation
    runs = load_list_runs() if runs is None else runs
    if ablation.empty or runs.empty:
        raise SystemExit("Figure 3 needs context_ablation/results/context_ablation.csv and "
                         "context_ablation/results_lists/runs.csv: see context_ablation/README.md.")
    colors = _ranked_category_colors()
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 9.6))
        # A | B, their tick labels, the key row, a spacer, then C.
        outer = fig.add_gridspec(5, 1, height_ratios=[4.0, 1.0, 1.2, 0.3, 3.2], hspace=0,
                                 left=0.075, right=0.985, top=0.97, bottom=0.06)
        ax_a, ax_b = _figure3_top(fig, outer, ablation, df, runs)

        bottom = outer[4].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.42], wspace=0.07)
        ax_c = [fig.add_subplot(bottom[0, 0])]
        ax_c.append(fig.add_subplot(bottom[0, 1], sharey=ax_c[0]))
        _missing_types_panel(ax_c, runs)
        ax_c_key = fig.add_subplot(bottom[0, 2])
        ax_c_key.axis("off")
        ax_c_key.legend(handles=[mpatches.Patch(facecolor=color, edgecolor="0.55", linewidth=0.4,
                                                label=MISSING_TYPE_KEY[key])
                                 for key, _label, color in LINEAGE_KEY],
                        loc="center left", frameon=False, fontsize=FS_LEGEND, borderaxespad=0,
                        borderpad=0, handlelength=1.6, labelspacing=0.7)
        left, right = ax_c[0].get_position(), ax_c[1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y0 - 0.03, "Share of the missing types' cells",
                 ha="center", va="top", fontsize=FS_LABEL)

        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c[0], "C")])
        return _save(fig, out_path, dpi)


def _marker_size_for(marker):
    return 9.5 if marker == "*" else 6.0


def _scatter_area_for(marker, scale=1.0):
    """``scatter`` size (an AREA in points^2) matching ``_marker_size_for``'s diameter.

    The two APIs disagree on units: ``plot(markersize=...)`` takes a diameter in points while
    ``scatter(s=...)`` takes an area. Passing one flat ``s`` to both a circle and a star therefore
    draws the star noticeably smaller, because a star spreads the same area over five thin points
    instead of a filled disc. Squaring the per-marker diameter keeps the star's visual weight equal
    to the round markers, exactly as it already is on the line panels.
    """
    return (_marker_size_for(marker) ** 2) * 1.78 * scale


def _arm_rows(frame, arm):
    """Rows for one arm at its DEFAULT variant, whatever that arm happens to index by.

    The two arm kinds are not indexed the same way: per-cell arms carry gene variants
    ("expression_ranked", "hvg") and the cluster arm carries clustering resolutions ("res0.5",
    "res1.0"). Filtering the frame to one variant before splitting by arm therefore deletes the
    cluster arm silently -- which it did, in BOTH panel A and panel C, leaving the per-cell/
    per-cluster comparison absent from the figure that exists to make it.

    Centralised here so the next panel cannot repeat it.
    """
    want = "res1.0" if arm.endswith("percluster") else "expression_ranked"
    sub = frame[(frame["arm"] == arm) & (frame["variant"] == want)]
    return sub if not sub.empty else frame[frame["arm"] == arm]


def _paired_slope(ax, frame, left_arm, right_arm, labels, cat_color):
    """One line per dataset between two arms, at the best mode both arms share.

    A paired slope rather than two bars: the question is whether adaptation beat its own base on
    the SAME data, and pairing shows a reversal on one dataset that averaged bars would hide.
    """
    shared = sorted(set(frame[frame["arm"] == left_arm]["mode"])
                    & set(frame[frame["arm"] == right_arm]["mode"]))
    if not shared:
        ax.text(0.5, 0.5, "no shared mode between the two arms", ha="center", va="center",
                transform=ax.transAxes, color="0.5", fontsize=FS_LEGEND)
        ax.axis("off")
        return
    mode = "M4" if "M4" in shared else shared[-1]
    for dataset in sorted(frame["dataset"].unique()):
        pair = []
        for arm in (left_arm, right_arm):
            vals = frame[(frame["arm"] == arm) & (frame["mode"] == mode)
                         & (frame["dataset"] == dataset)]["balanced_accuracy"]
            pair.append(vals.mean() if len(vals) else np.nan)
        if np.isnan(pair).any():
            continue
        colour = cat_color["scLLM"] if pair[1] >= pair[0] else cat_color["classical"]
        ax.plot([0, 1], pair, marker="o", markersize=5, linewidth=1.3, color=colour, zorder=3)
        ax.annotate(dataset, (1, pair[1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=FS_POINT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.25, 1.55)
    ax.set_ylim(0, None)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)


#: The one mode the cost panel is drawn at. M2 = genes + candidate cell types, which is the
#: benchmark's "unblind"/constrained setting, so the ported Gemini per-cell run is comparable.
COST_MODE = "M2"

#: Measured spend AT M2, normalised to dollars per 1,000 annotated cells.
#:
#: Per-arm totals were the wrong unit for this panel. They mixed modes that cost different amounts,
#: and they could not be compared against the ported Gemini per-cell run, which covers 3,997 cells
#: at one seed while our arms cover 11,991 (4 datasets x 3 seeds). Dollars per 1,000 cells divides
#: both out and is the same denominator the x axis already uses.
#:
#: Derived by summing ``usage.cost`` over the UNIQUE cache keys reached by each M2 prompt file
#: (sha256(model|prompt|seed)). Deduplication is essential for the cluster arms: their prompt CSVs
#: carry one row per cell with the cluster's prompt repeated, so counting rows overstates their
#: spend ~29x. Hardcoded rather than recomputed at plot time because the response caches are not
#: shipped with the figure code, and a missing cache should leave the label bare, not crash.
#:
#: Totals behind these, for the record: A1 $2.73, A4 $0.08, A8 $3.00 (each over 11,991 cells).
ARM_SPEND_PER_1K_M2 = {
    "A1_llm_percell":       0.2280,
    "A4_llm_percluster":    0.0066,
    "A8_gemini_percluster": 0.2501,
}

#: The Gemini per-cell point is PORTED from the benchmark, not run here: re-running it per cell
#: across this study's grid was not affordable. Its accuracy is read from the benchmark aggregate
#: at plot time (model ``gemini-3.1-pro-preview_unblind`` -- "unblind" supplies the candidate list,
#: which is exactly this study's M2), so only the cost is a constant: $34.24 over 3,997 cells.
#: Drawn hollow to mark it as the one point on the panel that did not come from this pipeline.
GEMINI_PERCELL = {"model": "gemini-3.1-pro-preview_unblind",
                  "cost_per_1k": 34.24 / 3997 * 1000,
                  "label": "Gemini 3.1 Pro, per cell"}


def _cost_panel(ax, frame, cat_color, bench=None):
    """Accuracy against calls made, AT ONE MODE. The x span IS the result: ~1,000 against ~41.

    Fixed to M2 (genes + candidate types) rather than averaged over modes. Two reasons: averaging
    mixed a mode where the LLM scores 0.27 with one where it scores 0.54, so no plotted point
    corresponded to a run anyone made; and M2 is the only setting in which the ported Gemini
    per-cell run exists, so it is the one mode where all four cost points are commensurable.

    Takes the FULL ablation frame and selects the right variant per arm, because the two arm kinds
    are indexed differently: per-cell arms carry gene variants ("expression_ranked", "hvg") while
    the cluster arms carry clustering resolutions ("res0.5", "res1.0"). A single variant filter
    upstream cannot serve both, and filtering to the per-cell variant drops the cluster arms.
    """
    labels, obstacles = [], []

    def _annotate(text, spend, calls, acc, colour, mark, hollow=False):
        ax.scatter(calls, acc, s=_scatter_area_for(mark, 1.19 if hollow else 1.0), marker=mark,
                   facecolor="none" if hollow else colour,
                   edgecolor=colour if hollow else "white",
                   linewidth=1.3 if hollow else 0.6, zorder=3)
        # Two decimals rounded the cheapest arm to "$0.01", which erases the ~34x spread this
        # panel exists to show. Scale the precision to the magnitude instead.
        if spend is None:
            money = None
        elif spend < 0.1:
            money = f"${spend:.3f}"
        elif spend < 10:
            money = f"${spend:.2f}"
        else:
            money = f"${spend:,.1f}"
        labels.append((f"{text}\n{money} / 1k cells" if money else text, calls, acc, colour))
        obstacles.append((calls, acc, 11.0))

    for arm, (family, marker, _ls, label) in ABLATION_ARMS.items():
        is_cluster = arm.endswith("percluster")
        sub = _arm_rows(frame, arm)
        sub = sub[sub["mode"] == COST_MODE]
        if sub.empty:
            continue
        # Calls per 1,000 cells: one per cell, or one per cluster the eval cells fall into.
        calls = (sub["n_clusters_in_eval"].mean()
                 if is_cluster and "n_clusters_in_eval" in sub
                 and sub["n_clusters_in_eval"].notna().any()
                 else 1000.0)
        # Dollars alongside the call count: calls generalise across models, dollars do not, but the
        # spend ratio is the number a reader wants and it is not recoverable from the axis alone.
        # The local arms (Gemma, C2S-Scale) ran on our own GPU and have no API cost; leaving their
        # label bare is correct -- printing "$0.00" would read as free rather than as not-purchased.
        _annotate(label, ARM_SPEND_PER_1K_M2.get(arm), calls,
                  sub["balanced_accuracy"].mean(), cat_color[family], marker)

    # The ported Gemini per-cell point, sourced from the benchmark aggregate rather than hardcoded
    # so it tracks any rescoring of the benchmark. Same model as the A8 cluster arm, one call per
    # cell, and the panel's whole point: it buys the most accuracy and costs ~34x more per cell
    # annotated than running the SAME model over clusters.
    if bench is not None and not bench.empty and "model" in bench:
        g = bench[bench["model"] == GEMINI_PERCELL["model"]]
        if not g.empty:
            _annotate(GEMINI_PERCELL["label"] + " (benchmark)", GEMINI_PERCELL["cost_per_1k"],
                      1000.0, g["balanced_accuracy"].mean(), cat_color["LLM"],
                      ABLATION_ARMS["A8_gemini_percluster"][1], hollow=True)

    ax.set_xscale("log")
    ax.set_xlabel("Model calls per 1,000 annotated cells")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, None)
    # Headroom on both sides: the cluster arm sits at ~40 calls and the per-cell arms at 1,000, so
    # without margin their labels run off the axis rather than merely colliding.
    ax.set_xlim(left=ax.get_xlim()[0] * 0.45, right=ax.get_xlim()[1] * 2.2)
    ax.grid(linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    # Collision-aware placement, the same helper Figure 2 uses. Hand-placed offsets put three
    # labels on top of the axis and each other, because the points are far apart in x and close
    # in y -- exactly the case a fixed offset cannot handle.
    # On a white box: a label at a point's height can sit on a grid line.
    _place_point_labels(ax, labels, obstacles, FS_POINT,
                        text_bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def _variant_slope(ax, frame, cat_color):
    """Expression-ranked against HVG sentences, paired per (arm, mode).

    Flat lines mean gene selection does not matter; crossing lines mean it interacts with context.
    The caption must carry the truncation caveat: at the 5,000-gene pool a large share of cells
    fall short of the 200-gene cap on some datasets, so gene choice and sentence length are partly
    confounded.
    """
    variants = ["expression_ranked", "hvg"]
    if not set(variants) <= set(frame["variant"]):
        ax.text(0.5, 0.5, "the HVG variant has not been run", ha="center", va="center",
                transform=ax.transAxes, color="0.5", fontsize=FS_LEGEND)
        ax.axis("off")
        return
    for arm, (family, marker, _ls, _label) in ABLATION_ARMS.items():
        for mode in MODE_ORDER:
            pair = [frame[(frame["arm"] == arm) & (frame["mode"] == mode)
                          & (frame["variant"] == v)]["balanced_accuracy"].mean()
                    for v in variants]
            if np.isnan(pair).any():
                continue
            ax.plot([0, 1], pair, marker=marker, markersize=5, linewidth=1.2,
                    color=cat_color[family], alpha=0.85, zorder=3)
    ax.set_xticks([0, 1])
    # Both sides rank by expression and both cap at 200 genes; only the ELIGIBLE VOCABULARY differs.
    # "expression ranked" vs "HVG pool" implied the ranking itself changed, which it does not.
    ax.set_xticklabels(["top 200 by expression\nfrom all genes",
                        "top 200 by expression\nfrom 5,000 HVGs"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, None)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)


def _ablation_dataset_panel(ax, ablation, dataset, gemini, arms, offsets, gemini_dx):
    """One dataset of the prompt-context ablation: each arm's mean over the seeds with its standard
    error, the seeds as small points, and Gemini per cell from the benchmark as a hollow star."""
    colors = _ranked_category_colors()
    modes = [m for m in MODE_ORDER if m in set(ablation["mode"])]
    x = np.arange(len(modes), dtype=float)
    if gemini_dx is not None:
        for mode, model in (("M1", "gemini-3.1-pro-preview_blind"),
                            ("M2", GEMINI_PERCELL["model"])):
            score = gemini.loc[(gemini["model"] == model) & (gemini["dataset"] == dataset),
                               "balanced_accuracy"]
            if len(score) and mode in modes:
                ax.scatter(x[modes.index(mode)] + gemini_dx, score.mean(), marker="*",
                           s=_scatter_area_for("*", 0.85 ** 2), facecolor="none",
                           edgecolor=colors["LLM"], linewidth=1.4, zorder=4,
                           label=GEMINI_PERCELL["label"] + " (benchmark)" if mode == "M2" else None)
    for dx, arm in zip(offsets, arms):
        family, marker, linestyle, label = ABLATION_ARMS[arm]
        sub = _arm_rows(ablation, arm)
        sub = sub[sub["dataset"] == dataset]
        means, errs = [], []
        for i, mode in enumerate(modes):
            vals = sub.loc[sub["mode"] == mode, "balanced_accuracy"].dropna().to_numpy()
            means.append(vals.mean() if len(vals) else np.nan)
            errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            ax.scatter(np.full(len(vals), x[i] + dx), vals, s=7, color=colors[family], alpha=0.55,
                       zorder=2, linewidths=0)
        ax.errorbar(x + dx, means, yerr=errs, linestyle=linestyle, marker=marker,
                    color=colors[family], linewidth=1.3, markersize=_marker_size_for(marker) * 0.85,
                    capsize=2.5, elinewidth=0.9, label=label, zorder=3)
    ax.set_title(dataset_label(dataset), fontsize=FS_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])
    ax.set_ylim(0, None)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)


def _name_column(ax, ends, x, fontsize):
    """Name lines in a column right of their last points ``ends`` = [(name, y)] at ``x``: the names
    keep the order of the ends and lie as close to them as one line of text between names allows
    (least squares, by isotonic regression), each joined to its end by a leader. Call after the
    y limits are set."""
    lo, hi = ax.get_ylim()
    points_high = ax.get_position().height * ax.figure.get_figheight() * 72
    placed = spread([y for _, y in ends], 1.25 * fontsize * (hi - lo) / points_high)
    for (name, y), y_text in zip(ends, placed):
        ax.annotate(name, xy=(x, y), xytext=(x + 0.5, y_text), va="center", fontsize=fontsize,
                    color="0.15", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", linewidth=0.4, color="0.6", shrinkA=4,
                                    shrinkB=2))


def _list_dataset_panel(ax, runs, reference, dataset):
    """One dataset of Figure 3B: each method as a line from no list through the exact list and
    A1-A3 to their union, coloured by family and marked by subcategory, named at the right."""
    steps = [0.0] + [REF_GAP + s for s in (0.0, 1.0, 2.0, 3.0, 3.0 + UNION_GAP)]
    means = (runs[runs["dataset"] == dataset]
             .groupby(["arm", "condition"])["balanced_accuracy"].mean())
    ends = []
    for arm in _list_study_arms(runs):
        colour, marker, size = _arm_style(arm)
        ys = [reference[arm]] + [means[(arm, c)] for c, _ in LIST_CONDITIONS]
        ax.plot(steps, ys, color=colour, linewidth=1.0, alpha=0.8, zorder=2,
                linestyle="--" if arm in PER_CLUSTER_ARMS else "-")
        ax.scatter(steps, ys, s=size * 0.7, marker=marker, color=colour, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ends.append((_display(STUDY_ARM_MODEL[arm]), ys[-1]))
    ax.set_title(dataset_label(dataset), fontsize=FS_TITLE)
    ax.set_xticks(steps)
    ax.set_xticklabels(["no list"] + [label for _, label in LIST_CONDITIONS], rotation=40,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.4, steps[-1] + 2.4)
    ax.set_ylim(0, None)  # per dataset, so close lines spread out
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    _name_column(ax, ends, steps[-1], FS_POINT)


def figure_context_per_dataset(df, out_path, dpi=600, ablation=None):
    """Supplementary for R2 — Figure 3A resolved per dataset, with the seeds shown: each arm's
    mean with its standard error over the seeds and the seeds as small points. Unlike Figure 3A it
    includes base Gemma and the shuffled gene order."""
    ablation = load_ablation() if ablation is None else ablation
    if ablation.empty:
        raise SystemExit("needs context_ablation/results/context_ablation.csv; run the study first")
    datasets = [d for d in DATASET_ORDER if d in set(ablation["dataset"])]
    arms = [a for a in ABLATION_ARMS if not _arm_rows(ablation, a).empty]
    gemini_ok = (df["model"] == GEMINI_PERCELL["model"]).any()
    offsets = np.linspace(-0.16, 0.16, len(arms) + gemini_ok)
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(11.6, 10.4))
        outer = fig.add_gridspec(2, 1, height_ratios=[9.4, 1.0], hspace=0.12)
        grid = outer[0].subgridspec(2, 2, hspace=0.45, wspace=0.18)
        panels = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(len(datasets))]
        for ax, dataset in zip(panels, datasets):
            _ablation_dataset_panel(ax, ablation, dataset, df, arms, offsets[:len(arms)],
                                    offsets[-1] if gemini_ok else None)
        ax_key = fig.add_subplot(outer[1])
        ax_key.axis("off")
        ax_key.legend(*panels[0].get_legend_handles_labels(), loc="center", ncol=3,
                      frameon=False, fontsize=FS_LEGEND)
        _panel_letters(fig, [(ax, chr(65 + i)) for i, ax in enumerate(panels)])
        return _save(fig, out_path, dpi)


def figure_lists_per_dataset(df, out_path, dpi=600, ablation=None, runs=None):
    """Supplementary for R2 — Figure 3B resolved per dataset, as Supplementary Figure S4 continued
    (panels E-H): each method as a line from no list through the exact list and A1-A3 to their
    union, the methods named at the right."""
    ablation = load_ablation() if ablation is None else ablation
    runs = load_list_runs() if runs is None else runs
    if ablation.empty or runs.empty:
        raise SystemExit("needs context_ablation/results/context_ablation.csv and "
                         "context_ablation/results_lists/runs.csv; run the studies first")
    datasets = [d for d in DATASET_ORDER if d in set(runs["dataset"])]
    arms = _list_study_arms(runs)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.4),
                                 gridspec_kw={"hspace": 0.5, "wspace": 0.18})
        panels = list(axes.ravel())
        for ax, dataset in zip(panels, datasets):
            _list_dataset_panel(ax, runs, _no_list_reference(ablation, df, arms, dataset), dataset)
        _panel_letters(fig, [(ax, chr(69 + i)) for i, ax in enumerate(panels)])
        return _save(fig, out_path, dpi)


def figure_context_ontology(df, out_path, dpi=600, ablation=None, runs=None):
    """Supplementary for R2 — Figure 3A and 3B on the agreement score, printed as Supplementary
    Figure S5 continued (panels E and F): half credit for a coarser type that is not broad or a
    finer type shows how much of a loss in balanced accuracy is near misses."""
    ablation = load_ablation() if ablation is None else ablation
    runs = load_list_runs() if runs is None else runs
    if ablation.empty or runs.empty:
        raise SystemExit("needs context_ablation/results/context_ablation.csv and "
                         "context_ablation/results_lists/runs.csv; run the studies first")
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 6.0))
        outer = fig.add_gridspec(3, 1, height_ratios=[4.0, 1.0, 1.2], hspace=0,
                                 left=0.075, right=0.985, top=0.94, bottom=0.02)
        ax_e, ax_f = _figure3_top(fig, outer, ablation, df, runs, "agreement")
        _panel_letters(fig, [(ax_e, "E"), (ax_f, "F")])
        return _save(fig, out_path, dpi)


def _unmapped_panel(ax, ablation):
    """The share of answers matching no Cell Ontology term, on Figure 3A's arms and conditions."""
    x = np.arange(len(PROMPT_MODES))
    for arm in PROMPT_ARMS:
        family, marker, linestyle, label = ABLATION_ARMS[arm]
        rows = _arm_rows(ablation, arm)
        ax.plot(x, [rows.loc[rows["mode"] == m, "non_cl_ratio"].mean() for m, _ in PROMPT_MODES],
                color=_ranked_category_colors()[family], marker=marker, linestyle=linestyle,
                markersize=_marker_size_for(marker) * 0.85, linewidth=1.4, label=label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in PROMPT_MODES], fontsize=FS_TICK - 1)
    ax.set_ylabel("Answers matching no ontology term")
    ax.set_ylim(0, None)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    ax.legend(loc="upper right", frameon=False, fontsize=FS_LEGEND - 1)


def figure_context_supporting(df, out_path, dpi=600, ablation=None):
    """Supplementary for R2 — the mechanism behind Figure 3A and three checks on the ablation.

    (A) Answers matching no Cell Ontology term, on Figure 3A's arms and conditions (Figure 3B until
        the rebuild of 2026-10-09). The base model is left to (B).
    (B) The base model's answer format, the caveat on any base-against-adapted comparison: most of
        Gemma-2-2B's answers resolve to no ontology term and many run past eight words, so its
        near-zero score mixes "wrong cell type" with "not a cell type".
    (C) The gene pool: the 200 genes drawn from 5,000 highly variable genes instead of all genes.
    (D) Clustering resolution, for both cluster arms; the numbers reported elsewhere use 1.0.
    """
    ablation = load_ablation() if ablation is None else ablation
    if ablation.empty:
        raise SystemExit("needs context_ablation/results/context_ablation.csv; run the study first")

    cat_color = _ranked_category_colors()
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 9.0))
        gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.42)
        # PANEL ORDER FOLLOWS R3, reordered 2026-08-15. R3.2 cites the base-model panel and
        # then the gene pool; R3.4 cites clustering resolution. The axes are named for
        # their CONTENT so that a future reorder changes only this one line.
        ax_unmapped, ax_base, ax_hvg, ax_res = (fig.add_subplot(gs[i // 2, i % 2])
                                                for i in range(4))
        _unmapped_panel(ax_unmapped, ablation)

        # (D) resolution sensitivity -----------------------------------------
        # BOTH cluster arms, not just GPT-OSS. The claim this panel supports is that the per-cluster
        # result is not an artefact of one lucky clustering; with a second model available at the
        # same two resolutions, showing only one of them tests the claim on half the evidence.
        variants = ["res0.5", "res1.0"]
        labels_a, obstacles_a = [], []
        for arm in ("A4_llm_percluster", "A8_gemini_percluster"):
            sub = ablation[ablation["arm"] == arm]
            if sub.empty or not set(variants) <= set(sub["variant"]):
                continue
            _fam, marker, linestyle, arm_label = ABLATION_ARMS[arm]
            for dataset in sorted(sub["dataset"].unique()):
                pair = [sub[(sub["dataset"] == dataset) & (sub["variant"] == v)]
                        ["balanced_accuracy"].mean() for v in variants]
                if np.isnan(pair).any():
                    continue
                ax_res.plot([0, 1], pair, marker=marker, linestyle=linestyle,
                          markersize=_marker_size_for(marker) * 0.85, linewidth=1.3,
                          color=cat_color["LLM"], alpha=0.85, zorder=3,
                          label=arm_label if dataset == sorted(sub["dataset"].unique())[0] else None)
                # Collision-aware placement instead of a fixed +6pt offset. With two arms the
                # right-hand endpoints for HFRA and Zeisel land within a few thousandths of each
                # other and their fixed-offset labels printed on top of one another.
                labels_a.append((dataset_label(dataset), 1, pair[1], "0.25"))
                obstacles_a.extend([(0, pair[0], 9.0), (1, pair[1], 9.0)])
        if labels_a:
            ax_res.set_xticks([0, 1])
            # Cluster counts, not just the resolution parameter: "0.5 vs 1.0" is meaningless without
            # knowing it roughly doubles the number of clusters the eval cells fall into. Counted on
            # the cluster arms present, which share one clustering per (dataset, resolution).
            cl = ablation[ablation["arm"].str.endswith("percluster")]
            counts = [int(round(cl[cl["variant"] == v]["n_clusters_in_eval"].mean()))
                      for v in variants]
            ax_res.set_xticklabels([f"resolution 0.5\n(~{counts[0]} clusters)",
                                  f"resolution 1.0\n(~{counts[1]} clusters)"])
            ax_res.set_xlim(-0.3, 1.75)
            ax_res.legend(loc="lower left", frameon=False, fontsize=FS_LEGEND - 1)
            _place_point_labels(ax_res, labels_a, obstacles_a, FS_POINT)
        ax_res.set_ylabel("Balanced accuracy")
        ax_res.set_ylim(0, None)
        ax_res.grid(axis="y", linestyle=":", alpha=0.5)
        sns.despine(ax=ax_res)

        # (B) base-model answer format ---------------------------------------
        a2 = ablation[(ablation["arm"] == "A2_base_percell")
                      & (ablation["variant"] == "expression_ranked")]
        modes = [m for m in MODE_ORDER if m in set(a2["mode"])]  # it ran no shuffled genes
        x = np.arange(len(modes), dtype=float)
        for col, style, label in [("non_cl_ratio", "-", "no ontology term"),
                                  ("frac_over_8_words", "--", "answer over 8 words")]:
            if col not in a2:
                continue
            vals = [a2[a2["mode"] == m][col].mean() for m in modes]
            ax_base.plot(x, vals, linestyle=style, marker="s", markersize=5,
                      color=cat_color["LLM"], linewidth=1.4, label=label, zorder=3)
        # The score it achieves, on the same axis, to make the gap between "answered" and "answered
        # usefully" immediate rather than something the reader has to hold in mind across figures.
        acc = [a2[a2["mode"] == m]["balanced_accuracy"].mean() for m in modes]
        ax_base.plot(x, acc, linestyle=":", marker="o", markersize=4, color="0.35",
                  linewidth=1.2, label="balanced accuracy", zorder=3)
        ax_base.set_xticks(x)
        ax_base.set_xticklabels([MODE_LABELS.get(m, m) for m in modes], fontsize=FS_TICK - 1)
        ax_base.set_ylabel("Fraction of predictions")
        ax_base.set_ylim(0, 1.05)
        ax_base.grid(axis="y", linestyle=":", alpha=0.5)
        # Nudged up from the default "center left" anchor (0, 0.5): the balanced-accuracy trace runs
        # along the bottom of this axes, and at dead centre the legend crowded it.
        ax_base.legend(loc="center left", bbox_to_anchor=(0, 0.60), frameon=False,
                    fontsize=FS_LEGEND - 1)
        ax_base.set_title("Gemma-2-2B (base)", fontsize=FS_TITLE)
        sns.despine(ax=ax_base)

        # (C) gene-pool ablation ---------------------------------------------
        _variant_slope(ax_hvg, ablation, cat_color)
        ax_hvg.set_ylabel("Balanced accuracy")

        _panel_letters(fig, [(ax_unmapped, "A"), (ax_base, "B"), (ax_hvg, "C"), (ax_res, "D")])
        return _save(fig, out_path, dpi)

# Every key IS the output filename (out = figure_{key}.{fmt}) and every filename matches the
# number LaTeX assigns, so the chain function -> file -> manuscript number holds end to end.
# RENUMBERED 2026-10-07: the model overview became Figure 1 (drawn outside this script; its PDF is
# figures/figure_1.pdf), so the benchmark results are Figure 2, the context ablation Figure 3 and
# the rare types Figure 4. The function names keep their earlier numbers.
FIGURES = {
    "2": figure_benchmark,               # benchmark results            -> Figure 2
    "2_composite": figure_1,             # panel B alone on the composite score, not in the paper
    "3": figure_2,                       # prompt-context ablation      -> Figure 3
    "4": figure_3,                       # rare cell types              -> Figure 4
    # Supplementary, numbered by first mention in the BiB build (Methods before Results), as
    # BiB asks (renumbered 2026-10-07): S1-S3 are cited in R1, S4-S6 in R2, S7-S11 in R3.
    "s1_harmonization": figure_harmonization,          # Figure 2G per method
    "s2_answer_types": figure_answer_types,            # Figure 2H per dataset and subcategory
    "s3_metric_agreement": figure_metric_agreement,    # task 2.3
    # R2, in the order it cites them (renumbered 2026-10-09): Figure 3A and 3B per dataset,
    # printed as S4 and S4 continued; the answer format, gene pool and clustering resolution; and
    # cost against accuracy.
    "s4_context_per_dataset": figure_context_per_dataset,
    "s4b_lists_per_dataset": figure_lists_per_dataset,
    "s5_context_supporting": figure_context_supporting,
    "s5b_ontology_scores": figure_context_ontology,
    "s6_cost_performance": figure_cost_performance,
    # R3: every rare cell type per method group, then (S7 continued) where the cells of the common
    # and of the rare types go.
    "s7_rare_full": figure_s7_rare_full,
    "s7b_rare_answer_types": figure_s7b_rare_answer_types,
    # One flow figure per dataset, S8 onward in RARE_FLOW_ORDER; HFRA has only two rare types at
    # or below the 3% floor.
    **{f"s{i + 8}_{ds.lower()}": _rare_flow_builder(ds)
       for i, ds in enumerate(RARE_FLOW_ORDER)},
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="Results/aggregated_by_dataset_model.csv",
                   help="aggregate snapshot CSV (default: %(default)s)")
    p.add_argument("--outdir", default="scLLMs_benchmark_paper/bioRxiv_template/figures",
                   help="where the composed figures are written (default: %(default)s)")
    p.add_argument("--figures", nargs="+", default=sorted(FIGURES), choices=sorted(FIGURES),
                   help="which figures to build (default: all)")
    p.add_argument("--subset-size", type=int, default=1000,
                   help="cells per model; the only size the LLMs ran at (default: %(default)s)")
    p.add_argument("--format", nargs="+", default=["pdf", "png"],
                   choices=["pdf", "png", "svg"],
                   help="output formats, one file each. Vector is preferred for submission; the "
                        "raster copy is for slides and previews (default: %(default)s)")
    p.add_argument("--dpi", type=int, default=300,
                   help="raster dpi; BiB asks for 300 for greyscale and 600 for line drawings, "
                        "and the vector copy is resolution-independent (default: %(default)s)")
    args = p.parse_args(argv)

    df = load_aggregate(args.input, subset_size=args.subset_size)
    outdir = Path(args.outdir)
    for n in args.figures:
        for fmt in args.format:
            out = FIGURES[n](df, outdir / f"figure_{n}.{fmt}", dpi=args.dpi)
            print(f"[figure {n}]  {df['model'].nunique()} models, "
                  f"{df['dataset'].nunique()} datasets -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
