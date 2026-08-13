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
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_model_comparison import (  # noqa: E402
    CATEGORY_TIERS,
    _RANKED_CATEGORY_LABELS,
    COMPOSITE_CATEGORIES,
    DEFAULT_DATES_CSV,
    EXCLUDED_BASES,
    _place_point_labels,
    _ranked_category_colors,
    categorize_model,
    derive_mode,
    load_confusion_pairs,
    load_model_dates,
    load_per_class_recall,
    model_base,
    model_tier,
    pretty,
    plot_composite_category,
    plot_composite_ranked_bars,
    plot_family_by_dataset,
    plot_metric_disagreement,
    plot_rare_type_flows,
    plot_recall_by_prevalence,
)

# --------------------------------------------------------------------------- #
# One typographic scale for every manuscript figure. Panel functions set their own sizes for
# standalone use; `_harmonize` overrides them here so panels drawn side by side match. Sizes are
# in points at the figure's native size -- do not scale the saved file afterwards.
# --------------------------------------------------------------------------- #
FONT_FAMILY = "DejaVu Sans"
FS_PANEL_LETTER = 15   # A, B, C
FS_TITLE = 10          # panel titles
FS_LABEL = 11          # axis labels
FS_TICK = 10           # tick labels
FS_GROUP = 12          # in-axes group headings (the family names under the strip panel)
FS_POINT = 7.5         # in-axes point annotations (model names in the scatter)
FS_LEGEND = 9
FS_ROSTER = 7.5        # model-name roster beside panel D

# Prose names for the two information regimes, applied to legend text at the end.
REGIME = {"blind": "unconstrained", "unblind": "constrained"}
# Axis-tick abbreviations. The full terms are three times longer than the code names and collide
# between adjacent sub-columns, so the strip panel carries single bold letters and the legend
# spells them out. The prose in the manuscript still uses the full terms everywhere (style Rule 5).
REGIME_SHORT = {"blind": "U", "unblind": "C"}

SUBSET_SIZE_COL = "subset_size"


def _rc():
    """rcParams shared by every manuscript figure."""
    return {
        "font.family": FONT_FAMILY,
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


def _roster_handles(df, wrap=30, with_tier_names=False, linestyle="none"):
    """Legend handles naming every model scored, grouped by method tier.

    One entry per tier: the tier's marker in its family color, then the models that carry it.
    Model names are mode-stripped, since a model's ``_blind`` and ``_unblind`` runs are the same
    model and both appear in the panels.

    ``with_tier_names`` prefixes each entry with the tier's name, which lets ONE legend block do
    the work of both a tier key and a roster. Without it the two would key the same seven symbols
    twice over — the reason Figure 2 merges them.

    Tiers with no scored model are skipped, so the roster describes this figure rather than the
    configuration file.
    """
    cat_color = _ranked_category_colors()
    scored = {}
    for model in df["model"].unique():
        category = categorize_model(model)
        if category == "simple":
            continue
        scored.setdefault((category, model_tier(model)), set()).add(model_base(model))

    handles = []
    for key, _clabel in COMPOSITE_CATEGORIES:
        for tier_key, tier_label, marker, _size in CATEGORY_TIERS.get(key, []):
            names = sorted(scored.get((key, tier_key), ()))
            if not names:
                continue
            text = ", ".join(names)
            text = f"{tier_label}: {text}" if with_tier_names else f": {text}"
            handles.append(plt.Line2D(
                [], [], linestyle=linestyle, marker=marker, color=cat_color[key], linewidth=1.4,
                markersize=11 if marker == "*" else 6.5,
                label=textwrap.fill(text, wrap)))
    return handles


def _panel_letters(fig, items, pad_x=0.012, pad_y=0.008):
    """Place bold panel letters outside the top-left corner of each axes in ``items``.

    ``items`` is ``[(ax, letter)]``. Two rules, both of which need the whole set rather than one
    panel at a time:

    * **Vertical** anchors on each panel's own TIGHT bounding box, so a letter clears whatever its
      panel carries — above the title where there is one, just above the frame where there is not.
    * **Horizontal** aligns every panel that starts in the same figure column to a single x, taken
      from the leftmost tight bbox in that column. Anchoring each letter to its own bbox would
      indent the letters of panels without y-axis decorations (the schematic placeholder) relative
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


def _placeholder(ax, text):
    """A reserved, empty panel: dashed outline, centered note, no data.

    Kept as a real axes rather than white space so the space is held at exactly the right size
    and the schematic can be dropped in later without re-tuning the layout.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for side, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linestyle((0, (4, 4)))
        spine.set_linewidth(0.8)
        spine.set_color("0.75")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=FS_LABEL, color="0.55")


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


def figure_1(df, out_path, dpi=600, schematic_note=None, metric=None):
    """Figure 1 — the protocol and the landscape it produces.

    (A) reserved for the pipeline and benchmark-setup schematic, drawn outside this script.
    (B) composite score by model family, unconstrained vs constrained, one line per model.

    Deliberately just those two. This is the Problem Solving Protocol move: what was run, and what
    came out, in one display item. The metric-behavior analysis that used to sit here as panels C
    and D is Figure 2, so that each Results section cites its own figure instead of sending the
    reader back to a figure they met two sections earlier.
    """
    note = schematic_note or "A  pipeline and benchmark overview\n(schematic placeholder)"
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(12.2, 11.2))
        # Panel A takes well over a third of the page: the schematic has to carry the pipeline,
        # the model roster and the two regimes, and anything smaller cannot.
        gs = fig.add_gridspec(2, 2, height_ratios=[6.2, 4.0], width_ratios=[1.0, 0.34],
                              hspace=0.30, wspace=0.04)

        ax_a = fig.add_subplot(gs[0, :])
        _placeholder(ax_a, note)
        ax_b = fig.add_subplot(gs[1, 0])
        ax_leg = fig.add_subplot(gs[1, 1])
        ax_leg.axis("off")

        handles = plot_composite_category(df, None, "", dpi, ax=ax_b, metric=metric)
        if not handles:
            raise SystemExit(f"Figure 1B: {metric or 'composite'} missing from the aggregate.")
        _harmonize(ax_b, text_size=FS_GROUP)
        _relabel_regimes(ax_b)

        handles = _regime_key_handles() + [_blank()] + list(handles)
        for h in handles:
            h.set_label(_to_prose_regime(h.get_label() or ""))
        ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=FS_LEGEND,
                      labelspacing=0.32, handletextpad=0.6, borderaxespad=0.0)

        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B")])
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


def _display(base):
    return DISPLAY_NAMES.get(base, pretty(base))


def _tier_marker(model):
    """Marker for a model's method tier, matching Figure 1B's key."""
    for key, _label, marker, _size in CATEGORY_TIERS.get(categorize_model(model), []):
        if key == model_tier(model):
            return marker
    return "o"


def figure_2(df, out_path, dpi=600, per_class=None):
    """Figure 2 — what ontology harmonization contributes, by method family.

    One panel per R2 paragraph, unconstrained setting throughout.
    (A) fraction of cells whose RAW label is not a Cell Ontology term, per method.
    (B) text-encoder score before and after mapping to the ontology. The gap IS the lift, so the
        panel shows the quantity R2 argues from rather than a derived number.
    (C) the two families R2's third paragraph compares, per dataset, on the two metrics that are
        invariant to how many labels a method uses.

    A and B share one row order (by lift, descending) so the reader can read across: what a
    method emits natively, and what the mapping step then buys it. The two are related but not
    monotonic, and the exceptions are the interesting part — scType and CellTypist both emit zero
    ontology terms yet sit far apart in B, because scType's labels are standard names in plural
    form while CellTypist's are model-internal cluster names.

    NOT drawn here: macro F1. Its macro average runs over the union of true and predicted classes,
    so it scores label-space size as well as classification, in both of its readings and in
    opposite directions (see the R2 header comment in 01_Article_MainText.tex). Panel C therefore
    uses balanced accuracy and Lin similarity, which are invariant to both effects.
    """
    need = ["non_cl_ratio", "text_emb_norm_balanced_raw", "text_emb_norm_balanced_mapped",
            "text_emb_norm_mapping_lift"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Figure 2 needs {', '.join(missing)}, which this aggregate does not carry. "
            "Regenerate it with the current bin/aggregate.py (it backfills these from the run "
            "JSONs without re-scoring) and re-snapshot.")

    d = df.copy()
    d["base"] = d["model"].map(model_base)
    d["cat"] = d["model"].map(categorize_model)
    d["regime"] = d["model"].map(derive_mode)
    d = d[d["cat"] != "simple"]          # chance floors have no label vocabulary to characterize
    if "text_emb_norm_clipped" not in d.columns:
        d["text_emb_norm_clipped"] = 0.0

    # Three row blocks rather than one. The baselines produce a SINGLE run, which the pipeline
    # names `_blind` for filename reasons only -- their label space is fixed by a marker database
    # or a reference, so the constrained/unconstrained axis does not apply to them. Pooling them
    # into an "unconstrained" block would assert a regime membership the paper denies (see the
    # abstract's comment block). Giving them their own block states it in the layout instead of
    # in a caption disclaimer, and it lets the same 17 language models appear under both settings.
    groups = [
        ("unconstrained", d[(d["regime"] == "blind") & (d["cat"] != "classical")]),
        ("constrained", d[d["regime"] == "unblind"]),
        ("baselines (single score)", d[d["cat"] == "classical"]),
    ]

    def _agg(frame):
        return (frame.groupby(["base", "cat"], as_index=False)
                     .agg(model=("model", "first"), non_cl=("non_cl_ratio", "mean"),
                          raw=("text_emb_norm_balanced_raw", "mean"),
                          mapped=("text_emb_norm_balanced_mapped", "mean"),
                          lift=("text_emb_norm_mapping_lift", "mean"),
                          clipped=("text_emb_norm_clipped", "mean"))
                     .sort_values("lift", ascending=False).reset_index(drop=True))

    GAP = 1.9                      # blank rows between blocks, leaving room for the heading
    blocks, rows, pos = [], [], 0.0
    for title, frame in groups:
        if frame.empty:
            continue
        pos += GAP
        block = _agg(frame)
        blocks.append((title, pos - 1.0, block))
        for _, r in block.iterrows():
            rows.append({**r.to_dict(), "ypos": pos, "regime": title})
            pos += 1
    agg = pd.DataFrame(rows)
    cat_color = _ranked_category_colors()

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(11.6, 13.0))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.30], wspace=0.02)
        left = outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.26], hspace=0.13)
        # A is narrow: it carries one bounded fraction. B gets the width because the LENGTH of
        # each segment is the quantity being read off it.
        top = left[0].subgridspec(1, 2, width_ratios=[0.52, 1.0], wspace=0.04)
        ax_a = fig.add_subplot(top[0, 0])
        ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
        bot = left[1].subgridspec(1, 2, width_ratios=[1.0, 0.001], wspace=0.02)
        ax_c1 = fig.add_subplot(bot[0, 0])
        ax_c2 = fig.add_subplot(bot[0, 1])
        ax_leg = fig.add_subplot(outer[0, 1])
        ax_leg.axis("off")

        y = agg["ypos"].to_numpy()
        colors = [cat_color[c] for c in agg["cat"]]

        def _cells(row, column):
            """The four per-dataset values behind one drawn row, matched on method AND block."""
            sel = (d["base"] == row["base"])
            if row["regime"] == "constrained":
                sel &= d["regime"] == "unblind"
            elif row["regime"] == "unconstrained":
                sel &= d["regime"] == "blind"
            return d.loc[sel, column].dropna()

        # --- A: what each method emits before any harmonization -------------------------
        ax_a.barh(y, agg["non_cl"], height=0.62, color=colors, alpha=0.85, zorder=2)
        for (_, row), yy in zip(agg.iterrows(), y):
            pts = _cells(row, "non_cl_ratio")
            ax_a.scatter(pts, np.full(len(pts), yy), s=7, color="0.25", alpha=0.5,
                         linewidths=0, zorder=3)
        ax_a.set_xlim(0, 1.04)
        ax_a.set_xlabel("non-ontology fraction")

        # --- B: what the harmonization step then buys ------------------------------------
        # Per-dataset values carry the SAME fill encoding as the means (hollow = as written,
        # filled = after mapping), which is what an earlier attempt got wrong: drawn as
        # undifferentiated gray dots they could not say which end of the segment they belonged
        # to. Drawn as points rather than as one thin segment per dataset, because 84 near-
        # horizontal lines read as stray rules across the panel.
        for (_, r), yy in zip(agg.iterrows(), y):
            col = cat_color[r["cat"]]
            raw_pts, map_pts = _cells(r, "text_emb_norm_balanced_raw"), _cells(r, "text_emb_norm_balanced_mapped")
            ax_b.scatter(raw_pts, np.full(len(raw_pts), yy), s=11, facecolor="none",
                         edgecolor=col, linewidths=0.7, alpha=0.6, zorder=3)
            ax_b.scatter(map_pts, np.full(len(map_pts), yy), s=11, color=col, alpha=0.5,
                         linewidths=0, zorder=3)
            ax_b.plot([r["raw"], r["mapped"]], [yy, yy], color=col, lw=1.8, alpha=0.95, zorder=2)
            ax_b.scatter(r["raw"], yy, s=34, facecolor="white", edgecolor=col, linewidths=1.3,
                         zorder=4)
            ax_b.scatter(r["mapped"], yy, s=46, marker=_tier_marker(r["model"]), color=col,
                         zorder=4)
        ax_b.set_xlabel("text-encoder score (unmapped $\\rightarrow$ mapped)")
        # Bound on the per-dataset points, not on the means: several datasets sit well outside
        # their model's mean and would otherwise be clipped at the axis.
        ax_b.set_xlim(-0.02, float(np.nanmax([d["text_emb_norm_balanced_raw"].max(),
                                              d["text_emb_norm_balanced_mapped"].max()])) + 0.04)

        # A dagger on the name, not a third marker shape in B: the marker slot there already
        # encodes method tier, and overloading it would break the key shared with Figure 1B.
        labels = [_display(b) + (" $\\dagger$" if c > 0 else "")
                  for b, c in zip(agg["base"], agg["clipped"])]
        ax_a.set_yticks(y)
        ax_a.set_yticklabels(labels)
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

        # --- C: which family leads, per dataset and per metric ---------------------------
        # Drawn as a DIFFERENCE, not as two lines. The claim is that the lead changes sign
        # between datasets and between metrics, and on overlaid family means those flips are a
        # few thousandths apart and invisible. Plotting LLM minus baseline puts the claim on the
        # y axis directly, and coloring each bar by whichever family is ahead means the reader
        # never has to remember which line was which.
        # UNCONSTRAINED runs only, matching the claim this panel supports. `d` now carries both
        # regimes for panels A and B, so without this filter the LLM mean would silently average
        # a model's constrained and unconstrained runs together and every bar would be inflated.
        dc = d[d["regime"] == "blind"]
        datasets = [ds for ds in DATASET_ORDER if ds in set(dc["dataset"])]
        metrics = [("balanced_accuracy", "balanced accuracy"),
                   ("lin_avg_similarity", "Lin similarity")]
        width = 0.34
        for k, (metric, _label) in enumerate(metrics):
            for j, ds in enumerate(datasets):
                sub = dc[dc["dataset"] == ds]
                diff = (sub.loc[sub["cat"] == "LLM", metric].mean()
                        - sub.loc[sub["cat"] == "classical", metric].mean())
                x = j + (k - 0.5) * width
                ax_c1.bar(x, diff, width=width * 0.92,
                          color=cat_color["LLM" if diff > 0 else "classical"],
                          alpha=0.9, zorder=2)
                ax_c1.text(x, diff + (0.006 if diff > 0 else -0.006), f"{diff:+.3f}",
                           ha="center", va="bottom" if diff > 0 else "top", fontsize=FS_POINT)
        ax_c1.axhline(0, color="0.25", lw=0.9, zorder=3)
        ax_c1.set_xticks(range(len(datasets)))
        ax_c1.set_xticklabels(datasets)
        ax_c1.set_ylabel("LLMs $-$ baselines")
        ax_c1.set_xlim(-0.55, len(datasets) - 0.45)
        ax_c1.margins(y=0.22)
        ax_c1.grid(axis="y", color="0.9", lw=0.6)
        ax_c1.set_axisbelow(True)
        for side in ("top", "right"):
            ax_c1.spines[side].set_visible(False)
        ax_c2.remove()

        for ax in (ax_a, ax_b, ax_c1):
            _harmonize(ax, text_size=FS_POINT)

        # --- legend ----------------------------------------------------------------------
        handles = [_blank("method family:")]
        handles += [plt.Line2D([0], [0], linestyle="none", marker="s", color=cat_color[k],
                               label=_RANKED_CATEGORY_LABELS[k])
                    for k in ("classical", "scLLM", "LLM") if k in set(agg["cat"])]
        handles += [_blank(), _blank("B — score:"),
                    plt.Line2D([0], [0], linestyle="none", marker="o", markerfacecolor="white",
                               markeredgecolor="0.35", label="as the model wrote it"),
                    plt.Line2D([0], [0], linestyle="none", marker="o", color="0.35",
                               label="after ontology mapping"),
                    ]
        handles += [_blank(), _blank("B — method tiers:")]
        seen = set()
        for cat in ("classical", "scLLM", "LLM"):
            for key, label, marker, _size in CATEGORY_TIERS.get(cat, []):
                if key in seen or not (agg["model"].map(model_tier) == key).any():
                    continue
                seen.add(key)
                handles.append(plt.Line2D([0], [0], linestyle="none", marker=marker,
                                          color=cat_color[cat], label=label))
        handles += [_blank(), _blank("A — dots: one per dataset"),
                    _blank("B — open/filled circles: one per dataset"),
                    _blank("C — unconstrained family means,"),
                    _blank("   LLMs minus baselines; bar colored"),
                    _blank("   by which family leads"),
                    _blank("   left bar: balanced accuracy"),
                    _blank("   right bar: Lin similarity"),
                    _blank("$\\dagger$ unmapped score at the random-pair"),
                    _blank("   floor, so the gain is a lower bound")]
        ax_leg.legend(handles=handles, loc="upper left", frameon=False, fontsize=FS_LEGEND,
                      labelspacing=0.45, handletextpad=0.6, borderaxespad=0.0, alignment="left")

        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c1, "C")])
        return _save(fig, out_path, dpi)


DEFAULT_RARE_CSV = "cache/rare_type_outcomes_1000_stratified.csv"
# Prevalence bands. Log-spaced by decade, which is the natural scale here: true prevalence spans
# 1.4e-05 to 0.41 across the 87 scored cell types. Equal-width bands would put 80 of them in one
# bin. The band edges are a presentational choice and the per-class values behind them are in the
# table this reads, so nothing depends on where exactly the cuts fall.
RARE_BANDS = ([0, 0.001, 0.01, 0.05, 1.0], ["<0.1%", "0.1-1%", "1-5%", ">5%"])
# Colour by family, line style by setting -- the same two encodings Figures 1 and 2 use.
RARE_BLOCKS = [
    ("constrained LLM", "LLM", "-"),
    ("unconstrained LLM", "LLM", "--"),
    ("constrained scLLM", "scLLM", "-"),
    ("unconstrained scLLM", "scLLM", "--"),
    ("baselines", "classical", "-"),
]


def load_rare_outcomes(path=DEFAULT_RARE_CSV):
    """Per-class recall and reachability by true dataset prevalence. Built by bin/rare_type_table.py."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"No rare-type table at {path}. Build it with:\n"
            f"    conda run -n evaluate python bin/rare_type_table.py")
    frame = pd.read_csv(path)
    edges, names = RARE_BANDS
    frame["band"] = pd.cut(frame["orig_prevalence"], edges, labels=names)
    return frame


def figure_3(df, out_path, dpi=600, rare=None):
    """Figure 3 -- how methods fail on rare cell types, by true dataset prevalence.

    (A) mean per-class recall against how common the cell type is in the FULL dataset.
    (B) the share of cell types a method never recovers at all.
    (C) for the rare types, what kind of failure it was: the label never appeared in the method's
        output, or it appeared but never on a cell of that type.

    Prevalence is the cell type's share of the full dataset, NOT of the evaluation subset. The
    subsets are stratified with a 20-cell floor, so 54 of the 87 cell types sit at exactly that
    floor and the subset's own prevalence is mostly the sampler. See bin/rare_type_table.py.
    """
    rare = load_rare_outcomes() if rare is None else rare
    _edges, names = RARE_BANDS
    cat_color = _ranked_category_colors()
    x = np.arange(len(names))

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(11.6, 4.2))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.28, 0.60], wspace=0.36)
        ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
        ax_leg = fig.add_subplot(gs[0, 3]); ax_leg.axis("off")

        for block, family, style in RARE_BLOCKS:
            sub = rare[rare["block"] == block]
            if sub.empty:
                continue
            recall = [sub.loc[sub["band"] == b, "recall"].mean() for b in names]
            missed = [(sub.loc[sub["band"] == b, "recall"] == 0).mean() for b in names]
            for ax, series in ((ax_a, recall), (ax_b, missed)):
                ax.plot(x, series, color=cat_color[family], linestyle=style, lw=1.8,
                        marker="o", ms=4.5, zorder=3)
        for ax, label in ((ax_a, "mean per-class recall"),
                          (ax_b, "share of cell types never recovered")):
            ax.set_xticks(x); ax.set_xticklabels(names)
            ax.set_xlabel("prevalence in the full dataset")
            ax.set_ylabel(label)
            ax.set_ylim(0, 1.0 if ax is ax_b else None)
            ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        # C: the rare classes only, decomposed. Pooling the two rarest bands gives n = 44 cell
        # types, which is what makes the three-way split readable rather than noisy.
        rarecls = rare[rare["orig_prevalence"] < 0.01]
        blocks = [b for b, _f, _s in RARE_BLOCKS if b in set(rarecls["block"])]
        shades = [("never emitted", "0.30"), ("emitted, never correct", "0.62"),
                  ("recovered", "0.88")]
        bottom = np.zeros(len(blocks))
        parts = []
        for name in ("never emitted", "emitted, never correct", "recovered"):
            vals = []
            for block in blocks:
                s = rarecls[rarecls["block"] == block]
                if name == "never emitted":
                    vals.append((~s["emitted"].astype(bool)).mean())
                elif name == "emitted, never correct":
                    vals.append(((s["recall"] == 0) & s["emitted"].astype(bool)).mean())
                else:
                    vals.append((s["recall"] > 0).mean())
            parts.append(np.array(vals))
        for (name, shade), vals in zip(shades, parts):
            ax_c.bar(np.arange(len(blocks)), vals, bottom=bottom, width=0.66,
                     color=shade, edgecolor="white", linewidth=0.7, zorder=2)
            bottom = bottom + vals
        ax_c.set_xticks(np.arange(len(blocks)))
        ax_c.set_xticklabels(blocks, rotation=28, ha="right", fontsize=FS_POINT)
        ax_c.set_ylabel("share of rare cell types")
        ax_c.set_ylim(0, 1)
        for side in ("top", "right"):
            ax_c.spines[side].set_visible(False)

        for ax in (ax_a, ax_b, ax_c):
            _harmonize(ax, text_size=FS_POINT)

        handles = [_blank("A, B — method group:")]
        for block, family, style in RARE_BLOCKS:
            handles.append(plt.Line2D([0], [0], color=cat_color[family], linestyle=style,
                                      lw=1.8, marker="o", ms=4.5, label=block))
        handles += [_blank(), _blank("C — outcome on cell types"), _blank("below 1% prevalence:")]
        handles += [plt.Line2D([0], [0], linestyle="none", marker="s", color=shade, label=name)
                    for name, shade in shades]
        ax_leg.legend(handles=handles, loc="upper left", frameon=False, fontsize=FS_LEGEND,
                      labelspacing=0.45, handletextpad=0.6, borderaxespad=0.0, alignment="left")
        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")])
        return _save(fig, out_path, dpi)


def select_flow_models(df, metric="balanced_accuracy"):
    """One representative per family for the rare-type flow figures, by a stated rule.

    Rule: the highest mean ``metric`` within its family, among models that ran on EVERY dataset.
    Two rows come back — constrained on top, unconstrained beneath, same model per column — with
    ``None`` where the baseline would sit in row 2, since it carries a single score.

    Why not the best model per dataset. It changes the scLLM in all four figures (langcell,
    mmContext, C2S-Scale-2B, InstructCell) and in doing so crosses the generative/retrieval split,
    whose members behave oppositely when unconstrained. The four supplementary figures would stop
    being comparable, and a change of sub-family would read as a change of failure mode.

    Why "ran on every dataset" is part of the rule rather than a caveat. mmContext has the highest
    scLLM mean (0.408 vs 0.369) but is blacklisted on mouse data, so it never ran on Zeisel;
    picking it would leave one figure with no scLLM at all.
    """
    d = df.copy()
    d["model_base"] = d["model"].map(model_base)
    d["category"] = d["model"].map(categorize_model)
    d["mode"] = d["model"].map(derive_mode)
    d = d[(~d["model_base"].isin(EXCLUDED_BASES)) & (d["category"] != "simple")]
    n_datasets = d["dataset"].nunique()

    picks = {}
    for category, mode in (("classical", "blind"), ("scLLM", "unblind"), ("LLM", "unblind")):
        sub = d[(d["category"] == category) & (d["mode"] == mode)]
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

    return [[named(picks["classical"], "blind"), named(picks["scLLM"], "unblind"),
             named(picks["LLM"], "unblind")],
            [None, named(picks["scLLM"], "blind"), named(picks["LLM"], "blind")]]
# How many rare types each dataset can support. HFRA has 9 classes and only 2 at or below the 3%
# floor, human_pancreas 4; asking for 5 everywhere would silently pad with common types.
RARE_FLOW_N_TYPES = {"HFRA": 2, "human_pancreas": 4, "HIHA": 5, "Zeisel": 5}
# Order the per-dataset flow figures appear in the supplement, and therefore their S-numbers.
# Alphabetical, matching the dataset order on Figure 2C so the two are read the same way.
RARE_FLOW_ORDER = ["HFRA", "HIHA", "Zeisel", "human_pancreas"]


def figure_rare_flows(df, out_path, dpi=600, dataset="HIHA", models=None, n_types=None):
    """Supplementary — where the cells of the rarest types actually go.

    Figure 2D says how OFTEN rare types are recovered; it cannot say what they are called instead,
    which is the difference between a near miss inside the right lineage and an unrelated call.
    One alluvial per model and regime: rare true types on the left, predicted labels on the right.

    ``df`` is unused — kept so every figure builder has the same signature.
    """
    conf = load_confusion_pairs()
    if conf is None or len(conf) == 0:
        raise SystemExit("Rare-type flows: no confusion-pair table; see cache/confusion_pairs_*.")
    n_types = n_types or RARE_FLOW_N_TYPES.get(dataset, 5)
    models = models or select_flow_models(df)
    with plt.rc_context(_rc()):
        # No in-figure title: the caption states the dataset and what is shown, and a title
        # duplicating it is redundant ink in a published figure.
        ok = plot_rare_type_flows(conf, out_path, dpi, dataset=dataset, models=models,
                                  n_types=n_types, title=None)
    if not ok:
        raise SystemExit(f"Rare-type flows: nothing to draw for {dataset}.")
    return Path(out_path)


def figure_1_balanced_accuracy(df, out_path, dpi=600, schematic_note=None):
    """Figure 1 with raw balanced accuracy on the B axis instead of the composite.

    Identical layout, so the two can be compared directly while the y-axis decision is open.
    Balanced accuracy needs no defense in Methods; the composite does, since it is our own
    construction. Only one of the two goes into the manuscript.
    """
    return figure_1(df, out_path, dpi, schematic_note, metric="balanced_accuracy")


# Keyed by the suffix used in the output filename, so a variant is a first-class figure rather
# than a flag: `figure_1.pdf`, `figure_1_balanced_accuracy.pdf`, `figure_2.pdf`.
def figure_ranked_balanced_accuracy(df, out_path, dpi=600):
    """Supplementary for R1 — every method ranked by balanced accuracy, both regimes.

    Figure 1B shows the families and their spread but does not name the models or order them, and
    it is drawn on the composite. This gives the per-model ranking on the primary metric instead,
    which is the view a reader wants when checking a specific model.

    Chance is marked as a dashed red line rather than as bars, which the composite version cannot
    do at all: the composite is normalized against random-stratified, so it is exactly 0 there by
    construction. On balanced accuracy the level is real, and several models sit at or below it.
    Bars for the three chance models would say the same thing three times and, worse, sort them
    into the ranking as though they were competing methods.
    """
    with plt.rc_context(_rc()):
        ok = plot_composite_ranked_bars(df, out_path, dpi, metric="balanced_accuracy",
                                        include_simple=False, prose_regimes=True)
    if not ok:
        raise SystemExit("Ranked balanced accuracy: metric missing from the aggregate.")
    return Path(out_path)


# Datasets whose public release post-dates several of the benchmarked models. HFRA and HIHA
# reached CELLxGENE in December 2025; Zeisel (2015) and human_pancreas (2016) have been public for
# a decade. The split is what makes the date trend testable against pretraining exposure.
RECENTLY_PUBLIC = {"HFRA", "HIHA"}


def figure_date_vs_performance(df, out_path, dpi=600, metric="balanced_accuracy"):
    """Supplementary for R2 — performance against model release date, and whether that trend is
    explained by the datasets that became public inside the release window.

    (A) One point per general-purpose LLM and regime. The trend is strong, and release date is
        collinear with weight availability, reasoning capability and vendor in this roster.
    (B) The same rank correlation computed per dataset. Two of the four became public in December
        2025, inside the release window of the strongest models; the other two have been public
        for a decade. A trend driven by exposure to the recent pair would be steeper on them.
    """
    dates = load_model_dates(DEFAULT_DATES_CSV, "artifact_available_date")
    if dates.empty or metric not in df.columns:
        raise SystemExit("Date figure: no usable model dates or metric missing.")

    d = df.copy()
    d["model_base"] = d["model"].map(model_base)
    d["category"] = d["model"].map(categorize_model)
    d["mode"] = d["model"].map(derive_mode)
    d = d[(d["category"] == "LLM") & (~d["model_base"].isin(EXCLUDED_BASES))].copy()
    d["date"] = d["model_base"].map(dates)
    d = d.dropna(subset=["date"])
    if d.empty:
        raise SystemExit("Date figure: no LLM rows carry a release date.")

    cat_color = _ranked_category_colors()
    with plt.rc_context(_rc()):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                         gridspec_kw={"width_ratios": [1.35, 1.0]})

        # (A) date vs metric, one point per model x regime.
        per = d.groupby(["model_base", "mode"])[metric].mean().reset_index()
        per["date"] = per["model_base"].map(dates)
        for mode, marker, face in (("unblind", "o", cat_color["LLM"]), ("blind", "o", "white")):
            sub = per[per["mode"] == mode]
            ax_a.scatter(sub["date"], sub[metric], s=46, marker=marker, facecolor=face,
                         edgecolor=cat_color["LLM"], linewidth=1.2, zorder=3,
                         label=REGIME[mode])
            rho = (sub["date"].map(pd.Timestamp.toordinal)
                   .corr(sub[metric], method="spearman"))
            ax_a.plot([], [], " ", label=f"   Spearman $\\rho$ = {rho:.2f} (n = {len(sub)})")
        ax_a.set_xlabel("model release date")
        ax_a.set_ylabel(f"{pretty(metric)} (mean over datasets)")
        ax_a.tick_params(axis="x", rotation=25)
        for lbl in ax_a.get_xticklabels():
            lbl.set_ha("right")
        ax_a.grid(linestyle=":", alpha=0.5)
        ax_a.legend(loc="upper left", frameon=False, fontsize=FS_LEGEND - 0.5)
        sns.despine(ax=ax_a)

        # (B) the same correlation, per dataset, grouped by how long the dataset has been public.
        datasets = sorted(d["dataset"].astype(str).unique())
        width, xs = 0.38, np.arange(len(datasets), dtype=float)
        for i, (mode, hatch) in enumerate((("unblind", None), ("blind", "///"))):
            vals = []
            for ds in datasets:
                g = d[(d["mode"] == mode) & (d["dataset"].astype(str) == ds)]
                vals.append(g["date"].map(pd.Timestamp.toordinal)
                            .corr(g[metric], method="spearman"))
            ax_b.bar(xs + (i - 0.5) * width, vals, width, color=cat_color["LLM"], hatch=hatch,
                     edgecolor="black" if hatch else "white", linewidth=0.5, zorder=3,
                     label=REGIME[mode])
        ax_b.set_xticks(xs)
        ax_b.set_xticklabels([f"{ds}\npublic {'12/2025' if ds in RECENTLY_PUBLIC else '2015-16'}"
                              for ds in datasets], fontsize=FS_TICK - 2.0)
        ax_b.set_ylabel(f"Spearman $\\rho$ (date, {pretty(metric).lower()})")
        ax_b.set_ylim(0, 1.0)
        ax_b.grid(axis="y", linestyle=":", alpha=0.5)
        ax_b.legend(loc="upper right", frameon=False, fontsize=FS_LEGEND - 0.5)
        sns.despine(ax=ax_b)

        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B")])
        return _save(fig, out_path, dpi)


def _rare_flow_builder(dataset):
    def build(df, out_path, dpi=600):
        return figure_rare_flows(df, out_path, dpi, dataset=dataset)
    build.__doc__ = f"Rare-type prediction flows for {dataset}."
    return build


# --------------------------------------------------------------------------- #
# Figure 5 — the context ablation (Results R5).
#
# Soft dependency, like the model dates and the per-class recall table: the study lives in
# ../context_ablation and has to be RUN before this can draw anything. Absent input is reported as
# a clear instruction rather than an exception, so `python bin/manuscript_figures.py` with no
# arguments still builds every other figure.
# --------------------------------------------------------------------------- #
DEFAULT_ABLATION_CSV = (Path(__file__).resolve().parents[1] / "context_ablation" / "results"
                        / "context_ablation.csv")

# Arm styling. Colors reuse the manuscript's family palette so an arm reads as the family it
# belongs to: the two general-purpose LLMs and the base model that C2S-Scale was fine-tuned from
# share the LLM color, C2S-Scale takes the scLLM color. The cluster arm is the same model as A1, so
# it keeps that color and is distinguished by marker and dash instead — the figure should not imply
# it is a different model when granularity is the only thing that changed.
ABLATION_ARMS = {
    "A1_llm_percell":    ("LLM",   "o", "-",  "GPT-OSS-120B, per cell"),
    "A5_llm20_percell":  ("LLM",   "^", "-",  "GPT-OSS-20B, per cell"),
    "A2_base_percell":   ("LLM",   "s", ":",  "Gemma-2-2B (base), per cell"),
    "A3_scllm_percell":  ("scLLM", "v", "-",  "C2S-Scale-2B, per cell"),
    "A4_llm_percluster": ("LLM",   "D", "--", "GPT-OSS-120B, per cluster"),
    # Gemini keeps the LLM family colour but takes a star marker and a dash-dot line. The comment
    # above reserves "same colour, different marker" for a granularity change at fixed model; here
    # BOTH change, so it needs to be separable from A4 at a glance without inventing a colour
    # outside the family palette.
    "A8_gemini_percluster": ("LLM", "*", "-.", "Gemini-3.1-Pro, per cluster"),
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
               "M2": "genes\n+ types", "M4": "genes\n+ tissue\n+ types"}


def load_ablation(path=DEFAULT_ABLATION_CSV):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def figure_5(df, out_path, dpi=600, ablation=None):
    """Figure 5 — what prompt context buys, and whether per-cell prompting is worth it.

    (A) balanced accuracy by context mode, one line per arm, from the shuffled-gene floor (M6)
        up to full context (M4).
    (B) the ontology miss rate on the same modes, stacked under A on a shared x -- the mechanism
        behind A's curve.
    (C) the base-model control: Gemma-2-2B against C2S-Scale-2B, paired per dataset.
    (D) cost against accuracy at M2, per-cell versus per-cluster, for both GPT-OSS-120B and
        Gemini-3.1-Pro.

    ``df`` (the benchmark aggregate) supplies panel D's ported Gemini per-cell point, which was
    not re-run in this study.
    """
    ablation = load_ablation() if ablation is None else ablation
    if ablation.empty:
        raise SystemExit(
            "Figure 5 needs context_ablation/results/context_ablation.csv, which does not exist "
            "yet. Run the study first: see context_ablation/README.md.")

    cat_color = _ranked_category_colors()
    with plt.rc_context(_rc()):
        # Panel A is two stacked axes sharing one x. The upper is balanced accuracy, the lower the
        # ontology miss rate on the SAME modes -- the mechanism behind the accuracy curve, which
        # only reads as a mechanism when the two are vertically aligned. A shared x also removes a
        # duplicate axis and its labels, so the pair costs barely more height than one panel.
        fig = plt.figure(figsize=(12.4, 11.0))
        # Two nested grids rather than one 3-row grid. The accuracy/miss-rate pair must sit tight
        # together to read as one stacked panel, while the bottom row needs a wide gap so its panel
        # letters do not land on the shared x-axis label above them -- two spacings one hspace
        # cannot provide.
        outer = fig.add_gridspec(2, 1, hspace=0.30, height_ratios=[1.45, 0.85])
        top = outer[0].subgridspec(2, 1, hspace=0.13, height_ratios=[1.0, 0.60])
        bottom = outer[1].subgridspec(1, 2, wspace=0.26)
        ax_a = fig.add_subplot(top[0])
        ax_miss = fig.add_subplot(top[1], sharex=ax_a)
        ax_b = fig.add_subplot(bottom[0])
        ax_c = fig.add_subplot(bottom[1])
        ax_d = None

        base = ablation[ablation["variant"] == "expression_ranked"]

        # (A) context modes -------------------------------------------------
        # ONE LINE PER ARM, means over 4 datasets x 3 seeds. A per-dataset version was built and
        # rejected: sixteen traces in two colours could not be attributed to a dataset, and the
        # faint lines spanning 0.09-0.84 swamped the arm means that are the panel's actual subject.
        # The per-dataset detail is real and worth showing -- Zeisel scores HIGHER with no genes
        # (0.44) than with them (0.38), and C2S-Scale's context gain reverses there -- but it needs
        # a supplementary figure with datasets labelled, per MANUSCRIPT_SKELETON.md, not sixteen
        # unlabelled lines here.
        modes = [m for m in MODE_ORDER if m in set(ablation["mode"])]
        x = np.arange(len(modes), dtype=float)
        # A small horizontal offset per arm. At `genes + types` three arms sit within 0.01 of each
        # other and their markers previously overlapped into one blob; the offset separates them
        # without moving any value, since x is categorical. The SAME offset is reused in panel B so
        # one model occupies one column position in both axes and can be tracked downward.
        drawn_arms = [a for a in ABLATION_ARMS if not _arm_rows(ablation, a).empty]
        # The ported Gemini per-cell run occupies a jitter slot of its own so it cannot land on the
        # Gemini per-cluster marker it is meant to be compared against -- they sit at the same mode
        # and, at M2, within 0.07 of each other.
        gem = (df[df["model"] == GEMINI_PERCELL["model"]]
               if df is not None and not df.empty and "model" in df else pd.DataFrame())
        ported_ok = not gem.empty and COST_MODE in modes
        slots = drawn_arms + (["_gemini_percell"] if ported_ok else [])
        jitter = dict(zip(slots, np.linspace(-0.10, 0.10, max(len(slots), 1))))
        for arm, (family, marker, linestyle, label) in ABLATION_ARMS.items():
            sub = _arm_rows(ablation, arm)
            if sub.empty:
                continue
            means = [sub[sub["mode"] == m]["balanced_accuracy"].mean() for m in modes]
            ax_a.plot(x + jitter[arm], means, linestyle=linestyle, marker=marker,
                      color=cat_color[family], linewidth=1.4,
                      markersize=_marker_size_for(marker), label=label, zorder=3)
        # Gemini per cell: ONE point, not a line. The benchmark ran it only in the constrained
        # setting (candidate types supplied), which is this study's M2, so drawing a line would
        # imply four runs that do not exist. Hollow, like its twin in panel D, to mark the single
        # value on this panel that came from the benchmark rather than from this pipeline. It is
        # the per-cell ceiling the per-cluster arms are being judged against, so panel A is
        # incomplete without it.
        if ported_ok:
            ax_a.scatter(modes.index(COST_MODE) + jitter["_gemini_percell"],
                         gem["balanced_accuracy"].mean(),
                         marker=ABLATION_ARMS["A8_gemini_percluster"][1],
                         s=_scatter_area_for(ABLATION_ARMS["A8_gemini_percluster"][1]),
                         facecolor="none", edgecolor=cat_color["LLM"], linewidth=1.5,
                         zorder=4, label=GEMINI_PERCELL["label"])
        ax_a.set_xticks(x)
        ax_a.set_ylabel("Balanced Accuracy")
        ax_a.set_ylim(0, None)
        ax_a.grid(axis="y", linestyle=":", alpha=0.5)
        ax_a.legend(loc="best", frameon=False, fontsize=FS_LEGEND - 1)
        ax_a.tick_params(labelbottom=False)      # the stacked axis below carries the tick labels
        sns.despine(ax=ax_a)

        # Miss rate, stacked under A on the same x. This is what turns "context helps" into a
        # mechanism: the candidate list cuts the LLM's unmappable-answer rate ~4.6x (0.37 -> 0.08)
        # while C2S-Scale sits at 0.05-0.08 in EVERY mode -- it already speaks Cell Ontology, so the
        # list buys it nothing on this axis, which is much of why it gains +0.05 where the LLM gains
        # +0.26. Tissue alone makes the rate WORSE (0.37 -> 0.44): naming the organ invites more
        # specific free-text answers that no ontology term matches.
        for arm, (family, marker, linestyle, label) in ABLATION_ARMS.items():
            sub = _arm_rows(ablation, arm)
            if sub.empty or "non_cl_ratio" not in sub:
                continue
            vals = [sub[sub["mode"] == m]["non_cl_ratio"].mean() for m in modes]
            ax_miss.plot(x + jitter[arm], vals, linestyle=linestyle, marker=marker,
                         color=cat_color[family], linewidth=1.4,
                         markersize=_marker_size_for(marker) * 0.85, zorder=3)
        ax_miss.set_xticks(x)
        ax_miss.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])
        ax_miss.set_ylabel("unmapped\nfraction")
        ax_miss.set_ylim(0, None)
        ax_miss.grid(axis="y", linestyle=":", alpha=0.5)
        sns.despine(ax=ax_miss)

        # (B) base vs fine-tuned, paired per dataset -------------------------
        _paired_slope(ax_b, base, "A2_base_percell", "A3_scllm_percell",
                      ("Gemma-2-2B\n(base)", "C2S-Scale-2B\n(fine-tuned)"), cat_color)
        ax_b.set_ylabel("Balanced Accuracy")

        # (C) per-cell vs per-cluster ---------------------------------------
        # The FULL frame, not `base`. `base` keeps only variant == "expression_ranked", but the
        # cluster arm's variants are the clustering resolutions ("res0.5", "res1.0"), so filtering
        # first silently dropped the very arm this panel exists to compare against -- leaving three
        # per-cell points and no contrast.
        _cost_panel(ax_c, ablation, cat_color, bench=df)

        # The gene-list ablation moves to the supplement. Its result is a null (+0.02 mean, and
        # -0.08 on Zeisel), which is worth reporting but does not earn main-text space against a
        # mechanism panel -- and its paired-slope form could not distinguish A1's M1 from its M4.
        _panel_letters(fig, [(ax_a, "A"), (ax_miss, "B"), (ax_b, "C"), (ax_c, "D")])
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
                  "label": "Gemini-3.1-Pro, per cell"}


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
            _annotate(GEMINI_PERCELL["label"] + "  (ported)", GEMINI_PERCELL["cost_per_1k"],
                      1000.0, g["balanced_accuracy"].mean(), cat_color["LLM"],
                      ABLATION_ARMS["A8_gemini_percluster"][1], hollow=True)

    ax.set_xscale("log")
    ax.set_xlabel("model calls per 1,000 annotated cells")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0, None)
    # Headroom on both sides: the cluster arm sits at ~40 calls and the per-cell arms at 1,000, so
    # without margin their labels run off the axis rather than merely colliding.
    ax.set_xlim(left=ax.get_xlim()[0] * 0.45, right=ax.get_xlim()[1] * 2.2)
    ax.grid(linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    # Collision-aware placement, the same helper Figure 1 uses. Hand-placed offsets put three
    # labels on top of the axis and each other, because the points are far apart in x and close
    # in y -- exactly the case a fixed offset cannot handle.
    _place_point_labels(ax, labels, obstacles, FS_POINT)


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
    ax.set_ylabel("Balanced Accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)


def figure_context_per_dataset(df, out_path, dpi=600, ablation=None):
    """Supplementary for R5 — Figure 5A resolved per dataset, with the repeats shown.

    Figure 5A averages over the four datasets because twenty unlabelled traces in one axes cannot
    be attributed to a dataset. That average is the right main-text summary and the wrong place to
    look for consistency, because the spread it collapses is large and structured: the seed-to-seed
    SD within a dataset is 0.0068, while the SD across datasets is 0.182 -- 27x larger. Two results
    live entirely in that gap:

      * THE GENE-ORDER EFFECT IS ALMOST ENTIRELY ONE DATASET. Destroying the expression ranking
        (M6) costs the LLM 0.166 on human pancreas but only 0.003-0.017 on the other three; for
        C2S-Scale it costs 0.198 on pancreas against 0.017-0.078 elsewhere. The headline averages
        (+0.048 and +0.089) are therefore carried by pancreas, and anyone reading them as "the
        ranking is worth ~0.05-0.09 in general" is reading a per-dataset result as a global one.
        This panel is where that becomes visible, which is the main reason it is worth its page.
      * C2S-Scale's context gain REVERSES on Zeisel (-0.01) while helping on the other three
        (+0.01 to +0.12), so "the adapted model barely uses context" is a statement about three
        datasets and not the fourth.

    Error bars are the standard error over the three seeds, and the seeds themselves are drawn as
    small points. Both are shown rather than one or the other: the bar states the precision of the
    mean, the points show that n is three and that the three agree closely -- which an error bar
    alone would leave the reader to assume. Arms are jittered on x so their intervals do not
    overlap; the jitter is cosmetic and the x positions are categorical.
    """
    ablation = load_ablation() if ablation is None else ablation
    if ablation.empty:
        raise SystemExit("needs context_ablation/results/context_ablation.csv; run the study first")

    cat_color = _ranked_category_colors()
    datasets = sorted(ablation["dataset"].unique())
    modes = [m for m in MODE_ORDER if m in set(ablation["mode"])]
    x = np.arange(len(modes), dtype=float)

    with plt.rc_context(_rc()):
        ncols = 2
        nrows = int(np.ceil(len(datasets) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.6, 4.6 * nrows),
                                 sharex=True, sharey=False)
        axes = np.atleast_1d(axes).ravel()

        arms = [a for a in ABLATION_ARMS if not _arm_rows(ablation, a).empty]
        # The ported Gemini per-cell run is resolved per dataset here, which is the whole point of
        # this supplement: its per-cell advantage is not uniform (0.93 on pancreas against 0.39 on
        # HIHA), and the main figure's single averaged point cannot show that. It gets its own
        # jitter slot for the same reason as in Figure 5A.
        gem = (df[df["model"] == GEMINI_PERCELL["model"]]
               if df is not None and not df.empty and "model" in df else pd.DataFrame())
        ported_ok = not gem.empty and COST_MODE in modes
        offsets = np.linspace(-0.16, 0.16, max(len(arms) + (1 if ported_ok else 0), 1))
        arm_offsets, ported_dx = offsets[:len(arms)], (offsets[-1] if ported_ok else None)

        for ax, dataset in zip(axes, datasets):
            if ported_ok:
                gm = gem[gem["dataset"] == dataset]["balanced_accuracy"]
                if len(gm):
                    ax.scatter(x[modes.index(COST_MODE)] + ported_dx, gm.mean(),
                               marker=ABLATION_ARMS["A8_gemini_percluster"][1],
                               # 0.85 diameter scale matches the errorbar markers on this panel.
                               s=_scatter_area_for(ABLATION_ARMS["A8_gemini_percluster"][1], 0.85 ** 2),
                               facecolor="none", edgecolor=cat_color["LLM"], linewidth=1.4,
                               zorder=4,
                               label=(GEMINI_PERCELL["label"]
                                      if dataset == datasets[0] else None))
            for dx, arm in zip(arm_offsets, arms):
                family, marker, linestyle, label = ABLATION_ARMS[arm]
                sub = _arm_rows(ablation, arm)
                sub = sub[sub["dataset"] == dataset]
                if sub.empty:
                    continue
                means, errs, xs = [], [], []
                for i, mode in enumerate(modes):
                    vals = sub[sub["mode"] == mode]["balanced_accuracy"].dropna().to_numpy()
                    if not len(vals):
                        means.append(np.nan); errs.append(np.nan); xs.append(x[i] + dx)
                        continue
                    means.append(vals.mean())
                    # SE of the mean over repeats; 0 when a single repeat survived scoring.
                    errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
                    xs.append(x[i] + dx)
                    ax.scatter(np.full(len(vals), x[i] + dx), vals, s=7,
                               color=cat_color[family], alpha=0.55, zorder=2, linewidths=0)
                ax.errorbar(xs, means, yerr=errs, linestyle=linestyle, marker=marker,
                            color=cat_color[family], linewidth=1.3,
                            markersize=_marker_size_for(marker) * 0.85, capsize=2.5,
                            elinewidth=0.9, label=label, zorder=3)
            ax.set_title(dataset, fontsize=FS_TITLE)
            ax.set_xticks(x)
            ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])
            ax.set_ylim(0, None)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_ylabel("Balanced Accuracy")
            sns.despine(ax=ax)

        for ax in axes[len(datasets):]:
            ax.axis("off")
        axes[len(datasets) - 1].set_xlabel("context supplied in the prompt")
        if len(datasets) > 1:
            axes[len(datasets) - 2].set_xlabel("context supplied in the prompt")

        # ORDER MATTERS. This used to place the figure legend first and call a bare tight_layout()
        # afterwards; tight_layout does not know about figure-level legends, so it relaid the axes
        # over the full canvas and the legend printed on top of the bottom row. Reserve the band
        # first via `rect`, then drop the legend into the reserved space.
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 3 if len(labels) > 4 else min(len(labels), 4)
        legend_rows = int(np.ceil(len(labels) / ncol)) if labels else 0
        # Height of the reserved band as a fraction of the figure: one line per legend row plus a
        # little breathing room, divided by the figure height so it scales with the dataset count.
        band = (legend_rows * (FS_LEGEND + 6) + 10) / (fig.get_figheight() * 72) if legend_rows else 0
        fig.tight_layout(rect=[0, band, 1, 1])
        if labels:
            fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                       fontsize=FS_LEGEND, bbox_to_anchor=(0.5, band * 0.06))
        # After the layout is final, so the letters anchor on the axes' real positions.
        _panel_letters(fig, [(ax, chr(65 + i)) for i, ax in enumerate(axes[:len(datasets)])])
        return _save(fig, out_path, dpi)


def figure_context_supporting(df, out_path, dpi=600, ablation=None):
    """Supplementary for R5 — three checks the main figure asserts but cannot show.

    (A) Clustering resolution, for BOTH cluster arms. The design ran two resolutions because
        resolution sensitivity IS the granularity claim: if a cluster-level score depended on the
        clustering, the per-cluster result would be an artefact of one lucky choice. Going from
        ~30 to ~41 clusters moves GPT-OSS by 0.025 on average and Gemini by 0.048. Neither is
        large enough to overturn the per-cluster comparison, but the two models behave
        differently: for Gemini the finer clustering wins on ALL FOUR datasets (+0.013 to +0.095),
        while GPT-OSS gains on three and loses slightly on one (-0.004 to +0.062). So the honest
        statement is that these scores are mildly resolution-dependent, systematically so for
        Gemini, and that the numbers reported elsewhere -- at resolution 1.0 -- sit at the more
        favourable end of that range.

    (B) Why the base-model comparison needs a caveat. Figure 5C shows Gemma-2-2B near zero against
        C2S-Scale far above it, which invites the reading that fine-tuning bought biological
        competence. Much of it is that the base model never emits a parseable label: 28-46% of its
        answers run past eight words and 65-99% resolve to no ontology term. Scoring cannot separate
        "wrong cell type" from "not a cell type", so the format rates have to be shown beside it.

    (C) The gene-pool ablation, moved here from the main figure. Restricting the vocabulary to 5,000
        HVGs with ribosomal, mitochondrial and haemoglobin genes removed changes nothing at all
        (-0.001 on average). A null, but a load-bearing one: it says the cell sentence's dominance
        by housekeeping genes is not what limits these models.
    """
    ablation = load_ablation() if ablation is None else ablation
    if ablation.empty:
        raise SystemExit("needs context_ablation/results/context_ablation.csv; run the study first")

    cat_color = _ranked_category_colors()
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(13.0, 4.4))
        gs = fig.add_gridspec(1, 3, wspace=0.34)
        ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

        # (A) resolution sensitivity -----------------------------------------
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
                ax_a.plot([0, 1], pair, marker=marker, linestyle=linestyle,
                          markersize=_marker_size_for(marker) * 0.85, linewidth=1.3,
                          color=cat_color["LLM"], alpha=0.85, zorder=3,
                          label=arm_label if dataset == sorted(sub["dataset"].unique())[0] else None)
                # Collision-aware placement instead of a fixed +6pt offset. With two arms the
                # right-hand endpoints for HFRA and Zeisel land within a few thousandths of each
                # other and their fixed-offset labels printed on top of one another.
                labels_a.append((dataset, 1, pair[1], "0.25"))
                obstacles_a.extend([(0, pair[0], 9.0), (1, pair[1], 9.0)])
        if labels_a:
            ax_a.set_xticks([0, 1])
            # Cluster counts, not just the resolution parameter: "0.5 vs 1.0" is meaningless without
            # knowing it roughly doubles the number of clusters the eval cells fall into. Counted on
            # the cluster arms present, which share one clustering per (dataset, resolution).
            cl = ablation[ablation["arm"].str.endswith("percluster")]
            counts = [int(round(cl[cl["variant"] == v]["n_clusters_in_eval"].mean()))
                      for v in variants]
            ax_a.set_xticklabels([f"resolution 0.5\n(~{counts[0]} clusters)",
                                  f"resolution 1.0\n(~{counts[1]} clusters)"])
            ax_a.set_xlim(-0.3, 1.75)
            ax_a.legend(loc="lower left", frameon=False, fontsize=FS_LEGEND - 1)
            _place_point_labels(ax_a, labels_a, obstacles_a, FS_POINT)
        ax_a.set_ylabel("Balanced Accuracy")
        ax_a.set_ylim(0, None)
        ax_a.grid(axis="y", linestyle=":", alpha=0.5)
        sns.despine(ax=ax_a)

        # (B) base-model answer format ---------------------------------------
        modes = [m for m in MODE_ORDER if m in set(ablation["mode"])]
        x = np.arange(len(modes), dtype=float)
        a2 = ablation[(ablation["arm"] == "A2_base_percell")
                      & (ablation["variant"] == "expression_ranked")]
        for col, style, label in [("non_cl_ratio", "-", "no ontology term"),
                                  ("frac_over_8_words", "--", "answer over 8 words")]:
            if col not in a2:
                continue
            vals = [a2[a2["mode"] == m][col].mean() for m in modes]
            ax_b.plot(x, vals, linestyle=style, marker="s", markersize=5,
                      color=cat_color["LLM"], linewidth=1.4, label=label, zorder=3)
        # The score it achieves, on the same axis, to make the gap between "answered" and "answered
        # usefully" immediate rather than something the reader has to hold in mind across figures.
        acc = [a2[a2["mode"] == m]["balanced_accuracy"].mean() for m in modes]
        ax_b.plot(x, acc, linestyle=":", marker="o", markersize=4, color="0.35",
                  linewidth=1.2, label="balanced accuracy", zorder=3)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([MODE_LABELS.get(m, m) for m in modes], fontsize=FS_TICK - 1)
        ax_b.set_ylabel("fraction of predictions")
        ax_b.set_ylim(0, 1.05)
        ax_b.grid(axis="y", linestyle=":", alpha=0.5)
        # Nudged up from the default "center left" anchor (0, 0.5): the balanced-accuracy trace runs
        # along the bottom of this axes, and at dead centre the legend crowded it.
        ax_b.legend(loc="center left", bbox_to_anchor=(0, 0.60), frameon=False,
                    fontsize=FS_LEGEND - 1)
        ax_b.set_title("Gemma-2-2B (base)", fontsize=FS_TITLE)
        sns.despine(ax=ax_b)

        # (C) gene-pool ablation ---------------------------------------------
        _variant_slope(ax_c, ablation, cat_color)
        ax_c.set_ylabel("Balanced Accuracy")

        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")])
        return _save(fig, out_path, dpi)

FIGURES = {
    "1": figure_1,
    "5": figure_5,
    "1_balanced_accuracy": figure_1_balanced_accuracy,
    "2": figure_2,
    "3": figure_3,
    # Supplementary. s1 serves R1 (per-model ranking on the primary metric); the flow figures
    # serve R2's rarity paragraph, one per dataset.
    "s1_ranked_balanced_accuracy": figure_ranked_balanced_accuracy,
    "s6_date_vs_performance": figure_date_vs_performance,
    # R5's per-dataset companion: Figure 5A resolved by dataset, repeats shown.
    "s7_context_per_dataset": figure_context_per_dataset,
    # R5 support: resolution sensitivity, the base model's answer format, gene pool.
    "s8_context_supporting": figure_context_supporting,
    # One supplementary flow figure per dataset; every dataset has rare types, though HFRA has
    # only two at or below the 3% floor.
    # S2 onward, numbered in RARE_FLOW_ORDER so the filenames match the LaTeX numbering.
    **{f"s{i + 2}_{ds.lower()}": _rare_flow_builder(ds)
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
