"""Typography, colours and drawing helpers shared by the figure modules.

Font sizes are in points at a figure's native size; the saved files are not scaled afterwards.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns

from methods import ANSWER_TYPES, CATEGORY_TIERS, FAMILIES, model_tier

# Helvetica, with Arial for a glyph Helvetica lacks.
FONT_FAMILY = ["Helvetica", "Arial"]
FS_PANEL_LETTER = 15
FS_TITLE = 10
FS_LABEL = 11
FS_TICK = 10
FS_GROUP = 12   # group headings inside an axes
FS_POINT = 7.5  # names beside points
FS_LEGEND = 9

RC = {
    "font.family": FONT_FAMILY,
    # TrueType rather than Type 3 fonts, so the text stays text in the PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Math text, the bold U and C of the setting keys, in Helvetica too.
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

# The first three colours of tab20 without its green pair.
FAMILY_COLORS = {"classical": "#1f77b4", "scLLM": "#aec7e8", "LLM": "#ff7f0e"}
TIER_STYLE = {tier: (marker, size) for tiers in CATEGORY_TIERS.values()
              for tier, _name, marker, size in tiers}
FAMILY_HEADER = {"classical": "Marker- and reference-based baselines",
                 "scLLM": "Single-cell language models (scLMs)",
                 "LLM": "General-purpose LLMs"}
SETTING_KEY = {"constrained": r"$\bf{C}$  constrained: expected cell types supplied",
               "unconstrained": r"$\bf{U}$  unconstrained: no expected cell types"}
# Legend text and colour of each of ANSWER_TYPES.
ANSWER_TYPE_STYLE = {
    "exact": ("exact type", "#1b7837"),
    "coarser": ("coarser, specific\n(e.g. T cell)", "#7fbf7b"),
    "broad": ("coarser, broad\n(e.g. neuron)", "#d9f0d3"),
    "finer": ("finer", "#c2a5cf"),
    "other": ("another cell type", "#bababa"),
    "none": ("no usable answer", "#ffffff"),
}


def _harmonize(ax, text_size=FS_POINT):
    """Set the shared sizes of axis labels, title and tick labels, and ``text_size`` for the
    text inside the axes."""
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    ax.title.set_size(FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    for t in ax.texts:
        t.set_fontsize(text_size)


def _blank(label=""):
    """A legend entry without a handle."""
    return plt.Line2D([], [], linestyle="none", marker="", label=label)


def _panel_letters(fig, items):
    """Bold letters above the top-left corner of each axes of ``items``, ``[(ax, letter)]``.

    The panels that start in one figure column share the x of the leftmost tight box among them, so
    that a panel without a y label is not indented against one with. Draws the canvas first.
    """
    pad_x, pad_y = 0.012, 0.008  # figure fraction
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


def _save(fig, out_path, dpi):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _tier_marker(model):
    """Marker and area of a run's subcategory."""
    return TIER_STYLE[model_tier(model)]


def _tiered_offsets(group_sizes, spacing=0.030, gap=0.080, max_half_width=0.24):
    """x offsets of points laid out in contiguous groups with a gap between groups, centred on 0
    and shrunk to at most ``max_half_width`` on either side."""
    pos, x = [], 0.0
    for i, k in enumerate(group_sizes):
        if i:
            x += gap
        for _ in range(k):
            pos.append(x)
            x += spacing
    arr = np.asarray(pos, dtype=float)
    arr -= (arr.min() + arr.max()) / 2.0
    half = float(arr.max())
    if half > max_half_width:
        arr *= max_half_width / half
    return arr


def _place_point_labels(ax, labels, obstacles, fontsize, candidates=None, leader_min=12,
                        text_bbox=None):
    """Name points without a name landing on a marker, on another name or outside the axes.

    ``labels`` is ``[(text, x, y, colour[, candidates])]`` and ``obstacles`` is ``[(x, y, radius in
    points)]``, both in data coordinates. A name takes the first offset of ``candidates``,
    ``[(dx, dy, ha, va)]`` in points, that overlaps nothing, or else the one that overlaps least,
    and a thin line joins it to its point beyond ``leader_min`` points. ``text_bbox`` boxes every
    name. Draws the canvas first.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    obstacle_boxes = []
    for x_data, y_data, r_pts in obstacles:
        xd, yd = ax.transData.transform((x_data, y_data))
        r = r_pts * fig.dpi / 72.0
        obstacle_boxes.append(mtransforms.Bbox.from_bounds(xd - r, yd - r, 2 * r, 2 * r))
    # Nearest offsets first, beside the point before above or below it.
    if candidates is None:
        candidates = [(dx, dy, "left" if dx >= 0 else "right",
                       "bottom" if dy > 0 else ("top" if dy < 0 else "center"))
                      for dx, dy in [(9, 0), (-9, 0), (9, 8), (-9, 8), (9, -8), (-9, -8),
                                     (0, 11), (0, -11), (18, 0), (-18, 0), (18, 14), (-18, 14),
                                     (18, -14), (-18, -14), (0, 20), (0, -20), (30, 0),
                                     (-30, 0)]]

    axes_box = ax.get_window_extent(renderer=renderer)

    def _cost(box, boxes):
        total = 0.0
        for other in boxes:
            inter = mtransforms.Bbox.intersection(box, other)
            if inter is not None and inter.width > 0 and inter.height > 0:
                total += inter.width * inter.height
        # The area outside the axes counts as overlap.
        inside = mtransforms.Bbox.intersection(box, axes_box)
        area = box.width * box.height
        total += area - (inside.width * inside.height if inside is not None else 0.0)
        return total

    placed = []
    for label in sorted(labels, key=lambda t: -t[2]):
        text, px, py, color = label[:4]
        best, best_cost, best_off = None, None, None
        for dx, dy, ha, va in (label[4] if len(label) > 4 else candidates):
            ann = ax.annotate(text, xy=(px, py), xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va=va, fontsize=fontsize, color=color, zorder=6,
                              bbox=text_bbox)
            box = ann.get_window_extent(renderer=renderer)
            cost = _cost(box, obstacle_boxes + placed)
            if cost == 0:
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
        placed.append(best.get_window_extent(renderer=renderer))
        if abs(best_off[0]) > leader_min or abs(best_off[1]) > leader_min:
            ax.annotate("", xy=(px, py), xytext=best_off, textcoords="offset points",
                        xycoords=best.xycoords,
                        arrowprops=dict(arrowstyle="-", linewidth=0.4, color="0.7",
                                        shrinkA=0, shrinkB=2), zorder=3)


def _family_key(ax, encodings, drawn=None, extra=(), spacing=0.8):
    """A key of one column of ``encodings``, then one column per family with its subcategories
    under the family's name in bold, then one column per ``extra`` ``(heading, [lines])``.
    ``drawn`` keeps only those ``(family, subcategory)`` pairs. Columns are padded to equal length,
    so the legend fills them column by column."""
    columns = [encodings] + [
        [_blank(FAMILY_HEADER[family])] + [
            plt.Line2D([], [], linestyle="none", marker=marker, color=FAMILY_COLORS[family],
                       markersize=11 if marker == "*" else 7, label=label)
            for tier, label, marker, _size in CATEGORY_TIERS[family]
            if drawn is None or (family, tier) in drawn]
        for family, _ in FAMILIES]
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


def _stacked_shares(ax, shares, y):
    """One stacked bar of the shares of ANSWER_TYPES per row of ``shares``, at heights ``y``."""
    left = np.zeros(len(shares))
    for key in ANSWER_TYPES:
        label, color = ANSWER_TYPE_STYLE[key]
        ax.barh(y, shares[key].to_numpy(), left=left, height=0.7, color=color,
                edgecolor="0.55", linewidth=0.4, label=label)
        left += shares[key].to_numpy()
    ax.set_xlim(0, 1)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)


def _row_marker(row):
    """Marker and area of a group row: a circle for a family in one setting, the subcategory's
    marker for a row ``(family, setting, label, subcategory)``."""
    return TIER_STYLE[row[3]] if len(row) > 3 else ("o", 46)


def _paired_rows(ax, means, rows, open_col, filled_col):
    """One row per entry of ``rows``, the first on top: an open marker at ``open_col`` and a filled
    one at ``filled_col`` of ``means``, joined by a line in the family's colour."""
    y = np.arange(len(rows))[::-1]
    for yy, row, (_label, mean) in zip(y, rows, means.iterrows()):
        color = FAMILY_COLORS[row[0]]
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
