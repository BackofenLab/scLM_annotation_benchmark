"""Figure 2 and Supplementary Figures S1 to S3: the benchmark results.

Figure 2 ranks every run (A), shows balanced accuracy per family and setting over all datasets and
per dataset (B to F), how much of each family's score rests on the curated mapping (G) and where
the answers sit in the Cell Ontology relative to the true type (H). S1 shows the curated mapping
per method, S2 the answer types per dataset and per subcategory, and S3 how alike the metrics rank
the methods. The inputs are the benchmark metrics and the confusion pairs; figures.py builds the
figures.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

from methods import (ANSWER_TYPES, CATEGORY_TIERS, CHANCE_MODEL, DATASET_ORDER, FAMILIES,
                     MAPPING_ROWS, METRIC_GROUPS, REGIME, REGIME_SHORT, _answer_shares,
                     _benchmark_pairs, _display, _group_means, _mapping_scores, _needs_mapping,
                     categorize_model, dataset_label, derive_mode, model_base, model_tier)
from style import (FAMILY_COLORS, FS_LABEL, FS_LEGEND, FS_POINT, RC, SETTING_KEY, _family_key,
                   _harmonize, _paired_rows, _panel_letters, _place_point_labels, _save,
                   _stacked_shares, _tier_marker, _tiered_offsets)

# The chance level, black and heavier than the grid so that it reads over the bars.
CHANCE_LINE = dict(linestyle="--", linewidth=1.2, color="black")
# One dot per dataset, light for a score without the curated mapping.
DOT_SIZE, DOT_COLOR, DOT_COLOR_NO_MAPPING = 6, "0.15", "0.6"
# Where _list_effect_panel tries a method's name, in points from its marker: above it, higher up
# over its own column, and only then beside it, so that a name stays over its column. Far enough
# out for the line that joins a name to its marker to be visible.
LABEL_CANDIDATES = [(0, 12, "center", "bottom"), (0, 20, "center", "bottom"),
                    (0, 28, "center", "bottom"), (-12, 0, "right", "center"),
                    (12, 0, "left", "center"), (-10, 8, "right", "bottom"),
                    (10, 8, "left", "bottom"), (0, -12, "center", "top"),
                    (-20, 0, "right", "center"), (20, 0, "left", "center")]


def _ranked_panel(ax, df):
    """Balanced accuracy of every run, best first; the dots are the datasets in DATASET_ORDER."""
    runs = df[df["model"].map(categorize_model) != "chance"]
    wide = runs.pivot_table(index="model", columns="dataset", values="balanced_accuracy")
    wide = wide[DATASET_ORDER].loc[wide.mean(axis=1).sort_values(ascending=False).index]
    x = np.arange(len(wide))
    for xi, model in enumerate(wide.index):
        unconstrained = derive_mode(model) == "blind"
        ax.bar(xi, wide.loc[model].mean(), width=0.8, color=FAMILY_COLORS[categorize_model(model)],
               hatch="///" if unconstrained else None, linewidth=0.5,
               edgecolor="black" if unconstrained else "white", zorder=3)
    for offset, dataset in zip(np.linspace(-0.27, 0.27, len(DATASET_ORDER)), DATASET_ORDER):
        ax.scatter(x + offset, wide[dataset], s=DOT_SIZE, color=DOT_COLOR, linewidth=0, zorder=4)
    chance = df.loc[df["model"] == CHANCE_MODEL, "balanced_accuracy"].mean()
    ax.axhline(chance, zorder=5, **CHANCE_LINE)
    ax.set_xticks(x)
    # The hatching gives the setting, so the names carry no U or C.
    ax.set_xticklabels([_display(model_base(m)) for m in wide.index], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(wide) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)


def _category_layout(family, members):
    """The points of one family: the members ordered by subcategory, their x offsets around the
    family's column, and per subcategory ``(slice of the members, marker, area)``."""
    ordered, segments = [], []
    for tier, _name, marker, size in CATEGORY_TIERS[family]:
        group = sorted(m for m in members if model_tier(m) == tier)
        if group:
            segments.append((slice(len(ordered), len(ordered) + len(group)), marker, size))
            ordered.extend(group)
    offsets = _tiered_offsets([sl.stop - sl.start for sl, _marker, _size in segments])
    return ordered, offsets, segments


def _list_effect_panel(ax, frame, title):
    """Balanced accuracy of every method in ``frame`` in both settings: per family a column of
    each setting, the two runs of a method joined by a line, the chance level dashed and the best
    constrained run of each family named."""
    per_model = frame["balanced_accuracy"].groupby(frame["model"]).mean()
    chance = float(per_model.loc[CHANCE_MODEL])
    scores = {}
    for model, value in per_model.items():
        family = categorize_model(model)
        if family != "chance":
            scores.setdefault((family, model_base(model)), {})[derive_mode(model)] = float(value)

    tick_pos, tick_lab, points = [], [], {}
    for i, (family, _name) in enumerate(FAMILIES):
        ordered, offsets, segments = _category_layout(
            family, sorted(base for (f, base) in scores if f == family))
        offsets = offsets * 0.6  # two columns share the family's slot
        column = {"blind": i - 0.25, "unblind": i + 0.25}
        for j, base in enumerate(ordered):
            run = scores[(family, base)]
            ax.plot([column["blind"] + offsets[j], column["unblind"] + offsets[j]],
                    [run["blind"], run["unblind"]], color="0.62", linewidth=0.7, zorder=2)
        for mode in ("blind", "unblind"):
            for sl, marker, size in segments:
                idx = range(sl.start, sl.stop)
                ax.scatter([column[mode] + offsets[j] for j in idx],
                           [scores[(family, ordered[j])][mode] for j in idx],
                           s=size, marker=marker, color=FAMILY_COLORS[family],
                           edgecolor="white", linewidth=0.5, zorder=4)
                points.update({(ordered[j], mode): (column[mode] + offsets[j],
                                                    scores[(family, ordered[j])][mode],
                                                    np.sqrt(size) / 2) for j in idx})
            tick_pos.append(column[mode])
            tick_lab.append(REGIME_SHORT[mode])

    ax.set_xticks(tick_pos)
    ax.set_xlim(-0.62, len(FAMILIES) - 0.38)
    below = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (_family, name) in enumerate(FAMILIES):
        ax.text(float(i), -0.13, name, transform=below, ha="center", va="top",
                fontsize=FS_POINT + 1)
    ax.set_ylim(0, 1)
    ax.axhline(chance, zorder=3, **CHANCE_LINE)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    _harmonize(ax, text_size=FS_POINT + 1)
    ax.set_xticklabels(tick_lab, fontweight="bold", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL)

    best = {}
    for (base, mode), (x, y, _r) in points.items():
        family = categorize_model(base)
        if mode == "unblind" and y > best.get(family, ("", 0, -1))[2]:
            best[family] = (base, x, y)

    # A name goes centred above its marker first, then beside it. The LLM's name is lifted and
    # centred over both LLM columns, clear of the scLMs' names, unless the best LLM is too close to
    # the top for that. When the space right above the scLM's marker is taken, its name goes
    # centred over both scLM columns, above the family's highest marker, before it tries the sides.
    def family_points(family):
        return [(px, py) for (b, _m), (px, py, _r) in points.items()
                if categorize_model(b) == family]

    llm_centre = np.mean([px for px, _py in family_points("LLM")])
    ax.figure.canvas.draw()  # the final axes position, to turn data offsets into points
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


def _mapping_panel(ax, pairs):
    """Balanced accuracy with the curated mapping (filled) and counting only the answers that
    needed none (open), per family and setting, averaged over the methods and the datasets."""
    means = _group_means(_mapping_scores(pairs).groupby("model").mean(), MAPPING_ROWS)
    _paired_rows(ax, means, MAPPING_ROWS, "without", "with")
    ax.set_xlim(0, 0.7)  # room for the key on the right
    ax.set_xlabel("Balanced accuracy (averaged over methods and datasets)")
    ax.legend(handles=[
        plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                   markeredgecolor="0.3", markersize=7, label="no curated mapping"),
        plt.Line2D([], [], linestyle="none", marker="o", color="0.3", markersize=7,
                   label="with curated mapping")],
        loc="upper right", frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0)


def _lineage_panel(ax, pairs):
    """Where the answers sit in the Cell Ontology relative to the true type, per family and
    setting, averaged over the methods and the datasets."""
    shares = _group_means(_answer_shares(pairs).groupby("model").mean(), MAPPING_ROWS)
    y = np.arange(len(MAPPING_ROWS))[::-1]
    _stacked_shares(ax, shares, y)
    ax.set_yticks(y)
    ax.set_yticklabels(shares.index)
    ax.set_ylim(-0.6, len(MAPPING_ROWS) - 0.4)
    ax.set_xlabel("Share of cells (averaged over cell types, methods and datasets)")


def figure_benchmark(df, out_path, dpi):
    """Figure 2: ranking, effect of the list per dataset, dependence on the curated mapping, and
    where the answers sit in the Cell Ontology."""
    pairs = _benchmark_pairs(df)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.2, 14.56))
        # A, its slanted run names, the key for A to F, a spacer, B to F, their family names and
        # G with H. The spacers make the white gaps above B to F and above G and H equal (0.3 in).
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

        # G, then H with its key close beside it, so that the key stays within the page width.
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


def figure_harmonization(df, out_path, dpi):
    """Supplementary Figure S1: per method and setting, the share of cells whose answer needed the
    curated mapping (A) and balanced accuracy without and with it (B), the rows ordered by the gap
    within each setting."""
    pairs = _benchmark_pairs(df)
    scores = _mapping_scores(pairs).rename(columns={"without": "ba_direct", "with": "ba_mapped"})
    scores["needs_mapping"] = _needs_mapping(pairs)
    d = df.merge(scores.reset_index(), on=["model", "dataset"], how="left")
    d["base"] = d["model"].map(model_base)
    d["cat"] = d["model"].map(categorize_model)
    d["regime"] = d["model"].map(derive_mode)
    d = d[d["cat"] != "chance"]

    def _agg(frame):
        out = (frame.groupby(["base", "cat"], as_index=False)
                    .agg(model=("model", "first"), needs=("needs_mapping", "mean"),
                         raw=("ba_direct", "mean"), mapped=("ba_mapped", "mean")))
        out["lift"] = out["mapped"] - out["raw"]
        return out.sort_values("lift", ascending=False).reset_index(drop=True)

    gap = 1.9  # blank rows between the blocks of the two settings, with room for the heading
    blocks, rows, pos = [], [], 0.0
    for mode in ("blind", "unblind"):
        pos += gap
        block = _agg(d[d["regime"] == mode])
        blocks.append((REGIME[mode], pos - 1.0))
        for _, r in block.iterrows():
            rows.append({**r.to_dict(), "ypos": pos, "regime": mode})
            pos += 1
    agg = pd.DataFrame(rows)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.2, 11.6))
        # The panels, then the key across the full width beneath them.
        outer = fig.add_gridspec(2, 1, height_ratios=[10.0, 0.9], hspace=0.25, left=0.15,
                                 right=0.985)
        # A carries one bounded share and is narrow; B is read off segment lengths and gets the
        # width. The gap keeps A's last tick label clear of B's first.
        top = outer[0].subgridspec(1, 2, width_ratios=[0.52, 1.0], wspace=0.08)
        ax_a = fig.add_subplot(top[0, 0])
        ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
        ax_key = fig.add_subplot(outer[1])
        ax_key.axis("off")

        y = agg["ypos"].to_numpy()
        colors = [FAMILY_COLORS[c] for c in agg["cat"]]

        def _cells(row, column):
            """The four per-dataset values behind one drawn row, matched on method and setting."""
            return d.loc[(d["base"] == row["base"]) & (d["regime"] == row["regime"]),
                         column].dropna()

        def _dots(ax, values, yy, color):
            ax.scatter(values, np.full(len(values), yy), s=DOT_SIZE, color=color, linewidths=0,
                       zorder=3)

        ax_a.barh(y, agg["needs"], height=0.62, color=colors, alpha=0.85, zorder=2)
        for (_, row), yy in zip(agg.iterrows(), y):
            _dots(ax_a, _cells(row, "needs_mapping"), yy, DOT_COLOR)
        ax_a.set_xlim(0, 1.04)
        ax_a.set_xticks(np.arange(0, 1.01, 0.2))
        ax_a.set_xlabel("Share of cells whose answer needs\nthe curated mapping")

        # The datasets as dots, light without the curated mapping and dark with it, the dark ones
        # drawn second so that they cover an equal light one.
        for (_, r), yy in zip(agg.iterrows(), y):
            col = FAMILY_COLORS[r["cat"]]
            _dots(ax_b, _cells(r, "ba_direct"), yy, DOT_COLOR_NO_MAPPING)
            _dots(ax_b, _cells(r, "ba_mapped"), yy, DOT_COLOR)
            ax_b.plot([r["raw"], r["mapped"]], [yy, yy], color=col, lw=1.8, alpha=0.95, zorder=2)
            ax_b.scatter(r["raw"], yy, s=34, facecolor="white", edgecolor=col, linewidths=1.3,
                         zorder=4)
            ax_b.scatter(r["mapped"], yy, s=46, marker=_tier_marker(r["model"])[0], color=col,
                         zorder=4)
        ax_b.set_xlabel("Balanced accuracy (averaged over datasets)")
        # Bounded by the per-dataset values, which reach beyond the means.
        ax_b.set_xlim(-0.02, float(np.nanmax([d["ba_direct"].max(), d["ba_mapped"].max()]))
                      + 0.04)

        ax_a.set_yticks(y)
        ax_a.set_yticklabels([_display(b) for b in agg["base"]])
        ax_a.set_ylim(agg["ypos"].max() + 1.0, 0.0)  # inverted: the first block on top
        plt.setp(ax_b.get_yticklabels(), visible=False)
        for ax in (ax_a, ax_b):
            ax.grid(axis="x", color="0.9", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        # The headings of the two blocks sit in the blank rows, just right of the left spine.
        trans = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
        for title, head_y in blocks:
            ax_a.text(0.02, head_y, title, transform=trans, ha="left", va="center",
                      fontsize=FS_LEGEND, fontweight="bold", color="0.25")
        for ax in (ax_a, ax_b):
            _harmonize(ax, text_size=FS_POINT)

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


def _subcategory_rows():
    """Rows ``(family, setting, label, subcategory)`` for every subcategory in both settings."""
    family_name = dict(FAMILIES)
    return [(family, mode, f"{family_name[family]}, {label}, {REGIME_SHORT[mode]}", tier)
            for family, _ in FAMILIES
            for tier, label, *_ in CATEGORY_TIERS[family]
            for mode in ("blind", "unblind")]


def figure_answer_types(df, out_path, dpi):
    """Supplementary Figure S2: where the answers sit in the Cell Ontology, per dataset (A) and
    per subcategory (B)."""
    per_run = _answer_shares(_benchmark_pairs(df))
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.2, 7.6))
        # The gap holds B's subcategory names; the key sits under both panels.
        grid = fig.add_gridspec(1, 2, left=0.11, right=0.985, wspace=0.78)
        ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

        # A: the rows of MAPPING_ROWS for each dataset, the dataset's name above its block.
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
                   bbox_to_anchor=(0.55, 0.0), ncol=len(ANSWER_TYPES), frameon=False,
                   fontsize=FS_LEGEND, handlelength=1.6, columnspacing=1.4)
        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B")])
        return _save(fig, out_path, dpi)


def figure_metric_agreement(df, out_path, dpi):
    """Supplementary Figure S3: Spearman correlation of the metrics over the runs within each
    dataset, averaged over the datasets, so that the difficulty of a dataset does not enter."""
    runs = df[df["model"].map(categorize_model) != "chance"]
    keys = [key for _, group in METRIC_GROUPS for key, _ in group]
    names = [name for _, group in METRIC_GROUPS for _, name in group]
    per_dataset = [g[keys].corr(method="spearman") for _, g in runs.groupby("dataset")]
    corr = (sum(per_dataset) / len(per_dataset)).to_numpy()
    with plt.rc_context(RC):
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
        for _, group in METRIC_GROUPS[:-1]:  # white rules between the groups
            edge += len(group)
            ax.axhline(edge, color="white", linewidth=2.5)
            ax.axvline(edge, color="white", linewidth=2.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        bar.set_label("Spearman correlation")
        return _save(fig, out_path, dpi)
