"""Figure 4 and Supplementary Figures S9 to S14: rare cell types.

A cell type is rare below RARE_PREVALENCE_MAX of the full dataset. The test sets hold at least 20
cells of every type, so a type's share of a test set says little about its rarity. Figure 4 shows
balanced accuracy over the common and the rare types (A), what became of each rare type (B) and,
for human pancreas, where the cells of the rarest types go (C). S9 shows every rare type per
method group, S10 the answer types of the common and the rare types, and S11 to S14 where the
cells of the rarest types go in every dataset. The inputs are the rare-type table
(paths.RARE_TYPE_OUTCOMES, built by rare_type_table.py), the per-type scores, the confusion pairs
and the benchmark metrics; figures.py builds the figures.
"""
from __future__ import annotations

import textwrap

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

import paths
from methods import (DATASET_ORDER, FAMILIES, MAPPING_ROWS, REGIME_SHORT, _answer_shares,
                     _benchmark_pairs, _display, _group_means, categorize_model, dataset_label,
                     derive_mode, group_name, load_confusion_pairs, model_base, model_tier,
                     run_label, spread)
from style import (FS_LEGEND, FS_POINT, FS_TITLE, RC, _harmonize, _paired_rows, _panel_letters,
                   _save, _stacked_shares)

RARE_PREVALENCE_MAX = 0.01
# The flows draw the rarest types below it, at most this many per dataset: HFRA has two, human
# pancreas three, HIHA and Zeisel more than five.
RARE_FLOW_MAX_TYPES = 5
# Answers drawn per true type in a flow; the others are pooled as "other".
FLOW_MAX_TARGETS = 6
# An answer's name in a flow is cut to this many characters, so that it ends before the next panel.
FLOW_LABEL_MAX = 48
# The dataset of the flows in figure_rare_types.
RARE_FLOW_EXAMPLE = "human_pancreas"
# The closed-weight LLMs, a group within the LLM family, as recorded in CATEGORY_TIER_MEMBERS.
CLOSED_WEIGHT_TIER = "closed_reasoning"
# The method groups of figure_rare_full, named and ordered as RARITY_ROWS.
BLOCK_LABEL = {"unconstrained baselines": "baselines, U", "constrained baselines": "baselines, C",
               "unconstrained scLLM": "scLMs, U", "constrained scLLM": "scLMs, C",
               "unconstrained LLM": "LLMs, U", "constrained LLM": "LLMs, C",
               "unconstrained closed-weight LLM": "closed-weight LLMs, U",
               "constrained closed-weight LLM": "closed-weight LLMs, C"}
BLOCK_ORDER = list(BLOCK_LABEL)
# Every family in both settings, then the closed-weight LLMs.
RARITY_ROWS = MAPPING_ROWS + [("LLM", mode, f"closed-weight LLMs, {REGIME_SHORT[mode]}",
                               CLOSED_WEIGHT_TIER) for mode in ("unconstrained", "constrained")]
# The outcomes of a rare cell type, best first: (key, legend text, grey). A type that a method never
# named is split by whether the method has a label for it at all (``reachable``, known for the
# constrained baselines); that part is hatched.
RARE_OUTCOMES = [
    ("most", "≥ 50% of cells correct", "0.94"),
    ("some", "< 50% of cells correct", "0.78"),
    ("wrong", "named, never correct", "0.56"),
    ("never", "never named", "0.30"),
    ("unreachable", "never named, no label", "0.30"),
]


def load_rare_outcomes():
    """Recall, whether the method named the type, and whether it has a label for it, per run and
    true cell type, with the type's share of the full dataset."""
    return pd.read_csv(paths.RARE_TYPE_OUTCOMES)


def load_per_class_scores():
    """Recall and Cell Ontology Lin similarity per run, dataset and true cell type."""
    d = pd.read_csv(paths.PER_CLASS_SCORES)
    d = d[(d["subset_size"] == 1000) & (d["cell_type"] != "__OVERALL__")
          & (d["model"].map(categorize_model) != "chance")].drop(columns="subset_size")
    # Lowercased as the key the rare-type table uses, with the Cell Ontology spelling for display.
    d["cell_type_name"] = d["cell_type"].astype(str).str.strip()
    d["cell_type"] = d["cell_type_name"].str.lower()
    return d.assign(block=d["model"].map(group_name))


def _with_closed_weight(d):
    """``d`` plus a copy of the closed-weight LLMs' rows under their own ``block``. The run is
    ``model`` in the per-type scores and ``method`` in the rare-type table."""
    col = "model" if "model" in d.columns else "method"
    closed = d[(d[col].map(categorize_model) == "LLM")
               & (d[col].map(model_tier) == CLOSED_WEIGHT_TIER)]
    return pd.concat([d, closed.assign(block=closed["block"].str.replace(
        " LLM", " closed-weight LLM", regex=False))], ignore_index=True)


def _prevalence(rare):
    """``{(dataset, cell type): share of the full dataset}``, cell types lowercased."""
    return (rare.drop_duplicates(["dataset", "k"])
                .set_index(["dataset", "k"])["orig_prevalence"].to_dict())


def _rarity_rates(rare):
    """Per run (index ``<method>_<mode>``): ``common`` and ``rare``, the mean recall of the common
    and of the rare cell types, averaged over a dataset's types and then over the datasets, since
    the rarest types sit almost all in two datasets; and the shares of the RARE_OUTCOMES among all
    rare types of the four datasets."""
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


def _outcome_bars(ax, shares, y):
    """One stacked bar of RARE_OUTCOMES per row of ``shares``, at heights ``y``."""
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


def _rare_dotplot(ax, rows, rare, per_class, blocks):
    """A dot per cell type of ``rows`` and method group of ``blocks``, its area the recall and its
    colour the Lin similarity, a ring with a cross for a type no method of the group named.

    ``rows`` is ``[("__header__", label, None) | (dataset, cell type, share)]`` in drawing order.
    Returns the colour norm, the area at recall 0 and the area span, for the key.
    """
    s_min, s_span = 22.0, 170.0
    never = {}
    for r in rare.itertuples():
        k = (r.dataset, r.k, r.block)
        never[k] = never.get(k, False) or bool(r.emitted)
    grouped = per_class.groupby(["dataset", "cell_type", "block"])[["recall", "lin"]].mean()
    names = (per_class.drop_duplicates(["dataset", "cell_type"])
             .set_index(["dataset", "cell_type"])["cell_type_name"])
    cmap, norm = plt.get_cmap("viridis"), mcolors.Normalize(0, 1)

    ypos, y, y_first = {}, 0.0, None
    for kind, name, p in rows:
        if kind == "__header__":
            y -= 0.78
            if y_first is None:
                y_first = y
            ax.text(-0.012, y, name, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=FS_POINT, fontweight="bold", clip_on=False)
            y -= 0.80
            continue
        ypos[(kind, name)] = y
        pv = f"{p * 100:.3f}%" if p < 0.1 else f"{p * 100:.1f}%"
        # The Cell Ontology name with the share, wrapped at 52 characters.
        label = textwrap.fill(f"{names[(kind, name)]} ({pv})", width=52)
        ax.text(-0.012, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=FS_POINT, linespacing=1.2, clip_on=False)
        y -= 1.0
    for (ds, t), yy in ypos.items():
        for xi, block in enumerate(blocks):
            rec, lin = grouped.loc[(ds, t, block), ["recall", "lin"]]
            size = s_min + s_span * float(rec)
            if not never[(ds, t, block)]:
                ax.scatter(xi, yy, s=size, facecolors="none", edgecolors=cmap(norm(lin)),
                           linewidths=1.4, zorder=3)
                ax.scatter(xi, yy, marker="x", s=15, c="0.35", linewidths=0.85, zorder=4)
            else:
                ax.scatter(xi, yy, s=size, color=cmap(norm(lin)), edgecolors="0.25",
                           linewidths=0.4, zorder=3)
    for xi in range(len(blocks)):
        ax.axvline(xi, color="0.93", lw=0.6, zorder=0)
    ax.set_xlim(-0.6, len(blocks) - 0.4)
    ax.set_ylim(y + 0.5, y_first + 0.45)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([BLOCK_LABEL[b] for b in blocks], fontsize=FS_POINT, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return norm, s_min, s_span


def figure_rare_full(df, out_path, dpi):
    """Supplementary Figure S9: every cell type below RARE_PREVALENCE_MAX per method group, sized
    to print at the full text width with its caption within the text height."""
    rare = load_rare_outcomes()
    per_class = load_per_class_scores()
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

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(11.5, 0.26 * units + 1.1))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_key = fig.add_subplot(gs[0, 1])
        ax_key.axis("off")
        norm, s_min, s_span = _rare_dotplot(ax, rows, _with_closed_weight(rare),
                                            _with_closed_weight(per_class), BLOCK_ORDER)
        # Inside the key axes, aligned with the column of markers.
        cax = ax_key.inset_axes([0.185, 0.30, 0.065, 0.32])
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("viridis")), cax=cax)
        cb.set_label("Lin similarity to the true label", fontsize=FS_POINT)
        cb.ax.tick_params(labelsize=FS_LEGEND)
        handles = [plt.Line2D([0], [0], linestyle="none", marker="o", mfc="0.62", mec="0.25",
                              ms=np.sqrt(s_min + s_span * v) / 1.7, label=f"recall {v:g}")
                   for v in (0.0, 0.5, 1.0)]
        # The marker of a type never named: a ring with a cross, one handle over the other.
        handles.append((plt.Line2D([0], [0], linestyle="none", marker="o", mfc="none",
                                   mec="0.4", ms=7.5, mew=1.4),
                        plt.Line2D([0], [0], linestyle="none", marker="x", color="0.35",
                                   ms=4.2, mew=0.9)))
        labels = [h.get_label() for h in handles[:3]] + ["never named"]
        ax_key.legend(handles=handles, labels=labels, loc="upper left", frameon=False,
                      fontsize=FS_LEGEND, labelspacing=1.0, handletextpad=0.6,
                      borderaxespad=0.0, alignment="left",
                      handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)})
        return _save(fig, out_path, dpi)


def figure_rare_answer_types(df, out_path, dpi):
    """Supplementary Figure S10: the answer types of the cells of the common (A) and of the rare (B)
    cell types, in RARITY_ROWS, averaged within each dataset first."""
    rare = load_rare_outcomes()
    pairs = _benchmark_pairs(df)
    prevalence = _prevalence(rare)
    is_rare = np.array([prevalence[(d, t.strip().lower())] < RARE_PREVALENCE_MAX
                        for d, t in zip(pairs["dataset"], pairs["true_label"])])

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(11.5, 3.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.36], wspace=0.14)
        axes = [fig.add_subplot(gs[0, 0])]
        axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
        ax_key = fig.add_subplot(gs[0, 2])
        ax_key.axis("off")
        y = np.arange(len(RARITY_ROWS))[::-1]
        for ax, keep, title in ((axes[0], ~is_rare, "Common cell types (≥ 1%)"),
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
        _panel_letters(fig, [(axes[0], "A"), (axes[1], "B")])
        return _save(fig, out_path, dpi)


def select_flow_models(df):
    """The runs of the flow figures: per family the method with the highest mean balanced accuracy
    of its constrained run. Two rows, the constrained runs and beneath them the unconstrained runs
    of the same methods."""
    picks = []
    for family, _name in FAMILIES:
        runs = df[(df["model"].map(categorize_model) == family)
                  & (df["model"].map(derive_mode) == "constrained")]
        picks.append(runs.groupby(runs["model"].map(model_base))["balanced_accuracy"].mean()
                     .idxmax())
    return [[f"{base}_{mode}" for base in picks] for mode in ("constrained", "unconstrained")]


def _run_label(model):
    """Name of a run's method with its setting, U or C."""
    return run_label(model, _display(model_base(model)))


def _flow_types(pairs, prevalence, dataset, model):
    """The rarest types of ``dataset`` below RARE_PREVALENCE_MAX, at most RARE_FLOW_MAX_TYPES, a
    colour for each, and their cells in the run ``model`` (the same in every run)."""
    share = pd.Series({t: prevalence[(dataset, t.strip().lower())]
                       for t in pairs["true_label"].unique()})
    rare = list(share[share < RARE_PREVALENCE_MAX].nsmallest(RARE_FLOW_MAX_TYPES).index)
    palette = sns.color_palette("tab10", n_colors=max(len(rare), 3)).as_hex()
    colors = {t: palette[i % len(palette)] for i, t in enumerate(rare)}
    counts = (pairs[(pairs["model"] == model) & (pairs["true_label"].isin(rare))]
              .groupby("true_label")["n_cells"].sum())
    return rare, colors, counts


def _flows(pairs, model, rare):
    """The ribbons ``(true type, answer, cells)`` of one run for the ``rare`` types, and the
    answers ordered by the mean position of their true types, which limits crossings."""
    m = pairs[(pairs["model"] == model) & (pairs["true_label"].isin(rare))]
    flows = []
    for true_label, grp in m.groupby("true_label"):
        grp = grp.sort_values("n_cells", ascending=False)
        keep = grp.head(FLOW_MAX_TARGETS)
        rest = grp.tail(max(len(grp) - FLOW_MAX_TARGETS, 0))
        flows += [(true_label, t.predicted_label, float(t.n_cells)) for t in keep.itertuples()]
        if len(rest):
            flows.append((true_label, "other", float(rest["n_cells"].sum())))
    weight = {}
    for src, tgt, val in flows:
        a, b = weight.get(tgt, (0.0, 0.0))
        weight[tgt] = (a + rare.index(src) * val, b + val)
    return flows, sorted(weight, key=lambda t: (t == "other", weight[t][0] / weight[t][1]))


def _sankey(ax, flows, left_order, right_order, colors, show_left_labels=True, label_pad=0.05,
            right_margin=1.9):
    """A two-column alluvial plot of ``flows``, ``[(true type, answer, cells)]``.

    Node heights are proportional to cells and ribbons are cubic Bezier curves. A ribbon to the
    true type itself is opaque with a dark edge; an answer node is dark when the answer is one of
    the true types on the left and light otherwise. The answers' names start at ``1 + label_pad``
    in axes width and are not clipped, so ``right_margin``, the right x limit, leaves them their
    room.
    """
    label_size = 7
    total = sum(v for _s, _t, v in flows)
    gap = 0.02 * total

    def stack(order, weights):
        pos, y = {}, 0.0
        for key in order:
            pos[key] = (y, y + weights[key])
            y += weights[key] + gap
        span = y - gap
        return {k: (a / span, b / span) for k, (a, b) in pos.items()}

    lw = {k: sum(v for s, _t, v in flows if s == k) for k in left_order}
    rw = {k: sum(v for _s, t, v in flows if t == k) for k in right_order}
    lpos, rpos = stack(left_order, lw), stack(right_order, rw)
    lcur = {k: a for k, (a, _b) in lpos.items()}
    rcur = {k: a for k, (a, _b) in rpos.items()}
    scale = {k: (lpos[k][1] - lpos[k][0]) / lw[k] for k in left_order}
    rscale = {k: (rpos[k][1] - rpos[k][0]) / rw[k] for k in right_order}

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
            codes.append(mpath.Path.LINETO if verts else mpath.Path.MOVETO)
            verts.append((xa, ya))
            verts += [((xa + xb) / 2, ya), ((xa + xb) / 2, yb), (xb, yb)]
            codes += [mpath.Path.CURVE4] * 3
        verts.append((x0, y0))
        codes.append(mpath.Path.CLOSEPOLY)
        ax.add_patch(mpatches.PathPatch(
            mpath.Path(verts, codes), facecolor=colors[src],
            alpha=0.85 if correct else 0.38, linewidth=0.6 if correct else 0.0,
            edgecolor="0.15" if correct else "none", zorder=3 if correct else 2))

    for key in left_order:
        a, b = lpos[key]
        ax.add_patch(mpatches.Rectangle((-0.035, a), 0.03, b - a, color=colors[key], zorder=4))
        if show_left_labels:
            ax.text(-label_pad, (a + b) / 2, key, ha="right", va="center", fontsize=label_size)
    # The names of small neighbouring answers would overlap: each goes as close to its node as one
    # line of text between names allows, joined to it by a line where it had to move.
    points_high = ax.get_position().height * ax.figure.get_figheight() * 72
    centres = [(a + b) / 2 for a, b in (rpos[k] for k in right_order)]
    label_y = spread(centres, 1.15 * label_size * 1.04 / points_high)
    for key, centre, y_text in zip(right_order, centres, label_y):
        a, b = rpos[key]
        hit = key in left_order
        ax.add_patch(mpatches.Rectangle((1.005, a), 0.03, b - a, zorder=4,
                                        facecolor="0.35" if hit else "0.78",
                                        edgecolor="0.35" if hit else "0.62", linewidth=0.4))
        text = key if len(key) <= FLOW_LABEL_MAX else key[:FLOW_LABEL_MAX - 1].rstrip() + "…"
        ax.text(1.0 + label_pad, y_text, text, ha="left", va="center", fontsize=label_size,
                fontweight="bold" if hit else "normal", color="0.1" if hit else "0.35")
        if abs(y_text - centre) > 1e-6:
            ax.plot([1.037, 1.0 + label_pad - 0.01], [centre, y_text], color="0.6",
                    linewidth=0.4, zorder=4)
    # Without the true types' names the left margin is not needed.
    ax.set_xlim(-0.9 if show_left_labels else -0.10, right_margin)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")


def _flow_key(ax, rare, colors, counts, label_size, notes, rows=None, ncol=None):
    """The ribbon encoding and the cells per rare type beneath a grid of flows.

    ``notes`` adds two lines on the ribbon width and on the answers drawn. With ``rows`` the
    entries are packed into that many centred rows; otherwise ``ncol`` columns span the axes.
    """
    handles = [
        mpatches.Patch(facecolor="0.55", edgecolor="0.15", linewidth=0.6, alpha=0.85,
                       label="prediction matches the true type"),
        mpatches.Patch(facecolor="0.55", edgecolor="none", alpha=0.38,
                       label="prediction differs"),
    ]
    if notes:
        handles += [
            plt.Line2D([], [], linestyle="none", marker="", label="ribbon width: number of cells"),
            plt.Line2D([], [], linestyle="none", marker="",
                       label=f"top {FLOW_MAX_TARGETS} predictions per type;\n"
                             f"remainder pooled as \"other\""),
        ]
    for t in rare:
        handles.append(mpatches.Patch(facecolor=colors[t], edgecolor="none",
                                      label=f"{t} (n = {int(counts[t])})"))
    if rows:
        ncol = -(-len(handles) // rows)
    kw = dict(handles=handles, frameon=False, fontsize=label_size, ncol=ncol,
              labelspacing=0.6, handlelength=1.6, handletextpad=0.7, alignment="left")
    if rows:
        ax.legend(loc="upper center", borderaxespad=0.0, columnspacing=2.5, **kw)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), mode="expand",
                  borderaxespad=0.0, columnspacing=1.2, **kw)


def _align_left(fig, gridspec, ax, reference):
    """Shift ``gridspec`` sideways so that ``ax``, labels included, starts where ``reference``
    does, so that two stacked rows of panels share one left edge."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    shift = (reference.get_tightbbox(renderer).transformed(inv).x0
             - ax.get_tightbbox(renderer).transformed(inv).x0)
    gridspec.update(left=gridspec.left + shift)


def figure_rare_types(df, out_path, dpi):
    """Figure 4: balanced accuracy over the common and over the rare cell types (A) and the
    outcomes of the rare types (B) in RARITY_ROWS, and where the cells of the rare types of human
    pancreas go in the runs of select_flow_models (C)."""
    rare = load_rare_outcomes()
    conf = load_confusion_pairs()
    flow_models = select_flow_models(df)
    rates = _group_means(_rarity_rates(rare), RARITY_ROWS)

    with plt.rc_context(RC):
        # At the output dpi, since _align_left measures text and the text widths depend on it.
        fig = plt.figure(figsize=(13.4, 9.4), dpi=dpi)
        # A and B share their rows, so B carries no row names. A's left edge is aligned with C's
        # at the end, so ``left`` here is only a start. The flows get the full width; ``wspace``
        # leaves room for the answers' names, which _sankey draws outside its axes.
        top = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.30], wspace=0.10,
                               left=0.16, right=0.985, top=0.985, bottom=0.64)
        ax_a = fig.add_subplot(top[0, 0])
        ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
        ax_bkey = fig.add_subplot(top[0, 2])
        ax_bkey.axis("off")
        bottom = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.11],
                                  width_ratios=[1.16, 1.0, 1.0, 1.0], wspace=0.30, hspace=0.12,
                                  left=0.05, right=0.985, top=0.53, bottom=0.0)
        ax_c = [fig.add_subplot(bottom[0, i]) for i in range(4)]
        ax_ckey = fig.add_subplot(bottom[1, :])
        ax_ckey.axis("off")

        _paired_rows(ax_a, rates, RARITY_ROWS, "common", "rare")
        ax_a.set_xlim(0, 0.8)
        ax_a.set_xticks(np.arange(0, 0.81, 0.2))
        ax_a.set_xlabel("Balanced accuracy")
        ax_a.legend(handles=[
            plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                       markeredgecolor="0.3", markersize=7, label="common (≥ 1%)"),
            plt.Line2D([], [], linestyle="none", marker="o", color="0.3", markersize=7,
                       label="rare (< 1%)")],
            loc="upper right", frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0)

        _outcome_bars(ax_b, rates, np.arange(len(RARITY_ROWS))[::-1])
        ax_b.tick_params(labelleft=False)
        ax_b.set_xlabel("Share of rare cell types")
        ax_bkey.legend(handles=ax_b.get_legend_handles_labels()[0], loc="center left",
                       frameon=False, fontsize=FS_LEGEND, borderaxespad=0, borderpad=0,
                       handlelength=1.6, labelspacing=0.9)
        for ax in (ax_a, ax_b):
            _harmonize(ax, text_size=FS_POINT)

        # The constrained runs of the three methods, with the unconstrained run of the LLM before
        # its constrained one.
        c_models = [flow_models[0][0], flow_models[0][1], flow_models[1][2], flow_models[0][2]]
        pairs = conf[conf["dataset"] == RARE_FLOW_EXAMPLE]
        rare_types, colors, counts = _flow_types(pairs, _prevalence(rare), RARE_FLOW_EXAMPLE,
                                                 c_models[0])
        for i, (ax, model) in enumerate(zip(ax_c, c_models)):
            flows, right = _flows(pairs, model, rare_types)
            # A wider right margin than in the supplementary flows, for the longest answer.
            _sankey(ax, flows, rare_types, right, colors, show_left_labels=(i == 0),
                    label_pad=0.11, right_margin=2.9)
            ax.set_title(_run_label(model), fontsize=FS_POINT)
        _flow_key(ax_ckey, rare_types, colors, counts, label_size=FS_LEGEND, notes=False, ncol=5)
        _align_left(fig, top, ax_a, ax_c[0])
        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c[0], "C")])
        return _save(fig, out_path, dpi)


def figure_rare_flows(df, out_path, dpi, dataset):
    """Supplementary Figures S11 to S14: where the cells of the rarest types of ``dataset`` go, one
    flow per run of select_flow_models, the constrained runs on top, with a key of the encoding and
    of the cells per type."""
    models = select_flow_models(df)
    conf = load_confusion_pairs()
    pairs = conf[conf["dataset"] == dataset]
    rare, colors, counts = _flow_types(pairs, _prevalence(load_rare_outcomes()), dataset,
                                       models[0][0])
    n_rows, n_cols = len(models), len(models[0])
    panel_height, key_height = 0.62 * len(rare) + 2.4, 1.1
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(6.2 * n_cols, panel_height * n_rows + key_height))
        grid = fig.add_gridspec(n_rows + 1, n_cols,
                                height_ratios=[panel_height] * n_rows + [key_height])
        key_ax = fig.add_subplot(grid[n_rows, :])
        key_ax.axis("off")
        _flow_key(key_ax, rare, colors, counts, label_size=8, notes=True, rows=2)
        for r, row in enumerate(models):
            for c, model in enumerate(row):
                ax = fig.add_subplot(grid[r, c])
                flows, right = _flows(pairs, model, rare)
                # The true types are the same in every panel, so only the first column names them.
                _sankey(ax, flows, rare, right, colors, show_left_labels=(c == 0))
                ax.set_title(_run_label(model), fontsize=10)
        return _save(fig, out_path, dpi)
