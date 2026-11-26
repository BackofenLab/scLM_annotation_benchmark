"""Figure 3 and Supplementary Figures S4 to S8: the prompt-context ablation and the imperfect-list
study.

Figure 3 shows balanced accuracy by what the prompt carries (A), with the exact list and with
imperfect lists (B), and where the cells of the types missing from a list go (C). S4 shows the
ablation per dataset with the seeds, S5 the answer format of the base model Gemma-2-2B, S6
annotation quality against cost, S7 the list study per dataset, and S8 the list study on the
agreement score. The inputs are the results of the two studies (paths.ABLATION_RESULTS,
paths.LIST_RUNS), the benchmark metrics and, for S6, the OpenRouter log
(paths.OPENROUTER_ACTIVITY) and the ablation's spend per run (paths.ABLATION_SPEND); figures.py
builds the figures.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

import paths
from methods import (ANSWER_TYPES, CATEGORY_TIERS, DATASET_ORDER, FAMILIES, MODE_LABELS,
                     MODE_ORDER, STUDY_ARM_MODEL, _display, _list_study_arms, categorize_model,
                     dataset_label, derive_mode, model_base, model_tier, spread)
from style import (ANSWER_TYPE_STYLE, FAMILY_COLORS, FS_GROUP, FS_LABEL, FS_LEGEND, FS_POINT,
                   FS_TICK, FS_TITLE, RC, SETTING_KEY, _family_key, _harmonize, _panel_letters,
                   _place_point_labels, _save, _stacked_shares, _tier_marker, _tiered_offsets)

# The ablation's arms: family colour, marker, line style and name. The marker is that of the arm's
# subcategory, solid lines are per cell and dashed ones per cluster. Gemma-2-2B, the base model of
# C2S-Scale and an open-weight non-reasoning LLM, takes that subcategory's square, dotted.
ABLATION_ARMS = {
    "A1_llm_percell": ("LLM", "^", "-", "GPT-OSS-120B, per cell"),
    "A2_base_percell": ("LLM", "s", ":", "Gemma-2-2B (base), per cell"),
    "A3_scllm_percell": ("scLLM", "o", "-", "C2S-Scale 2B, per cell"),
    "A4_llm_percluster": ("LLM", "^", "--", "GPT-OSS-120B, per cluster"),
    "A8_gemini_percluster": ("LLM", "*", "--", "Gemini 3.1 Pro, per cluster"),
}
# The arms that label clusters: the per-cluster language models, and scType and SCSA, which score
# their marker sets per cluster.
PER_CLUSTER_ARMS = {"A4_llm_percluster", "A8_gemini_percluster", "B3_sctype", "B4_scsa"}
# The arms of _prompt_content_panel, and its conditions in the order the model is given more.
PROMPT_ARMS = ["A1_llm_percell", "A4_llm_percluster", "A8_gemini_percluster", "A3_scllm_percell"]
PROMPT_MODES = ["M1", "M3", "M2", "M4"]
# The conditions of the list study that the figures show: the exact list, the three imperfections
# and their union. The key spells out A1 to A3.
LIST_CONDITIONS = [("E", "exact list"), ("L1", "A1"), ("L2-5", "A2"), ("L3", "A3"),
                   ("L4", "A1 + A2 + A3")]
LIST_KEY = ("Imperfect lists", ["A1  parents or children added",
                                "A2  five absent types added",
                                "A3  a quarter of the types removed"])
# _missing_types_panel: its conditions, and the answer types worded for the cells of a type the
# list left out.
MISSING_TYPE_CONDITIONS = [("L3", "a quarter of the types removed (A3)"),
                           ("L4", "all three errors (A1 + A2 + A3)")]
MISSING_TYPE_KEY = {"exact": "the missing type itself",
                    "coarser": "coarser, specific\n(e.g. a listed parent)",
                    "broad": "coarser, broad", "finer": "finer", "other": "another cell type",
                    "none": "no usable answer\n(incl. uncertain, Unknown)"}
# x positions in _imperfect_lists_panel: the no-list reference and the union stand apart, and the
# gap between families holds the methods' names.
REF_GAP, UNION_GAP, FAMILY_GAP = 1.3, 1.3, 2.6
# The scores of _imperfect_lists_panel: (column in the ablation and the benchmark, column in the
# list study, axis label).
STUDY_METRICS = {"balanced_accuracy": ("balanced_accuracy", "balanced_accuracy",
                                       "Balanced accuracy"),
                 "agreement": ("agreement_score", "agreement", "Agreement score")}
# Where _imperfect_lists_panel places a method's name: beside its last point first.
NAME_BESIDE = [(9, 0, "left", "center"), (9, 7, "left", "bottom"), (9, -7, "left", "top"),
               (16, 0, "left", "center"), (16, 10, "left", "bottom"), (16, -10, "left", "top"),
               (0, 9, "center", "bottom"), (0, -9, "center", "top")]

# _cost_panel shows the ablation at genes + list, the benchmark's constrained setting.
COST_MODE = "M2"
# Gemini 3.1 Pro per cell is taken from the benchmark: its constrained run, priced as in panel A.
GEMINI_PERCELL = {"model": "gemini-3.1-pro-preview_unblind", "label": "Gemini 3.1 Pro, per cell"}
# OpenRouter model identifiers in the log and the benchmark's method names.
OPENROUTER_MODELS = {
    "anthropic/claude-4.7-opus-20260416": "claude-opus-4.7",
    "anthropic/claude-4.6-sonnet-20260217": "claude-sonnet-4.6",
    "deepseek/deepseek-v3.2-20251201": "deepseek-v3.2",
    "google/gemini-3.1-pro-preview-20260219": "gemini-3.1-pro-preview",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "llama-4-maverick",
    "moonshotai/kimi-k2.6-20260420": "kimi-k2.6",
    "openai/gpt-5.5-20260423": "gpt-5.5",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
}


def load_ablation():
    return pd.read_csv(paths.ABLATION_RESULTS)


def load_list_runs():
    return pd.read_csv(paths.LIST_RUNS)


def _arm_rows(frame, arm):
    """The rows of one arm at its default variant: all genes per cell, resolution 1.0 per
    cluster."""
    variant = "res1.0" if arm.endswith("percluster") else "expression_ranked"
    return frame[(frame["arm"] == arm) & (frame["variant"] == variant)]


def _arm_style(arm):
    """Colour, marker and scatter area of an arm, from its method's family and subcategory."""
    model = STUDY_ARM_MODEL[arm]
    marker, size = _tier_marker(model)
    return FAMILY_COLORS[categorize_model(model)], marker, size


def _marker_size_for(marker):
    return 9.5 if marker == "*" else 6.0


def _scatter_area_for(marker, scale):
    """Scatter area (points squared) for the marker diameter of ``_marker_size_for``, so that a
    star keeps the visual weight of the round markers."""
    return (_marker_size_for(marker) ** 2) * 1.78 * scale


def _lighter(color):
    """``color`` mixed half and half with white."""
    return mcolors.to_hex([c + (1 - c) * 0.5 for c in mcolors.to_rgb(color)])


def _prompt_content_panel(ax, ablation, bench):
    """Balanced accuracy by what the prompt carries, one line per arm, and Gemini 3.1 Pro per cell
    from the benchmark, whose two prompts are those of genes only and genes + list, as hollow
    stars."""
    x = np.arange(len(PROMPT_MODES))
    handles = []
    for arm in PROMPT_ARMS:
        rows = _arm_rows(ablation, arm)
        colour, marker, _size = _arm_style(arm)
        unit = "cluster" if arm in PER_CLUSTER_ARMS else "cell"
        line, = ax.plot(x, [rows.loc[rows["mode"] == m, "balanced_accuracy"].mean()
                            for m in PROMPT_MODES],
                        color=colour, marker=marker, markersize=11 if marker == "*" else 7,
                        markeredgecolor="white", markeredgewidth=0.5, linewidth=1.4, zorder=3,
                        linestyle="--" if unit == "cluster" else "-",
                        label=f"{_display(STUDY_ARM_MODEL[arm])}, per {unit}")
        handles.append(line)
    colour = FAMILY_COLORS["LLM"]
    for mode, setting in (("M1", "blind"), ("M2", "unblind")):
        score = bench.loc[bench["model"] == f"gemini-3.1-pro-preview_{setting}",
                          "balanced_accuracy"]
        ax.scatter(PROMPT_MODES.index(mode), score.mean(), marker="*", s=150,
                   facecolor="white", edgecolor=colour, linewidth=1.3, zorder=4)
    handles.append(plt.Line2D([], [], linestyle="none", marker="*", markersize=11,
                              markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.3,
                              label="Gemini 3.1 Pro, per cell (benchmark)"))
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in PROMPT_MODES])
    ax.set_xlim(-0.35, len(PROMPT_MODES) - 0.65)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=FS_LEGEND - 0.5,
              handlelength=2.6, borderaxespad=0.1)


def _no_list_reference(ablation, bench, arms, dataset=None, metric="balanced_accuracy"):
    """Each list-study method without a list, the list being the only difference: the ablation's
    genes-only runs for the prompted arms, the unconstrained benchmark runs on the same cells for
    the others. Over the four datasets, or for ``dataset``."""
    column = STUDY_METRICS[metric][0]
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


def _imperfect_lists_panel(ax, runs, reference, metric):
    """``metric`` over all true types, a missing type counting as missed, per family: no list, the
    exact list, A1 to A3 and their union. A thin line joins the points of one method."""
    _column, list_column, label = STUDY_METRICS[metric]
    means = runs.groupby(["arm", "condition"])[list_column].mean()
    steps = [0.0, 1.0, 2.0, 3.0, 3.0 + UNION_GAP]
    span = REF_GAP + steps[-1]
    arms = _list_study_arms(runs)
    ticks, tick_labels, points, labels = [], [], [], []
    for g, (family, group_label) in enumerate(FAMILIES):
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
        tick_labels += ["no list"] + [name for _, name in LIST_CONDITIONS]
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
    """Where the cells of the types left off the list go, one bar per method and condition,
    averaged over the missing types as balanced accuracy averages."""
    arms = _list_study_arms(runs)[::-1]  # the first method on top
    family = [categorize_model(STUDY_ARM_MODEL[a]) for a in arms]
    y = np.cumsum([0.0] + [1.0 + 0.5 * (b != a) for a, b in zip(family, family[1:])])
    for ax, (condition, title) in zip(axes, MISSING_TYPE_CONDITIONS):
        shares = (runs[runs["condition"] == condition].groupby("arm")
                  [[f"removed_{k}" for k in ANSWER_TYPES]].mean().loc[arms]
                  .set_axis(ANSWER_TYPES, axis=1))
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
    # Each name carries its method's marker.
    beside = mtransforms.blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for yy, arm in zip(y, arms):
        colour, marker, size = _arm_style(arm)
        axes[0].scatter(-0.022, yy, s=size * 0.8, marker=marker, color=colour, edgecolor="white",
                        linewidth=0.4, transform=beside, clip_on=False, zorder=5)


def _list_study_key(fig, cell, same_method=True, spacing=1.5):
    """The key of the list study's panels in the gridspec ``cell``: line styles, the families and
    subcategories of the methods, and A1 to A3. ``same_method`` keys the thin grey line that joins
    one method's points; ``spacing`` is the gap between the key's columns."""
    drawn = {(categorize_model(m), model_tier(m)) for m in STUDY_ARM_MODEL.values()}
    lines = [plt.Line2D([], [], color="0.35", linewidth=1.4, label="per cell"),
             plt.Line2D([], [], color="0.35", linewidth=1.4, linestyle="--", label="per cluster")]
    if same_method:
        lines.append(plt.Line2D([], [], color="0.62", linewidth=0.7, label="same method"))
    ax_key = fig.add_subplot(cell)
    ax_key.axis("off")
    _family_key(ax_key, lines, drawn=drawn, extra=[LIST_KEY], spacing=spacing)


def figure_context_and_lists(df, out_path, dpi):
    """Figure 3: what the list of expected cell types adds (A, the prompt-context ablation), what an
    imperfect list costs (B, the list study) and where the cells of the types missing from the list
    go (C). The benchmark metrics supply Gemini 3.1 Pro per cell in A and the runs without a list
    of the methods that take no prompt in B."""
    ablation, runs = load_ablation(), load_list_runs()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.2, 9.6))
        # A and B, their tick labels, the key, a spacer, then C.
        outer = fig.add_gridspec(5, 1, height_ratios=[4.0, 1.0, 1.2, 0.3, 3.2], hspace=0,
                                 left=0.075, right=0.985, top=0.97, bottom=0.06)
        top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 2.55], wspace=0.13)
        ax_a = fig.add_subplot(top[0, 0])
        ax_b = fig.add_subplot(top[0, 1], sharey=ax_a)
        _prompt_content_panel(ax_a, ablation, df)
        _imperfect_lists_panel(ax_b, runs, _no_list_reference(ablation, df, _list_study_arms(runs)),
                               "balanced_accuracy")
        ax_b.tick_params(labelleft=True)
        ax_a.set_ylim(0, 0.72)
        # An invisible heading as tall as B's family names, so that the letters of A and B align.
        ax_a.text(0.5, 1.02, "baseline", transform=ax_a.transAxes, ha="center", va="bottom",
                  fontsize=FS_GROUP, alpha=0)
        _list_study_key(fig, outer[2])

        bottom = outer[4].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.42], wspace=0.07)
        ax_c = [fig.add_subplot(bottom[0, 0])]
        ax_c.append(fig.add_subplot(bottom[0, 1], sharey=ax_c[0]))
        _missing_types_panel(ax_c, runs)
        ax_c_key = fig.add_subplot(bottom[0, 2])
        ax_c_key.axis("off")
        ax_c_key.legend(handles=[mpatches.Patch(facecolor=ANSWER_TYPE_STYLE[key][1],
                                                edgecolor="0.55", linewidth=0.4,
                                                label=MISSING_TYPE_KEY[key])
                                 for key in ANSWER_TYPES],
                        loc="center left", frameon=False, fontsize=FS_LEGEND, borderaxespad=0,
                        borderpad=0, handlelength=1.6, labelspacing=0.7)
        left, right = ax_c[0].get_position(), ax_c[1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y0 - 0.03, "Share of the missing types' cells",
                 ha="center", va="top", fontsize=FS_LABEL)

        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c[0], "C")])
        return _save(fig, out_path, dpi)


def _ablation_dataset_panel(ax, ablation, dataset, bench, arms, offsets, gemini_dx):
    """One dataset of the prompt-context ablation: each arm's mean over the seeds with its standard
    error, the seeds as small points, and Gemini 3.1 Pro per cell from the benchmark as a hollow
    star."""
    x = np.arange(len(MODE_ORDER), dtype=float)
    for mode, model in (("M1", "gemini-3.1-pro-preview_blind"), ("M2", GEMINI_PERCELL["model"])):
        score = bench.loc[(bench["model"] == model) & (bench["dataset"] == dataset),
                          "balanced_accuracy"]
        ax.scatter(x[MODE_ORDER.index(mode)] + gemini_dx, score.mean(), marker="*",
                   s=_scatter_area_for("*", 0.85 ** 2), facecolor="none",
                   edgecolor=FAMILY_COLORS["LLM"], linewidth=1.4, zorder=4,
                   label=GEMINI_PERCELL["label"] + " (benchmark)" if mode == "M2" else None)
    for dx, arm in zip(offsets, arms):
        family, marker, linestyle, label = ABLATION_ARMS[arm]
        sub = _arm_rows(ablation, arm)
        sub = sub[sub["dataset"] == dataset]
        means, errs = [], []
        for i, mode in enumerate(MODE_ORDER):
            vals = sub.loc[sub["mode"] == mode, "balanced_accuracy"].dropna().to_numpy()
            means.append(vals.mean() if len(vals) else np.nan)
            errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            ax.scatter(np.full(len(vals), x[i] + dx), vals, s=7, color=FAMILY_COLORS[family],
                       alpha=0.55, zorder=2, linewidths=0)
        ax.errorbar(x + dx, means, yerr=errs, linestyle=linestyle, marker=marker,
                    color=FAMILY_COLORS[family], linewidth=1.3,
                    markersize=_marker_size_for(marker) * 0.85, capsize=2.5, elinewidth=0.9,
                    label=label, zorder=3)
    ax.set_title(dataset_label(dataset), fontsize=FS_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER])
    ax.set_ylim(0, None)
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)


def figure_context_per_dataset(df, out_path, dpi):
    """Supplementary Figure S4: the prompt-context ablation per dataset, with every arm and the
    shuffled gene order, each arm's mean and standard error over the seeds and the seeds as small
    points."""
    ablation = load_ablation()
    arms = list(ABLATION_ARMS)
    offsets = np.linspace(-0.16, 0.16, len(arms) + 1)  # the last one is Gemini's
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(11.6, 10.4))
        outer = fig.add_gridspec(2, 1, height_ratios=[9.4, 1.0], hspace=0.12)
        grid = outer[0].subgridspec(2, 2, hspace=0.45, wspace=0.18)
        panels = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(len(DATASET_ORDER))]
        for ax, dataset in zip(panels, DATASET_ORDER):
            _ablation_dataset_panel(ax, ablation, dataset, df, arms, offsets[:len(arms)],
                                    offsets[-1])
        ax_key = fig.add_subplot(outer[1])
        ax_key.axis("off")
        ax_key.legend(*panels[0].get_legend_handles_labels(), loc="center", ncol=3,
                      frameon=False, fontsize=FS_LEGEND)
        _panel_letters(fig, [(ax, chr(65 + i)) for i, ax in enumerate(panels)])
        return _save(fig, out_path, dpi)


def figure_context_supporting(df, out_path, dpi):
    """Supplementary Figure S5: the answer format of the base model Gemma-2-2B, the share of its
    answers that name no Cell Ontology term and that run past eight words, with its balanced
    accuracy on the same axis."""
    a2 = _arm_rows(load_ablation(), "A2_base_percell")
    modes = [m for m in MODE_ORDER if m in set(a2["mode"])]  # the arm had no shuffled order
    x = np.arange(len(modes), dtype=float)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12.2, 4.6))
        for col, style, label in [("non_cl_ratio", "-", "no ontology term"),
                                  ("frac_over_8_words", "--", "answer over 8 words")]:
            vals = [a2[a2["mode"] == m][col].mean() for m in modes]
            ax.plot(x, vals, linestyle=style, marker="s", markersize=5,
                    color=FAMILY_COLORS["LLM"], linewidth=1.4, label=label, zorder=3)
        acc = [a2[a2["mode"] == m]["balanced_accuracy"].mean() for m in modes]
        ax.plot(x, acc, linestyle=":", marker="o", markersize=4, color="0.35", linewidth=1.2,
                label="balanced accuracy", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in modes], fontsize=FS_TICK - 1)
        ax.set_ylabel("Share")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        # Lower right, above the balanced-accuracy line: on the left the lines rise steeply.
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), frameon=False,
                  fontsize=FS_LEGEND - 1)
        ax.set_title("Gemma-2-2B (base)", fontsize=FS_TITLE)
        sns.despine(ax=ax)
        return _save(fig, out_path, dpi)


def _llm_costs():
    """Spend per 1,000 calls and share of answered calls per LLM run, from the OpenRouter log.
    A run makes one call per cell, and a call counts as answered when it finished normally."""
    log = pd.read_csv(paths.OPENROUTER_ACTIVITY)
    log = log[log["in_benchmark"]]
    calls = pd.DataFrame({
        "model": log["model_permaslug"].map(OPENROUTER_MODELS) + "_" + log["mode"],
        "answered": log["finish_reason_normalized"].fillna("").str.lower().eq("stop"),
        "cost": log["cost_total"]}).groupby("model")
    n = calls.size()
    return pd.DataFrame({"cost_per_1k_cells": calls["cost"].sum() / n * 1000.0,
                         "answered_frac": calls["answered"].sum() / n})


def _arm_spend():
    """Spend per 1,000 annotated cells of each OpenRouter arm of the ablation at COST_MODE and its
    default variant (paths.ABLATION_SPEND): the cost of the distinct calls of its runs, summed from
    the response caches, over the cells of those runs. A cluster's prompt is one call for all of
    its cells."""
    spend = pd.read_csv(paths.ABLATION_SPEND)
    spend = spend[spend["mode"] == COST_MODE]
    totals = {arm: _arm_rows(spend, arm)[["cost_usd", "cells"]].sum()
              for arm in spend["arm"].unique()}
    return {arm: t["cost_usd"] / t["cells"] * 1000.0 for arm, t in totals.items()}


def _pareto_front(points):
    """Indices of the (cost, score) points that no cheaper point scores higher than."""
    front, best = [], -np.inf
    for i in sorted(range(len(points)), key=lambda k: points[k][0]):
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def _cost_quality_panel(ax, df):
    """Balanced accuracy of every LLM run, averaged over the datasets, against its spend per 1,000
    cells on a log axis. A step line per setting marks the runs that no cheaper run scores higher
    than; those runs and the runs that answered fewer than 98% of the calls are named."""
    costs = _llm_costs()
    scores = df.groupby("model")["balanced_accuracy"].mean()
    rows = [(model_base(m), derive_mode(m), float(costs.loc[m, "cost_per_1k_cells"]),
             float(score), float(costs.loc[m, "answered_frac"]))
            for m, score in scores.items() if categorize_model(m) == "LLM"]
    # The family's colour, lighter for the unconstrained runs.
    colors = {"unblind": FAMILY_COLORS["LLM"], "blind": _lighter(FAMILY_COLORS["LLM"])}
    obstacles = []
    for base, mode, cost, score, _answered in rows:
        marker, size = _tier_marker(f"{base}_{mode}")
        ax.scatter(cost, score, marker=marker, s=size * 1.6, color=colors[mode],
                   edgecolor="white", linewidth=0.8, zorder=4)
        obstacles.append((cost, score, np.sqrt(size * 1.6) / 2))

    labels = {}
    for base, mode, cost, score, answered in rows:
        if answered < 0.98:
            labels[base, mode] = [f"{_display(base)} ({answered:.0%} answered)", cost, score,
                                  "0.15"]
    for mode in ("blind", "unblind"):
        pts = [(c, s) for _b, m, c, s, _a in rows if m == mode]
        front = _pareto_front(pts)
        fx = [pts[i][0] for i in front]
        fy = [pts[i][1] for i in front]
        ax.step(fx, fy, where="post", color=colors[mode], linewidth=1.4, alpha=0.9, zorder=2)
        # The step line is an obstacle for the names too.
        for x0, x1, y0, y1 in zip(fx, fx[1:], fy, fy[1:]):
            obstacles += [(x, y0, 1.5) for x in np.geomspace(x0, x1, 60)]
            obstacles += [(x1, y, 1.5) for y in np.linspace(y0, y1, 20)]
        for i in front:
            base = [b for b, m, c, s, _a in rows if m == mode and (c, s) == pts[i]][0]
            labels.setdefault((base, mode), [_display(base), *pts[i], None])[3] = "0.15"

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 cells annotated (USD, log scale)")
    ax.set_ylabel("Balanced accuracy (averaged over datasets)")
    handles = [mpatches.Patch(facecolor=colors[m], label=SETTING_KEY[m])
               for m in ("unblind", "blind")]
    handles += [plt.Line2D([], [], linestyle="none", marker=marker, color="0.4", label=name)
                for _tier, name, marker, _size in CATEGORY_TIERS["LLM"]]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _place_point_labels(ax, [tuple(label) for label in labels.values()], obstacles, FS_POINT)


def _cost_panel(ax, ablation, bench):
    """Balanced accuracy at genes + list against the model calls per 1,000 annotated cells, one per
    cell or one per cluster the cells fall into, named with each arm's spend; Gemini 3.1 Pro per
    cell from the benchmark as a hollow star. The local arms have no spend."""
    labels, obstacles = [], []

    def _annotate(text, spend, calls, acc, colour, mark, hollow=False):
        ax.scatter(calls, acc, s=_scatter_area_for(mark, 1.19 if hollow else 1.0), marker=mark,
                   facecolor="none" if hollow else colour,
                   edgecolor=colour if hollow else "white",
                   linewidth=1.3 if hollow else 0.6, zorder=3)
        if spend is not None:
            # Three decimals below $0.10, so that the cheapest arm does not round to $0.01.
            text += f"\n${spend:.3f} / 1k cells" if spend < 0.1 else f"\n${spend:.2f} / 1k cells"
        labels.append((text, calls, acc, colour))
        obstacles.append((calls, acc, 11.0))

    spend = _arm_spend()
    for arm, (family, marker, _linestyle, label) in ABLATION_ARMS.items():
        rows = _arm_rows(ablation, arm)
        rows = rows[rows["mode"] == COST_MODE]
        calls = rows["n_clusters_in_eval"].mean() if arm.endswith("percluster") else 1000.0
        _annotate(label, spend.get(arm), calls, rows["balanced_accuracy"].mean(),
                  FAMILY_COLORS[family], marker)
    gemini = bench[bench["model"] == GEMINI_PERCELL["model"]]
    _annotate(GEMINI_PERCELL["label"] + " (benchmark)",
              _llm_costs().loc[GEMINI_PERCELL["model"], "cost_per_1k_cells"], 1000.0,
              gemini["balanced_accuracy"].mean(), FAMILY_COLORS["LLM"],
              ABLATION_ARMS["A8_gemini_percluster"][1], hollow=True)

    ax.set_xscale("log")
    ax.set_xlabel("Model calls per 1,000 annotated cells")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, None)
    # Room on both sides for the names of the points at the ends of the axis.
    ax.set_xlim(left=ax.get_xlim()[0] * 0.45, right=ax.get_xlim()[1] * 2.2)
    ax.grid(linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    # On a white box, since a name at a point's height can sit on a grid line.
    _place_point_labels(ax, labels, obstacles, FS_POINT,
                        text_bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def figure_cost_performance(df, out_path, dpi):
    """Supplementary Figure S6: annotation quality against the spend of every LLM of the benchmark
    (A) and against the model calls of the prompt-context ablation, per cell or per cluster (B)."""
    ablation = load_ablation()
    with plt.rc_context(RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"wspace": 0.3})
        _cost_quality_panel(ax_a, df)
        _harmonize(ax_a)
        _cost_panel(ax_b, ablation, df)
        _harmonize(ax_b)
        _panel_letters(fig, [(ax_a, "A"), (ax_b, "B")])
        return _save(fig, out_path, dpi)


def _name_column(ax, ends, x, fontsize):
    """Names in a column right of the last points, ``ends`` = [(name, y)] at ``x``: in the order
    of the ends and as close to them as one line of text between names allows, each joined to its
    end by a line. Call after the y limits are set."""
    lo, hi = ax.get_ylim()
    points_high = ax.get_position().height * ax.figure.get_figheight() * 72
    placed = spread([y for _, y in ends], 1.25 * fontsize * (hi - lo) / points_high)
    for (name, y), y_text in zip(ends, placed):
        ax.annotate(name, xy=(x, y), xytext=(x + 0.5, y_text), va="center", fontsize=fontsize,
                    color="0.15", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", linewidth=0.4, color="0.6", shrinkA=4,
                                    shrinkB=2))


def _list_dataset_panel(ax, runs, reference, dataset):
    """One dataset of the list study: each method as a line from no list through the exact list
    and A1 to A3 to their union, coloured by family, marked by subcategory and named at the
    right."""
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
    ax.set_ylim(0, None)  # per dataset, so that close lines spread out
    ax.set_ylabel("Balanced accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    _name_column(ax, ends, steps[-1], FS_POINT)


def figure_lists_per_dataset(df, out_path, dpi):
    """Supplementary Figure S7: the list study per dataset."""
    ablation, runs = load_ablation(), load_list_runs()
    arms = _list_study_arms(runs)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(11.6, 11.6))
        # The four datasets, then the key below the rotated tick labels.
        outer = fig.add_gridspec(2, 1, height_ratios=[10.4, 1.2], hspace=0.26)
        grid = outer[0].subgridspec(2, 2, hspace=0.5, wspace=0.18)
        panels = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(len(DATASET_ORDER))]
        for ax, dataset in zip(panels, DATASET_ORDER):
            _list_dataset_panel(ax, runs, _no_list_reference(ablation, df, arms, dataset), dataset)
        # The lines take the family's colour here, so no grey line joins a method's points.
        _list_study_key(fig, outer[1], same_method=False, spacing=0.7)
        _panel_letters(fig, [(ax, chr(65 + i)) for i, ax in enumerate(panels)])
        return _save(fig, out_path, dpi)


def figure_context_ontology(df, out_path, dpi):
    """Supplementary Figure S8: the list study on the agreement score, which gives half credit for
    a finer type or a coarser type that is not broad."""
    ablation, runs = load_ablation(), load_list_runs()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.2, 6.0))
        outer = fig.add_gridspec(3, 1, height_ratios=[4.0, 1.0, 1.2], hspace=0,
                                 left=0.075, right=0.985, top=0.94, bottom=0.02)
        ax = fig.add_subplot(outer[0])
        _imperfect_lists_panel(ax, runs, _no_list_reference(ablation, df, _list_study_arms(runs),
                                                            metric="agreement"), "agreement")
        ax.set_ylim(0, 0.72)
        _list_study_key(fig, outer[2])
        return _save(fig, out_path, dpi)
