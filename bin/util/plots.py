"""Plots written by the evaluation of a run."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

FONT_SIZE = 8
WHITE_TO_TEAL = LinearSegmentedColormap.from_list(
    "teal_white", ["#ffffff", plt.get_cmap("tab20c")(0 / 19)])


def set_style():
    sns.set_context("paper", font_scale=1)
    sns.set_style("white")
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # A "$" in an answer is text, not the start of a formula.
        "text.parse_math": False,
    })


def plot_confusion_matrix(true_labels, pred_labels, output_path, title):
    """Heatmap of true against predicted labels, each row scaled to proportions."""
    set_style()
    true_labels = pd.Series(true_labels).astype(str)
    pred_labels = pd.Series(pred_labels).astype(str)
    labels = sorted(set(true_labels) | set(pred_labels))
    counts = confusion_matrix(true_labels, pred_labels, labels=labels)
    with np.errstate(invalid="ignore"):
        shares = np.nan_to_num(counts.astype("float") / counts.sum(axis=1, keepdims=True))

    side = max(6, 0.7 * len(labels))
    plt.figure(figsize=(side, side))
    ax = sns.heatmap(
        shares, annot=True, square=True, fmt=".2f", cmap=WHITE_TO_TEAL,
        xticklabels=labels, yticklabels=labels, vmin=0, vmax=1.0,
        cbar_kws={"label": "Proportion", "shrink": 0.6, "aspect": 20},
        annot_kws={"size": FONT_SIZE})
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.outline.set_edgecolor("black")
    plt.xlabel("Predicted Label", fontsize=FONT_SIZE)
    plt.ylabel("True Label", fontsize=FONT_SIZE)
    plt.title(title, fontsize=FONT_SIZE, fontweight="normal", family="Arial")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
