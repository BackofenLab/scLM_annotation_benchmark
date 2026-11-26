#!/usr/bin/env python
"""Standard plots of a benchmark run.

Reads ``benchmark_metrics.csv`` and writes, for every metric and test set size,

  heatmap_<metric>_sub<size>.pdf   models by datasets, sorted by the mean over the datasets
  ranking_<metric>_sub<size>.pdf   mean rank of every model, with its rank in each dataset

    plot_results.py --input results/benchmark_metrics.csv --output results/plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = {
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "lin_avg_similarity": "Lin similarity",
    "agreement_score": "Agreement score",
}
ROW_HEIGHT = 0.28  # inches per model


def heatmap(table, title, path):
    """Models by datasets, the models sorted by their mean."""
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(2.5 + 1.1 * table.shape[1], 1.2 + ROW_HEIGHT * len(table)))
    image = ax.imshow(table.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table)), table.index)
    for row, values in enumerate(table.to_numpy(dtype=float)):
        for column, value in enumerate(values):
            if pd.notna(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.6 else "black")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ranking(table, title, path):
    """Mean rank of every model over the datasets (1 is best) and its rank in each dataset."""
    ranks = table.rank(ascending=False, method="min")
    order = ranks.mean(axis=1).sort_values().index
    ranks = ranks.loc[order]
    fig, ax = plt.subplots(figsize=(6, 1.2 + ROW_HEIGHT * len(ranks)))
    positions = range(len(ranks))
    ax.barh(positions, ranks.mean(axis=1), color="lightgrey")
    for dataset in ranks.columns:
        ax.scatter(ranks[dataset], positions, s=18, label=dataset, zorder=3)
    ax.set_yticks(positions, ranks.index)
    ax.invert_yaxis()
    ax.set_xlabel("Rank (1 is best)")
    ax.set_title(title)
    ax.legend(title="Dataset", fontsize=7, frameon=False, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="benchmark_metrics.csv of aggregate.py")
    parser.add_argument("--output", required=True, help="folder for the plots")
    args = parser.parse_args()

    metrics = pd.read_csv(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for size, runs in metrics.groupby("subset_size"):
        for metric, label in METRICS.items():
            if metric not in runs.columns or runs[metric].notna().sum() == 0:
                continue
            table = runs.pivot_table(index="model", columns="dataset", values=metric)
            title = f"{label}, {int(size):,} cells per test set"
            heatmap(table, title, out_dir / f"heatmap_{metric}_sub{int(size)}.pdf")
            ranking(table, title, out_dir / f"ranking_{metric}_sub{int(size)}.pdf")
    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    main()
