"""Summary statistics of the benchmark that the paper quotes: dataset means and best scores,
per-method means, mean ranks, and the comparison of the 1,000- and the 10,000-cell runs.

The runs are those of the benchmark metrics without the chance model. Statistics across datasets
(per-method means, mean ranks) use the 1,000-cell runs, the only size every method ran at;
per-dataset statistics and the paired comparison of the sizes use every run. Writes four TSV
files to paths.STATISTICS.

    python paper/scripts/extract_statistics.py [--input CSV]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# The scripts folder, also when this file is loaded by its path rather than run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from methods import categorize_model, load_aggregate  # noqa: E402

METRICS = ["balanced_accuracy", "accuracy", "weighted_f1", "lin_avg_similarity"]
OUT = paths.STATISTICS


def dataset_summaries(df):
    rows = []
    for (dataset, size), g in df.groupby(["dataset", "subset_size"]):
        row = {"dataset": dataset, "subset_size": size, "n_models": g["model"].nunique()}
        for m in METRICS:
            row[f"mean_{m}"] = g[m].mean()
        for m in ("weighted_f1", "lin_avg_similarity"):
            best = g.loc[g[m].idxmax()]
            row[f"best_{m}"], row[f"best_{m}_model"] = best[m], best["model"]
        rows.append(row)
    return pd.DataFrame(rows)


def model_means(df):
    g = df.groupby("model")
    out = pd.DataFrame({f"mean_{m}": g[m].mean() for m in METRICS})
    out["sd_weighted_f1"] = g["weighted_f1"].std()
    out["sd_lin_avg_similarity"] = g["lin_avg_similarity"].std()
    out = out.sort_values("mean_weighted_f1", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index()


def mean_ranks(df):
    """Rank within each dataset, then summarize per model; ordered by the mean weighted-F1 rank,
    ties broken by the mean balanced-accuracy rank."""
    parts = []
    for m in METRICS:
        rank = df.groupby("dataset")[m].rank(method="min", ascending=False).groupby(df["model"])
        parts.append(pd.DataFrame({f"{m}_wins": rank.apply(lambda r: (r == 1).sum()),
                                   f"{m}_mean_rank": rank.mean(), f"{m}_median_rank": rank.median(),
                                   f"{m}_top3": rank.apply(lambda r: (r <= 3).sum()),
                                   f"{m}_mean": df.groupby("model")[m].mean()}))
    out = pd.concat(parts, axis=1).sort_values(["weighted_f1_mean_rank",
                                                "balanced_accuracy_mean_rank"])
    out.insert(0, "overall_rank", range(1, len(out) + 1))
    return out.rename_axis("model").reset_index()


def subset_size_statistics(df):
    """Paired comparison over the (dataset, model) runs that exist at both sizes."""
    rows = []
    for m in METRICS:
        pair = df.pivot_table(index=["dataset", "model"], columns="subset_size", values=m).dropna()
        x, y = pair[1000], pair[10000]
        diff, w, t = x - y, wilcoxon(x, y), ttest_rel(x, y)
        rows.append({"Metric": m, "N pairs": len(diff), "Mean (1000)": x.mean(),
                     "Mean (10000)": y.mean(), "Mean difference": diff.mean(),
                     "Wilcoxon W": w.statistic, "Wilcoxon p": w.pvalue, "Paired t": t.statistic,
                     "Paired t p": t.pvalue, "Cohen's d": diff.mean() / diff.std(ddof=1)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(paths.BENCHMARK_METRICS))
    args = p.parse_args()

    runs = load_aggregate(args.input, subset_size=None)
    runs = runs[runs["model"].map(categorize_model) != "chance"]
    small = runs[runs["subset_size"] == 1000]

    dataset_summaries(runs).to_csv(OUT / "dataset_summaries.tsv", sep="\t", index=False)
    model_means(small).to_csv(OUT / "rankings.tsv", sep="\t", index=False)
    mean_ranks(small).to_csv(OUT / "ranking_summary_all_metrics.tsv", sep="\t", index=False)
    subset_size_statistics(runs).to_csv(OUT / "subset_size_statistics.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
