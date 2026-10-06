"""Summary statistics quoted in the Results text: dataset means and best scores, per-model means,
mean ranks, and the 1,000- against 10,000-cell comparison.

The model set is the figures' one: the runs the paper keeps (``load_aggregate``) without the chance
models. Cross-dataset statistics (per-model means, mean ranks) use the 1,000-cell runs, the only
size every model ran at, and count mmContext's Zeisel run as a run without answers
(``with_unsupported_as_zero``). Per-dataset statistics and the paired size comparison use the runs
as they are.

    python cache/result/extract_statistics.py --input cache/result/aggregated_by_dataset_model_<v>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bin"))
from manuscript_figures import load_aggregate  # noqa: E402
from plot_model_comparison import categorize_model, with_unsupported_as_zero  # noqa: E402

METRICS = ["balanced_accuracy", "accuracy", "weighted_f1", "lin_avg_similarity"]
OUT = Path(__file__).resolve().parent


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
    out["sd_weighted_f1"], out["sd_lin_avg_similarity"] = g["weighted_f1"].std(), g["lin_avg_similarity"].std()
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
    out = pd.concat(parts, axis=1).sort_values(["weighted_f1_mean_rank", "balanced_accuracy_mean_rank"])
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="Results/aggregated_by_dataset_model.csv")
    args = p.parse_args()

    runs = load_aggregate(args.input, subset_size=None)
    runs = runs[runs["model"].map(categorize_model) != "simple"]  # chance models are references
    small = with_unsupported_as_zero(runs[runs["subset_size"] == 1000], model_cols=("model", "mode"))

    dataset_summaries(runs).to_csv(OUT / "dataset_summaries.tsv", sep="\t", index=False)
    model_means(small).to_csv(OUT / "rankings.tsv", sep="\t", index=False)
    mean_ranks(small).to_csv(OUT / "ranking_summary_all_metrics.tsv", sep="\t", index=False)
    subset_size_statistics(runs).to_csv(OUT / "subset_size_statistics.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
