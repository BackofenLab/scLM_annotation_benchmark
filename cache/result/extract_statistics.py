import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
import numpy as np

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv("aggregated_by_dataset_model.tsv", sep='\t')
# ---------------------------------------------------------------------
# Dataset × subset size summary
# ---------------------------------------------------------------------
summary = (
    df.groupby(["dataset", "subset_size"])
      .agg(
          mean_balanced_accuracy=("balanced_accuracy", "mean"),
          mean_accuracy=("accuracy", "mean"),
          mean_weighted_f1=("weighted_f1", "mean"),
          best_weighted_f1=("weighted_f1", "max"),
          mean_lin_sim=("lin_avg_similarity", "mean"),
          best_lin_sim=("lin_avg_similarity", "max"),
          n_models=("model", "nunique"),
      )
      .reset_index()
)

summary = summary.sort_values(["dataset", "subset_size"])
summary.to_csv("dataset_summaries.tsv", sep='\t', index=False)

ranking = (
    df.groupby("model")
      .agg(
          mean_weighted_f1=("weighted_f1", "mean"),
          mean_balanced_accuracy=("balanced_accuracy", "mean"),
          mean_accuracy=("accuracy", "mean"),
          sd_weighted_f1=("weighted_f1", "std"),
          lin_avg_sim=("lin_avg_similarity", "mean"),
          lin_sd_sim=("lin_avg_similarity", "std"),
      )
      .sort_values("mean_weighted_f1", ascending=False)
      .reset_index()
)

ranking["rank"] = range(1, len(ranking) + 1)

ranking = ranking[
    [
        "rank",
        "model",
        "mean_weighted_f1",
        "mean_balanced_accuracy",
        "mean_accuracy",
        "sd_weighted_f1",
        "lin_avg_sim",
        "lin_sd_sim",
    ]
]

ranking.to_csv("rankings.tsv", sep='\t', index=False)

dataset_rank = (
    df.groupby(["dataset", "model"])
      .agg(
          mean_weighted_f1=("weighted_f1", "mean"),
          mean_balanced_accuracy=("balanced_accuracy", "mean"),
      )
      .reset_index()
)

dataset_rank["rank"] = (
    dataset_rank.groupby("dataset")["mean_weighted_f1"]
    .rank(method="dense", ascending=False)
)

dataset_rank = dataset_rank.sort_values(["dataset", "rank"])

winners = (
    dataset_rank.sort_values(
        ["dataset", "mean_weighted_f1"],
        ascending=[True, False]
    )
    .groupby("dataset")
    .first()
    .reset_index()
)

metrics = [
    "balanced_accuracy",
    "accuracy",
    "weighted_f1",
    "lin_avg_similarity",
]

tables = []

for metric in metrics:

    tmp = df.copy()

    # Rank within each dataset × subset size
    tmp["rank"] = (
        tmp.groupby(["dataset", "subset_size"])[metric]
           .rank(method="min", ascending=False)
    )

    summary = (
        tmp.groupby("model")
           .agg(
               **{
                   f"{metric}_wins": ("rank", lambda x: (x == 1).sum()),
                   f"{metric}_mean_rank": ("rank", "mean"),
                   f"{metric}_median_rank": ("rank", "median"),
                   f"{metric}_top3": ("rank", lambda x: (x <= 3).sum()),
                   f"{metric}_mean": (metric, "mean"),
               }
           )
           .reset_index()
    )

    tables.append(summary)

# ------------------------------------------------------------
# Merge all metrics into one table
# ------------------------------------------------------------

ranking = tables[0]

for t in tables[1:]:
    ranking = ranking.merge(t, on="model")

ranking = ranking.sort_values(
    "weighted_f1_mean_rank"
).reset_index(drop=True)

# Optional overall rank column
ranking.insert(0, "overall_rank", range(1, len(ranking) + 1))

ranking.to_csv("ranking_summary_all_metrics.tsv", sep='\t', index=False)

metrics = [
    "balanced_accuracy",
    "accuracy",
    "weighted_f1",
    "lin_avg_similarity",
]

subset_small = 1000
subset_large = 10000

results = []

for metric in metrics:

    # ------------------------------------------------------------------
    # Pivot so each row is Dataset × Model
    # ------------------------------------------------------------------

    pivot = (
        df.pivot_table(
            index=["dataset", "model"],
            columns="subset_size",
            values=metric,
            aggfunc="mean",
        )
        .reset_index()
    )

    # Keep only models evaluated on BOTH subset sizes
    pivot = pivot.dropna(subset=[subset_small, subset_large])

    x = pivot[subset_small]
    y = pivot[subset_large]

    # --------------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------------

    w_stat, w_p = wilcoxon(x, y, alternative="two-sided")

    t_stat, t_p = ttest_rel(x, y)

    # --------------------------------------------------------------
    # Paired Cohen's d
    # --------------------------------------------------------------

    diff = x - y
    cohens_d = diff.mean() / diff.std(ddof=1)

    results.append({
        "Metric": metric,
        "N pairs": len(diff),
        "Mean (1000)": x.mean(),
        "Mean (10000)": y.mean(),
        "Mean difference": diff.mean(),
        "Wilcoxon W": w_stat,
        "Wilcoxon p": w_p,
        "Paired t": t_stat,
        "Paired t p": t_p,
        "Cohen's d": cohens_d,
    })

results = pd.DataFrame(results)

for metric in metrics:

    for dataset, sub in df.groupby("dataset"):

        pivot = (
            sub.pivot_table(
                index="model",
                columns="subset_size",
                values=metric
            )
            .dropna()
        )

        stat, p = wilcoxon(pivot[1000], pivot[10000])

pd.set_option("display.precision", 3)

results.to_csv("subset_size_statistics.tsv", sep='\t', index=False)
