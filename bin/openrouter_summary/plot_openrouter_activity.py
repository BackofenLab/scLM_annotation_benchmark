#!/usr/bin/env python3
"""Generate publication-style plots from OpenRouter activity logs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from summarize_openrouter_activity import DEFAULT_INPUT, REPO_ROOT, grouped_summary, load_activity


PLOT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=PLOT_COLORS,
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def plot_cost_and_tokens(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    summary = grouped_summary(df, "model").sort_values("total_cost_usd")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)

    axes[0].barh(summary["model"], summary["total_cost_usd"], color=PLOT_COLORS[0])
    axes[0].set_xlabel("Total cost (USD)")
    axes[0].set_ylabel("")
    axes[0].set_title("A. Spend")

    token_cols = ["prompt_tokens", "visible_completion_tokens", "reasoning_tokens"]
    left = pd.Series(0, index=summary.index, dtype=float)
    labels = ["Prompt", "Visible completion", "Reasoning"]
    for col, label, color in zip(token_cols, labels, PLOT_COLORS[:3]):
        axes[1].barh(summary["model"], summary[col] / 1e6, left=left / 1e6, label=label, color=color)
        left += summary[col]
    axes[1].set_xlabel("Tokens (millions)")
    axes[1].set_title("B. Token composition")
    axes[1].legend(frameon=False, loc="lower right")
    save(fig, out_dir, "figure_1_cost_and_token_use", formats)


def plot_latency_distributions(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    order = grouped_summary(df, "model").sort_values("median_generation_time_s")["model"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    sns.boxplot(data=df, y="model", x="generation_time_s", order=order, ax=axes[0], fliersize=0, color=PLOT_COLORS[0])
    sns.boxplot(data=df, y="model", x="ttft_s", order=order, ax=axes[1], fliersize=0, color=PLOT_COLORS[1])
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Generation time (s, log scale)")
    axes[1].set_xlabel("Time to first token (s, log scale)")
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    axes[0].set_title("A. End-to-end latency")
    axes[1].set_title("B. First-token latency")
    save(fig, out_dir, "figure_2_latency_distributions", formats)


def plot_efficiency(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    summary = grouped_summary(df, "model")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    sns.scatterplot(
        data=summary,
        x="median_generation_time_s",
        y="cost_per_1k_total_tokens_usd",
        size="requests",
        hue="model",
        sizes=(80, 300),
        ax=ax,
        legend=False,
    )
    for _, row in summary.iterrows():
        ax.text(row["median_generation_time_s"] * 1.03, row["cost_per_1k_total_tokens_usd"], row["model"], fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median generation time (s, log scale)")
    ax.set_ylabel("Cost per 1k total tokens (USD, log scale)")
    ax.set_title("Cost-latency trade-off by model")
    save(fig, out_dir, "figure_3_cost_latency_tradeoff", formats)


def plot_provider_mix(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    mix = pd.crosstab(df["model"], df["provider_name"], normalize="index")
    provider_order = df["provider_name"].value_counts().index.tolist()
    mix = mix.reindex(columns=provider_order).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bottom = pd.Series(0, index=mix.index, dtype=float)
    for idx, provider in enumerate(mix.columns):
        vals = mix[provider]
        ax.bar(mix.index, vals, bottom=bottom, label=provider, color=sns.color_palette("tab20", len(mix.columns))[idx])
        bottom += vals
    ax.set_ylabel("Fraction of requests")
    ax.set_xlabel("")
    ax.set_title("Provider routing mix by model")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, out_dir, "figure_4_provider_mix", formats)


def plot_temporal_activity(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    ts = (
        df.set_index("created_at")
        .sort_index()
        .resample("15min")
        .agg(requests=("generation_id", "count"), cost_usd=("cost_total", "sum"))
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(7.2, 2.8))
    ax1.plot(ts["created_at"], ts["requests"], color=PLOT_COLORS[0], lw=1.5, label="Requests")
    ax1.set_ylabel("Requests per 15 min")
    ax1.set_xlabel("")
    ax2 = ax1.twinx()
    ax2.plot(ts["created_at"], ts["cost_usd"], color=PLOT_COLORS[1], lw=1.5, label="Cost")
    ax2.set_ylabel("Cost (USD)")
    ax1.set_title("Temporal workload profile")
    fig.autofmt_xdate()
    save(fig, out_dir, "figure_5_temporal_activity", formats)


def plot_finish_reasons(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    reasons = pd.crosstab(df["model"], df["finish_reason_normalized"], normalize="index")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    reasons.plot(kind="bar", stacked=True, ax=ax, color=PLOT_COLORS)
    ax.set_ylabel("Fraction of requests")
    ax.set_xlabel("")
    ax.set_title("Finish reasons by model")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="Finish reason")
    save(fig, out_dir, "figure_6_finish_reasons", formats)


def plot_cost_distribution(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    order = grouped_summary(df, "model").sort_values("median_cost_usd")["model"]
    plot_df = df[df["cost_total"] > 0].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    sns.boxplot(
        data=plot_df,
        y="model",
        x="cost_total",
        order=order,
        ax=ax,
        fliersize=0,
        color=PLOT_COLORS[0],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Cost per request (USD, log scale)")
    ax.set_ylabel("")
    ax.set_title("Per-request cost distribution by model")
    save(fig, out_dir, "figure_7_cost_distribution", formats)


def plot_cost_vs_completion_tokens(df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    plot_df = df[(df["cost_total"] > 0) & (df["tokens_completion"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for idx, (model, group) in enumerate(plot_df.groupby("model", sort=False)):
        ax.scatter(
            group["tokens_completion"],
            group["cost_total"],
            s=10,
            alpha=0.35,
            label=model,
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            linewidths=0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Completion tokens per request (log scale)")
    ax.set_ylabel("Cost per request (USD, log scale)")
    ax.set_title("Request cost scales with completion length")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, out_dir, "figure_8_cost_vs_completion_tokens", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="OpenRouter activity CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "cache/openrouter_logs/figures_2026-05-07",
        help="Directory for plot files.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    df = load_activity(args.input)
    plot_cost_and_tokens(df, args.output_dir, args.formats)
    plot_latency_distributions(df, args.output_dir, args.formats)
    plot_efficiency(df, args.output_dir, args.formats)
    plot_provider_mix(df, args.output_dir, args.formats)
    plot_temporal_activity(df, args.output_dir, args.formats)
    plot_finish_reasons(df, args.output_dir, args.formats)
    plot_cost_distribution(df, args.output_dir, args.formats)
    plot_cost_vs_completion_tokens(df, args.output_dir, args.formats)
    print(f"Wrote OpenRouter figures to {args.output_dir}")


if __name__ == "__main__":
    main()
