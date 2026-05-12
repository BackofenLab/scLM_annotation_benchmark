#!/usr/bin/env python3
"""Summarize OpenRouter activity logs for manuscript reporting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "cache/openrouter_logs/openrouter_activity_2026-05-07.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cache/openrouter_logs/summary_2026-05-07"
NUMERIC_COLS = [
    "cost_total",
    "cost_web_search",
    "cost_cache",
    "cost_file_processing",
    "byok_usage_inference",
    "tokens_prompt",
    "tokens_completion",
    "tokens_reasoning",
    "tokens_cached",
    "generation_time_ms",
    "time_to_first_token_ms",
]
PROJECTION_CELLS_PER_DATASET = 1000
PROJECTION_DATASETS = 4


def model_label(slug: str) -> str:
    """Compact display name for OpenRouter model slugs."""
    label = str(slug).split("/")[-1]
    label = re.sub(r"-20\d{6}$", "", label)
    label = label.replace("-17b-128e-instruct", "")
    return label


def q(series: pd.Series, p: float) -> float:
    return float(series.dropna().quantile(p)) if series.notna().any() else np.nan


def fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def fmt_usd(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"${x:,.4f}" if abs(x) < 1 else f"${x:,.2f}"


def load_activity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["created_at"])
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["model"] = df["model_permaslug"].map(model_label)
    df["completed"] = (~df["cancelled"].astype(bool)) & (
        df["finish_reason_normalized"].fillna("").str.lower() == "stop"
    )
    df["total_tokens"] = df["tokens_prompt"] + df["tokens_completion"]
    df["visible_completion_tokens"] = (
        df["tokens_completion"] - df["tokens_reasoning"]
    ).clip(lower=0)
    df["generation_time_s"] = df["generation_time_ms"] / 1000
    df["ttft_s"] = df["time_to_first_token_ms"] / 1000
    df["output_tokens_per_s"] = np.where(
        df["generation_time_s"] > 0,
        df["tokens_completion"] / df["generation_time_s"],
        np.nan,
    )
    df["cost_per_1k_total_tokens"] = np.where(
        df["total_tokens"] > 0, df["cost_total"] / df["total_tokens"] * 1000, np.nan
    )
    df["cost_per_1k_completion_tokens"] = np.where(
        df["tokens_completion"] > 0,
        df["cost_total"] / df["tokens_completion"] * 1000,
        np.nan,
    )
    df["reasoning_share"] = np.where(
        df["tokens_completion"] > 0,
        df["tokens_reasoning"] / df["tokens_completion"],
        np.nan,
    )
    df["cached_prompt_share"] = np.where(
        df["tokens_prompt"] > 0, df["tokens_cached"] / df["tokens_prompt"], np.nan
    )
    return df


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    span_h = (df["created_at"].max() - df["created_at"].min()).total_seconds() / 3600
    rows = [
        ("time_window_start", df["created_at"].min().isoformat(sep=" ")),
        ("time_window_end", df["created_at"].max().isoformat(sep=" ")),
        ("observation_window_hours", span_h),
        ("requests", len(df)),
        ("unique_models", df["model_permaslug"].nunique()),
        ("unique_providers", df["provider_name"].nunique()),
        ("completed_requests", int(df["completed"].sum())),
        ("completion_rate", df["completed"].mean()),
        ("cancelled_requests", int(df["cancelled"].sum())),
        ("cancelled_rate", df["cancelled"].mean()),
        ("streamed_rate", df["streamed"].mean()),
        ("total_cost_usd", df["cost_total"].sum()),
        ("total_cache_discount_usd", df["cost_cache"].sum()),
        ("median_cost_per_request_usd", df["cost_total"].median()),
        ("mean_cost_per_request_usd", df["cost_total"].mean()),
        ("cost_per_1k_total_tokens_usd", df["cost_total"].sum() / df["total_tokens"].sum() * 1000),
        ("prompt_tokens", int(df["tokens_prompt"].sum())),
        ("completion_tokens", int(df["tokens_completion"].sum())),
        ("reasoning_tokens", int(df["tokens_reasoning"].sum())),
        ("visible_completion_tokens", int(df["visible_completion_tokens"].sum())),
        ("cached_tokens", int(df["tokens_cached"].sum())),
        ("reasoning_share_of_completion_tokens", df["tokens_reasoning"].sum() / df["tokens_completion"].sum()),
        ("cached_share_of_prompt_tokens", df["tokens_cached"].sum() / df["tokens_prompt"].sum()),
        ("requests_with_cached_tokens_rate", (df["tokens_cached"] > 0).mean()),
        ("median_generation_time_s", df["generation_time_s"].median()),
        ("iqr_generation_time_s", q(df["generation_time_s"], 0.75) - q(df["generation_time_s"], 0.25)),
        ("p95_generation_time_s", q(df["generation_time_s"], 0.95)),
        ("median_ttft_s", df["ttft_s"].median()),
        ("iqr_ttft_s", q(df["ttft_s"], 0.75) - q(df["ttft_s"], 0.25)),
        ("p95_ttft_s", q(df["ttft_s"], 0.95)),
        ("median_output_tokens_per_s", df["output_tokens_per_s"].median()),
        ("requests_per_minute_observed_window", len(df) / (span_h * 60) if span_h > 0 else np.nan),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def grouped_summary(df: pd.DataFrame, keys: str | Iterable[str]) -> pd.DataFrame:
    keys = [keys] if isinstance(keys, str) else list(keys)
    g = df.groupby(keys, dropna=False)
    out = g.agg(
        requests=("created_at", "count"),
        completed_requests=("completed", "sum"),
        completion_rate=("completed", "mean"),
        total_cost_usd=("cost_total", "sum"),
        median_cost_usd=("cost_total", "median"),
        prompt_tokens=("tokens_prompt", "sum"),
        completion_tokens=("tokens_completion", "sum"),
        reasoning_tokens=("tokens_reasoning", "sum"),
        visible_completion_tokens=("visible_completion_tokens", "sum"),
        cached_tokens=("tokens_cached", "sum"),
        median_generation_time_s=("generation_time_s", "median"),
        p95_generation_time_s=("generation_time_s", lambda s: q(s, 0.95)),
        median_ttft_s=("ttft_s", "median"),
        p95_ttft_s=("ttft_s", lambda s: q(s, 0.95)),
        median_output_tokens_per_s=("output_tokens_per_s", "median"),
        median_cost_per_1k_total_tokens_usd=("cost_per_1k_total_tokens", "median"),
        mean_reasoning_share=("reasoning_share", "mean"),
        mean_cached_prompt_share=("cached_prompt_share", "mean"),
    ).reset_index()
    out["cost_per_1k_total_tokens_usd"] = (
        out["total_cost_usd"] / (out["prompt_tokens"] + out["completion_tokens"]) * 1000
    )
    out["reasoning_share_of_completion_tokens"] = (
        out["reasoning_tokens"] / out["completion_tokens"].replace(0, np.nan)
    )
    out["cached_share_of_prompt_tokens"] = (
        out["cached_tokens"] / out["prompt_tokens"].replace(0, np.nan)
    )
    return out.sort_values(["requests", "total_cost_usd"], ascending=False)


def finish_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "finish_reason_normalized"]
    out = df.groupby(cols, dropna=False).size().reset_index(name="requests")
    out["fraction_within_model"] = out["requests"] / out.groupby("model")["requests"].transform("sum")
    return out.sort_values(["model", "requests"], ascending=[True, False])


def cost_projection_1000_cells(df: pd.DataFrame, n_datasets: int = PROJECTION_DATASETS) -> pd.DataFrame:
    """Project per-model cost for 1,000 cells per dataset across n datasets."""
    projected_requests = PROJECTION_CELLS_PER_DATASET * n_datasets
    rows = []
    for model, group in df.groupby("model", dropna=False):
        costs = group["cost_total"].dropna()
        row = {
            "model": model,
            "observed_requests": int(len(group)),
            "cells_per_dataset": PROJECTION_CELLS_PER_DATASET,
            "n_datasets": n_datasets,
            "projected_requests": projected_requests,
            "cost_mean_per_request_usd": costs.mean(),
            "cost_median_per_request_usd": costs.median(),
            "cost_std_per_request_usd": costs.std(),
            "cost_min_per_request_usd": costs.min(),
            "cost_p25_per_request_usd": costs.quantile(0.25),
            "cost_p75_per_request_usd": costs.quantile(0.75),
            "cost_max_per_request_usd": costs.max(),
        }
        for stat in ["mean", "median", "min", "p25", "p75", "max"]:
            row[f"projected_{stat}_cost_usd"] = (
                row[f"cost_{stat}_per_request_usd"] * projected_requests
            )
        row["recommended_budget_p75_plus_10pct_usd"] = row["projected_p75_cost_usd"] * 1.10
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("projected_mean_cost_usd")
    total = {
        "model": "TOTAL",
        "observed_requests": int(out["observed_requests"].sum()),
        "cells_per_dataset": PROJECTION_CELLS_PER_DATASET,
        "n_datasets": n_datasets,
        "projected_requests": int(out["projected_requests"].sum()),
    }
    for col in out.columns:
        if col.startswith("projected_") or col == "recommended_budget_p75_plus_10pct_usd":
            total[col] = out[col].sum()
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def manuscript_markdown(df: pd.DataFrame, by_model: pd.DataFrame) -> str:
    top_cost = by_model.sort_values("total_cost_usd", ascending=False).iloc[0]
    fastest = by_model.sort_values("median_generation_time_s").iloc[0]
    slowest = by_model.sort_values("median_generation_time_s", ascending=False).iloc[0]
    return "\n".join(
        [
            "# OpenRouter Activity Summary",
            "",
            (
                f"The activity log contains {len(df):,} generations across "
                f"{df['model_permaslug'].nunique()} models and {df['provider_name'].nunique()} providers "
                f"from {df['created_at'].min()} to {df['created_at'].max()}."
            ),
            (
                f"Total spend was {fmt_usd(df['cost_total'].sum())}; the median request cost was "
                f"{fmt_usd(df['cost_total'].median())}."
            ),
            (
                f"Requests used {df['tokens_prompt'].sum():,} prompt tokens and "
                f"{df['tokens_completion'].sum():,} completion tokens, including "
                f"{df['tokens_reasoning'].sum():,} reasoning tokens."
            ),
            (
                f"Median generation latency was {fmt_num(df['generation_time_s'].median())} s "
                f"(IQR {fmt_num(q(df['generation_time_s'], 0.25))}-"
                f"{fmt_num(q(df['generation_time_s'], 0.75))} s; "
                f"95th percentile {fmt_num(q(df['generation_time_s'], 0.95))} s)."
            ),
            (
                f"Median time to first token was {fmt_num(df['ttft_s'].median())} s "
                f"(IQR {fmt_num(q(df['ttft_s'], 0.25))}-"
                f"{fmt_num(q(df['ttft_s'], 0.75))} s)."
            ),
            (
                f"The highest total-cost model was {top_cost['model']} "
                f"({fmt_usd(top_cost['total_cost_usd'])}); the fastest median model was "
                f"{fastest['model']} ({fmt_num(fastest['median_generation_time_s'])} s), "
                f"and the slowest was {slowest['model']} "
                f"({fmt_num(slowest['median_generation_time_s'])} s)."
            ),
            (
                f"Completion rate was {df['completed'].mean():.1%}; cancellation rate was "
                f"{df['cancelled'].mean():.1%}."
            ),
            "",
            "Generated files: `overall_summary.csv`, `by_model.csv`, `by_provider.csv`, "
            "`by_model_provider.csv`, `finish_reasons.csv`, `cost_projection_1000_cells.csv`, "
            "and `summary.json`.",
        ]
    )


def write_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = overall_summary(df)
    by_model = grouped_summary(df, "model")
    by_provider = grouped_summary(df, "provider_name")
    by_model_provider = grouped_summary(df, ["model", "provider_name"])
    finish_reasons = finish_reason_summary(df)
    projection = cost_projection_1000_cells(df)

    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    by_model.to_csv(out_dir / "by_model.csv", index=False)
    by_provider.to_csv(out_dir / "by_provider.csv", index=False)
    by_model_provider.to_csv(out_dir / "by_model_provider.csv", index=False)
    finish_reasons.to_csv(out_dir / "finish_reasons.csv", index=False)
    projection.to_csv(out_dir / "cost_projection_1000_cells.csv", index=False)
    (out_dir / "manuscript_numbers.md").write_text(
        manuscript_markdown(df, by_model) + "\n", encoding="utf-8"
    )
    summary = {
        "overall": dict(overall.itertuples(index=False, name=None)),
        "by_model": by_model.to_dict(orient="records"),
        "cost_projection_1000_cells": projection.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="OpenRouter activity CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary tables and manuscript text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_activity(args.input)
    write_outputs(df, args.output_dir)
    print(f"Wrote OpenRouter summary tables to {args.output_dir}")


if __name__ == "__main__":
    main()
