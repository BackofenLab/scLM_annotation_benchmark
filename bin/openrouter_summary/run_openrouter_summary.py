#!/usr/bin/env python3
"""Run OpenRouter activity summaries and paper plots in one step."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_openrouter_activity import (
    plot_cost_and_tokens,
    plot_cost_distribution,
    plot_cost_vs_completion_tokens,
    plot_efficiency,
    plot_finish_reasons,
    plot_latency_distributions,
    plot_provider_mix,
    plot_temporal_activity,
    setup_style,
)
from summarize_openrouter_activity import DEFAULT_INPUT, REPO_ROOT, load_activity, write_outputs


DEFAULT_OUTPUT_DIR = REPO_ROOT / "cache" / "openrouter_logs" / "openrouter_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="OpenRouter activity CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for tables and figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure output formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_dir = args.output_dir / "statistics"
    figures_dir = args.output_dir / "figures"

    df = load_activity(args.input)
    write_outputs(df, stats_dir)

    setup_style()
    plot_cost_and_tokens(df, figures_dir, args.formats)
    plot_latency_distributions(df, figures_dir, args.formats)
    plot_efficiency(df, figures_dir, args.formats)
    plot_provider_mix(df, figures_dir, args.formats)
    plot_temporal_activity(df, figures_dir, args.formats)
    plot_finish_reasons(df, figures_dir, args.formats)
    plot_cost_distribution(df, figures_dir, args.formats)
    plot_cost_vs_completion_tokens(df, figures_dir, args.formats)

    print(f"Wrote OpenRouter statistics to {stats_dir}")
    print(f"Wrote OpenRouter figures to {figures_dir}")


if __name__ == "__main__":
    main()
