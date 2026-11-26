#!/usr/bin/env python
"""Build the paper's figures, each as figure_<key>.pdf and figure_<key>.png.

The keys are the figure numbers of the paper (2 to 4, s1 to s14). The benchmark metrics are read
from --input; the other inputs are the result tables, data files and experiment results listed in
paths.py.

    python paper/scripts/figures.py [--input CSV] [--outdir DIR] [--dpi N] [--figures KEY ...]
"""
import argparse
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import paths  # noqa: E402
from figures_benchmark import (figure_answer_types, figure_benchmark,  # noqa: E402
                               figure_harmonization, figure_metric_agreement)
from figures_experiments import (figure_context_and_lists, figure_context_ontology,  # noqa: E402
                                 figure_context_per_dataset, figure_context_supporting,
                                 figure_cost_performance, figure_lists_per_dataset)
from figures_rare_types import (figure_rare_answer_types, figure_rare_flows,  # noqa: E402
                                figure_rare_full, figure_rare_types)
from methods import DATASET_ORDER, load_aggregate  # noqa: E402

FIGURES = {
    "2": figure_benchmark,
    "3": figure_context_and_lists,
    "4": figure_rare_types,
    "s1_harmonization": figure_harmonization,
    "s2_answer_types": figure_answer_types,
    "s3_metric_agreement": figure_metric_agreement,
    "s4_context_per_dataset": figure_context_per_dataset,
    "s5_context_supporting": figure_context_supporting,
    "s6_cost_performance": figure_cost_performance,
    "s7_lists_per_dataset": figure_lists_per_dataset,
    "s8_ontology_scores": figure_context_ontology,
    "s9_rare_full": figure_rare_full,
    "s10_rare_answer_types": figure_rare_answer_types,
    **{f"s{i + 11}_{dataset.lower()}": partial(figure_rare_flows, dataset=dataset)
       for i, dataset in enumerate(DATASET_ORDER)},
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(paths.BENCHMARK_METRICS),
                   help="benchmark metrics (default: %(default)s)")
    p.add_argument("--outdir", default=str(paths.FIGURES),
                   help="output folder (default: %(default)s)")
    p.add_argument("--dpi", type=int, default=300,
                   help="raster resolution (default: %(default)s)")
    p.add_argument("--figures", nargs="+", default=sorted(FIGURES), choices=sorted(FIGURES),
                   help="figures to build (default: all)")
    args = p.parse_args(argv)

    df = load_aggregate(args.input)
    for key in args.figures:
        for fmt in ("pdf", "png"):
            out = FIGURES[key](df, Path(args.outdir) / f"figure_{key}.{fmt}", dpi=args.dpi)
            print(f"[figure {key}]  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
