#!/usr/bin/env python
"""Recall per method and true cell type, with the share of the type in the full dataset.

Joins the benchmark's per-class table (``paper/results/per_class_scores.csv``, 1,000-cell test
sets) with the share of every cell type in the full dataset before subsampling
(``paper/data/cell_type_distributions/<dataset>.csv``). A stratified test set holds at least 20
cells of every type, so only the share in the full dataset tells how rare a type is.

One row per method, setting, dataset and true cell type. ``k`` is the type's Cell Ontology name
in lower case, ``emitted`` says whether the method named the type for any cell and ``reachable``
whether a constrained baseline has a label at or below the type (empty for other runs).

    python paper/scripts/rare_type_table.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.PIPELINE_BIN))
from plot_model_comparison import categorize_model, derive_mode, group_name, model_base  # noqa: E402

SUBSET_SIZE = 1000
OVERALL = "__OVERALL__"


def full_prevalence(pattern, mappings, cache_dir):
    """Share and number of cells of every cell type in the full dataset. Labels that name the
    same Cell Ontology term are summed."""
    from evaluate import load_ontology_context
    from util.cell_ontology_similarity import download_cl_ontology
    from util.ontology_mapping import canonical_names, load_review_status

    label_index, _taxonomy = load_ontology_context(download_cl_ontology(cache_dir))
    _reviewed, review_map = load_review_status(mappings)
    frames = [pd.read_csv(f).assign(dataset=Path(f).stem) for f in sorted(glob.glob(pattern))]
    if not frames:
        raise SystemExit(f"No cell type distributions match {pattern}")
    distribution = pd.concat(frames)
    distribution = distribution[~distribution["dropped"].fillna(False)].copy()
    distribution["k"] = canonical_names(distribution["cell_type"], label_index,
                                        review_map)[0].str.lower()
    return distribution.groupby(["dataset", "k"], as_index=False).agg(
        orig_prevalence=("original_proportion", "sum"),
        orig_count=("original_count", "sum"))


def outcomes(per_class_path):
    """Recall, emitted and reachable per run and true cell type at SUBSET_SIZE cells."""
    per_class = pd.read_csv(per_class_path, float_precision="round_trip", dtype={"emitted": str,
                                                                                 "reachable": str})
    per_class = per_class[(per_class["subset_size"] == SUBSET_SIZE)
                          & (per_class["cell_type"] != OVERALL)]
    reachable = {"True": True, "False": False}
    return pd.DataFrame({
        "method": per_class["model"].map(model_base),
        "mode": per_class["model"].map(derive_mode),
        "dataset": per_class["dataset"],
        "k": per_class["cell_type"].str.lower(),
        "n_cells": per_class["n_cells"],
        "recall": per_class["recall"],
        "emitted": per_class["emitted"] == "True",
        "reachable": per_class["reachable"].map(reachable),
    })


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-class", default=str(paths.PER_CLASS_SCORES))
    p.add_argument("--distributions", default=str(paths.CELL_TYPE_DISTRIBUTIONS / "*.csv"))
    p.add_argument("--mappings", default=str(paths.REPO / "resources" / "cell_type_ontology_mappings.csv"))
    p.add_argument("--cache-dir", default=str(paths.REPO / "cache"),
                   help="where the Cell Ontology release is stored")
    p.add_argument("--out", default=str(paths.RARE_TYPE_OUTCOMES))
    args = p.parse_args(argv)

    prevalence = full_prevalence(args.distributions, args.mappings, Path(args.cache_dir))
    merged = outcomes(args.per_class).merge(prevalence, on=["dataset", "k"], how="left")
    unmatched = merged["orig_prevalence"].isna()
    if unmatched.any():
        names = merged.loc[unmatched, ["dataset", "k"]].drop_duplicates()
        print(f"WARNING: {unmatched.sum()} rows have no share in the full dataset and are left "
              f"out, covering {len(names)} labels:\n{names.to_string(index=False)}",
              file=sys.stderr)
    merged = merged[~unmatched].copy()
    merged["family"] = merged["method"].map(lambda m: categorize_model(f"{m}_blind"))
    merged = merged[merged["family"].isin(["LLM", "scLLM", "classical"])]
    merged["block"] = [group_name(f"{m}_{mode}") for m, mode in zip(merged["method"], merged["mode"])]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    types = merged.drop_duplicates(["dataset", "k"])
    print(f"{len(merged)} rows, {merged['method'].nunique()} methods, {len(types)} cell types -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
