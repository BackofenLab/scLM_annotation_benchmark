#!/usr/bin/env python
"""Per-class ontology-graded scores for every 1,000-cell run.

Companion to ``bin/rare_type_table.py``. That script answers "did the method find this cell type"
(recall, and whether the label was emitted at all); this one answers "how close was it when it
missed", by carrying the Cell Ontology Lin similarity down to the level of a single true class.

The distinction matters for Results R3. A method scoring recall 0 on a rare type has either
produced an unrelated label or produced a coarser relative of the right one, and only the graded
score separates those. Aggregate-level Lin cannot: it is a per-cell mean over the whole run.

WHY THIS IS A SEPARATE SCRIPT. ``Results/`` carries ``true_labels_scored.csv`` for the baselines
and the scLLMs but for NO general-purpose LLM run, so the graded scores simply do not exist on
disk for the family the section is about. Rather than re-run the pipeline (whose publishDir would
overwrite ``Results/``), this loads the CL ontology once and reuses ``bin/evaluate.py``'s
preprocessing -- trust rule, coverage gate, symmetric CL canonicalization -- and its ontology
scoring, then aggregates per TRUE CLASS instead of over all cells.

VALIDATED against both existing artifacts, 2026-08-28:
  * overall Lin reproduces the aggregate snapshot to max |diff| 0.0011 over all 162 runs;
  * per-class recall reproduces ``cache/rare_type_outcomes_1000_stratified.csv`` to 4 decimals
    for every method group. The two tables are independent computations that agree, and the
    figure builder joins them on (dataset, cell type) rather than recomputing either.

Needs ``pronto`` and the cached CL ontology, so run it in the evaluation environment:
    conda run -n evaluate python bin/per_class_scores.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_OUT = "cache/per_class_scores_1000_stratified.csv"
DEFAULT_PRED_GLOB = "cache/models/predictions/*/*/1000_stratified/*_mapped.csv"


def model_name(wrapper: str, stem: str) -> str:
    """Recover the scored model name from the wrapper directory and the file stem.

    The OpenRouter wrappers hold one file per served model
    (``predictions_openai--gpt-5.5_mapped.csv``), so the wrapper alone is not the model.
    """
    if not stem.startswith("predictions_") or stem == "predictions":
        return wrapper
    sub = stem[len("predictions_"):]
    if "--" in sub:
        sub = sub.split("--", 1)[1]
    for suffix in ("_unblind", "_blind"):
        if wrapper.endswith(suffix):
            return f"{sub}{suffix}"
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-glob", default=DEFAULT_PRED_GLOB)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--min-scoring-coverage", type=float, default=0.9,
                    help="matches params.min_scoring_coverage in nextflow.config")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)

    from evaluate import (load_ontology_context, load_review_status, prepare_for_scoring,
                          trusted_answers)
    from util.cell_ontology_similarity import compute_ontology_similarities, download_cl_ontology

    label_index, taxonomy = load_ontology_context(download_cl_ontology(cache_dir))
    reviewed, review_map = load_review_status(cache_dir)
    print(f"  {len(reviewed)} reviewed mappings", flush=True)

    files = sorted(glob.glob(args.pred_glob))
    print(f"{len(files)} mapped prediction files", flush=True)
    rows, gated = [], 0
    for i, path in enumerate(files, 1):
        p = Path(path)
        wrapper, dataset = p.parts[-4], p.parts[-3]
        model = model_name(wrapper, p.name.replace("_mapped.csv", ""))
        frame = pd.read_csv(path)
        if not {"true_labels", "predicted_labels"} <= set(frame.columns):
            continue
        raw_col = ("original_predicted_labels" if "original_predicted_labels" in frame.columns
                   else "predicted_labels")
        _, trusted = trusted_answers(frame[raw_col], label_index, reviewed)
        coverage = trusted.mean() if len(frame) else 0.0
        if coverage < args.min_scoring_coverage:
            gated += 1
            print(f"  [{i}/{len(files)}] GATED {dataset}/{model} coverage={coverage:.3f}",
                  flush=True)
            continue
        frame, _, _ = prepare_for_scoring(frame, trusted, ["true_labels"], label_index, review_map)
        pairs, avg = compute_ontology_similarities(frame, "true_labels", "predicted_labels",
                                                   label_index, taxonomy)
        pairs["hit"] = (pairs["true_labels"] == pairs["predicted_labels"]).astype(float)
        for cell_type, group in pairs.groupby("true_labels"):
            weights = group["cell_count"]
            rows.append({
                "dataset": dataset, "model": model, "cell_type": cell_type,
                "n_cells": int(weights.sum()),
                **{name: float(np.average(group[col], weights=weights)) for name, col in
                   (("recall", "hit"), ("agreement", "agreement"), ("lin", "lin_similarity"),
                    ("rada", "rada_similarity"))},
            })
        rows.append({"dataset": dataset, "model": model, "cell_type": "__OVERALL__",
                     "n_cells": int(len(frame)),
                     "recall": float((frame["true_labels"] == frame["predicted_labels"]).mean()),
                     "agreement": avg["agreement_score"],
                     "lin": avg["lin_avg_similarity"], "rada": avg["rada_avg_similarity"]})
        print(f"  [{i}/{len(files)}] {dataset}/{model}  lin={avg['lin_avg_similarity']:.4f}",
              flush=True)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_types = out.loc[out["cell_type"] != "__OVERALL__", "cell_type"].nunique()
    print(f"\n{len(out)} rows | {out['model'].nunique()} models | {n_types} cell types "
          f"| {gated} run(s) gated below coverage\n-> {args.out}")


if __name__ == "__main__":
    main()
