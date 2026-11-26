#!/usr/bin/env python
"""Map the answers of a run to Cell Ontology terms.

For every prediction file (columns ``cell_id``, ``predicted_labels`` and, in a benchmark run,
``true_labels``) this writes

  <stem>_mapped.csv        the rows with ``predicted_labels`` mapped and the normalized answer
                           in ``original_predicted_labels``; the input of evaluate.py
  <stem>_annotations.csv   per cell the answer, the cell type, its Cell Ontology id and how the
                           answer was resolved
  <stem>_annotations.json  the number of cells per way of resolving

An answer that names a Cell Ontology term is kept. Any other answer is looked up in the mapping
table. Answers the table does not hold yet are appended to it as unreviewed rows unless
``--append false`` is given.

    map_predictions.py --pred predictions.csv --method celltypist_unconstrained --dataset HFRA \
        --out-dir mapped --mappings resources/cell_type_ontology_mappings.csv --cache_dir cache
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import pronto

from util.cell_ontology_similarity import (
    CL_RELEASE, build_label_index, download_cl_ontology, map_string_to_term,
)
from util.label_normalization import (
    EMPTY_PREDICTION_LABEL, normalize_predicted_labels, normalize_true_labels,
)
from util.ontology_mapping import load_review_status, read_mappings, resolve_labels
from util.runs import mapping_origin

PRED_COL = "predicted_labels"
TRUE_COL = "true_labels"
# How an answer was resolved, in the order in which the rules apply.
ONTOLOGY_TERM, REVIEWED, UNREVIEWED, UNMAPPED = (
    "ontology term", "reviewed mapping", "unreviewed mapping", "unmapped")


def boolean(text) -> bool:
    return str(text).lower() == "true"


def annotations(mapped: pd.DataFrame, label_index, mappings) -> pd.DataFrame:
    """Per cell the answer, its cell type and Cell Ontology id, and how it was resolved."""
    reviewed, _ = load_review_status(mappings)
    in_table = set(read_mappings(mappings)["predicted_label"])
    answers = mapped[f"original_{PRED_COL}"].astype(str)
    terms = {label: map_string_to_term(label, label_index) for label in mapped[PRED_COL].unique()}

    def resolved_by(answer):
        if map_string_to_term(answer, label_index) is not None:
            return ONTOLOGY_TERM
        if answer in reviewed:
            return REVIEWED
        return UNREVIEWED if answer in in_table else UNMAPPED

    term = mapped[PRED_COL].map(terms)
    table = pd.DataFrame({
        "answer": answers,
        "cell_type": term.map(lambda t: t.name if t is not None else ""),
        "cell_ontology_id": term.map(lambda t: t.id if t is not None else ""),
        "mapping": answers.map({answer: resolved_by(answer) for answer in answers.unique()}),
    })
    if "cell_id" in mapped.columns:
        table.insert(0, "cell_id", mapped["cell_id"])
    return table


def map_file(path, method, dataset, out_dir, label_index, mappings, cache_dir, append):
    frame = pd.read_csv(path)
    if PRED_COL not in frame.columns:
        raise SystemExit(f"{path}: no column '{PRED_COL}'")
    stem = Path(path).stem
    frame[PRED_COL] = normalize_predicted_labels(frame[PRED_COL])
    if TRUE_COL in frame.columns:
        frame[TRUE_COL] = normalize_true_labels(frame[TRUE_COL])
    frame[f"original_{PRED_COL}"] = frame[PRED_COL]
    frame[PRED_COL], replaced = resolve_labels(
        frame[PRED_COL].tolist(), label_index, mappings, cache_dir,
        origin=mapping_origin(method, stem, dataset), append=append)
    frame.to_csv(out_dir / f"{stem}_mapped.csv", index=False)

    table = annotations(frame, label_index, mappings)
    table.to_csv(out_dir / f"{stem}_annotations.csv", index=False)
    counts = table["mapping"].value_counts()
    summary = {
        "method": method, "dataset": dataset, "prediction_file": Path(path).name,
        "cells": len(table),
        "cell_ontology_release": CL_RELEASE,
        "answers": {way: int(counts.get(way, 0))
                    for way in (ONTOLOGY_TERM, REVIEWED, UNREVIEWED, UNMAPPED)},
        "empty_answers": int((table["answer"] == EMPTY_PREDICTION_LABEL).sum()),
        "distinct_answers": int(table["answer"].nunique()),
        "distinct_cell_types": int(table.loc[table["cell_type"] != "", "cell_type"].nunique()),
    }
    (out_dir / f"{stem}_annotations.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{path}: {len(frame)} cells, {len(replaced)} label(s) replaced by their match, "
          f"{summary['answers'][UNREVIEWED] + summary['answers'][UNMAPPED]} cell(s) without a "
          "reviewed cell type")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", required=True,
                        help="prediction file, or several separated by commas")
    parser.add_argument("--method", required=True, help="name of the method folder in models/")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mappings", required=True, help="the mapping table")
    parser.add_argument("--cache_dir", default="cache",
                        help="where the Cell Ontology release is stored")
    parser.add_argument("--append", type=boolean, default=True,
                        help="add answers that are new to the mapping table as unreviewed rows")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    label_index = build_label_index(pronto.Ontology(str(download_cl_ontology(cache_dir))))
    for path in [p.strip() for p in args.pred.split(",") if p.strip()]:
        map_file(path, args.method, args.dataset, out_dir, label_index, args.mappings, cache_dir,
                 args.append)


if __name__ == "__main__":
    main()
