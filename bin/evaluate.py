#!/usr/bin/env python
"""Score the mapped predictions of one method on one dataset.

Reads the ``*_mapped.csv`` files that map_predictions.py writes. Every file is one run. For
each run this writes, named ``<dataset>__<model>__sub<size>``:

  <run>.json                   metrics, scoring coverage and the identity of the run
  <run>_pairs.csv              cells per pair of true and predicted cell type, with the ontology
                               scores of the pair
  <run>_per_class.csv          recall and ontology scores per true cell type
  <run>_confusion_matrix.pdf
  <run>_text_similarity.csv    only with --use_text_encoder true

An answer is trusted if it names a Cell Ontology term or has a reviewed row in the mapping
table. Untrusted answers count as wrong. A run in which fewer than ``--min-scoring-coverage`` of
the answers are trusted gets no metrics.

    evaluate.py --pred mapped/predictions_mapped.csv --method celltypist_blind --dataset HFRA \
        --subset-size 1000 --out-dir scores --mappings resources/cell_type_ontology_mappings.csv
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pronto
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from util.cell_ontology_similarity import (
    ONTOLOGY_KEYS, CellTaxonomy, build_label_index, compute_ontology_similarities,
    download_cl_ontology, map_string_to_term,
)
from util.label_normalization import EMPTY_PREDICTION_LABEL, label_key
from util.ontology_mapping import canonical_names, load_review_status
from util.plots import plot_confusion_matrix
from util.runs import prediction_stem, run_name
from util.text_embedding_similarity import (
    compute_text_embedding_similarities, load_text_encoder, reference_baseline,
)

TRUE_COL = "true_labels"
PRED_COL = "predicted_labels"
RAW_COL = "original_predicted_labels"
# Placeholder of an answer without a usable cell type. It equals no true label.
UNMAPPED = "__unmapped__"
# Reviewed mappings send answers that name no cell type to the root of the ontology.
CL_ROOT_NAME = "cell"
CL_ID = re.compile(r"cl:\d{7}", re.IGNORECASE)
OVERALL = "__OVERALL__"


def boolean(text) -> bool:
    return str(text).lower() == "true"


def load_ontology_context(cl_path):
    """The label index and the taxonomy of the Cell Ontology file."""
    ontology = pronto.Ontology(str(cl_path))
    return build_label_index(ontology), CellTaxonomy.from_ontology(ontology)


def trusted_answers(raw_labels, label_index, reviewed_labels):
    """Masks ``(is_cl_term, trusted)``: an answer is trusted if it is a Cell Ontology term or has
    a reviewed mapping."""
    raw = raw_labels.astype(str)
    is_cl = raw.map(lambda s: map_string_to_term(s, label_index) is not None)
    return is_cl, is_cl | raw.isin(reviewed_labels)


def prepare_for_scoring(df, trusted, levels, label_index, review_map, pred_col=PRED_COL):
    """A copy of ``df`` with untrusted answers set to UNMAPPED and with true and predicted labels
    as Cell Ontology names.

    Returns ``(df, unresolved, collisions)``: per column of true labels, the labels without a
    term and the distinct labels that share one term.
    """
    df = df.copy()
    df.loc[~trusted, pred_col] = UNMAPPED
    for col in [*levels, pred_col, f"original_{pred_col}"]:
        if col in df.columns:
            df[col] = df[col].fillna(UNMAPPED).astype(str)
    df[pred_col], _ = canonical_names(df[pred_col], label_index, review_map)
    unresolved, collisions = {}, {}
    for col in levels:
        raw = df[col].copy()
        df[col], without_term = canonical_names(raw, label_index, review_map)
        if without_term:
            unresolved[col] = sorted(set(without_term))
        pairs = pd.DataFrame({"raw": raw, "canon": df[col]}).drop_duplicates()
        merged = {canon: sorted(group) for canon, group in pairs.groupby("canon")["raw"]
                  if len(group) > 1}
        if merged:
            collisions[col] = merged
    return df, unresolved, collisions


def label_metrics(true_labels, pred_labels):
    """Accuracy, balanced accuracy and the classification report over the true classes.

    The report is restricted to the true classes, so a label that no cell carries adds no zero
    to a macro average, and macro recall equals balanced accuracy.
    """
    return {
        "balanced_accuracy": balanced_accuracy_score(true_labels, pred_labels),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "summary_dict": classification_report(true_labels, pred_labels,
                                              labels=np.unique(true_labels),
                                              output_dict=True, zero_division=0),
    }


def ids_as_names(labels, label_index):
    """Replace each Cell Ontology id by the name of its term and keep every other label."""
    def name(label):
        term = map_string_to_term(label, label_index) if CL_ID.fullmatch(str(label).strip()) else None
        return term.name if term is not None else label
    return labels.map(name)


def pair_table(df, pairs, direct):
    """Cells per pair of true and predicted label, with the share of the true type in the test
    set and the number of cells whose answer needed no curated mapping."""
    n_direct = df.assign(direct=direct).groupby([TRUE_COL, PRED_COL])["direct"].sum()
    return pd.DataFrame({
        "true_label": pairs[TRUE_COL],
        "predicted_label": pairs[PRED_COL],
        "n_cells": pairs["cell_count"],
        "prevalence": (pairs.groupby(TRUE_COL)["cell_count"].transform("sum")
                       / pairs["cell_count"].sum()),
        "n_direct": [int(n_direct[key]) for key in zip(pairs[TRUE_COL], pairs[PRED_COL])],
        "lineage": pairs["lineage"],
        "lin_similarity": pairs["lin_similarity"],
        "rada_similarity": pairs["rada_similarity"],
        "agreement": pairs["agreement"],
    })


def per_class_table(df, pairs, averages):
    """Recall and mean ontology scores per true cell type, whether the type was ever predicted
    and, for a constrained baseline, whether the method can name it. The last row holds the
    values over all cells."""
    predicted = set(df[PRED_COL])
    reachable = (df["reachable"].astype(bool).groupby(df[TRUE_COL]).first()
                 if "reachable" in df.columns else {})
    hit = (pairs[TRUE_COL] == pairs[PRED_COL]).astype(float)
    rows = []
    for cell_type, group in pairs.assign(hit=hit).groupby(TRUE_COL):
        weights = group["cell_count"]
        rows.append({
            "cell_type": cell_type, "n_cells": int(weights.sum()),
            **{name: float(np.average(group[col], weights=weights)) for name, col in
               (("recall", "hit"), ("agreement", "agreement"), ("lin", "lin_similarity"),
                ("rada", "rada_similarity"))},
            "emitted": cell_type in predicted,
            "reachable": reachable.get(cell_type, ""),
        })
    rows.append({"cell_type": OVERALL, "n_cells": int(len(df)),
                 "recall": float((df[TRUE_COL] == df[PRED_COL]).mean()),
                 "agreement": averages["agreement_score"], "lin": averages["lin_avg_similarity"],
                 "rada": averages["rada_avg_similarity"], "emitted": "", "reachable": ""})
    return pd.DataFrame(rows)


def score_run(frame, label_index, taxonomy, reviewed_labels, review_map, min_coverage,
              text_scoring=None):
    """Score one mapped prediction file.

    Returns ``(coverage, metrics, tables)``. ``metrics`` is None and ``tables`` is empty if the
    run is below the coverage threshold. ``tables`` holds the pair table, the per-class table,
    the prepared frame and, with a text encoder, the text similarities.
    """
    raw_col = RAW_COL if RAW_COL in frame.columns else PRED_COL
    raw = frame[raw_col].astype(str)
    is_cl, trusted = trusted_answers(raw, label_index, reviewed_labels)
    n_total = len(frame)
    share_trusted = int(trusted.sum()) / n_total if n_total else 0.0
    coverage = {
        "reviewed_fraction": round(share_trusted, 6), "n_trusted": int(trusted.sum()),
        "n_total": n_total,
        "non_cl_ratio": round(float((~is_cl).mean()) if n_total else 0.0, 6),
        "n_non_cl_labels": int(raw[~is_cl].nunique()),
        "n_unique_pred_raw": int(raw.nunique()),
        "min_scoring_coverage": min_coverage,
    }
    if share_trusted < min_coverage:
        return coverage, None, {}

    # A Cell Ontology term and one of the dataset's own labels need no curated mapping.
    own_labels = set(frame[TRUE_COL].astype(str).map(label_key))
    direct = is_cl | raw.map(label_key).isin(own_labels)
    df, unresolved, collisions = prepare_for_scoring(
        frame, trusted, [TRUE_COL], label_index, review_map)
    coverage["true_unresolved"] = unresolved
    coverage["true_coarsening_collisions"] = collisions

    metrics = label_metrics(df[TRUE_COL], df[PRED_COL])
    pairs, averages = compute_ontology_similarities(df, TRUE_COL, PRED_COL, label_index, taxonomy)
    metrics.update(averages)
    tables = {"pairs": pair_table(df, pairs, direct),
              "per_class": per_class_table(df, pairs, averages), "frame": df}
    if text_scoring is not None:
        text_df = df.assign(**{raw_col: ids_as_names(df[raw_col], label_index)})
        tables["text_similarity"], text_metrics = compute_text_embedding_similarities(
            text_df, TRUE_COL, raw_col, PRED_COL, *text_scoring,
            no_answer=(UNMAPPED, EMPTY_PREDICTION_LABEL, CL_ROOT_NAME))
        metrics.update(text_metrics)
    return coverage, metrics, tables


def rounded(metrics):
    """Numbers rounded to four decimals; nested values are kept as they are."""
    return {key: round(value, 4) if isinstance(value, (int, float)) and not np.isnan(value)
            else value for key, value in metrics.items()}


def json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", required=True,
                        help="mapped prediction file, or several separated by commas")
    parser.add_argument("--method", required=True, help="name of the method folder in models/")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset-size", type=int, default=None,
                        help="number of cells in the test set")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mappings", required=True, help="the mapping table")
    parser.add_argument("--cache_dir", default="cache",
                        help="where the Cell Ontology release and the text encoder are stored")
    parser.add_argument("--min-scoring-coverage", type=float, default=0.9,
                        help="share of trusted answers below which a run gets no metrics")
    parser.add_argument("--use_text_encoder", type=boolean, default=False,
                        help="also score the similarity of label texts")
    parser.add_argument("--text_encoder_model",
                        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    args = parser.parse_args()

    out_dir, cache_dir = Path(args.out_dir), Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_index, taxonomy = load_ontology_context(download_cl_ontology(cache_dir))
    reviewed_labels, review_map = load_review_status(args.mappings)
    text_scoring = None
    if args.use_text_encoder:
        encoder = load_text_encoder(args.text_encoder_model, cache_dir=cache_dir)
        text_scoring = (encoder, reference_baseline(encoder, taxonomy.names.values(), cache_dir))

    for path in [p.strip() for p in args.pred.split(",") if p.strip()]:
        frame = pd.read_csv(path)
        missing = {TRUE_COL, PRED_COL} - set(frame.columns)
        if missing:
            raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
        stem = prediction_stem(path)
        model = run_name(args.method, stem)
        run = f"{args.dataset}__{model}__sub{args.subset_size}"
        coverage, metrics, tables = score_run(
            frame, label_index, taxonomy, reviewed_labels, review_map,
            args.min_scoring_coverage, text_scoring)

        if metrics is None:
            metrics = {key: np.nan for key in ("balanced_accuracy", "accuracy", *ONTOLOGY_KEYS)}
            print(f"{run}: {coverage['n_trusted']} of {coverage['n_total']} answers trusted, "
                  f"below {args.min_scoring_coverage}: no metrics")
        else:
            tables["pairs"].to_csv(out_dir / f"{run}_pairs.csv", index=False)
            tables["per_class"].to_csv(out_dir / f"{run}_per_class.csv", index=False)
            if "text_similarity" in tables:
                tables["text_similarity"].to_csv(out_dir / f"{run}_text_similarity.csv", index=False)
            try:
                plot_confusion_matrix(
                    tables["frame"][TRUE_COL], tables["frame"][PRED_COL],
                    out_dir / f"{run}_confusion_matrix.pdf",
                    f"Confusion Matrix {TRUE_COL} "
                    f"(Balanced Accuracy = {metrics['balanced_accuracy']:.2f})")
            except Exception as error:
                print(f"{run}: confusion matrix not drawn ({error})")
            print(f"{run}: balanced accuracy {metrics['balanced_accuracy']:.4f}, "
                  f"Lin {metrics['lin_avg_similarity']:.4f}")

        metrics = rounded(metrics)
        metrics.update({key: coverage[key] for key in
                        ("reviewed_fraction", "non_cl_ratio", "n_non_cl_labels",
                         "n_unique_pred_raw")})
        metrics["subset_size"] = args.subset_size
        result = {"dataset": args.dataset, "method": args.method, "model": model,
                  "subset_size": args.subset_size, "prediction_file": stem,
                  "coverage": coverage, "metrics": metrics}
        (out_dir / f"{run}.json").write_text(json.dumps(result, indent=4, default=json_default))


if __name__ == "__main__":
    main()
