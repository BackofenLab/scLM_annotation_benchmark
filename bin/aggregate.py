#!/usr/bin/env python
"""Combine the scored runs into the result tables of a benchmark.

Reads the ``<run>.json`` files of evaluate.py and, next to each, ``<run>_pairs.csv`` and
``<run>_per_class.csv``. Writes into the output folder

  benchmark_metrics.csv   one row per dataset, model and test set size
  per_class_scores.csv    recall and ontology scores per true cell type and run
  confusion_pairs.csv     cells per pair of true and predicted cell type and run

    aggregate.py --input scores/*.json --output results
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.runs import setting_of  # noqa: E402
from util.text_embedding_similarity import TEXT_EMB_KEYS  # noqa: E402

RUN = ["dataset", "model", "subset_size"]
SCALAR_KEYS = (
    "balanced_accuracy", "accuracy", "agreement_score", "lin_avg_similarity",
    "rada_avg_similarity", "reviewed_fraction", "non_cl_ratio", "n_non_cl_labels",
    "n_unique_pred_raw", "n_unique_pred_mapped",
    *TEXT_EMB_KEYS,  # present only in runs scored with the text encoder
)
# Averages of the classification report, as (column, report row, report column).
REPORT_KEYS = tuple((f"{average}_{stat}", f"{average} avg", "f1-score" if stat == "f1" else stat)
                    for average in ("macro", "weighted")
                    for stat in ("precision", "recall", "f1"))
PER_CLASS_COLUMNS = ["dataset", "model", "cell_type", "n_cells", "recall", "agreement", "lin",
                     "rada", "subset_size", "emitted", "reachable"]
PAIR_COLUMNS = ["true_label", "predicted_label", "n_cells", "model", "dataset", "prevalence",
                "n_direct", "lineage", "subset_size"]


def metrics_row(result):
    metrics = result["metrics"]
    report = metrics.get("summary_dict") or {}
    return {
        **{key: result[key] for key in RUN},
        **{key: metrics.get(key) for key in SCALAR_KEYS},
        **{column: (report.get(row) or {}).get(stat) for column, row, stat in REPORT_KEYS},
    }


def side_table(path, result, columns):
    """A per-run table with the identity of the run added, or None if the run has none."""
    if not path.exists():
        return None
    table = pd.read_csv(path, float_precision="round_trip")
    return table.assign(**{key: result[key] for key in RUN})[columns]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", nargs="+", required=True, help="result files of evaluate.py")
    parser.add_argument("--output", required=True, help="folder for the tables")
    args = parser.parse_args()

    runs = [(Path(path), json.loads(Path(path).read_text())) for path in args.input]
    # Per-class and pair tables list the runs by test set size, method folder, dataset and file.
    runs.sort(key=lambda run: (run[1]["subset_size"], run[1]["method"], run[1]["dataset"],
                               run[1]["prediction_file"]))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame([metrics_row(result) for _, result in runs])
    values = [c for c in rows.columns if c not in RUN]
    rows[values] = rows[values].apply(pd.to_numeric, errors="coerce").astype(float)
    # A run scored twice is averaged. The mean also writes a negative zero as 0.0.
    metrics = rows.groupby(RUN, dropna=False)[values].mean().reset_index()
    metrics.insert(2, "mode", metrics["model"].map(setting_of))
    metrics.to_csv(out_dir / "benchmark_metrics.csv", index=False)

    for name, suffix, columns in (("per_class_scores", "_per_class", PER_CLASS_COLUMNS),
                                  ("confusion_pairs", "_pairs", PAIR_COLUMNS)):
        tables = [side_table(path.with_name(path.stem + suffix + ".csv"), result, columns)
                  for path, result in runs]
        tables = [table for table in tables if table is not None]
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(out_dir / f"{name}.csv", index=False)

    scored = metrics["balanced_accuracy"].notna().sum()
    print(f"{len(metrics)} runs, {scored} with metrics, written to {out_dir}")
    print(metrics.pivot_table(index=["subset_size", "dataset"], columns="model",
                              values="balanced_accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
