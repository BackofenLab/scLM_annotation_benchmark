#!/usr/bin/env python
"""Collect the scored runs of the context ablation into ``results/context_ablation.csv``.

One row per arm, dataset, mode, gene variant and seed, with the metrics of ``bin/evaluate.py``,
the answer-format rates of ``parse_stats.json``, the sentence statistics of ``sentences.py`` and,
for the cluster arms, the number of clusters among the eval cells. Prints a check of A1 at M2
against the benchmark's balanced accuracy of gpt-oss-120b with the list.

    python src/aggregate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

# Metrics at the top level of the "metrics" block of evaluate.py's JSON.
METRICS = ("balanced_accuracy", "accuracy", "agreement_score", "lin_avg_similarity",
           "rada_avg_similarity", "reviewed_fraction", "non_cl_ratio", "n_unique_pred_raw",
           "n_unique_pred_mapped")

# Precision, recall and F1 from the classification report (summary_dict), as in bin/aggregate.py.
REPORT_METRICS = {
    "macro_precision": ("macro avg", "precision"),
    "macro_recall": ("macro avg", "recall"),
    "macro_f1": ("macro avg", "f1-score"),
    "weighted_precision": ("weighted avg", "precision"),
    "weighted_recall": ("weighted avg", "recall"),
    "weighted_f1": ("weighted avg", "f1-score"),
}


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def collect(root, cfg):
    rows = []
    tree = root / cfg["studies"]["ablation"]
    for result in sorted((tree / "scored").glob("*/*__sub*.json")):
        arm, dataset, mode, variant, seed = result.parent.name.split("__")
        block = json.loads(result.read_text()).get("metrics") or {}
        if "balanced_accuracy" not in block:
            print(f"  [skip] {result.parent.name}: no metrics found")
            continue
        row = {"arm": arm, "dataset": dataset, "mode": mode, "variant": variant,
               "seed": int(seed.removeprefix("seed"))}

        # Answer-format rates from parse_stats.json, which the runners write next to
        # predictions.csv.
        run_dir = (tree / "predictions" / arm / dataset / mode / variant /
                   f"seed{int(seed.removeprefix('seed'))}")
        ps = None
        stats_file = run_dir / "parse_stats.json"
        if stats_file.exists():
            try:
                ps = json.loads(stats_file.read_text())
            except json.JSONDecodeError:
                print(f"  [warn] unreadable {stats_file}")
        elif (run_dir / "predictions.csv").exists():
            # Without parse_stats.json, the same rates from predictions.csv.
            pred = pd.read_csv(run_dir / "predictions.csv")["predicted_labels"]
            text = pred.astype(str)
            ps = {"n": len(pred),
                  "empty": int((text.str.strip() == "").sum() + pred.isna().sum()),
                  "over_8_words": int((text.str.split().str.len() > 8).sum())}
        if ps:
            n = ps.get("n") or 0
            row["n_predictions"] = n
            row["frac_empty"] = (ps.get("empty") or 0) / n if n else None
            row["frac_over_8_words"] = (ps.get("over_8_words") or 0) / n if n else None

        row.update({m: block.get(m) for m in METRICS})
        report = block.get("summary_dict") or {}
        for name, (group, field) in REPORT_METRICS.items():
            row[name] = (report.get(group) or {}).get(field)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)

    # Sentence statistics of sentences.py, per dataset and gene variant.
    stats = tree / "sentence_stats.csv"
    if stats.exists():
        frame = frame.merge(pd.read_csv(stats)[["dataset", "variant", "median_sentence_len",
                                                "frac_below_cap"]],
                            on=["dataset", "variant"], how="left")

    # Clusters among the eval cells, for the cluster arms. cluster_summary.csv has one row per
    # dataset and resolution; a cluster run's variant is its resolution ("res0.5"), and per-cell
    # rows stay unmatched.
    summary = tree / "clusters" / "cluster_summary.csv"
    if summary.exists():
        frame["resolution"] = pd.to_numeric(
            frame["variant"].astype(str).str.extract(r"^res(.+)$", expand=False), errors="coerce")
        clusters = pd.read_csv(summary)[["dataset", "resolution", "n_clusters_in_eval"]]
        before = len(frame)
        frame = frame.merge(clusters, on=["dataset", "resolution"], how="left")
        if len(frame) != before:
            raise SystemExit(f"cluster merge changed the row count ({before} -> {len(frame)}); "
                             f"cluster_summary.csv has duplicate (dataset, resolution) keys")
    return frame


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--config", default=str(root / "config.yml"))
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    frame = collect(root, cfg)
    if frame.empty:
        raise SystemExit("nothing scored yet; run score.py first")

    out = root / cfg["output"]["aggregate_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"{len(frame)} rows -> {out}")
    print(f"  arms={frame['arm'].nunique()} datasets={frame['dataset'].nunique()} "
          f"modes={sorted(frame['mode'].unique())} variants={sorted(frame['variant'].unique())}")

    # A1 at M2 with expression-ranked sentences against 0.554, the benchmark's mean balanced
    # accuracy of gpt-oss-120b with the list on the four 1,000-cell test sets.
    anchor = frame[(frame.arm == "A1_llm_percell") & (frame["mode"] == "M2")
                   & (frame.variant == "expression_ranked")]
    if not anchor.empty:
        mean = anchor.groupby("seed")["balanced_accuracy"].mean().mean()
        delta = mean - 0.554
        # The tolerance of 0.06 allows for the providers, which differ between the two runs.
        verdict = ("consistent (within provider variation)" if abs(delta) < 0.06
                   else "OUT OF RANGE -- investigate before using any number")
        print(f"\nanchor check: A1/M2 balanced accuracy {mean:.3f} vs benchmark 0.554 "
              f"(delta {delta:+.3f}) -> {verdict}")
        print("  Different providers served the two runs, so this is a consistency check, not an "
              "exact reproduction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
