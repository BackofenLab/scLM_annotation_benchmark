#!/usr/bin/env python
"""Collect the scored runs into one tidy table for the figure.

One row per (arm, dataset, mode, gene variant, seed) with the metrics ``evaluate.py`` emitted, so
the repeats stay as individual rows rather than being averaged here. That is deliberate: the
manuscript's convention is to draw the three repeats as points rather than as an error bar, so
collapsing them at this stage would throw away exactly what the figure shows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

# Metrics that sit at the top level of the "true_labels" block.
METRICS = ("balanced_accuracy", "accuracy", "lin_avg_similarity", "rada_avg_similarity",
           "id_accuracy", "balanced_id_accuracy", "reviewed_fraction", "non_cl_ratio",
           "n_unique_pred_raw", "n_unique_pred_mapped")

# The F1 family is NOT top level: evaluate.py leaves it inside the sklearn classification report,
# under summary_dict["macro avg"]["f1-score"]. Listing them in METRICS would have yielded a column
# of Nones. Derived the same way bin/aggregate.py does (see its lines 227-251) so this study's
# macro F1 means exactly what the main table's does.
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
    scored_root = root / "results" / "scored"
    # Each scored directory holds TWO json files: evaluate.py's metrics and map_predictions.py's
    # `predictions_mapped.stats.json` sidecar. Globbing "*/*.json" picks up both and would emit a
    # duplicate row per run -- caught by the filename audit, which found 6 json files across 3
    # directories. Match the metrics file by name instead.
    for result in sorted(scored_root.glob("*/*.json")):
        if result.name.endswith(".stats.json"):
            continue
        arm, dataset, mode, variant, seed = result.parent.name.split("__")
        try:
            payload = json.loads(result.read_text())
        except json.JSONDecodeError:
            print(f"  [skip] unreadable {result}")
            continue

        # evaluate.py nests the metrics TWO levels down: {<prediction-file stem>: {"true_labels":
        # {...metrics...}}}, alongside sibling filter_stats / coverage_stats blocks. Verified
        # against a real scored file and against how bin/aggregate.py reads them
        # (`base = [annotation_key, "true_labels"]`).
        #
        # An earlier version of this searched one level down and silently produced a table of
        # Nones -- the worst failure mode here, since it looks like a successful run.
        block = None
        for outer in payload.values():
            if isinstance(outer, dict) and isinstance(outer.get("true_labels"), dict):
                block = outer["true_labels"]
                break
        if block is None or "balanced_accuracy" not in block:
            print(f"  [skip] {result.parent.name}: no metrics found. evaluate.py may have gated "
                  f"this run on scoring coverage, which yields NaN metrics by design.")
            continue
        row = {"arm": arm, "dataset": dataset, "mode": mode, "variant": variant,
               "seed": int(seed.removeprefix("seed"))}

        # Answer-format rates, from the sidecar the runners write next to predictions.csv.
        #
        # These belong in the results table, not just in per-file JSON, because for a BASE model
        # they are load-bearing: A2 (gemma-2-2b) returns prose rather than a bare label in 39.5% of
        # answers and echoes the prompt placeholder ("<cell type>", "<YOUR ANSWER HERE>") in another
        # 1.0%, where A3 does neither. Any A3-minus-A2 gap is therefore part biology and part
        # formatting, and reporting the score without these numbers would credit fine-tuning for
        # both. Carried for every arm so the comparison has a baseline.
        run_dir = (root / "results" / "predictions" / arm / dataset / mode / variant /
                   f"seed{int(seed.removeprefix('seed'))}")
        ps = None
        stats_file = run_dir / "parse_stats.json"
        if stats_file.exists():
            try:
                ps = json.loads(stats_file.read_text())
            except json.JSONDecodeError:
                print(f"  [warn] unreadable {stats_file}")
        elif (run_dir / "predictions.csv").exists():
            # Derived on the spot rather than demanding a re-run. These rates are a pure function of
            # predicted_labels, so recomputing them is exact -- not an estimate standing in for the
            # real thing. Needed because the sidecar was added after most files were written, and
            # regenerating 74,346 API calls to obtain a statistic already implied by the output on
            # disk would be indefensible.
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

    # Sentence-length diagnostics travel with the results: the HVG arm truncates on some datasets
    # (49% of HFRA cells at the 5,000-gene pool), so gene choice and sentence length are partly
    # confounded and the rate must be readable next to the score rather than looked up separately.
    stats = root / "results" / "sentence_stats.csv"
    if stats.exists():
        frame = frame.merge(pd.read_csv(stats)[["dataset", "variant", "median_sentence_len",
                                                "frac_below_cap"]],
                            on=["dataset", "variant"], how="left")

    # Cluster arms: how many calls actually bought the score, which is half of the per-cell vs
    # per-cluster result.
    #
    # Merged on dataset AND resolution, never on dataset alone. cluster_summary.csv holds one row
    # per (dataset, resolution), so a dataset-only merge silently DOUBLES every row in the table --
    # which is exactly what it did until the filename audit surfaced 6 rows where 3 were expected.
    # The resolution lives in the `variant` field for cluster runs ("res0.5"); per-cell rows have
    # no resolution and are left unmatched.
    summary = root / "results" / "clusters" / "cluster_summary.csv"
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

    # The comparability check the whole study rests on, surfaced automatically rather than left to
    # be remembered: A1 at M2 with expression-ranked sentences should reproduce the benchmark's
    # published gpt-oss-120b_unblind balanced accuracy of 0.554.
    anchor = frame[(frame.arm == "A1_llm_percell") & (frame["mode"] == "M2")
                   & (frame.variant == "expression_ranked")]
    if not anchor.empty:
        mean = anchor.groupby("seed")["balanced_accuracy"].mean().mean()
        delta = mean - 0.554
        # Tolerance is 0.06, not 0.03, and deliberately loose. The benchmark's 120B number came
        # almost entirely from DekaLLM (3,971 of 3,997 requests), which OpenRouter no longer
        # serves, so this study runs on DeepInfra instead. A difference here can be the provider
        # rather than a defect, and a tight band would flag that as a failure. It still catches a
        # gross error, which is what the check is for.
        verdict = ("consistent (within provider variation)" if abs(delta) < 0.06
                   else "OUT OF RANGE -- investigate before using any number")
        print(f"\nanchor check: A1/M2 balanced accuracy {mean:.3f} vs benchmark 0.554 "
              f"(delta {delta:+.3f}) -> {verdict}")
        print("  NB the benchmark's provider for this model (DekaLLM) is no longer available, so "
              "this is a consistency check, not an exact reproduction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
