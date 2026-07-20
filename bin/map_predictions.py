#!/usr/bin/env python
"""
Stage 1 of evaluation: normalize + review-filter + ontology-map predictions.

This is the *deterministic, cacheable* half of the old monolithic ``evaluate.py``. It is the
ONLY stage that reads/writes ``cache/cell_type_ontology_mappings.csv`` and performs the
stateful side-effect of appending genuinely-new unmapped labels for manual review.

For each input ``predictions*.csv`` it writes:
  - ``<stem>_mapped.csv``       : all rows + the columns ``predicted_labels`` (ontology-mapped),
                                  ``original_predicted_labels`` (normalized raw), and ``kept``
                                  (False for rows whose label has an unreviewed mapping, i.e.
                                  the rows the scoring stage must exclude).
  - ``<stem>_mapped.stats.json``: the exact ``filter_stats`` dict the scoring JSON needs, so
                                  scoring never has to touch the mappings CSV.

Splitting this out means re-running the pipeline (e.g. to iterate on a scoring metric) reuses
the cached ``*_mapped.csv`` instead of re-doing ontology mapping on unchanged predictions.

Mapping is applied to the *kept* rows only (mirroring the old behaviour where mapping ran on
the review-filtered frame), so both the scored numbers and the set of newly-discovered labels
are identical to the pre-split pipeline.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pronto

from util.cell_ontology_similarity import download_cl_ontology, build_label_index, map_string_to_term
from util.label_normalization import normalize_predicted_labels, normalize_true_labels
from util.ontology_helpers import apply_ontology_mapping_to_cell_type_list

TRUE_COLS = ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3']


def parse_pred_input(pred_input):
    """Convert prediction input (single path or comma-separated list) to a list of existing paths."""
    if isinstance(pred_input, list):
        return pred_input
    if not isinstance(pred_input, str):
        raise ValueError("pred_input must be a string or list")
    paths = [p.strip() for p in pred_input.split(',')] if ',' in pred_input else [pred_input]
    valid = []
    for p in paths:
        if p and Path(p).exists():
            valid.append(p)
        elif p:
            print(f"Warning: File not found: {p}")
    if not valid:
        raise ValueError(f"No valid prediction files found in: {pred_input}")
    return valid


def apply_ontology_mapping_to_predictions(df: pd.DataFrame, pred_col: str, cache_dir: Path,
                                          verbose: bool = True, origin: str = "unknown",
                                          label_index=None, discover_only: bool = False) -> pd.DataFrame:
    """
    Apply ontology mapping to predicted labels, backing up the raw labels into
    ``original_<pred_col>``. Lifted verbatim from the old ``evaluate.py`` so behaviour
    (including the append-new-labels side-effect) is unchanged. ``label_index`` is the
    preloaded ontology label index (parsed once in ``main``) so coverage checks do not
    re-parse the OWL per file.
    """
    if verbose:
        print(f"Applying ontology mapping to predicted labels in column '{pred_col}'...")

    original_pred_col = f"original_{pred_col}"
    df[original_pred_col] = df[pred_col].copy()

    predicted_labels = df[pred_col].tolist()
    try:
        mapped_labels, replacements = apply_ontology_mapping_to_cell_type_list(
            cell_types=predicted_labels,
            cache_dir=cache_dir,
            force_remapping=False,
            verbose=verbose,
            origin=origin,
            label_index=label_index,
            discover_only=discover_only,
        )
        df[pred_col] = mapped_labels
        if verbose and replacements:
            print(f"  Successfully mapped {len(replacements)} predicted label types")
            print(f"  Backup created in column '{original_pred_col}'")
        elif verbose:
            print("  No mappings were applied (all labels already mapped or no mappings available)")
    except Exception as e:
        if verbose:
            print(f"  Warning: Ontology mapping failed: {e}")
            print("  Continuing with original predicted labels...")

    return df


def _disabled_filter_stats(n):
    return {
        "total_predictions": n,
        "filtered_predictions": n,
        "unreviewed_predictions": 0,
        "unreviewed_ratio": 0.0,
        "unreviewed_labels": [],
        "filtered": False,
        "note": "Ontology mapping disabled",
    }


def map_one_file(pred_file, model, dataset, cache_dir, apply_mapping, out_dir,
                 label_index=None):
    """Map a single prediction CSV -> (<stem>_mapped.csv, <stem>_mapped.stats.json). Returns True on success."""
    pred_col = 'predicted_labels'
    print(f"\n{'='*60}\nMapping prediction file: {pred_file}\n{'='*60}")

    try:
        df = pd.read_csv(pred_file)
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return False

    if pred_col not in df.columns:
        print(f"Warning: Column '{pred_col}' not found in {pred_file}. Skipping.")
        return False

    # OpenRouter LLM wrappers: each LLM writes predictions_<SLUG>.csv into one model dir; tag
    # the origin with the slug so mapping rows record which specific LLM produced the label.
    file_origin = f"{model}_{dataset}"
    if 'openrouter' in model.lower():
        stem = Path(pred_file).stem
        if stem.startswith('predictions_'):
            file_origin = f"{model}_{stem[len('predictions_'):]}_{dataset}"

    # 1) Normalize (single source of truth; same helpers the scoring + coverage check use).
    print("Normalizing labels...")
    df[pred_col] = normalize_predicted_labels(df[pred_col])
    for col in TRUE_COLS:
        if col in df.columns:
            df[col] = normalize_true_labels(df[col])

    out = df.copy()
    out["original_predicted_labels"] = out[pred_col]  # normalized raw, for every row

    # Discovery of novel labels ALWAYS runs (append every non-CL, not-yet-in-CSV label to the
    # mappings CSV as reviewed=False for human review) -- this is independent of apply_mapping so
    # that marker methods on cellxgene sources (apply_ontology_mapping=false), whose free-text
    # labels carry no CL id, still surface for curation. What apply_mapping controls is only
    # whether the labels are then REWRITTEN to their chosen_match:
    #   apply_mapping=True  -> discovery + resolution (labels rewritten to chosen_match)
    #   apply_mapping=False -> discovery only (labels left as-is; mapped == raw)
    # The review-filter + coverage gate live in the scoring stage (bin/evaluate.py), which reads
    # the live `reviewed` flags, so a single pass discovers ALL non-CL labels regardless of how
    # many are still unreviewed. Never aborts: on failure we keep raw labels.
    mode = "discovery + resolution" if apply_mapping else "discovery only (mapped == raw)"
    print(f"Applying ontology mapping ({mode})...")
    try:
        out = apply_ontology_mapping_to_predictions(
            out, pred_col, cache_dir, verbose=True, origin=file_origin,
            label_index=label_index, discover_only=not apply_mapping,
        )
    except Exception as e:
        print(f"Warning: Could not apply ontology mapping: {e}; keeping raw labels.")
    if apply_mapping:
        filter_stats = {
            "total_predictions": len(df),
            "note": "discovery + resolution; review-filter + coverage gate applied at scoring (evaluate.py)",
        }
    else:
        filter_stats = _disabled_filter_stats(len(df))
        filter_stats["note"] = "discovery only (labels unchanged); novel labels appended for review"

    # "Needs human correction" diagnostic: fraction of ALL cells for which the model's RAW
    # prediction (normalized model output, BEFORE ontology mapping) is not itself a Cell-Ontology
    # term. These are the cells whose label a human had to curate/map by hand, so it measures how
    # much the scored results rely on the manual mapping rather than on labels the model already
    # emitted as valid CL terms. Computed on `original_predicted_labels` (NOT the mapped column,
    # which would hide every garbage label that a reviewed mapping rescued). Requires the CL label
    # index; with mapping disabled (label_index is None) we record nulls rather than a misleading 0.
    raw_col = "original_predicted_labels" if "original_predicted_labels" in out.columns else pred_col
    if label_index is not None:
        raw_labels = out[raw_col].astype(str)
        non_cl_mask = raw_labels.map(lambda s: map_string_to_term(s, label_index) is None)
        n_non_cl = int(non_cl_mask.sum())
        total = len(out)
        filter_stats["non_cl_predictions"] = n_non_cl
        filter_stats["non_cl_ratio"] = round(n_non_cl / total, 6) if total else 0.0
        filter_stats["non_cl_labels"] = sorted(raw_labels[non_cl_mask].unique().tolist())
    else:
        filter_stats["non_cl_predictions"] = None
        filter_stats["non_cl_ratio"] = None
        filter_stats["non_cl_labels"] = []

    stem = Path(pred_file).stem
    mapped_csv = out_dir / f"{stem}_mapped.csv"
    stats_json = out_dir / f"{stem}_mapped.stats.json"
    out.to_csv(mapped_csv, index=False)
    with stats_json.open("w") as f:
        json.dump(filter_stats, f, indent=2, default=lambda o: list(o) if isinstance(o, set) else str(o))
    print(f"Wrote {mapped_csv}  (rows={len(out)})")
    print(f"Wrote {stats_json}")
    return True


def main(args):
    pred_files = parse_pred_input(args.pred)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("cache")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the CL ontology is present (mapping/coverage need it). Cached after first download.
    try:
        cl_file_path = download_cl_ontology(cache_dir)
    except RuntimeError as e:
        raise RuntimeError(f"Error: {e}")

    # Parse the OWL + build the label index ONCE for this process, then reuse it across every
    # prediction file. The coverage check inside apply_ontology_mapping_to_cell_type_list used to
    # re-parse the 63MB OWL per file (e.g. once per OpenRouter slug file). Built unconditionally:
    # discovery of novel labels now runs even when apply_ontology_mapping=false (discover-only),
    # and the non-CL diagnostic needs the index too. A parse failure is non-fatal: fall back to
    # per-file parsing (label_index=None) to preserve old behaviour.
    label_index = None
    try:
        print(f"Parsing CL ontology once from {cl_file_path}...")
        label_index = build_label_index(pronto.Ontology(str(cl_file_path)))
    except Exception as e:
        print(f"Warning: could not preload ontology label index ({e}); will parse per file.")
        label_index = None

    any_ok = False
    for pred_file in pred_files:
        ok = map_one_file(pred_file, args.model, args.dataset, cache_dir,
                          args.apply_ontology_mapping, out_dir,
                          label_index=label_index)
        any_ok = any_ok or ok

    if not any_ok:
        raise SystemExit("No prediction files were mapped successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize + review-filter + ontology-map predictions (eval stage 1)")
    parser.add_argument("--pred", required=True, type=str,
                        help="Prediction CSV(s): single path or comma-separated list")
    parser.add_argument("--model", required=True, type=str, help="Model name (for mapping origin)")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name (for mapping origin / markers)")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to write *_mapped.csv + *_mapped.stats.json")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir (default: 'cache')")
    parser.add_argument("--apply_ontology_mapping", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Apply ontology mapping to predicted labels (default: True)")
    args = parser.parse_args()
    main(args)
