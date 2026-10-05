#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import argparse
import json
import sys
import re

import pronto

from util.plots import UMAPCellPlotter  # Replace with correct import if needed
from util.cell_ontology_similarity import (
    SIMILARITY_KEYS, CellTaxonomy, build_label_index, compute_ontology_similarities,
    download_cl_ontology, map_string_to_term,
)
from util.label_normalization import EMPTY_PREDICTION_LABEL
from util.ontology_mapper import load_ontology_mappings_with_review_status
from util.text_embedding_similarity import (
    compute_text_embedding_similarities, load_text_encoder, reference_baseline,
)


# Placeholder for a prediction that has no usable label: a blank answer, or one that is neither a
# CL term nor a reviewed mapping. It matches no true label, so it counts as wrong in every metric.
UNMAPPED = "__unmapped__"


def _nan_metrics():
    """Metric dict with all core scores NaN — written when a file is below the scoring-coverage floor
    so the aggregated table carries no number computed on too little human-verified data."""
    return {k: np.nan for k in ("balanced_accuracy", "accuracy", *SIMILARITY_KEYS)}


def _canonicalize_to_cl(series, label_index, review_map):
    """Map each label to a stable canonical CL identity so true and predicted labels are
    compared on the SAME surface form.

    Upstream, predictions are ontology-mapped but ground truth is only lowercased, so exact
    string comparison mismatched on (a) case (``pancreatic a cell`` vs the CL-canonical
    ``pancreatic A cell``) and (b) truth labels that need mapping (``delta`` -> ``pancreatic D
    cell``). This resolves both by mapping EACH side identically. For every unique label:

      1. if it resolves to a CL term (case/synonym-insensitive) -> the term's canonical name
         (1:1 with the CL id, human-readable);
      2. else if a reviewed mapping resolves it -> that target's canonical CL name;
      3. else the lowercased/stripped string (identity-preserving fallback; never collapses
         two distinct classes into one).

    Returns ``(canonical_series, unresolved_labels)`` where ``unresolved_labels`` is the list of
    raw labels that fell through to the string fallback (expected empty for the current datasets).
    """
    s = series.astype(str)
    if label_index is None:
        return s, []
    canon = {}
    unresolved = []
    for raw in s.unique():
        term = map_string_to_term(raw, label_index)
        if term is None:
            tgt = review_map.get(raw.strip().lower())
            if tgt:
                term = map_string_to_term(tgt, label_index)
        if term is not None:
            canon[raw] = term.name
        else:
            canon[raw] = raw.strip().lower()
            unresolved.append(raw)
    return s.map(canon), unresolved

# -- Utility functions --

def parse_pred_input(pred_input):
    """Convert prediction input to list of file paths.
    
    Args:
        pred_input: Either a string (single file or comma-separated) or list
        
    Returns:
        List of file paths
    """
    if isinstance(pred_input, str):
        # Check if it's a comma-separated string
        if ',' in pred_input:
            paths = [path.strip() for path in pred_input.split(',')]
        else:
            paths = [pred_input]
        
        # Filter out empty strings and validate paths exist
        valid_paths = []
        for path in paths:
            if path and Path(path).exists():
                valid_paths.append(path)
            elif path:  # Non-empty but doesn't exist
                print(f"Warning: File not found: {path}")
        
        if not valid_paths:
            raise ValueError(f"No valid prediction files found in: {pred_input}")
        
        return valid_paths
    elif isinstance(pred_input, list):
        return pred_input
    else:
        raise ValueError("pred_input must be a string or list")


def load_ontology_context(cl_path):
    """Parse the CL ontology once per run. The label index resolves label strings to terms; the
    taxonomy carries the is_a graph and information content the similarity metrics need."""
    print(f"Loading CL ontology (once) from {cl_path}...")
    ont = pronto.Ontology(str(cl_path))
    return build_label_index(ont), CellTaxonomy.from_ontology(ont)


def load_review_status(cache_dir):
    """Reviewed labels and their targets, read from the live mappings CSV.

    Returns ``(reviewed_labels, review_map)``. A raw answer is trusted iff it is a CL term or one of
    ``reviewed_labels``. ``review_map`` resolves reviewed labels that are not CL terms themselves
    (e.g. "delta") to their chosen CL target, keyed by the lowercased label so the lowercased truth
    column resolves through it too.
    """
    mdf = load_ontology_mappings_with_review_status(cache_dir)
    if mdf is None or not {"predicted_label", "reviewed"} <= set(mdf.columns):
        return set(), {}
    is_reviewed = mdf["reviewed"].map(
        lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes"))
    rows = mdf.loc[is_reviewed]
    review_map = {}
    if "chosen_match" in rows.columns:
        review_map = {str(label).strip().lower(): str(target)
                      for label, target in zip(rows["predicted_label"], rows["chosen_match"])
                      if pd.notna(target) and str(target).strip()}
    return set(rows["predicted_label"].astype(str)), review_map


def trusted_answers(raw_labels, label_index, reviewed_labels):
    """Return ``(is_cl_term, trusted)`` masks: an answer is trusted when it is a CL term itself or
    has a human-reviewed mapping."""
    raw = raw_labels.astype(str)
    is_cl = raw.map(lambda s: map_string_to_term(s, label_index) is not None)
    return is_cl, is_cl | raw.isin(reviewed_labels)


def prepare_for_scoring(df, trusted, levels, label_index, review_map, pred_col="predicted_labels"):
    """Copy of ``df`` ready for scoring, plus diagnostics on the truth labels.

    Untrusted answers become UNMAPPED, so they count as wrong instead of leaving the denominator;
    dropping them would inflate every score. Truth and predictions are then canonicalized to CL
    names on both sides. Returns ``(df, unresolved, collisions)``: true labels that did not resolve
    to CL, and distinct true labels that canonicalization merged, both keyed by truth column.
    """
    df = df.copy()
    df.loc[~trusted, pred_col] = UNMAPPED
    for col in [*levels, pred_col, f"original_{pred_col}"]:
        if col in df.columns:
            df[col] = df[col].fillna(UNMAPPED).astype(str)
    df[pred_col], _ = _canonicalize_to_cl(df[pred_col], label_index, review_map)
    unresolved, collisions = {}, {}
    for col in levels:
        raw = df[col].copy()
        df[col], unres = _canonicalize_to_cl(raw, label_index, review_map)
        if unres:
            unresolved[col] = sorted(set(unres))
        pairs = pd.DataFrame({"raw": raw, "canon": df[col]}).drop_duplicates()
        merged = {canon: sorted(group) for canon, group in pairs.groupby("canon")["raw"]
                  if len(group) > 1}
        if merged:
            collisions[col] = merged
    return df, unresolved, collisions


def label_metrics(true_labels, pred_labels):
    """Exact-match metrics on canonical labels.

    Macro averages run over the true classes only. sklearn's default averages over every label that
    occurs, so each label a method emits that no cell carries adds a zero: that penalizes a broad
    vocabulary, not wrong cells. With ``labels`` given, the report omits accuracy, so it is computed
    separately. Macro recall then equals balanced accuracy.
    """
    return {
        "balanced_accuracy": balanced_accuracy_score(true_labels, pred_labels),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "summary_dict": classification_report(true_labels, pred_labels,
                                              labels=np.unique(true_labels),
                                              output_dict=True, zero_division=0),
    }


def run_eval(df, true_col, pred_col, output_dir, ont_ctx, text_scoring=None):
    # Truth and predictions arrive canonicalized and placeholder-filled (prepare_for_scoring).
    true_labels = df[true_col]
    pred_labels = df[pred_col]
    results = label_metrics(true_labels, pred_labels)

    label_index, taxonomy = ont_ctx
    pairs, similarities = compute_ontology_similarities(df, true_col, pred_col, label_index, taxonomy)
    results.update(similarities)
    scored_csv_path = output_dir / f"{true_col}_scored.csv"
    pairs.to_csv(scored_csv_path, index=False)
    print(f"Lin: {similarities['lin_avg_similarity']:.4f}  Rada: {similarities['rada_avg_similarity']:.4f}"
          f"  Agreement: {similarities['agreement_score']:.4f}"
          f"  ({len(pairs)} label pairs -> {scored_csv_path})")

    # Mapping-free text-encoder similarity (complementary to the ontology metrics). Scores the raw
    # model output and the ontology-mapped label on one scale, so it also covers free-text labels
    # and yields a mapping-lift diagnostic. Opt-in: text_scoring is (encoder, baseline) or None.
    if text_scoring is not None:
        try:
            # Raw = actual model output (post-normalization, pre-CL-mapping); mapped = pred_col.
            raw_pred_col = f"original_{pred_col}" if f"original_{pred_col}" in df.columns else pred_col
            te_combos, te_metrics = compute_text_embedding_similarities(
                df, true_col, raw_pred_col, pred_col, *text_scoring,
                no_answer=(UNMAPPED, EMPTY_PREDICTION_LABEL))
            results.update(te_metrics)
            te_csv_path = output_dir / f"{true_col}_text_emb_scored.csv"
            te_combos.to_csv(te_csv_path, index=False)
            print(f"Text-emb rescaled: raw={te_metrics['text_emb_norm_balanced_raw']:.4f}  "
                  f"mapped={te_metrics['text_emb_norm_balanced_mapped']:.4f}  "
                  f"lift={te_metrics['text_emb_norm_mapping_lift']:.4f}  -> {te_csv_path}")
        except Exception as e:
            print(f"Warning: Could not compute text-encoder similarities: {e}")
            print("Continuing without text-encoder metric...")

    # Generate confusion matrix. This is a secondary artifact: pathological predicted labels
    # (e.g. gibberish blind-LLM output with zero-width/unrenderable unicode) can crash the
    # matplotlib PDF backend, so a plotting failure must NOT discard the already-computed
    # metrics / JSON.
    output_plot_path = output_dir / f"{true_col}_confusion_matrix.pdf"
    title = f"Confusion Matrix {true_col} (Balanced Accuracy = {results['balanced_accuracy']:.2f})"
    try:
        plotter = UMAPCellPlotter()
        plotter.plot_cm(
            true_labels=true_labels,
            pred_labels=pred_labels,
            output_path=output_plot_path,
            title=title,
            normalize=True
        )
        print(f"Confusion matrix saved to: {output_plot_path}")
    except Exception as e:
        print(f"Warning: Could not render confusion matrix ({e}); keeping metrics anyway.")

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")

    return results

def main(args):
    
    # Parse prediction input (can be string or list)
    pred_files = parse_pred_input(args.pred)
    # Extract dataset and model from output path (format: dataset_model.json)
    # Remove .json extension first, then split
    filename_stem = Path(args.out).stem  # e.g., "HIHA_c2s_scale_2B"
    model = args.model
    if filename_stem.endswith(model):
        dataset = filename_stem[:-(len(model) + 1)]  # also remove the preceding _
    else:
        print(f"WARNING: json file name seems to be misformatted")
        dataset, model = filename_stem.split('_', 1)  # Split from left: ['HIHA', 'c2s_scale_2B']
    output_dir = Path(args.out).parent / dataset / model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up cache directory and CL ontology path
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path("cache")
    
    # Download CL ontology if not present
    try:
        cl_path = download_cl_ontology(cache_dir)
    except RuntimeError as e:
        raise RuntimeError(f"Error: {e}")
    
    print(f"Input argument: {args.pred}")
    print(f"Parsed prediction files: {pred_files}")
    print(f"Output directory: {output_dir}")
    print(f"Cell Ontology path: {cl_path}")

    # Parse the CL ontology ONCE for this whole run (shared across files + true-label levels).
    # Every metric needs it, so a failure here stops the run instead of producing partial scores.
    ont_ctx = load_ontology_context(cl_path)
    label_index = ont_ctx[0]

    # Load the text encoder once per process (weights cached under cache_dir), with its rescaling
    # baseline over the Cell Ontology's cell-type names (cached there too). Graceful: a failure
    # disables the metric but does not stop evaluation.
    text_scoring = None
    if args.use_text_encoder:
        try:
            print(f"Loading text encoder '{args.text_encoder_model}' (cache: {cache_dir})...")
            encoder = load_text_encoder(args.text_encoder_model, cache_dir=cache_dir)
            baseline = reference_baseline(encoder, ont_ctx[1].names.values(), cache_dir)
            print(f"Text-encoder baseline (mean cosine of two cell-type names): {baseline:.4f}")
            text_scoring = (encoder, baseline)
        except Exception as e:
            print(f"Warning: Could not load text encoder '{args.text_encoder_model}': {e}")
            print("Continuing without the text-encoder metric...")

    # Load the human-reviewed labels ONCE (the review filter moved from mapping to scoring).
    reviewed_labels, review_map = set(), {}
    try:
        reviewed_labels, review_map = load_review_status(cache_dir)
        print(f"Loaded {len(reviewed_labels)} reviewed mapping labels "
              f"({len(review_map)} with a chosen_match) for scoring-coverage + canonicalization.")
    except Exception as e:
        print(f"Warning: could not load review status ({e}); treating no labels as reviewed.")

    all_results = {}
    
    # Process each prediction file
    for pred_file in pred_files:
        print(f"\n{'='*60}")
        print(f"Processing prediction file: {pred_file}")
        print(f"{'='*60}")
        
        # Load the prediction file
        try:
            df = pd.read_csv(pred_file)
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
            continue
        
        # Check for predicted_labels column
        if 'predicted_labels' not in df.columns:
            print(f"Warning: Column 'predicted_labels' not found in {pred_file}. Skipping.")
            continue
        
        pred_col = 'predicted_labels'
        file_results = {}

        # Predictions are already normalized + ontology-mapped by bin/map_predictions.py (discovery
        # stage). The mapped CSV carries: predicted_labels (mapped/resolved) and
        # original_predicted_labels (normalized raw). The review-filter + coverage gate are applied
        # HERE (below), reading the live mappings CSV — mapping itself no longer decides what is kept.
        # A slim sidecar <stem>.stats.json (discovery diagnostics) is read for reference if present.
        stats_path = Path(pred_file).with_name(Path(pred_file).stem + ".stats.json")
        if stats_path.exists():
            try:
                with stats_path.open() as f:
                    file_results["filter_stats"] = json.load(f)
            except Exception as e:
                print(f"Warning: could not read sidecar stats {stats_path}: {e}")

        # --- Coverage (moved here from the mapping stage) -------------------------------------
        # A cell is trusted iff its RAW label (original_predicted_labels) is a direct CL term OR has
        # a reviewed=True mapping in the LIVE mappings CSV. coverage = trusted / total. Below
        # min_scoring_coverage we emit NO metrics (NaN) so the table never carries a number built
        # on too little human-verified data; otherwise untrusted cells are scored as wrong.
        raw_col = 'original_predicted_labels' if 'original_predicted_labels' in df.columns else pred_col
        n_total = len(df)
        # Label-diversity diagnostics, computed on the FULL file so they describe the model's
        # complete output:
        #   n_unique_pred_raw    = distinct RAW model outputs (before ontology mapping) -> the model's vocabulary
        #   n_unique_pred_mapped = distinct labels AFTER ontology mapping -> effective CL granularity used
        #   n_non_cl_labels      = distinct RAW labels that are not themselves CL terms
        raw_all = df[raw_col].astype(str)
        n_unique_pred_raw = int(raw_all.nunique())
        n_unique_pred_mapped = int(df[pred_col].astype(str).nunique())
        is_cl, trusted_mask = trusted_answers(raw_all, label_index, reviewed_labels)
        non_cl_ratio = float((~is_cl).mean()) if n_total else 0.0  # cells whose RAW label is not a CL term
        n_non_cl_labels = int(raw_all[~is_cl].nunique())
        n_trusted = int(trusted_mask.sum())
        coverage = (n_trusted / n_total) if n_total else 0.0
        file_results["coverage_stats"] = {
            "reviewed_fraction": round(coverage, 6), "n_trusted": n_trusted, "n_total": n_total,
            "non_cl_ratio": round(non_cl_ratio, 6), "n_non_cl_labels": n_non_cl_labels,
            "n_unique_pred_raw": n_unique_pred_raw, "n_unique_pred_mapped": n_unique_pred_mapped,
            "min_scoring_coverage": args.min_scoring_coverage,
        }
        gated = coverage < args.min_scoring_coverage
        print(f"Scoring coverage (trusted/total) = {n_trusted}/{n_total} = {coverage:.4f} "
              f"(floor {args.min_scoring_coverage}) -> {'GATED: NaN metrics' if gated else 'scoring all rows, untrusted as wrong'}")

        levels = [c for c in ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3'] if c in df.columns]

        if not gated:
            # Untrusted answers count as wrong, then truth and predictions are put on the same
            # canonical CL names. original_predicted_labels stays raw: it drives the coverage gate
            # and is the model's real output for the text-encoder metric.
            df, true_unresolved, true_collisions = prepare_for_scoring(
                df, trusted_mask, levels, label_index, review_map, pred_col)
            file_results["coverage_stats"]["true_unresolved"] = true_unresolved
            file_results["coverage_stats"]["true_coarsening_collisions"] = true_collisions
            if true_unresolved:
                _n = sum(len(v) for v in true_unresolved.values())
                print(f"  WARNING: {_n} true label(s) did not resolve to CL — scored on string "
                      f"fallback (may auto-mismatch mapped predictions): {true_unresolved}")
            if true_collisions:
                print(f"  WARNING: truth coarsening — distinct classes merged by canonicalization: "
                      f"{true_collisions}")

        # Process each level that exists in the dataframe
        for true_col in levels:
            print(f"\nProcessing {true_col}...")
            if gated:
                eval_results = _nan_metrics()
            else:
                eval_results = run_eval(df, true_col, pred_col, output_dir, ont_ctx, text_scoring)

            # Round numerical results for JSON output and store per true_col
            rounded_results = {}
            for key, value in eval_results.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    rounded_results[key] = round(value, 4)
                else:
                    rounded_results[key] = value
            rounded_results["reviewed_fraction"] = round(coverage, 6)
            rounded_results["non_cl_ratio"] = round(non_cl_ratio, 6)
            rounded_results["n_non_cl_labels"] = n_non_cl_labels
            rounded_results["n_unique_pred_raw"] = n_unique_pred_raw
            rounded_results["n_unique_pred_mapped"] = n_unique_pred_mapped
            rounded_results["subset_size"] = args.subset_size
            file_results[true_col] = file_results.get(true_col, {})
            file_results[true_col].update(rounded_results)
        
        # Check if any true_labels columns were found for this file
        if not file_results:
            print(f"Warning: No true_labels columns found in {pred_file}. Expected 'true_labels', 'true_labels_1', 'true_labels_2', or 'true_labels_3'.")
        else:
            # Use filename (without extension) as key for results. Strip the "_mapped" suffix
            # added by the mapping stage so the key matches the original prediction stem
            # ("predictions" / "predictions_<slug>") that aggregate.py expects.
            file_key = Path(pred_file).stem
            if file_key.endswith("_mapped"):
                file_key = file_key[:-len("_mapped")]
            all_results[file_key] = file_results

    # Check if any results were generated
    if not all_results:
        raise ValueError("No valid prediction files were processed successfully.")

    # Save results to JSON
    eval_path = args.out
    def _json_default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)
    with open(eval_path, "w+") as f:
        json.dump(all_results, f, indent=4, default=_json_default)

    # Print summary of filtering statistics
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    total_files = len(all_results)
    files_with_filtering = 0
    total_unreviewed_excluded = 0
    total_predictions_processed = 0
    
    for file_key, file_results in all_results.items():
        if "filter_stats" in file_results:
            filter_stats = file_results["filter_stats"]
            total_predictions_processed += filter_stats.get("total_predictions", 0)
            if filter_stats.get("filtered", False):
                files_with_filtering += 1
                total_unreviewed_excluded += filter_stats.get("unreviewed_predictions", 0)
    
    print(f"Files processed: {total_files}")
    print(f"Files with unreviewed predictions filtered: {files_with_filtering}")
    print(f"Total predictions processed: {total_predictions_processed}")
    print(f"Total unreviewed predictions excluded: {total_unreviewed_excluded}")
    
    if total_unreviewed_excluded > 0:
        exclusion_ratio = total_unreviewed_excluded / total_predictions_processed if total_predictions_processed > 0 else 0
        print(f"Overall exclusion ratio: {exclusion_ratio:.1%}")
        print(f"\n⚠️  WARNING: {total_unreviewed_excluded} predictions were excluded due to unreviewed ontology mappings")
        print(f"   Please review the ontology mappings CSV file and set 'reviewed' to 'True' for verified mappings")
        print(f"   Mapping file location: {cache_dir.parent / 'cell_type_ontology_mappings.csv'}")
    else:
        print(f"\n✅ All predictions used had reviewed ontology mappings")
    
    print(f"\nAll evaluation results saved to: {eval_path}")
    print(f"{'='*60}")
    
            

# -- CLI Interface --
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate label predictions")

    parser.add_argument("--pred", required=True, type=str, 
                       help="CSV file(s) with true and predicted labels. Can be a single file path or comma-separated list of file paths")
    parser.add_argument("--out", required=True, type=str, help="Path to results file")
    parser.add_argument("--model", required=True, type=str, help="Model name for accurate splitting")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for ontology files. If not provided, uses 'cache' in current directory")
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Number of cells the model was run on for this dataset (threaded from the "
                            "pipeline). Recorded per result so the aggregated table carries a subset_size "
                            "column. Optional; omitted -> null.")
    parser.add_argument("--min-scoring-coverage", type=float, default=0.9,
                       help="Minimum trusted-mapping coverage (fraction of cells whose raw label is a "
                            "direct CL term OR has reviewed=True) required to emit metrics. Below this, "
                            "all metrics are NaN (no number written). Default: 0.9")
    parser.add_argument("--use_text_encoder", type=lambda x: (str(x).lower() == 'true'),
                       default=False,
                       help="Also compute mapping-free text-encoder cosine similarity (default: False)")
    parser.add_argument("--text_encoder_model", type=str,
                       default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                       help="HuggingFace text encoder for the text-similarity metric "
                            "(default: SapBERT; e.g. 'sentence-transformers/all-MiniLM-L6-v2' for a zero-download fallback)")

    args = parser.parse_args()
    main(args)
