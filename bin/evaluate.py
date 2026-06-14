#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, classification_report
import argparse
import json
import sys
import re

import pronto

from util.plots import UMAPCellPlotter  # Replace with correct import if needed
from util.cell_ontology_similarity import (
    compute_ontology_similarities, download_cl_ontology,
    build_label_index, compute_depths, descendant_counts, map_string_to_term,
)
from util.ontology_mapper import load_ontology_mappings_with_review_status
from util.text_embedding_similarity import load_text_encoder, compute_text_embedding_similarities


def _nan_metrics():
    """Metric dict with all core scores NaN — written when a file is below the scoring-coverage floor
    so the aggregated table carries no number computed on too little human-verified data."""
    return {k: np.nan for k in ("balanced_accuracy", "accuracy", "rada_avg_similarity",
                                "lin_avg_similarity", "id_accuracy", "balanced_id_accuracy")}

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
    """
    Parse the CL ontology ONCE and precompute the graph structures the similarity metrics
    need, so a single ``evaluate.py`` run no longer re-parses the 63MB OWL per true-label
    level (it used to parse it 3-9x). Returns a tuple threaded through ``run_eval`` ->
    ``compute_ontology_similarities``.
    """
    print(f"Loading CL ontology (once) from {cl_path}...")
    ont = pronto.Ontology(str(cl_path))
    label_index = build_label_index(ont)
    depth = compute_depths(ont)
    desc_count = descendant_counts(ont)
    max_desc = max(desc_count.values()) if desc_count else 0
    return (ont, label_index, depth, desc_count, max_desc)


def run_eval(df, true_col, pred_col, output_dir, cl_path=None, max_unmapped_ratio=0.0, cache_dir=None, text_encoder=None, ont_ctx=None):
    # NaN/blank labels are coerced to "__unmapped__" once in main() (before the true-label loop),
    # so df reaches here already cleaned for every level.
    true_labels = df[true_col]
    pred_labels = df[pred_col]  # Use mapped predicted labels directly

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)

    # Compute additional metrics
    summary_dict = classification_report(true_labels, pred_labels, output_dict=True)
    
    # Extract accuracy from summary_dict for easier access
    accuracy = summary_dict.get("accuracy", np.nan)

    # Initialize results dictionary
    results = {"balanced_accuracy": balanced_acc, "accuracy": accuracy, "summary_dict": summary_dict}
    
    # Compute ontology similarities if CL path is provided
    ontology_df = None
    if cl_path and Path(cl_path).exists():
        try:
            print(f"Computing ontology similarities using {cl_path}...")
            # Create a temporary dataframe for ontology computation
            # Use original predicted labels for the predicted_labels column in the scored CSV
            original_pred_col = f"original_{pred_col}"
            if original_pred_col in df.columns:
                # Use original predicted labels in the scored CSV, but mapped labels for similarity computation
                temp_df = df[[true_col, original_pred_col, pred_col]].copy()
                temp_df = temp_df.rename(columns={
                    true_col: "true_labels", 
                    original_pred_col: "predicted_labels",
                    pred_col: "mapped_predicted_labels"
                })
            else:
                # Fallback if no original column exists
                temp_df = df[[true_col, pred_col]].copy()
                temp_df = temp_df.rename(columns={true_col: "true_labels", pred_col: "predicted_labels"})
                temp_df["mapped_predicted_labels"] = temp_df["predicted_labels"]
            
            # Use mapped labels for similarity computation, but keep original labels in output
            similarity_pred_col = "mapped_predicted_labels" if "mapped_predicted_labels" in temp_df.columns else "predicted_labels"
            # Pass the preloaded ontology (parsed once in main) so this does not re-parse the OWL.
            ont, label_index, depth, desc_count, max_desc = ont_ctx if ont_ctx else (None, None, None, None, None)
            ontology_df, avg_similarities = compute_ontology_similarities(
                temp_df, cl_path, "true_labels", similarity_pred_col, max_unmapped_ratio,
                ont=ont, label_index=label_index, depth=depth, desc_count=desc_count, max_desc=max_desc,
            )
            
            # Fix the column structure: swap predicted_labels and mapped_labels if we used mapped labels for computation
            if "mapped_predicted_labels" in temp_df.columns and similarity_pred_col == "mapped_predicted_labels":
                # The ontology_df now has mapped labels in the column named by similarity_pred_col
                # We need to swap it so original predictions are in 'predicted_labels' and mapped are in 'mapped_labels'
                
                # Create a mapping from (true_label, mapped_pred) back to original_pred
                original_mapping = temp_df.groupby(['true_labels', 'mapped_predicted_labels'])['predicted_labels'].first().reset_index()
                original_mapping = original_mapping.rename(columns={'predicted_labels': 'original_predicted_labels'})
                
                # Merge to get original predicted labels
                # Note: ontology_df has the mapped labels in the column named by similarity_pred_col
                ontology_df = ontology_df.merge(
                    original_mapping, 
                    left_on=['true_labels', similarity_pred_col], 
                    right_on=['true_labels', 'mapped_predicted_labels'], 
                    how='left'
                )
                
                # Rename and reorganize columns: 
                # - Put original predictions in 'predicted_labels'
                # - Put mapped predictions in 'mapped_labels' 
                ontology_df['predicted_labels'] = ontology_df['original_predicted_labels']  # Use original labels
                ontology_df['mapped_labels'] = ontology_df[similarity_pred_col]  # Put mapped labels in mapped_labels column
                
                # Clean up temporary columns
                ontology_df = ontology_df.drop(columns=['original_predicted_labels', similarity_pred_col, 'mapped_predicted_labels'], errors='ignore')
                
                # Reorder columns to put predicted_labels right after mapped_labels
                # Get current column order and reorganize
                cols = list(ontology_df.columns)
                # Remove predicted_labels from its current position
                cols.remove('predicted_labels')
                # Find the position of mapped_labels and insert predicted_labels after it
                mapped_labels_idx = cols.index('mapped_labels')
                cols.insert(mapped_labels_idx + 1, 'predicted_labels')
                # Reorder the dataframe
                ontology_df = ontology_df[cols]
            
            # Add ontology results to main results
            results.update(avg_similarities)
            
            print(f"Rada average similarity: {avg_similarities.get('rada_avg_similarity', 'N/A'):.4f}")
            print(f"Lin average similarity: {avg_similarities.get('lin_avg_similarity', 'N/A'):.4f}")
            print(f"ID accuracy: {avg_similarities.get('id_accuracy', 'N/A'):.4f}")
            print(f"Balanced ID accuracy: {avg_similarities.get('balanced_id_accuracy', 'N/A'):.4f}")
            
            # Save scored CSV with ontology similarities (unique combinations)
            scored_csv_path = output_dir / f"{true_col}_scored.csv"
            
            # ontology_df now contains unique combinations with aggregated scores
            ontology_df.to_csv(scored_csv_path, index=False)
            print(f"Scored CSV with unique combinations saved to: {scored_csv_path}")
            print(f"Found {len(ontology_df)} unique label combinations")
            
        except Exception as e:
            raise RuntimeError(f"Could not compute ontology similarities: {e}")
            results["rada_avg_similarity"] = np.nan
            results["lin_avg_similarity"] = np.nan
            results["id_accuracy"] = np.nan
            results["balanced_id_accuracy"] = np.nan

    # Mapping-free text-encoder similarity (complementary to the ontology metrics).
    # Scores both the raw model output and the ontology-mapped label on one scale, so it
    # also covers free-text labels the ontology path drops, and yields a mapping-lift
    # diagnostic. Opt-in (text_encoder is None when --use_text_encoder is false).
    if text_encoder is not None:
        try:
            # Raw = actual model output (post-normalization, pre-CL-mapping); mapped = pred_col.
            raw_pred_col = f"original_{pred_col}" if f"original_{pred_col}" in df.columns else pred_col
            print(f"Computing text-encoder similarities ({text_encoder.model_name})...")
            te_combos, te_avgs = compute_text_embedding_similarities(
                df, true_col, raw_pred_col, pred_col, text_encoder
            )
            results.update(te_avgs)

            te_csv_path = output_dir / f"{true_col}_text_emb_scored.csv"
            te_combos.to_csv(te_csv_path, index=False)

            print(f"  Text-emb raw  : avg={te_avgs['text_emb_avg_similarity_raw']:.4f} "
                  f"norm={te_avgs['text_emb_norm_avg_raw']:.4f}")
            print(f"  Text-emb mapped: avg={te_avgs['text_emb_avg_similarity_mapped']:.4f} "
                  f"norm={te_avgs['text_emb_norm_avg_mapped']:.4f}")
            print(f"  Mapping lift  : {te_avgs['text_emb_mapping_lift']:.4f} "
                  f"(baseline={te_avgs['text_emb_baseline']:.4f})")
            print(f"  Text-emb scored CSV saved to: {te_csv_path}")
        except Exception as e:
            print(f"Warning: Could not compute text-encoder similarities: {e}")
            print("Continuing without text-encoder metric...")

    # Generate confusion matrix. This is a secondary artifact: pathological predicted labels
    # (e.g. gibberish blind-LLM output with zero-width/unrenderable unicode) can crash the
    # matplotlib PDF backend, so a plotting failure must NOT discard the already-computed
    # metrics / JSON.
    output_plot_path = output_dir / f"{true_col}_confusion_matrix.pdf"
    title = f"Confusion Matrix {true_col} (Balanced Accuracy = {balanced_acc:.2f})"
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

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

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
    print(f"Max unmapped ratio: {args.max_unmapped_ratio}")

    # Load the text encoder once per process (weights cached under cache_dir) so it is not
    # reloaded for every prediction file / true-label level. Graceful: a load failure
    # disables the metric but does not stop evaluation.
    text_encoder = None
    if args.use_text_encoder:
        try:
            print(f"Loading text encoder '{args.text_encoder_model}' (cache: {cache_dir})...")
            text_encoder = load_text_encoder(args.text_encoder_model, cache_dir=cache_dir)
        except Exception as e:
            print(f"Warning: Could not load text encoder '{args.text_encoder_model}': {e}")
            print("Continuing without the text-encoder metric...")
            text_encoder = None

    # Parse the CL ontology ONCE for this whole run (shared across files + true-label levels).
    ont_ctx = None
    if cl_path and Path(cl_path).exists():
        try:
            ont_ctx = load_ontology_context(cl_path)
        except Exception as e:
            print(f"Warning: Could not load CL ontology context: {e}; will lazy-load per call.")
            ont_ctx = None

    # Load the set of human-reviewed labels ONCE (the review-filter moved from mapping to scoring).
    # A predicted label is "trusted/scoreable" iff it is a direct CL term OR has reviewed=True here.
    reviewed_labels = set()
    try:
        _mdf = load_ontology_mappings_with_review_status(cache_dir)
        if _mdf is not None and 'predicted_label' in _mdf.columns and 'reviewed' in _mdf.columns:
            _rev = _mdf['reviewed'].map(
                lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ('true', '1', 'yes'))
            reviewed_labels = set(_mdf.loc[_rev, 'predicted_label'].astype(str))
        print(f"Loaded {len(reviewed_labels)} reviewed mapping labels for the scoring-coverage filter.")
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

        # --- Coverage filter (moved here from the mapping stage) -------------------------------
        # A cell is "trusted/scoreable" iff its RAW label (original_predicted_labels) is a direct CL
        # term OR has a reviewed=True mapping in the LIVE mappings CSV. coverage = trusted / total.
        # Below min_scoring_coverage we emit NO metrics (NaN) so the table never carries a number
        # built on too little human-verified data; otherwise we score the trusted rows only.
        raw_col = 'original_predicted_labels' if 'original_predicted_labels' in df.columns else pred_col
        label_index = ont_ctx[1] if ont_ctx else None
        n_total = len(df)
        # Label-diversity diagnostics, computed on the FULL file (pre-gating, pre-trust-filter) so they
        # describe the model's complete output:
        #   n_unique_pred_raw    = distinct RAW model outputs (before ontology mapping) -> the model's vocabulary
        #   n_unique_pred_mapped = distinct labels AFTER ontology mapping -> effective CL granularity used
        #   n_non_cl_labels      = distinct RAW labels that are not themselves CL terms (needs the label index)
        _raw_all = df[raw_col].astype(str)
        n_unique_pred_raw = int(_raw_all.nunique())
        n_unique_pred_mapped = int(df[pred_col].astype(str).nunique())
        if label_index is not None and n_total > 0:
            _raw = _raw_all
            _is_cl = _raw.map(lambda s: map_string_to_term(s, label_index) is not None)  # raw label is a CL term
            trusted_mask = _is_cl | _raw.isin(reviewed_labels)
            non_cl_ratio = float((~_is_cl).mean())  # fraction of cells whose RAW label is NOT a CL term
            n_non_cl_labels = int(_raw[~_is_cl].nunique())  # distinct non-CL raw labels (subset of the raw vocab)
        else:
            trusted_mask = pd.Series(True, index=df.index)  # ontology/mapping unavailable -> score all
            non_cl_ratio = 0.0
            n_non_cl_labels = None  # cannot determine CL membership without the label index
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
              f"(floor {args.min_scoring_coverage}) -> {'GATED: NaN metrics' if gated else 'scoring trusted rows'}")

        levels = [c for c in ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3'] if c in df.columns]

        if not gated:
            df = df[trusted_mask].copy()
            # Coerce blank/NaN labels to an explicit placeholder ONCE for all true-label levels.
            # Some models emit gibberish that normalizes to an empty string -> NaN; left as float NaN
            # it crashes sklearn metrics (float/str mix in np.unique) and the ontology/text steps.
            for _c in ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3',
                       pred_col, f"original_{pred_col}"]:
                if _c in df.columns:
                    df[_c] = df[_c].fillna("__unmapped__").astype(str)

        # Process each level that exists in the dataframe
        for true_col in levels:
            print(f"\nProcessing {true_col}...")
            if gated:
                eval_results = _nan_metrics()
            else:
                # Run evaluation with mapped predicted labels (trusted rows only)
                eval_results = run_eval(df, true_col, pred_col, output_dir, cl_path, args.max_unmapped_ratio, cache_dir, text_encoder=text_encoder, ont_ctx=ont_ctx)

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
    parser.add_argument("--max-unmapped-ratio", type=float, default=0.0,
                       help="Maximum acceptable ratio of unmapped cell types (0.0-1.0). Default: 0.0 (all must be mapped)")
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
