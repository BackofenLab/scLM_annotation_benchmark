#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, classification_report
import argparse
import json
import sys
import re

from util.plots import UMAPCellPlotter  # Replace with correct import if needed
from util.cell_ontology_similarity import compute_ontology_similarities, download_cl_ontology
from util.ontology_helpers import apply_ontology_mapping_to_cell_type_list
from util.ontology_mapper import filter_predictions_by_review_status

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

# -- Preprocessing utilities --
def remove_trailing_periods_from_predictions(df: pd.DataFrame, pred_col: str, verbose: bool = True) -> pd.DataFrame:
    """
    Remove trailing periods from predicted labels in a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictions
    pred_col : str
        Column name containing predicted labels
    verbose : bool, default True
        If True, print progress information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with trailing periods removed from predicted labels
    """
    if verbose:
        # Count labels with trailing periods before stripping
        labels_with_periods = df[pred_col].astype(str).str.endswith('.').sum()
        if labels_with_periods > 0:
            print(f"  Found {labels_with_periods} predicted labels with trailing periods - stripping them")
    
    # Strip trailing periods from predicted labels
    df[pred_col] = df[pred_col].astype(str).str.rstrip('.').replace('', np.nan)
    
    return df

# -- Ontology mapping utilities --
def apply_ontology_mapping_to_predictions(df: pd.DataFrame, pred_col: str, cache_dir: Path, verbose: bool = True, origin: str = "unknown") -> pd.DataFrame:
    """
    Apply ontology mapping to predicted labels in a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictions
    pred_col : str
        Column name containing predicted labels
    cache_dir : Path
        Cache directory for ontology files
    verbose : bool, default True
        If True, print progress information
    origin : str, default "unknown"
        Origin of the predicted labels (e.g., model name)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with mapped predicted labels and backup column
    """
    if verbose:
        print(f"Applying ontology mapping to predicted labels in column '{pred_col}'...")
    
    # Create backup column
    original_pred_col = f"original_{pred_col}"
    df[original_pred_col] = df[pred_col].copy()
    
    # Get predicted labels as list
    predicted_labels = df[pred_col].tolist()
    
    try:
        # Apply ontology mapping
        mapped_labels, replacements = apply_ontology_mapping_to_cell_type_list(
            cell_types=predicted_labels,
            cache_dir=cache_dir,
            force_remapping=False,
            verbose=verbose,
            origin=origin
        )
        
        # Update the dataframe
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

def run_eval(df, true_col, pred_col, output_dir, cl_path=None, max_unmapped_ratio=0.0, cache_dir=None):
    
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
            ontology_df, avg_similarities = compute_ontology_similarities(
                temp_df, cl_path, "true_labels", similarity_pred_col, max_unmapped_ratio, 
                cache_dir=Path(cache_dir) if cache_dir else None
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

    # Set up plotter
    plotter = UMAPCellPlotter()

    # Generate confusion matrix
    output_plot_path = output_dir / f"{true_col}_confusion_matrix.pdf"
    title = f"Confusion Matrix {true_col} (Balanced Accuracy = {balanced_acc:.2f})"
    
    plotter.plot_cm(
        true_labels=true_labels,
        pred_labels=pred_labels,
        output_path=output_plot_path,
        title=title,
        normalize=True
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Confusion matrix saved to: {output_plot_path}")
    
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

        # For OpenRouter LLM wrappers, multiple LLMs share one model directory and
        # each LLM writes its own predictions_<SLUG>.csv. Append the slug to the
        # origin so ontology-mapping rows record which specific LLM produced the
        # label, instead of all rows collapsing onto the wrapper name.
        file_origin = f"{model}_{dataset}"
        if 'openrouter' in model.lower():
            stem = Path(pred_file).stem
            if stem.startswith('predictions_'):
                file_origin = f"{model}_{stem[len('predictions_'):]}_{dataset}"
        
        # Normalize predicted labels: strip whitespace (incl. trailing \n) BEFORE
        # removing a trailing period, then strip again to clean up any space the
        # period removal exposed, lowercase, and convert blanks to NaN.
        # The strip-first ordering is required because some models (e.g.
        # c2s_scale_27B_blind) emit "label.\n" — a naive rstrip('.') sees the
        # newline as the trailing char and silently leaves the period in place.
        print(f"\nPreprocessing predicted labels...")
        df[pred_col] = (df[pred_col].astype(str)
                                    .str.strip()
                                    .str.rstrip('.')
                                    .str.strip()
                                    .str.lower()
                                    .replace('', np.nan))
        for col in ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3']:
            if col in df.columns:
                df[col] = df[col].str.strip().str.lower()
        
        # Filter predictions by review status before applying ontology mapping (only if mapping is enabled)
        if args.apply_ontology_mapping:
            print(f"\nFiltering predictions by ontology mapping review status...")
            try:
                df, filter_stats = filter_predictions_by_review_status(
                    df, pred_col, cache_dir, args.max_unreviewed_ratio, verbose=True
                )
                
                # Store filter statistics for reporting
                file_results["filter_stats"] = filter_stats
                
                if filter_stats["filtered"]:
                    print(f"⚠️  WARNING: {filter_stats['unreviewed_predictions']} predictions with unreviewed mappings were excluded")
                    print(f"   Remaining predictions: {filter_stats['filtered_predictions']}/{filter_stats['total_predictions']}")
                    if filter_stats["unreviewed_labels"]:
                        print(f"   Unreviewed labels: {sorted(filter_stats['unreviewed_labels'])}")
                        
            except RuntimeError as e:
                print(f"❌ ERROR: {e}")
                print("Evaluation stopped due to too many unreviewed predictions.")
                marker = cache_dir / "ONTOLOGY_REVIEW_PENDING.txt"
                with marker.open("a") as f:
                    f.write(f"[{dataset} x {model}]\n{e}\n\n")
                continue
            except Exception as e:
                print(f"Warning: Could not filter predictions by review status: {e}")
                print("Continuing with all predictions...")
                file_results["filter_stats"] = {
                    "total_predictions": len(df),
                    "filtered_predictions": len(df),
                    "unreviewed_predictions": 0,
                    "unreviewed_ratio": 0.0,
                    "unreviewed_labels": [],
                    "filtered": False,
                    "error": str(e)
                }
        else:
            # If mapping is disabled, no filtering needed
            file_results["filter_stats"] = {
                "total_predictions": len(df),
                "filtered_predictions": len(df),
                "unreviewed_predictions": 0,
                "unreviewed_ratio": 0.0,
                "unreviewed_labels": [],
                "filtered": False,
                "note": "Ontology mapping disabled"
            }

        # Apply ontology mapping to predicted labels once for all evaluations (only if enabled)
        if args.apply_ontology_mapping:
            print(f"\nApplying ontology mapping to predicted labels...")
            try:
                df = apply_ontology_mapping_to_predictions(df, pred_col, cache_dir, verbose=True, origin=file_origin)
            except Exception as e:
                print(f"Warning: Could not apply ontology mapping: {e}")
                print("Continuing with original predicted labels...")
        else:
            print(f"\nOntology mapping disabled - using original predicted labels")
        
        # Process each level that exists in the dataframe
        for true_col in ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3']:
            if true_col in df.columns:
                print(f"\nProcessing {true_col}...")
                
                # Run evaluation with mapped predicted labels
                eval_results = run_eval(df, true_col, pred_col, output_dir, cl_path, args.max_unmapped_ratio, cache_dir)

                # Round numerical results for JSON output and store per true_col
                rounded_results = {}
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        rounded_results[key] = round(value, 4)
                    else:
                        rounded_results[key] = value
                file_results[true_col] = file_results.get(true_col, {})
                file_results[true_col].update(rounded_results)
        
        # Check if any true_labels columns were found for this file
        if not file_results:
            print(f"Warning: No true_labels columns found in {pred_file}. Expected 'true_labels', 'true_labels_1', 'true_labels_2', or 'true_labels_3'.")
        else:
            # Use filename (without extension) as key for results
            file_key = Path(pred_file).stem
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
    parser.add_argument("--max-unmapped-ratio", type=float, default=0.0,
                       help="Maximum acceptable ratio of unmapped cell types (0.0-1.0). Default: 0.0 (all must be mapped)")
    parser.add_argument("--max-unreviewed-ratio", type=float, default=0.1,
                       help="Maximum acceptable ratio of predictions with unreviewed ontology mappings (0.0-1.0). Default: 0.1 (10%)")
    parser.add_argument("--apply_ontology_mapping", type=lambda x: (str(x).lower() == 'true'),
                       default=True,
                       help="Apply ontology mapping to predicted labels (default: True)")

    args = parser.parse_args()
    main(args)
