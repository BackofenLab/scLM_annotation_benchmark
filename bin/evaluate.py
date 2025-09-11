#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
import argparse
import json
import sys
import re

from util.plots import UMAPCellPlotter  # Replace with correct import if needed
from util.cell_ontology_similarity import compute_ontology_similarities, download_cl_ontology
from util.ontology_helpers import apply_ontology_mapping_to_cell_type_list

# -- Utility functions --
def extract_label_suffix(filepath):
    """Extract label suffix from prediction CSV filename.
    
    Extracts the label identifier after 'predictions_' and before '.csv'
    Examples: 
    - 'predictions_AIFI_L1.csv' -> 'AIFI_L1'
    - 'predictions_AIFI_L2.csv' -> 'AIFI_L2'
    - 'predictions.csv' -> None
    """
    filename = Path(filepath).name
    # Try to match predictions_LABEL_SUFFIX.csv
    match = re.search(r'predictions_(.+?)\.csv$', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None

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

# -- Ontology mapping utilities --
def apply_ontology_mapping_to_predictions(df: pd.DataFrame, pred_col: str, cache_dir: Path, verbose: bool = True) -> pd.DataFrame:
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
            verbose=verbose
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

    # Initialize results dictionary
    results = {"balanced_accuracy": balanced_acc}
    
    # Compute ontology similarities if CL path is provided
    ontology_df = None
    if cl_path and Path(cl_path).exists():
        try:
            print(f"Computing ontology similarities using {cl_path}...")
            # Create a temporary dataframe for ontology computation using mapped predicted labels
            temp_df = df[[true_col, pred_col]].copy()
            temp_df = temp_df.rename(columns={true_col: "true_labels", pred_col: "predicted_labels"})
            
            ontology_df, avg_similarities = compute_ontology_similarities(
                temp_df, cl_path, "true_labels", "predicted_labels", max_unmapped_ratio, 
                cache_dir=Path(cache_dir) if cache_dir else None
            )
            
            # Add ontology results to main results
            results.update(avg_similarities)
            
            print(f"Rada average similarity: {avg_similarities.get('rada_avg_similarity', 'N/A'):.4f}")
            print(f"Lin average similarity: {avg_similarities.get('lin_avg_similarity', 'N/A'):.4f}")
            
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

    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Confusion matrix saved to: {output_plot_path}")
    
    return results

def main(args):
    
    # Parse prediction input (can be string or list)
    pred_files = parse_pred_input(args.pred)
    output_dir = Path(args.out)
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
        
        # Extract label suffix for identification purposes
        label_suffix = extract_label_suffix(pred_file)
        if label_suffix:
            print(f"Processing prediction file with label suffix: {label_suffix}")
        else:
            print(f"Processing basic prediction file (no label suffix)")
        
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
        
        # Apply ontology mapping to predicted labels once for all evaluations
        print(f"\nApplying ontology mapping to predicted labels...")
        try:
            df = apply_ontology_mapping_to_predictions(df, pred_col, cache_dir, verbose=True)
        except Exception as e:
            print(f"Warning: Could not apply ontology mapping: {e}")
            print("Continuing with original predicted labels...")
        
        # Process each level that exists in the dataframe
        for true_col in ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3']:
            if true_col in df.columns:
                print(f"\nProcessing {true_col}...")
                
                # Run evaluation with mapped predicted labels
                eval_results = run_eval(df, true_col, pred_col, output_dir, cl_path, args.max_unmapped_ratio, cache_dir)
                
                # Round numerical results for JSON output
                file_results[true_col] = {}
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        file_results[true_col][key] = round(value, 4)
                    else:
                        file_results[true_col][key] = value
        
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
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w+") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n{'='*60}")
    print(f"All evaluation results saved to: {eval_path}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary of results:")
    for file_key, file_results in all_results.items():
        print(f"\n{file_key}:")
        for true_col, metrics in file_results.items():
            print(f"  {true_col}:")
            print(f"    Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A'):.4f}")
            if 'rada_avg_similarity' in metrics and not np.isnan(metrics['rada_avg_similarity']):
                print(f"    Rada Avg Similarity: {metrics['rada_avg_similarity']:.4f}")
            if 'lin_avg_similarity' in metrics and not np.isnan(metrics['lin_avg_similarity']):
                print(f"    Lin Avg Similarity: {metrics['lin_avg_similarity']:.4f}")
            

# -- CLI Interface --
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate label predictions")

    parser.add_argument("--pred", required=True, type=str, 
                       help="CSV file(s) with true and predicted labels. Can be a single file path or comma-separated list of file paths")
    parser.add_argument("--out", required=True, type=str, help="Path to results folder")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for ontology files. If not provided, uses 'cache' in current directory")
    parser.add_argument("--max-unmapped-ratio", type=float, default=0.0,
                       help="Maximum acceptable ratio of unmapped cell types (0.0-1.0). Default: 0.0 (all must be mapped)")

    args = parser.parse_args()
    main(args)
