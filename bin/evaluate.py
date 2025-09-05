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
from util.cell_ontology_similarity import compute_ontology_similarities

# -- Utility functions --
def extract_model_name(filepath):
    """Extract model name from prediction CSV filename.
    
    Extracts everything after 'prediction' and before '.csv'
    Example: 'predictions_cellwhisperer_model1.csv' -> 'cellwhisperer_model1'
    """
    filename = Path(filepath).name
    match = re.search(r'prediction[s]?[_-]?(.+?)\.csv$', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # Fallback: use filename without extension
        return Path(filepath).stem

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

# -- Normalization utilities --
def normalize(text):
    return str(text).lower().replace("-", " ").replace(",", "").replace(".", "").strip()

def clean_celltype_L1(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    clean_pred_col = f"clean_{pred_col}"

    # Normalize all unique true labels once
    normalized_true_labels = {
        normalize(label): label
        for label in df[true_col].unique()
    }

    def match_true_label(pred_label):
        norm_pred = normalize(pred_label)
        for norm_true, original_true in normalized_true_labels.items():
            if norm_true in norm_pred or norm_pred in norm_true:
                return original_true
        return "Unknown"  

    df[clean_pred_col] = df[pred_col].apply(match_true_label)

    print(df.head())
    return df

def clean_celltype_L2(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    # Placeholder function for level 2 - does nothing for now
    clean_pred_col = f"clean_{pred_col}"
    df[clean_pred_col] = df[pred_col]
    return df

def clean_celltype_L3(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    # Placeholder function for level 3 - does nothing for now
    clean_pred_col = f"clean_{pred_col}"
    df[clean_pred_col] = df[pred_col]
    return df

def run_eval(df, true_col, pred_col, output_dir, model_name=None, cl_path=None, max_unmapped_ratio=1.0):
    
    true_labels = df[true_col]
    pred_labels = df[f"clean_{pred_col}"]

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)

    # Initialize results dictionary
    results = {"balanced_accuracy": balanced_acc}
    
    # Compute ontology similarities if CL path is provided
    ontology_df = None
    if cl_path and Path(cl_path).exists():
        try:
            print(f"Computing ontology similarities using {cl_path}...")
            # Create a temporary dataframe for ontology computation
            temp_df = df[[true_col, f"clean_{pred_col}"]].copy()
            temp_df = temp_df.rename(columns={true_col: "true_labels", f"clean_{pred_col}": "predicted_labels"})
            
            ontology_df, avg_similarities = compute_ontology_similarities(
                temp_df, cl_path, "true_labels", "predicted_labels", max_unmapped_ratio
            )
            
            # Add ontology results to main results
            results.update(avg_similarities)
            
            print(f"Rada average similarity: {avg_similarities.get('rada_avg_similarity', 'N/A'):.4f}")
            print(f"Lin average similarity: {avg_similarities.get('lin_avg_similarity', 'N/A'):.4f}")
            
            # Save scored CSV with ontology similarities (unique combinations)
            if model_name:
                scored_csv_path = output_dir / f"{model_name}_{true_col}_scored.csv"
            else:
                scored_csv_path = output_dir / f"{true_col}_scored.csv"
            
            # ontology_df now contains unique combinations with aggregated scores
            ontology_df.to_csv(scored_csv_path, index=False)
            print(f"Scored CSV with unique combinations saved to: {scored_csv_path}")
            print(f"Found {len(ontology_df)} unique label combinations")
            
        except Exception as e:
            print(f"Warning: Could not compute ontology similarities: {e}")
            results["rada_avg_similarity"] = np.nan
            results["lin_avg_similarity"] = np.nan

    # Set up plotter
    plotter = UMAPCellPlotter()

    # Generate confusion matrix with model name in filename
    if model_name:
        output_plot_path = output_dir / f"{model_name}_{true_col}_confusion_matrix.pdf"
        title = f"Confusion Matrix {true_col} - {model_name} (Balanced Accuracy = {balanced_acc:.2f})"
    else:
        output_plot_path = output_dir / f"{true_col}_confusion_matrix.pdf"
        title = f"Confusion Matrix {true_col} (Balanced Accuracy = {balanced_acc:.2f})"
    
    plotter.plot_cm(
        true_labels=true_labels,
        pred_labels=pred_labels,
        output_path=output_plot_path,
        title=title,
        normalize=True
    )

    print(f"Model: {model_name if model_name else 'Unknown'}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Confusion matrix saved to: {output_plot_path}")
    
    return results

def main(args):
    
    # Parse prediction input (can be string or list)
    pred_files = parse_pred_input(args.pred)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up CL ontology path (cl-base.owl in same directory as script)
    script_dir = Path(__file__).parent
    cl_path = script_dir / "cl-base.owl"
    
    print(f"Input argument: {args.pred}")
    print(f"Parsed prediction files: {pred_files}")
    print(f"Output directory: {output_dir}")
    print(f"Cell Ontology path: {cl_path}")
    print(f"Max unmapped ratio: {args.max_unmapped_ratio}")
    
    all_results = {}
    
    # Define cleaning functions for each level
    clean_functions = {
        'true_labels': clean_celltype_L1,
        'true_labels_1': clean_celltype_L1,
        'true_labels_2': clean_celltype_L2,
        'true_labels_3': clean_celltype_L3
    }
    
    # Process each prediction file
    for pred_file in pred_files:
        print(f"\n{'='*60}")
        print(f"Processing prediction file: {pred_file}")
        print(f"{'='*60}")
        
        # Extract model name from filename
        model_name = extract_model_name(pred_file)
        print(f"Detected model name: {model_name}")
        
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
        model_results = {}
        
        # Process each level that exists in the dataframe
        for true_col in ['true_labels', 'true_labels_1', 'true_labels_2', 'true_labels_3']:
            if true_col in df.columns:
                print(f"\nProcessing {true_col} for {model_name}...")
                
                # Apply appropriate cleaning function
                clean_func = clean_functions[true_col]
                df_clean = clean_func(df, true_col, pred_col)
                
                # Run evaluation
                eval_results = run_eval(df_clean, true_col, pred_col, output_dir, model_name, cl_path, args.max_unmapped_ratio)
                
                # Round numerical results for JSON output
                model_results[true_col] = {}
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        model_results[true_col][key] = round(value, 4)
                    else:
                        model_results[true_col][key] = value
        
        # Check if any true_labels columns were found for this model
        if not model_results:
            print(f"Warning: No true_labels columns found in {pred_file}. Expected 'true_labels', 'true_labels_1', 'true_labels_2', or 'true_labels_3'.")
        else:
            all_results[model_name] = model_results

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
    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        for true_col, metrics in model_results.items():
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
    parser.add_argument("--max-unmapped-ratio", type=float, default=1.0,
                       help="Maximum acceptable ratio of unmapped cell types (0.0-1.0). Default: 1.0 (accept any ratio)")

    args = parser.parse_args()
    main(args)
