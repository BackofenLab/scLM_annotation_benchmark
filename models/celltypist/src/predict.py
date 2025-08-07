#!/usr/bin/env python3
"""
CellTypist prediction script for single-cell data annotation.
"""

import argparse
import pandas as pd
import scanpy as sc
import celltypist
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load single-cell data from h5ad file.
    
    Args:
        data_path (str): Path to the h5ad file
        
    Returns:
        scanpy.AnnData: Loaded AnnData object
    """
    logger.info(f"Loading data from: {data_path}")
    try:
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded data with shape: {adata.shape}")
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

def run_celltypist_prediction(adata, model="Immune_All_high.pkl"):
    """
    Run CellTypist prediction on the data.
    
    Args:
        adata (scanpy.AnnData): Input data
        model (str): Model name to use for prediction
        
    Returns:
        celltypist.AnnotationResult: Prediction results
    """
    logger.info(f"Running CellTypist prediction with model: {model}")
    try:
        predictions = celltypist.annotate(adata, model=model)
        logger.info("Prediction completed successfully")
        return predictions
    except Exception as e:
        logger.error(f"Failed to run CellTypist prediction: {e}")
        sys.exit(1)

def create_results_dataframe(adata, predictions, label_key="cell_type"):
    """
    Create a DataFrame with true and predicted labels.
    
    Args:
        adata (scanpy.AnnData): Input data with true labels
        predictions (celltypist.AnnotationResult): Prediction results
        label_key (str): Key in adata.obs containing true labels
        
    Returns:
        pandas.DataFrame: DataFrame with true and predicted labels
    """
    logger.info("Creating results DataFrame")
    
    # Get predicted labels
    predicted_labels = list(predictions.predicted_labels.values)
    predicted_idx = predictions.predicted_labels.index
    
    # Get true labels from adata.obs[label_key]
    if label_key in adata.obs.columns:
        true_labels = list(adata.obs[label_key][predicted_idx].values)
    else:
        logger.error(f"'{label_key}' not found in adata.obs")
        sys.exit(1)
    # Create DataFrame
    results_df = pd.DataFrame({
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    })
    
    # Add index if it's not already the index
    if not results_df.index.equals(adata.obs.index):
        results_df.index = adata.obs.index
    
    logger.info(f"Created results DataFrame with {len(results_df)} rows")
    return results_df

def save_results(results_df, output_path):
    """
    Save results to CSV file.
    
    Args:
        results_df (pandas.DataFrame): Results DataFrame
        output_path (str): Output file path
    """
    logger.info(f"Saving results to: {output_path}")
    try:
        results_df.to_csv(output_path, index=True)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

def main():
    """
    Main function to run CellTypist prediction pipeline.
    """
    # Get default input path relative to this script
    default_input = str(Path(__file__).resolve().parents[3] / "util" / "test_adata.h5ad")
    default_output = str(Path.cwd() / "celltypist_predictions.csv")

    parser = argparse.ArgumentParser(description="CellTypist prediction for single-cell data")
    parser.add_argument("--input", "-i", default=default_input,
                       help=f"Input h5ad file path (default: {default_input})")
    parser.add_argument("--output", "-o", default=default_output,
                       help=f"Output CSV file path (default: {default_output})")
    parser.add_argument("--model", "-m", default="Immune_All_High.pkl", 
                       help="CellTypist model to use (default: Immune_All_High.pkl)")
    parser.add_argument("--label-key", "-l", default="cell_ontology_class",
                       help="Key in adata.obs containing true labels (default: cell_ontology_class)")
    
    args = parser.parse_args()
    
    # Load data
    adata = load_data(args.input)
    
    # Run prediction
    predictions = run_celltypist_prediction(adata, args.model)
    
    # Create results DataFrame
    results_df = create_results_dataframe(adata, predictions, args.label_key)
    
    # Save results
    print(f"Saving results to: {args.output}")
    save_results(results_df, args.output)
    
    logger.info("CellTypist prediction pipeline completed successfully")
    logger.info(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()

