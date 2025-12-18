#!/usr/bin/env python3
"""
CellTypist prediction script for single-cell data annotation.
"""

import argparse
import os
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

def run_celltypist_prediction(adata, model="Immune_All_High.pkl"):
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
        celltypist.models.download_models(model=model)
        model_obj = celltypist.models.Model.load(model)
        all_start_with_ENSG = all(str(idx).startswith("ENSG") for idx in adata.var.index)
        if all_start_with_ENSG: # Only if all genes start with ESNG, assume the index contains ensebml ids and convert the model
            model_obj.convert('GENCODEv44_Gene_id2name.csv')
        predictions = celltypist.annotate(adata, model=model_obj)
        logger.info("Prediction completed successfully")
        return predictions
    except Exception as e:
        logger.error(f"Failed to run CellTypist prediction: {e}")
        sys.exit(1)

def create_results_dataframe_for_key(adata, predictions, label_key):
    """
    Create a DataFrame with true and predicted labels for a single annotation key.
    
    Args:
        adata (scanpy.AnnData): Input data with true labels
        predictions (celltypist.AnnotationResult): Prediction results
        label_key (str): Key in adata.obs containing true labels
        
    Returns:
        pandas.DataFrame: DataFrame with true and predicted labels
    """
    logger.info(f"Creating results DataFrame for annotation key: {label_key}")
    
    # Get predicted labels
    predicted_labels = predictions.predicted_labels["predicted_labels"].values.tolist()
    predicted_idx = predictions.predicted_labels.index
    
    # Check if label key exists
    if label_key not in adata.obs.columns:
        logger.error(f"'{label_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        return None
    
    # Get true labels for this annotation key
    true_labels = list(adata.obs[label_key][predicted_idx].values)
    
    # Create DataFrame with cell_id, true_labels, and predicted_labels
    results_df = pd.DataFrame({
        'cell_id': predicted_idx,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    })
    
    logger.info(f"Created results DataFrame for {label_key} with {len(results_df)} rows")
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
        results_df.to_csv(output_path, index=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

def preprocess_data(adata):
    """
    Preprocess data for CellTypist prediction.
    
    Args:
        adata (scanpy.AnnData): Input data
        
    Returns:
        scanpy.AnnData: Preprocessed data
    """
    logger.info("Preprocessing data")
    try:
        sc.pp.normalize_total(adata, target_sum = 1e4)
        sc.pp.log1p(adata)
        return adata
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        sys.exit(1)


def model_dataset_mapper(path_dataset):
    """
    Map dataset path to appropriate CellTypist model name.
    """
    print(f"Mapping model {path_dataset} to dataset")
    filename = os.path.basename(path_dataset)
    filename = filename.lower()
    if "liver" in filename:
        model_name = "Healthy_Human_Liver.pkl"
    elif "heart" in filename.lower():
        model_name = "Healthy_Adult_Heart.pkl"
    else:
        model_name = "Immune_All_High.pkl"
    return model_name

        
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
                       help=f"Output CSV file path base name (default: {default_output})")
    parser.add_argument("--model", "-m", default="Immune_All_High.pkl", 
                       help="CellTypist model to use (default: Immune_All_High.pkl)")
    parser.add_argument("--annotation_keys", nargs='+', default=["cell_ontology_class"],
                       help="Keys in adata.obs containing true labels (default: cell_ontology_class)")
    
    args = parser.parse_args()
    
    # Load data
    adata = load_data(args.input)
    adata = preprocess_data(adata)

    model_to_use = args.model#model_dataset_mapper(args.input)

    # Run prediction once
    predictions = run_celltypist_prediction(adata, model_to_use)
    
    # Get base output path info (same as CellWhisperer)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    
    # Process each annotation key separately
    for annotation_key in args.annotation_keys:
        logger.info(f"Processing annotation key: {annotation_key}")
        
        # Create results DataFrame for this annotation key
        results_df = create_results_dataframe_for_key(adata, predictions, annotation_key)
        
        if results_df is not None:
            # Create output filename for this annotation level (same naming as CellWhisperer)
            output_csv = output_dir / f"{output_stem}_{annotation_key}{output_suffix}"
            
            # Save results
            print(f"Saving results to: {output_csv}")
            save_results(results_df, str(output_csv))
            logger.info(f"Results for {annotation_key} saved to: {output_csv}")
        else:
            logger.warning(f"Skipping {annotation_key} due to errors")
    
    logger.info("CellTypist prediction pipeline completed successfully")

if __name__ == "__main__":
    main()

