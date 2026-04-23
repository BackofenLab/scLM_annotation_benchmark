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

def _download_model(model: str) -> None:
    """Download a CellTypist model, falling back to the Organ Atlas URL if needed."""
    try:
        celltypist.models.download_models(model=model)
    except (ValueError, SystemExit):
        # Model not in standard repertoire — try Organ Atlas URL.
        # Organ Atlas URL pattern:
        #   https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/{organ}/models/{model}
        import urllib.request
        from pathlib import Path

        stem = model.removesuffix(".pkl")
        parts = stem.split("_", 2)
        if len(parts) < 3:
            raise ValueError(
                f"Cannot derive organ from model name '{model}'. "
                "Expected pattern: <Descriptor>_<Species>_<Organ>.pkl"
            )
        organ = parts[2]
        organ_atlas_url = (
            f"https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/"
            f"{organ}/models/{model}"
        )

        models_dir = Path(celltypist.models.models_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / model
        if not dest.exists():
            logger.info(f"Model not in standard list; downloading from Organ Atlas: {organ_atlas_url}")
            urllib.request.urlretrieve(organ_atlas_url, dest)
        else:
            logger.info(f"Organ Atlas model already cached: {dest}")


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
        # CellTypist requires gene symbols in var.index; check multiple possible sources
        gene_symbol_cols = ["gene_names", "feature_name", "gene_symbols", "gene_name", "symbol"]
        for col in gene_symbol_cols:
            if col in adata.var.columns:
                logger.info(f"Swapping var.index to gene symbols from '{col}' column")
                adata.var_names = adata.var[col].astype(str).values
                adata.var_names_make_unique()
                break
        
        _download_model(model)
        model_obj = celltypist.models.Model.load(model)
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
        # Restore raw counts before CellTypist's required normalize_total(1e4) + log1p;
        # the pipeline's preprocessed X is already log1p-normalized, so re-normalizing
        # it would double-log.
        if "counts" in adata.layers:
            adata = adata.copy()
            adata.X = adata.layers["counts"]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
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
                       help=f"Output CSV file path base name (default: {default_output})")
    parser.add_argument("--model", "-m", default="Immune_All_High.pkl", 
                       help="CellTypist model to use (default: Immune_All_High.pkl)")
    parser.add_argument("--annotation_key", default="cell_ontology_class",
                       help="Key in adata.obs containing true labels (default: cell_ontology_class)")
    
    args = parser.parse_args()
    
    # Load data
    adata = load_data(args.input)
    adata = preprocess_data(adata)

    model_to_use = args.model

    # Run prediction once
    predictions = run_celltypist_prediction(adata, model_to_use)
    
    annotation_key = args.annotation_key
    logger.info(f"Processing annotation key: {annotation_key}")
    
    # Create results DataFrame
    results_df = create_results_dataframe_for_key(adata, predictions, annotation_key)
    
    if results_df is not None:
        # Save results
        print(f"Saving results to: {args.output}")
        save_results(results_df, args.output)
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.error("Failed to create results DataFrame")
        sys.exit(1)
    
    logger.info("CellTypist prediction pipeline completed successfully")

if __name__ == "__main__":
    main()

