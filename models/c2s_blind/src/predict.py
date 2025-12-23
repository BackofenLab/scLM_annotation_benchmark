#!/usr/bin/env python3
"""
Cell2Sentence prediction script for single-cell data annotation.
"""

import argparse
import pandas as pd
import scanpy as sc
import cell2sentence as cs
from cell2sentence.tasks import embed_cells, predict_cell_types_of_data
from datasets import Dataset
from pathlib import Path
import sys
import logging
import os
import random
import numpy as np
import torch
logger = logging.getLogger(__name__)

# Set seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

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
        
        # Set required metadata for cell2sentence
        if 'organism' not in adata.obs.columns:
            adata.obs['organism'] = 'human'  # Default to human
        
        # Use highly variable genes if available
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.var['highly_variable']]
            logger.info(f"Filtered to highly variable genes, new shape: {adata.shape}")
        
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

def load_model(model_path="vandijklab/C2S-Pythia-410m-cell-type-prediction", 
               model_name="cell_embedding_prediction_pythia_410M", 
               cache_dir=None):
    """
    Load Cell2Sentence model.
    
    Args:
        model_path (str): Model name or path
        model_name (str): Name for saving the model
        cache_dir (str): Directory for caching models
        
    Returns:
        cell2sentence.CSModel: Loaded model
    """
    logger.info(f"Loading Cell2Sentence model: {model_path}")
    try:
        # Use cache_dir if provided, otherwise use current directory
        save_dir = cache_dir if cache_dir else "."
        
        csmodel = cs.CSModel(
            model_name_or_path=model_path,
            save_dir=save_dir,
            save_name=model_name
        )
        logger.info("Model loaded successfully")
        return csmodel
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

def convert_to_c2s_format(adata, annotation_key, cache_dir=None):
    """
    Convert AnnData to Cell2Sentence format for a specific annotation key.
    
    Args:
        adata (scanpy.AnnData): Input data
        annotation_key (str): Single annotation key to use as cell_type
        cache_dir (str): Directory for caching
        
    Returns:
        cell2sentence.CSData: Converted data
    """
    logger.info(f"Converting data to Cell2Sentence format for annotation key: {annotation_key}")
    try:
        # Create a copy of adata to avoid modifying the original
        adata_copy = adata.copy()

        if "counts" not in adata.layers:
            raise ValueError("Counts layer not found in adata.layers. Run prepreocessing step first. C2S needs raw counts")
        #set X to raw counts and apply log10 transformation
        adata_copy.X = adata.layers["counts"]
        logger.info(f"Setting X to raw counts, normalizing and log-transforming with base 10. (see https://github.com/vandijklab/cell2sentence/blob/master/tutorials/c2s_tutorial_0_data_preparation.ipynb)")
        sc.pp.normalize_total(adata_copy)
        sc.pp.log1p(adata_copy, base=10)
        
        # and var names need to contain gene names
        adata_copy.var_names = adata_copy.var["gene_names"]
        # Set the cell_type column to the current annotation key
        if annotation_key not in adata_copy.obs.columns:
            raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")
        
        adata_copy.obs['cell_type'] = adata_copy.obs[annotation_key]
        
        # Determine which columns to keep (organism and cell_type)
        adata_obs_cols_to_keep = ['organism', 'cell_type']
        # Only keep columns that exist in the data
        adata_obs_cols_to_keep = [col for col in adata_obs_cols_to_keep if col in adata_copy.obs.columns]
        
        logger.info(f"Keeping observation columns: {adata_obs_cols_to_keep}")
        
        # Use cache_dir if provided
        c2s_save_dir = cache_dir if cache_dir else "."
        c2s_save_name = f"temp_c2s_data_{annotation_key}"
        
        arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
            adata=adata_copy, 
            random_state=SEED, 
            sentence_delimiter=' ',
            label_col_names=adata_obs_cols_to_keep
        )
        print("Exemplary Cell2Sentence data:")
        print(arrow_ds[0])
        print("Exemplary vocabulary:")
        print(list(vocabulary.items())[:10])
        csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds, 
            vocabulary=vocabulary,
            save_dir=c2s_save_dir,
            save_name=c2s_save_name,
            dataset_backend="arrow"
        )
        
        logger.info("Data conversion completed successfully")
        return csdata
    except Exception as e:
        logger.error(f"Failed to convert data: {e}")
        sys.exit(1)

def run_cell2sentence_prediction(csdata, csmodel, n_genes=200):
    """
    Run Cell2Sentence prediction on the data.
    
    Args:
        csdata (cell2sentence.CSData): Input data in C2S format
        csmodel (cell2sentence.CSModel): Model for prediction
        n_genes (int): Number of genes to use for prediction
        
    Returns:
        list: Predicted cell types
    """
    logger.info(f"Running Cell2Sentence prediction with {n_genes} genes")
    try:
        predicted_cell_types = predict_cell_types_of_data(
            csdata=csdata,
            csmodel=csmodel,
            n_genes=n_genes
        )
        logger.info("Prediction completed successfully")

        #filter the special token \n<ctrl100>' from the end of the predicted cell types
        predicted_cell_types = [ct.replace('\n<ctrl100>', '') for ct in predicted_cell_types]

        return predicted_cell_types
    except Exception as e:
        logger.error(f"Failed to run Cell2Sentence prediction: {e}")
        sys.exit(1)


def create_results_dataframe_for_key(adata, predicted_cell_types, label_key):
    """
    Create a DataFrame with true and predicted labels for a single annotation key.
    
    Args:
        adata (scanpy.AnnData): Input data with true labels
        predicted_cell_types (list): Predicted cell types
        label_key (str): Key in adata.obs containing true labels
        
    Returns:
        pandas.DataFrame: DataFrame with true and predicted labels
    """
    logger.info(f"Creating results DataFrame for annotation key: {label_key}")
    
    # Check if label key exists
    if label_key not in adata.obs.columns:
        logger.error(f"'{label_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        return None
    
    # Get true labels for this annotation key
    true_labels = list(adata.obs[label_key].values)
    
    # Create DataFrame with cell_id, true_labels, and predicted_labels
    results_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'true_labels': true_labels,
        'predicted_labels': predicted_cell_types
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

def main():
    """
    Main function to run Cell2Sentence prediction pipeline.
    """
    # Get default input path relative to this script
    default_input = str(Path(__file__).resolve().parents[3] / "util" / "test_adata.h5ad")
    default_output = str(Path.cwd() / "cell2sentence_predictions.csv")

    parser = argparse.ArgumentParser(description="Cell2Sentence prediction for single-cell data")
    parser.add_argument("--input", "-i", default=default_input,
                       help=f"Input h5ad file path (default: {default_input})")
    parser.add_argument("--output", "-o", default=default_output,
                       help=f"Output CSV file path base name (default: {default_output})")
    parser.add_argument("--model", "-m", default="vandijklab/C2S-Pythia-410m-cell-type-prediction", 
                       help="Cell2Sentence model to use (default: vandijklab/C2S-Pythia-410m-cell-type-prediction)")
    parser.add_argument("--model_name", default="cell_embedding_prediction_pythia_410M",
                       help="Model name for caching (default: cell_embedding_prediction_pythia_410M)")
    parser.add_argument("--annotation_key", default="cell_ontology_class",
                       help="Key in adata.obs containing true labels (default: cell_ontology_class)")
    parser.add_argument("--n_genes", type=int, default=200,
                       help="Number of genes to use for prediction (default: 200)")
    parser.add_argument("--cache_dir", default=None,
                       help="Directory for caching models and data")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load data
    adata = load_data(args.input)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # Load model
    csmodel = load_model(args.model, args.model_name, args.cache_dir)
    csmodel.device = device
    
    annotation_key = args.annotation_key
    logger.info(f"Processing annotation key: {annotation_key}")
    
    # Check if annotation key exists
    if annotation_key not in adata.obs.columns:
        logger.error(f"'{annotation_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        sys.exit(1)
    
    # Convert to Cell2Sentence format
    csdata = convert_to_c2s_format(adata, annotation_key, args.cache_dir)
    
    # Run prediction
    predicted_cell_types = run_cell2sentence_prediction(csdata, csmodel, args.n_genes)
    
    # Create results DataFrame
    results_df = create_results_dataframe_for_key(adata, predicted_cell_types, annotation_key)
    
    if results_df is not None:
        # Save results to the specified output path
        print(f"Saving results to: {args.output}")
        save_results(results_df, args.output)
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.error("Failed to create results DataFrame")
        sys.exit(1)
    
    logger.info("Cell2Sentence prediction pipeline completed successfully")

if __name__ == "__main__":
    main()
