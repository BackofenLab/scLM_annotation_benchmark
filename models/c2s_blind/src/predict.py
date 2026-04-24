#!/usr/bin/env python3
"""
Cell2Sentence prediction script for single-cell data annotation.
"""

import argparse
import pandas as pd
import scanpy as sc
import cell2sentence as cs
from cell2sentence.tasks import embed_cells, predict_cell_types_of_data
from datasets import Dataset, load_from_disk
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

def convert_to_c2s_format(adata, annotation_key, cache_dir=None, input_tag=None):
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
        
        # Cell2Sentence uppercases var_names internally. Ensure we provide
        # case-insensitive unique feature names to avoid index mismatches.
        if "gene_names" in adata_copy.var.columns:
            gene_names = adata_copy.var["gene_names"].astype(str)
        elif "feature_name" in adata_copy.var.columns:
            gene_names = adata_copy.var["feature_name"].astype(str)
        elif "gene_symbols" in adata_copy.var.columns:
            gene_names = adata_copy.var["gene_symbols"].astype(str)
        else:
            gene_names = pd.Series(adata_copy.var_names.astype(str), index=adata_copy.var_names)
            logger.warning("No gene name column found; using adata.var_names as feature names.")

        gene_names = gene_names.fillna("").str.strip()
        missing_mask = gene_names.eq("")
        if missing_mask.any():
            fallback = pd.Series(adata_copy.var_names.astype(str), index=adata_copy.var_names)
            gene_names.loc[missing_mask] = fallback.loc[missing_mask]
            logger.warning(f"Filled {int(missing_mask.sum())} missing gene names from adata.var_names.")

        adata_copy.var_names = pd.Index(gene_names.values).str.upper()
        n_vars = adata_copy.n_vars
        n_before = adata_copy.var_names.nunique()
        duplicate_count = n_vars - n_before
        adata_copy.var_names_make_unique()
        if duplicate_count > 0:
            logger.warning(
                f"Adjusted {duplicate_count} case-insensitive duplicate gene names for Cell2Sentence compatibility."
            )

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
        safe_annotation_key = str(annotation_key).replace("/", "_")
        safe_input_tag = str(input_tag if input_tag else "input").replace("/", "_")
        # Include PID to avoid collisions across parallel Nextflow tasks.
        c2s_save_name = f"temp_c2s_data_{safe_input_tag}_{safe_annotation_key}_{os.getpid()}"
        
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

        # Read prediction cell IDs from the same C2S dataset used for prompting.
        # This avoids relying on positional alignment when preprocessing or temp
        # dataset collisions alter row counts.
        pred_ds = load_from_disk(csdata.data_path)
        prediction_cell_ids = list(pred_ds["cell_name"]) if "cell_name" in pred_ds.column_names else None

        if prediction_cell_ids is not None and len(prediction_cell_ids) != len(predicted_cell_types):
            logger.warning(
                "Prediction output length mismatch inside C2S dataset: %d ids vs %d predictions. "
                "Trimming to minimum length for safe alignment.",
                len(prediction_cell_ids),
                len(predicted_cell_types),
            )
            n = min(len(prediction_cell_ids), len(predicted_cell_types))
            prediction_cell_ids = prediction_cell_ids[:n]
            predicted_cell_types = predicted_cell_types[:n]

        return predicted_cell_types, prediction_cell_ids
    except Exception as e:
        logger.error(f"Failed to run Cell2Sentence prediction: {e}")
        sys.exit(1)


def create_results_dataframe_for_key(adata, predicted_cell_types, prediction_cell_ids, label_key):
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
    
    truth_df = pd.DataFrame({
        'cell_id': adata.obs.index.astype(str),
        'true_labels': list(adata.obs[label_key].values),
    })

    if prediction_cell_ids is not None:
        pred_df = pd.DataFrame({
            'cell_id': [str(x) for x in prediction_cell_ids],
            'predicted_labels': predicted_cell_types,
        })
        # Use left join to preserve input cell order and fill any missing predictions.
        results_df = truth_df.merge(pred_df, on='cell_id', how='left')
        missing = int(results_df['predicted_labels'].isna().sum())
        if missing > 0:
            logger.warning(
                "Missing predictions for %d/%d cells after ID alignment; filling with 'unknown'.",
                missing,
                len(results_df),
            )
            results_df['predicted_labels'] = results_df['predicted_labels'].fillna('unknown')
    else:
        logger.warning(
            "No prediction cell IDs available; falling back to positional alignment "
            "(true=%d, pred=%d).",
            len(truth_df),
            len(predicted_cell_types),
        )
        n = min(len(truth_df), len(predicted_cell_types))
        results_df = truth_df.iloc[:n].copy()
        results_df['predicted_labels'] = predicted_cell_types[:n]
    
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
    input_tag = Path(args.input).stem

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # Load model
    csmodel = load_model(args.model, args.model_name, args.cache_dir)
    csmodel.device = device
   
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix

    annotation_key = args.annotation_key
    logger.info(f"Processing annotation key: {annotation_key}")
    
    # Check if annotation key exists
    if annotation_key not in adata.obs.columns:
        logger.error(f"'{annotation_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        logger.warning(f"Skipping {annotation_key}")
    
    # Convert to Cell2Sentence format for this specific annotation key
    csdata = convert_to_c2s_format(adata, annotation_key, args.cache_dir, input_tag=input_tag)
    
    # Run prediction for this annotation key
    predicted_cell_types, prediction_cell_ids = run_cell2sentence_prediction(csdata, csmodel, args.n_genes)
    
    # Create results DataFrame for this annotation key
    results_df = create_results_dataframe_for_key(adata, predicted_cell_types, prediction_cell_ids, annotation_key)
    
    if results_df is not None:
        # Create output filename for this annotation level (same naming as CellWhisperer)
        output_csv = output_dir / f"{output_stem}{output_suffix}"
        
        # Save results
        print(f"Saving results to: {output_csv}")
        save_results(results_df, str(output_csv))
        logger.info(f"Results for {annotation_key} saved to: {output_csv}")
    else:
        logger.warning(f"Skipping {annotation_key} due to errors")
    
    logger.info("Cell2Sentence prediction pipeline completed successfully")

if __name__ == "__main__":
    main()
