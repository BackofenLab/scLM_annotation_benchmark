#!/usr/bin/env python3
"""
InstructCell prediction script for unblind cell type annotation.
Processes single-cell data and generates cell type predictions using InstructCell model.
Includes cell type options in the prompt.
"""

import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import yaml
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List
import torch
from scipy import sparse

# Add InstructCell to path (cloned by prepare.sh)
# The repo is cloned to models/instructcell_unblind/src/InstructCell
sys.path.insert(0, str(Path(__file__).parent / "InstructCell"))
from mmllm.module import InstructCell
from utils import unify_gene_features

logger = logging.getLogger(__name__)


def load_data(data_path: str):
    """Load single-cell data from h5ad file."""
    logger.info(f"Loading data from: {data_path}")
    try:
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded data with shape: {adata.shape}")
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def load_gene_vocab(vocab_path: str):
    """Load gene vocabulary."""
    logger.info(f"Loading gene vocabulary from: {vocab_path}")
    try:
        gene_vocab = np.load(vocab_path)
        logger.info(f"Loaded {len(gene_vocab)} genes in vocabulary")
        return gene_vocab
    except Exception as e:
        logger.error(f"Failed to load gene vocabulary: {e}")
        sys.exit(1)


def extract_metadata(adata: anndata.AnnData, cell_idx: int, config: Dict, dataset_metadata: Dict[str, str] = None) -> Dict[str, str]:
    """
    Extract metadata from adata.obs for a specific cell, or use dataset-wide values.
    
    Args:
        adata: AnnData object
        cell_idx: Index of the cell
        config: Dataset configuration dictionary
        dataset_metadata: Pre-computed dataset-wide metadata (if using direct values)
        
    Returns:
        Dictionary with metadata fields (species, tissue, sequencing_method)
    """
    metadata = {}
    
    # Use pre-computed dataset metadata if available (direct values from config)
    if dataset_metadata:
        metadata.update(dataset_metadata)
        return metadata
    
    # Extract species - check for direct value first, then key
    if 'species' in config:
        # Direct value provided
        metadata['species'] = str(config['species']).strip()
    else:
        # Key provided - extract per-cell value
        species_key = config.get('species_key')
        if species_key and species_key in adata.obs.columns:
            species_val = adata.obs.iloc[cell_idx][species_key]
            if pd.notna(species_val) and str(species_val).strip():
                metadata['species'] = str(species_val).strip()
    
    # Extract tissue - check for direct value first, then key
    if 'tissue' in config:
        # Direct value provided
        metadata['tissue'] = str(config['tissue']).strip()
    else:
        # Key provided - extract per-cell value
        tissue_key = config.get('tissue_key')
        if tissue_key and tissue_key in adata.obs.columns:
            tissue_val = adata.obs.iloc[cell_idx][tissue_key]
            if pd.notna(tissue_val) and str(tissue_val).strip():
                metadata['tissue'] = str(tissue_val).strip()
    
    # Extract sequencing method - check for direct value first, then key
    if 'sequencing_method' in config:
        # Direct value provided
        metadata['sequencing_method'] = str(config['sequencing_method']).strip()
    else:
        # Key provided - extract per-cell value
        seq_key = config.get('sequencing_method_key')
        if seq_key and seq_key in adata.obs.columns:
            seq_val = adata.obs.iloc[cell_idx][seq_key]
            if pd.notna(seq_val) and str(seq_val).strip():
                metadata['sequencing_method'] = str(seq_val).strip()
    
    return metadata


def build_prompt(metadata: Dict[str, str], include_cell_types: Optional[List[str]] = None) -> str:
    """
    Build the prompt for cell type annotation.
    
    Args:
        metadata: Dictionary with metadata fields (species, tissue, sequencing_method)
        include_cell_types: List of cell types to include in prompt (required for unblind mode)
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    # Start with the main question
    if 'species' in metadata:
        parts.append(f"Can you help me annotate this single cell from a {metadata['species']}?")
    else:
        parts.append("Can you help me annotate this single cell?")
    
    # Add sequencing method if available
    if 'sequencing_method' in metadata:
        parts.append(f"It was sequenced using {metadata['sequencing_method']}.")
    
    # Add tissue if available
    if 'tissue' in metadata:
        parts.append(f"It is derived from {metadata['tissue']}.")
    
    # Add cell type list for unblind mode (required)
    if include_cell_types:
        cell_types_str = ", ".join(include_cell_types)
        parts.append(f"Pick from this list: {cell_types_str}.")
    else:
        logger.warning("No cell types provided for unblind mode")
    
    # Add gene expression profile placeholder
    parts.append("The gene expression profile is {input}. Thanks!")
    
    return " ".join(parts)


def extract_prediction(response: str) -> str:
    """
    Extract the predicted cell type from model response.
    
    Args:
        response: Full model response text
        
    Returns:
        Cleaned predicted cell type
    """
    # The response should be the cell type directly
    # Remove common prefixes/suffixes and clean up
    pred = response.strip()
    
    # Remove trailing punctuation
    pred = pred.rstrip('.,!?')
    
    # Remove "Assistant:" prefix if present
    if pred.startswith("Assistant:"):
        pred = pred[len("Assistant:"):].strip()
    
    # Remove "text:" prefix if present
    if pred.lower().startswith("text:"):
        pred = pred[len("text:"):].strip()
    
    return pred.strip()


def get_unique_cell_types(adata: anndata.AnnData, annotation_key: str) -> List[str]:
    """
    Extract unique cell types from the anndata object.
    
    Args:
        adata: Input AnnData object
        annotation_key: Key in adata.obs containing cell type labels
        
    Returns:
        List of unique cell type labels
    """
    if annotation_key not in adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")
    
    unique_cell_types = sorted(adata.obs[annotation_key].dropna().unique().tolist())
    logger.info(f"Found {len(unique_cell_types)} unique cell types for annotation key '{annotation_key}'")
    return unique_cell_types


def main():
    """Main function to run InstructCell prediction pipeline."""
    parser = argparse.ArgumentParser(description="InstructCell prediction for single-cell data (unblind mode)")
    parser.add_argument("--input", "-i", required=True,
                       help="Input h5ad file path")
    parser.add_argument("--output", "-o", required=True,
                       help="Output CSV file path")
    parser.add_argument("--config", "-c", required=True,
                       help="Dataset configuration YAML file")
    parser.add_argument("--cache_dir", default=None,
                       help="Directory for caching models")
    parser.add_argument("--model_variant", default="instruct", choices=["chat", "instruct"],
                       help="Model variant to use: 'chat' or 'instruct' (default: instruct)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing cells (default: 1)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum new tokens to generate (default: 256)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    annotation_key = config.get('annotation_key', 'cell_type')
    logger.info(f"Using annotation key: {annotation_key}")
    
    # Load data
    adata = load_data(args.input)
    
    # Check if annotation key exists
    if annotation_key not in adata.obs.columns:
        logger.error(f"'{annotation_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        sys.exit(1)
    
    # Get unique cell types for unblind mode
    cell_type_list = get_unique_cell_types(adata, annotation_key)
    logger.info(f"Including {len(cell_type_list)} cell types in prompt")
    
    # Load gene vocabulary
    # Path relative to this script: InstructCell/exp_log/gene_vocab.npy (cloned by prepare.sh)
    script_dir = Path(__file__).parent
    vocab_path = script_dir / "InstructCell" / "exp_log" / "gene_vocab.npy"
    if not vocab_path.exists():
        logger.error(f"Gene vocabulary not found at: {vocab_path}")
        sys.exit(1)
    
    gene_vocab = load_gene_vocab(str(vocab_path))
    
    # Unify gene features
    logger.info("Unifying gene features with vocabulary...")
    # Try with uppercase first (vocabulary is typically uppercase)
    try:
        adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=True)
        logger.info("Successfully unified genes with uppercase matching")
    except ValueError as e:
        # If uppercase fails, try without uppercase
        logger.warning(f"Uppercase matching failed: {e}. Trying case-sensitive matching...")
        adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)
    logger.info(f"Data shape after unification: {adata.shape}")
    
    # Load model
    model_id = f"zjunlp/InstructCell-{args.model_variant}"
    logger.info(f"Loading model: {model_id}")
    try:
        model = InstructCell.from_pretrained(model_id, cache_dir=args.cache_dir)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Pre-compute dataset-wide metadata if direct values are provided (for efficiency)
    dataset_metadata = {}
    if 'species' in config:
        dataset_metadata['species'] = str(config['species']).strip()
        logger.info(f"Using dataset-wide species: {dataset_metadata['species']}")
    if 'tissue' in config:
        dataset_metadata['tissue'] = str(config['tissue']).strip()
        logger.info(f"Using dataset-wide tissue: {dataset_metadata['tissue']}")
    if 'sequencing_method' in config:
        dataset_metadata['sequencing_method'] = str(config['sequencing_method']).strip()
        logger.info(f"Using dataset-wide sequencing_method: {dataset_metadata['sequencing_method']}")
    
    # Log which keys are being used for per-cell metadata
    if not dataset_metadata:
        logger.info("Using per-cell metadata from obs columns")
        if config.get('species_key'):
            logger.info(f"  - species_key: {config.get('species_key')}")
        if config.get('tissue_key'):
            logger.info(f"  - tissue_key: {config.get('tissue_key')}")
        if config.get('sequencing_method_key'):
            logger.info(f"  - sequencing_method_key: {config.get('sequencing_method_key')}")
    
    # Generate predictions
    logger.info(f"Generating predictions for {len(adata)} cells...")
    predicted_cell_types = []
    
    # Process cells
    for i in range(len(adata)):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing cell {i+1}/{len(adata)}")
        
        # Extract gene counts for this cell
        if sparse.issparse(adata.X):
            gene_counts = adata[i, :].X.toarray().flatten()
        else:
            gene_counts = adata[i, :].X.flatten()
        
        # Extract metadata (uses pre-computed dataset_metadata if available, otherwise extracts per-cell)
        cell_metadata = extract_metadata(adata, i, config, dataset_metadata if dataset_metadata else None)
        
        # Build prompt with cell type list
        prompt = build_prompt(cell_metadata, include_cell_types=cell_type_list)
        
        # Generate prediction
        try:
            result = model.predict(
                prompt,
                gene_counts=gene_counts,
                sc_metadata=cell_metadata,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                max_new_tokens=args.max_new_tokens,
            )
            
            # Extract text prediction
            text_response = result.get("text", "")
            pred = extract_prediction(text_response)
            predicted_cell_types.append(pred)
            
        except Exception as e:
            logger.warning(f"Error predicting cell {i}: {e}")
            predicted_cell_types.append("")
    
    # Create results DataFrame
    logger.info("Creating results DataFrame...")
    true_labels = list(adata.obs[annotation_key].values)
    
    results_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'true_labels': true_labels,
        'predicted_labels': predicted_cell_types
    })
    
    # Save results
    logger.info(f"Saving results to: {args.output}")
    results_df.to_csv(args.output, index=False)
    logger.info("Prediction completed successfully")


if __name__ == "__main__":
    main()

