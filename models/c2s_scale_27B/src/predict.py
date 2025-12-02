#!/usr/bin/env python3
"""
Simplified C2S-Scale prediction script for single-cell data annotation.

This script loads the model directly using transformers and generates
predictions without the full CSData/CSModel abstractions.
"""

import argparse
import pandas as pd
import scanpy as sc
import cell2sentence as cs
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from pathlib import Path
import sys
import logging
import random
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Set seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data(data_path: str):
    """
    Load single-cell data from h5ad file.
    
    Args:
        data_path: Path to the h5ad file
        
    Returns:
        Loaded AnnData object
    """
    logger.info(f"Loading data from: {data_path}")
    try:
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded data with shape: {adata.shape}")
        
        # Set required metadata for cell2sentence
        if 'organism' not in adata.obs.columns:
            adata.obs['organism'] = 'Homo sapiens'  # Default to human
            logger.info("Set default organism to 'Homo sapiens'")
        
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def load_model(model_id: str, device: str, cache_dir: str = None):
    """
    Load model and tokenizer directly from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on
        cache_dir: Optional cache directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def convert_to_arrow_dataset(adata, annotation_key: str):
    """
    Convert AnnData to arrow dataset format for C2S-Scale.
    
    Args:
        adata: Input AnnData object
        annotation_key: Key in adata.obs to use as cell_type
        
    Returns:
        HuggingFace arrow dataset with cell sentences
    """
    logger.info(f"Converting data to arrow format for annotation key: {annotation_key}")
    try:
        # Create a copy to avoid modifying the original
        adata_copy = adata.copy()
        
        # Set X to raw counts and apply log10 transformation
        if "counts" not in adata.layers:
            raise ValueError(
                "Counts layer not found in adata.layers. "
                "Run preprocessing step first. C2S-Scale needs raw counts."
            )
        adata_copy.X = adata.layers["counts"]
        logger.info("Setting X to raw counts, normalizing and log-transforming with base 10")
        sc.pp.normalize_total(adata_copy)
        sc.pp.log1p(adata_copy, base=10)

        # Set var names to gene names
        adata_copy.var_names = adata_copy.var["gene_names"]

        # Set cell_type column
        if annotation_key not in adata_copy.obs.columns:
            raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")
        adata_copy.obs['cell_type'] = adata_copy.obs[annotation_key]
        
        # Ensure organism is set
        if 'organism' not in adata_copy.obs.columns:
            adata_copy.obs['organism'] = 'Homo sapiens'

        # Keep organism and cell_type columns
        label_col_names = ['organism', 'cell_type']
        
        # Convert to arrow dataset
        arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
            adata=adata_copy, 
            random_state=SEED, 
            sentence_delimiter=' ',
            label_col_names=label_col_names
        )
        
        logger.info(f"Created arrow dataset with {arrow_ds.num_rows} cells")
        logger.info(f"Dataset features: {arrow_ds.features}")
        logger.info(f"Example sample keys: {list(arrow_ds[0].keys())}")
        
        return arrow_ds
        
    except Exception as e:
        logger.error(f"Failed to convert data: {e}")
        sys.exit(1)


def truncate_cell_sentence(cell_sentence: str, n_genes: int) -> str:
    """
    Truncate cell sentence to top N genes.
    
    Cell sentences are space-separated gene names ordered by expression level.
    
    Args:
        cell_sentence: Space-separated gene names
        n_genes: Number of top genes to keep
        
    Returns:
        Truncated cell sentence with top n_genes
    """
    genes = cell_sentence.split()
    truncated = genes[:n_genes]
    return ' '.join(truncated)


def format_prompts(arrow_ds: Dataset, n_genes: int) -> Dataset:
    """
    Format prompts for cell type prediction.
    
    Uses the C2S-Scale prompt template with truncated cell sentences.
    
    Args:
        arrow_ds: Arrow dataset with cell_sentence and organism columns
        n_genes: Number of top genes to include in prompt
        
    Returns:
        Dataset with 'model_input' column containing formatted prompts
    """
    logger.info(f"Formatting prompts with top {n_genes} genes")
    
    prompts = []
    for i in range(arrow_ds.num_rows):
        sample = arrow_ds[i]
        
        # Truncate cell sentence to top N genes
        cell_sentence = truncate_cell_sentence(sample["cell_sentence"], n_genes)
        organism = sample.get("organism", "Homo sapiens")
        
        # Format prompt using C2S-Scale template
        prompt = f"""The following is a list of {n_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the cell type which this cell belongs to based on its gene expression.
Cell sentence: {cell_sentence}.
The cell type corresponding to these genes is:"""
        
        prompts.append(prompt)
    
    # Create dataset with prompts
    formatted_ds = Dataset.from_dict({"model_input": prompts})
    
    logger.info(f"Formatted {formatted_ds.num_rows} prompts")
    return formatted_ds


def extract_prediction(response: str) -> str:
    """
    Extract the predicted cell type from model response.
    
    Splits on the prompt ending and removes special tokens.
    
    Args:
        response: Full model response including prompt
        
    Returns:
        Cleaned predicted cell type
    """
    # Split on prompt ending and take the generated part
    pred = response.split("The cell type corresponding to these genes is:")[-1].strip()
    
    # Remove special tokens
    pred = pred.replace('<ctrl100>', '')
    pred = pred.replace('<eos>', '')
    pred = pred.replace('<pad>', '')
    
    # Remove trailing punctuation and whitespace
    pred = pred.strip().rstrip('.')
    
    return pred


def generate_predictions(
    model, 
    tokenizer, 
    formatted_ds: Dataset, 
    device: str,
    max_new_tokens: int = 20,
    batch_size: int = 8
):
    """
    Generate cell type predictions from formatted prompts using batched inference.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded tokenizer
        formatted_ds: Dataset with formatted prompts
        device: Device to run inference on
        max_new_tokens: Maximum new tokens to generate
        batch_size: Number of samples to process in parallel
        
    Returns:
        List of predicted cell types
    """
    logger.info(f"Generating predictions for {formatted_ds.num_rows} cells (batch_size={batch_size})")
    
    # Ensure tokenizer has padding configured
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation - pad on left
    
    predicted_cell_types = []
    all_prompts = formatted_ds["model_input"]
    num_samples = len(all_prompts)
    
    # Process in batches
    for batch_start in tqdm(range(0, num_samples, batch_size), desc="Predicting batches"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_prompts = all_prompts[batch_start:batch_end]
        
        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate for entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False  # Greedy decoding for reproducibility
            )
        
        # Decode each response in batch
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=False)
            pred = extract_prediction(response)
            predicted_cell_types.append(pred)
    
    logger.info("Prediction completed successfully")
    return predicted_cell_types


def create_results_dataframe(adata, predicted_cell_types: list, label_key: str):
    """
    Create a DataFrame with true and predicted labels.
    
    Args:
        adata: Input AnnData with true labels
        predicted_cell_types: List of predicted cell types
        label_key: Key in adata.obs containing true labels
        
    Returns:
        DataFrame with cell_id, true_labels, and predicted_labels
    """
    logger.info(f"Creating results DataFrame for annotation key: {label_key}")
    
    if label_key not in adata.obs.columns:
        logger.error(f"'{label_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        return None
    
    true_labels = list(adata.obs[label_key].values)
    
    results_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'true_labels': true_labels,
        'predicted_labels': predicted_cell_types
    })
    
    logger.info(f"Created results DataFrame with {len(results_df)} rows")
    return results_df


def save_results(results_df: pd.DataFrame, output_path: str):
    """Save results to CSV file."""
    logger.info(f"Saving results to: {output_path}")
    try:
        results_df.to_csv(output_path, index=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)


def main():
    """Main function to run C2S-Scale prediction pipeline."""
    # Get default paths
    default_input = str(Path(__file__).resolve().parents[3] / "util" / "test_adata.h5ad")
    default_output = str(Path.cwd() / "c2s_scale_predictions.csv")

    parser = argparse.ArgumentParser(description="C2S-Scale prediction for single-cell data")
    parser.add_argument("--input", "-i", default=default_input,
                       help=f"Input h5ad file path (default: {default_input})")
    parser.add_argument("--output", "-o", default=default_output,
                       help=f"Output CSV file path base name (default: {default_output})")
    parser.add_argument("--model", "-m", default="vandijklab/C2S-Scale-Gemma-2-27B", 
                       help="HuggingFace model ID (default: vandijklab/C2S-Scale-Gemma-2-27B)")
    parser.add_argument("--annotation_keys", nargs='+', default=["cell_ontology_class"],
                       help="Keys in adata.obs containing true labels (default: cell_ontology_class)")
    parser.add_argument("--n_genes", type=int, default=200,
                       help="Number of genes to use for prediction (default: 200)")
    parser.add_argument("--cache_dir", default=None,
                       help="Directory for caching models")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                       help="Maximum new tokens to generate (default: 20)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference (default: 8). Increase for faster inference if memory allows.")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    adata = load_data(args.input)
    
    # Load model and tokenizer (once, reuse for all annotation keys)
    model, tokenizer = load_model(args.model, device, args.cache_dir)
    
    # Get output path components
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    
    # Process each annotation key
    for annotation_key in args.annotation_keys:
        logger.info(f"Processing annotation key: {annotation_key}")
        
        # Check if annotation key exists
        if annotation_key not in adata.obs.columns:
            logger.error(f"'{annotation_key}' not found in adata.obs")
            logger.error(f"Available keys: {list(adata.obs.columns)}")
            logger.warning(f"Skipping {annotation_key}")
            continue
        
        # Convert to arrow dataset
        arrow_ds = convert_to_arrow_dataset(adata, annotation_key)
        
        # Format prompts with custom formatter
        formatted_ds = format_prompts(arrow_ds, args.n_genes)
        
        # Print example prompt
        if formatted_ds.num_rows > 0:
            logger.info(f"Example prompt:\n{formatted_ds[0]['model_input'][:500]}...")
        
        # Generate predictions
        predicted_cell_types = generate_predictions(
            model, 
            tokenizer, 
            formatted_ds, 
            device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )
        
        # Create results DataFrame
        results_df = create_results_dataframe(adata, predicted_cell_types, annotation_key)
        
        if results_df is not None:
            # Create output filename for this annotation key
            output_csv = output_dir / f"{output_stem}_{annotation_key}{output_suffix}"
            
            # Save results
            save_results(results_df, str(output_csv))
            logger.info(f"Results for {annotation_key} saved to: {output_csv}")
        else:
            logger.warning(f"Skipping {annotation_key} due to errors")
    
    logger.info("C2S-Scale prediction pipeline completed successfully")


if __name__ == "__main__":
    main()

