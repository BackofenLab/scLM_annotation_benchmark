#!/usr/bin/env python3
"""
Test script for C2S-Scale model with batch processing.

This script loads the model and runs inference on the entire dataset
in batches, similar to the main predict.py script.
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
    """Load single-cell data from h5ad file."""
    logger.info(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)
    logger.info(f"Loaded data with shape: {adata.shape}")
    
    if 'organism' not in adata.obs.columns:
        adata.obs['organism'] = 'Homo sapiens'
        logger.info("Set default organism to 'Homo sapiens'")
    
    return adata


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(model_id: str, cache_dir: str = None, multi_gpu: bool = False):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    
    if multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Multi-GPU mode: distributing model across {num_gpus} GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        device = "cuda:0"
        if hasattr(model, 'hf_device_map'):
            logger.info(f"Device map: {model.hf_device_map}")
    else:
        device = get_device()
        logger.info(f"Single device mode: {device}")
        
        # Determine dtype based on device
        # MPS supports float16 but some operations may need float32
        if device == "cuda":
            dtype = torch.float16
        elif device == "mps":
            # Use float32 for better MPS compatibility
            dtype = torch.float32
        else:
            dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        ).to(device)
    
    model.eval()
    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer, device


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
    
    adata_copy = adata.copy()
    
    if "counts" not in adata.layers:
        raise ValueError("Counts layer not found. C2S-Scale needs raw counts.")
    
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
    
    if 'organism' not in adata_copy.obs.columns:
        adata_copy.obs['organism'] = 'Homo sapiens'
    
    # Convert to arrow dataset
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata_copy, 
        random_state=SEED, 
        sentence_delimiter=' ',
        label_col_names=['organism', 'cell_type']
    )
    
    logger.info(f"Created arrow dataset with {arrow_ds.num_rows} cells")
    return arrow_ds


def truncate_cell_sentence(cell_sentence: str, n_genes: int) -> str:
    """Truncate cell sentence to top N genes."""
    genes = cell_sentence.split()
    truncated = genes[:n_genes]
    return ' '.join(truncated)


def format_prompts(arrow_ds: Dataset, n_genes: int) -> Dataset:
    """
    Format prompts for cell type prediction.
    
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
    """Extract the predicted cell type from model response."""
    pred = response.split("The cell type corresponding to these genes is:")[-1].strip()
    pred = pred.replace('<ctrl100>', '').replace('<eos>', '').replace('<pad>', '')
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
        for output in outputs:
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
    results_df.to_csv(output_path, index=False)
    logger.info("Results saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Test C2S-Scale on dataset with batch processing")
    parser.add_argument("--input", "-i", required=True, help="Input h5ad file path")
    parser.add_argument("--output", "-o", default=None,
                       help="Output CSV file path (default: predictions.csv in current dir)")
    parser.add_argument("--model", "-m", default="vandijklab/C2S-Scale-Gemma-2-27B",
                       help="HuggingFace model ID")
    parser.add_argument("--annotation_key", default="AIFI_L2",
                       help="Key in adata.obs containing true labels")
    parser.add_argument("--n_genes", type=int, default=200,
                       help="Number of genes to use")
    parser.add_argument("--cache_dir", default=None, help="Model cache directory")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                       help="Maximum new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference (default: 8)")
    parser.add_argument("--multi_gpu", action="store_true",
                       help="Enable multi-GPU mode")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = str(Path.cwd() / f"predictions_{args.annotation_key}.csv")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log device info
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"         Memory: {mem:.1f} GB")
    elif torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
    else:
        logger.info("No GPU available, will use CPU")
    
    # Load data
    adata = load_data(args.input)
    
    # Load model
    model, tokenizer, device = load_model(args.model, args.cache_dir, args.multi_gpu)
    
    # Convert to arrow dataset
    arrow_ds = convert_to_arrow_dataset(adata, args.annotation_key)
    
    # Format prompts
    formatted_ds = format_prompts(arrow_ds, args.n_genes)
    
    # Print example prompt
    if formatted_ds.num_rows > 0:
        logger.info(f"\n{'='*60}\nEXAMPLE PROMPT:\n{'='*60}\n{formatted_ds[0]['model_input'][:500]}...\n{'='*60}")
    
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
    results_df = create_results_dataframe(adata, predicted_cell_types, args.annotation_key)
    
    if results_df is not None:
        # Save results
        save_results(results_df, args.output)
        
        # Print summary statistics
        logger.info(f"\n{'='*60}\nRESULTS SUMMARY:\n{'='*60}")
        logger.info(f"Total cells: {len(results_df)}")
        
        # Calculate accuracy (case-insensitive)
        matches = sum(
            1 for t, p in zip(results_df['true_labels'], results_df['predicted_labels'])
            if str(t).lower() == str(p).lower()
        )
        accuracy = matches / len(results_df) * 100
        logger.info(f"Exact match accuracy: {accuracy:.2f}%")
        
        # Show first few predictions
        logger.info(f"\nFirst 5 predictions:")
        for i, row in results_df.head(5).iterrows():
            match = '✓' if str(row['true_labels']).lower() == str(row['predicted_labels']).lower() else '✗'
            logger.info(f"  {match} True: {row['true_labels']:<30} Pred: {row['predicted_labels']}")
        
        logger.info(f"\nResults saved to: {args.output}")
        logger.info(f"{'='*60}")
    else:
        logger.error("Failed to create results DataFrame")
        sys.exit(1)


if __name__ == "__main__":
    main()
