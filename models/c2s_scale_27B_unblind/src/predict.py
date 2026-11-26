#!/usr/bin/env python3
"""Cell type annotation with C2S-Scale-Gemma-2-27B, with the dataset's labels as candidates (unblind).

Each cell becomes a cell sentence (gene names by descending expression), and the model
completes the C2S-Scale cell type prompt for the top --n_genes genes; the prompt also
lists the sorted cell type labels of the dataset to pick from. The model is loaded with
transformers; cell2sentence only builds the cell sentences.

Input: preprocessed h5ad with raw counts in layers["counts"] and gene symbols in
var["gene_names"]. Output: CSV with cell_id, true_labels and predicted_labels.
Called by run.sh with --multi_gpu, which spreads the 27B model over all visible GPUs.
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

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data(data_path: str):
    """Read the h5ad file; add obs["organism"] = "Homo sapiens" if the column is missing."""
    logger.info(f"Loading data from: {data_path}")
    try:
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded data with shape: {adata.shape}")
        
        if 'organism' not in adata.obs.columns:
            adata.obs['organism'] = 'Homo sapiens'
            logger.info("Set default organism to 'Homo sapiens'")
        
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def get_dtype_for_device(device: str):
    if device == "cuda":
        return torch.bfloat16
    else:
        # The model was trained in bfloat16; MPS and CPU use float16.
        return torch.float16


def load_model(model_id: str, device: str, cache_dir: str = None, multi_gpu: bool = False):
    """Return model, tokenizer and the device for the inputs; multi_gpu splits the model over all GPUs."""
    logger.info(f"Loading model: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        
        if multi_gpu and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Multi-GPU mode enabled. Found {num_gpus} GPU(s)")
            
            dtype = torch.bfloat16
            logger.info(f"Using dtype: {dtype} for CUDA multi-GPU")
            
            if num_gpus > 1:
                logger.info(f"Distributing model across {num_gpus} GPUs using device_map='auto'")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                # With device_map="auto", the inputs go to the first GPU.
                effective_device = "cuda:0"
                
                if hasattr(model, 'hf_device_map'):
                    logger.info(f"Model device map: {model.hf_device_map}")
            else:
                logger.info("Only 1 GPU found, falling back to single-GPU mode")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    torch_dtype=dtype,
                ).to(device)
                effective_device = device
        else:
            dtype = get_dtype_for_device(device)
            logger.info(f"Using dtype: {dtype} for device: {device}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                torch_dtype=dtype,
            ).to(device)
            effective_device = device
        
        model.eval()
        logger.info(f"Model loaded successfully. Effective input device: {effective_device}")
        return model, tokenizer, effective_device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def convert_to_arrow_dataset(adata, annotation_key: str):
    """Normalize the raw counts (total counts, log10) and build the cell sentences as an arrow dataset."""
    logger.info(f"Converting data to arrow format for annotation key: {annotation_key}")
    try:
        adata_copy = adata.copy()
        
        if "counts" not in adata.layers:
            raise ValueError(
                "Counts layer not found in adata.layers. "
                "Run preprocessing step first. C2S-Scale needs raw counts."
            )
        adata_copy.X = adata.layers["counts"]
        logger.info("Setting X to raw counts, normalizing and log-transforming with base 10")
        sc.pp.normalize_total(adata_copy)
        sc.pp.log1p(adata_copy, base=10)

        adata_copy.var_names = adata_copy.var["gene_names"].astype(str)
        adata_copy.var_names_make_unique()

        # cell2sentence uppercases genes when building the vocabulary. If `var_names` contains
        # case-insensitive duplicates (e.g. 'PISD' and 'Pisd'), the vocabulary becomes shorter
        # than `n_vars`, and sentence generation can crash with IndexError.
        upper = np.array([str(x).upper() for x in adata_copy.var_names], dtype=object)
        if len(np.unique(upper)) != adata_copy.n_vars:
            adata_copy.var_names = upper
            adata_copy.var_names_make_unique()

        if annotation_key not in adata_copy.obs.columns:
            raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")
        adata_copy.obs['cell_type'] = adata_copy.obs[annotation_key]
        
        if 'organism' not in adata_copy.obs.columns:
            adata_copy.obs['organism'] = 'Homo sapiens'

        label_col_names = ['organism', 'cell_type']
        
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
    """Keep the first n_genes names of a cell sentence (space-separated, by descending expression)."""
    genes = cell_sentence.split()
    truncated = genes[:n_genes]
    return ' '.join(truncated)


def get_unique_cell_types(adata, annotation_key: str) -> list:
    """Return the sorted labels of obs[annotation_key], the candidates in the prompt."""
    if annotation_key not in adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")
    
    unique_cell_types = sorted(adata.obs[annotation_key].unique().tolist())
    logger.info(f"Found {len(unique_cell_types)} unique cell types for annotation key '{annotation_key}'")
    return unique_cell_types


def format_prompts(arrow_ds: Dataset, n_genes: int, cell_type_list: list) -> Dataset:
    """Return a dataset with one C2S-Scale cell type prompt per cell in column model_input."""
    logger.info(f"Formatting prompts with top {n_genes} genes")
    logger.info(f"Including {len(cell_type_list)} cell types in prompt for selection")
    
    cell_types_str = ", ".join(cell_type_list)
    
    prompts = []
    for i in range(arrow_ds.num_rows):
        sample = arrow_ds[i]
        
        cell_sentence = truncate_cell_sentence(sample["cell_sentence"], n_genes)
        organism = sample.get("organism", "Homo sapiens")
        
        # C2S-Scale prompt template for cell type prediction, extended by the candidate list
        prompt = f"""The following is a list of {n_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the cell type which this cell belongs to based on its gene expression. Pick from this list: {cell_types_str}.
Cell sentence: {cell_sentence}.
The cell type corresponding to these genes is:"""
        
        prompts.append(prompt)
    
    formatted_ds = Dataset.from_dict({"model_input": prompts})
    
    logger.info(f"Formatted {formatted_ds.num_rows} prompts")
    return formatted_ds


def extract_prediction(response: str) -> str:
    """Return the text generated after the prompt, without special tokens and trailing periods."""
    pred = response.split("The cell type corresponding to these genes is:")[-1].strip()
    
    pred = pred.replace('<ctrl100>', '')
    pred = pred.replace('<eos>', '')
    pred = pred.replace('<pad>', '')
    
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
    """Generate the answers in batches with greedy decoding and return the predicted labels."""
    logger.info(f"Generating predictions for {formatted_ds.num_rows} cells (batch_size={batch_size})")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # batched generation needs left padding
    
    predicted_cell_types = []
    all_prompts = formatted_ds["model_input"]
    num_samples = len(all_prompts)
    
    for batch_start in tqdm(range(0, num_samples, batch_size), desc="Predicting batches"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_prompts = all_prompts[batch_start:batch_end]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=False)
            pred = extract_prediction(response)
            predicted_cell_types.append(pred)
    logger.info(f"Full response of the last cell: {response}")
    
    logger.info("Prediction completed successfully")
    return predicted_cell_types


def create_results_dataframe(adata, predicted_cell_types: list, label_key: str):
    """Return a DataFrame with cell_id, true_labels and predicted_labels, or None if label_key is missing."""
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
    logger.info(f"Saving results to: {output_path}")
    try:
        results_df.to_csv(output_path, index=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)


def main():
    default_input = str(Path(__file__).resolve().parents[3] / "util" / "test_adata.h5ad")
    default_output = str(Path.cwd() / "c2s_scale_predictions.csv")

    parser = argparse.ArgumentParser(description="C2S-Scale prediction for single-cell data")
    parser.add_argument("--input", "-i", default=default_input,
                       help=f"Input h5ad file path (default: {default_input})")
    parser.add_argument("--output", "-o", default=default_output,
                       help=f"Output CSV file path base name (default: {default_output})")
    parser.add_argument("--model", "-m", default="vandijklab/C2S-Scale-Gemma-2-27B", 
                       help="HuggingFace model ID (default: vandijklab/C2S-Scale-Gemma-2-27B)")
    parser.add_argument("--annotation_key", default="cell_ontology_class",
                       help="Key in adata.obs containing true labels (default: cell_ontology_class)")
    parser.add_argument("--n_genes", type=int, default=200,
                       help="Number of genes to use for prediction (default: 200)")
    parser.add_argument("--cache_dir", default=None,
                       help="Directory for caching models")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                       help="Maximum new tokens to generate (default: 20)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference (default: 8). Increase for faster inference if memory allows.")
    parser.add_argument("--multi_gpu", action="store_true",
                       help="Enable multi-GPU mode to distribute model across all available GPUs")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    adata = load_data(args.input)
    
    model, tokenizer, effective_device = load_model(
        args.model, device, args.cache_dir, multi_gpu=args.multi_gpu
    )
    
    annotation_key = args.annotation_key
    logger.info(f"Processing annotation key: {annotation_key}")
    
    if annotation_key not in adata.obs.columns:
        logger.error(f"'{annotation_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        sys.exit(1)
    
    arrow_ds = convert_to_arrow_dataset(adata, annotation_key)
    
    # Candidates for the prompt: the dataset's own labels
    cell_type_list = get_unique_cell_types(adata, annotation_key)
    
    formatted_ds = format_prompts(arrow_ds, args.n_genes, cell_type_list)
    
    if formatted_ds.num_rows > 0:
        logger.info(f"Example prompt:\n{formatted_ds[0]['model_input'][:500]}...")
    
    predicted_cell_types = generate_predictions(
        model, 
        tokenizer, 
        formatted_ds, 
        effective_device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    
    results_df = create_results_dataframe(adata, predicted_cell_types, annotation_key)
    
    if results_df is not None:
        print(f"Saving results to: {args.output}")
        save_results(results_df, args.output)
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.error("Failed to create results DataFrame")
        sys.exit(1)
    
    logger.info("C2S-Scale prediction pipeline completed successfully")


if __name__ == "__main__":
    main()

