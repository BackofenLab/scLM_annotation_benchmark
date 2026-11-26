#!/usr/bin/env python3
"""Cell type annotation with InstructCell, with the dataset's labels as candidates (constrained).

The model (zjunlp/InstructCell-instruct from the Hugging Face Hub) gets each cell's raw counts
together with a text prompt that names the species, tissue and sequencing method when the
dataset config gives them, as values (species, tissue, sequencing_method) or as obs columns
(species_key, tissue_key, sequencing_method_key), and it lists the dataset's cell type labels
to pick from. The model samples its answer, so the script sets a seed, and it runs on the CPU as
for the paper, because on a GPU some answers differ.

Input: preprocessed h5ad with raw counts in layers["counts"] and gene symbols in
var["gene_names"], and the dataset config YAML. Output: CSV with cell_id, true_labels and
predicted_labels. Called by run.sh.
"""

import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import yaml
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List
from scipy import sparse

# prepare.sh clones InstructCell next to this script.
sys.path.insert(0, str(Path(__file__).parent / "InstructCell"))
from mmllm.module import InstructCell
from utils import set_global_random_seed, unify_gene_features

logger = logging.getLogger(__name__)


def load_data(data_path: str):
    logger.info(f"Loading data from: {data_path}")
    try:
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded data with shape: {adata.shape}")
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def load_gene_vocab(vocab_path: str):
    logger.info(f"Loading gene vocabulary from: {vocab_path}")
    try:
        gene_vocab = np.load(vocab_path)
        logger.info(f"Loaded {len(gene_vocab)} genes in vocabulary")
        return gene_vocab
    except Exception as e:
        logger.error(f"Failed to load gene vocabulary: {e}")
        sys.exit(1)


def extract_metadata(adata: anndata.AnnData, cell_idx: int, config: Dict, dataset_metadata: Dict[str, str] = None) -> Dict[str, str]:
    """Species, tissue and sequencing method of one cell; a non-empty dataset_metadata is returned as is."""
    metadata = {}
    
    if dataset_metadata:
        metadata.update(dataset_metadata)
        return metadata
    
    # A value in the config wins over a per-cell obs column.
    if 'species' in config:
        metadata['species'] = str(config['species']).strip()
    else:
        species_key = config.get('species_key')
        if species_key and species_key in adata.obs.columns:
            species_val = adata.obs.iloc[cell_idx][species_key]
            if pd.notna(species_val) and str(species_val).strip():
                metadata['species'] = str(species_val).strip()
    
    if 'tissue' in config:
        metadata['tissue'] = str(config['tissue']).strip()
    else:
        tissue_key = config.get('tissue_key')
        if tissue_key and tissue_key in adata.obs.columns:
            tissue_val = adata.obs.iloc[cell_idx][tissue_key]
            if pd.notna(tissue_val) and str(tissue_val).strip():
                metadata['tissue'] = str(tissue_val).strip()
    
    if 'sequencing_method' in config:
        metadata['sequencing_method'] = str(config['sequencing_method']).strip()
    else:
        seq_key = config.get('sequencing_method_key')
        if seq_key and seq_key in adata.obs.columns:
            seq_val = adata.obs.iloc[cell_idx][seq_key]
            if pd.notna(seq_val) and str(seq_val).strip():
                metadata['sequencing_method'] = str(seq_val).strip()
    
    return metadata


def build_prompt(metadata: Dict[str, str], include_cell_types: Optional[List[str]] = None) -> str:
    """Build the annotation prompt from the metadata, with the candidate cell types if given."""
    parts = []
    
    if 'species' in metadata:
        parts.append(f"Can you help me annotate this single cell from a {metadata['species']}?")
    else:
        parts.append("Can you help me annotate this single cell?")
    
    if 'sequencing_method' in metadata:
        parts.append(f"It was sequenced using {metadata['sequencing_method']}.")
    
    if 'tissue' in metadata:
        parts.append(f"It is derived from {metadata['tissue']}.")
    
    if include_cell_types:
        cell_types_str = ", ".join(include_cell_types)
        parts.append(f"Pick from this list: {cell_types_str}.")
    else:
        logger.warning("No cell types provided for constrained mode")
    
    # {input} is the placeholder for the cell's expression profile.
    parts.append("The gene expression profile is {input}. Thanks!")
    
    return " ".join(parts)


def extract_prediction(response: str) -> str:
    """Strip whitespace, trailing punctuation and an "Assistant:" or "text:" prefix from the answer."""
    pred = response.strip()
    
    pred = pred.rstrip('.,!?')
    
    if pred.startswith("Assistant:"):
        pred = pred[len("Assistant:"):].strip()
    
    if pred.lower().startswith("text:"):
        pred = pred[len("text:"):].strip()
    
    return pred.strip()


def expected_types(found):
    """The expected cell types: the names in EXPECTED_TYPES_FILE, one per line, if it is set (a
    dataset's own list in annotation mode), else the cell types found in the labels."""
    path = os.environ.get("EXPECTED_TYPES_FILE")
    if not path:
        return found
    with open(path, encoding="utf-8") as fh:
        return list(dict.fromkeys(line.strip() for line in fh if line.strip()))


def get_unique_cell_types(adata: anndata.AnnData, annotation_key: str) -> List[str]:
    """Return the sorted labels of obs[annotation_key], without missing values, or the sorted
    names of the dataset's own list (expected_types)."""
    if annotation_key not in adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")
    
    unique_cell_types = sorted(expected_types(adata.obs[annotation_key].dropna().unique().tolist()))
    logger.info(f"Found {len(unique_cell_types)} unique cell types for annotation key '{annotation_key}'")
    return unique_cell_types


def main():
    parser = argparse.ArgumentParser(description="InstructCell prediction for single-cell data (constrained mode)")
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
    parser.add_argument("--seed", type=int, default=1234,
                       help="Random seed for sampling the answer (default: 1234)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # InstructCell samples its answer; fix the seed so that a run can be reproduced.
    set_global_random_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # The model stays on the CPU, as for the paper: on a GPU it samples other answers for some
    # cells.
    logger.info("Running on the CPU")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    annotation_key = config.get('annotation_key', 'cell_type')
    logger.info(f"Using annotation key: {annotation_key}")
    
    adata = load_data(args.input)
    
    if annotation_key not in adata.obs.columns:
        logger.error(f"'{annotation_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        sys.exit(1)
    
    # Candidates for the prompt: the dataset's own labels
    cell_type_list = get_unique_cell_types(adata, annotation_key)
    logger.info(f"Including {len(cell_type_list)} cell types in prompt")
    
    # InstructCell takes raw counts and normalizes them itself: the released config sets
    # normalize_total and log_variational, and its own preprocessing keeps integer counts.
    # X holds log-normalized values here, so pass the counts layer.
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
        logger.info("Using raw counts from layers['counts']")
    else:
        values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
        if not np.allclose(values, np.round(values)):
            logger.error("InstructCell needs raw counts: no 'counts' layer, and X is not integer")
            sys.exit(1)
    
    # InstructCell's gene vocabulary, from the repository that prepare.sh clones
    script_dir = Path(__file__).parent
    vocab_path = script_dir / "InstructCell" / "exp_log" / "gene_vocab.npy"
    if not vocab_path.exists():
        logger.error(f"Gene vocabulary not found at: {vocab_path}")
        sys.exit(1)
    
    gene_vocab = load_gene_vocab(str(vocab_path))
    
    logger.info("Unifying gene features with vocabulary...")
    # The vocabulary holds gene symbols; preprocessing stores them in var["gene_names"].
    adata.var_names = adata.var["gene_names"]
    # Stop if too few genes match the vocabulary, e.g. mouse data that were not mapped to
    # human orthologs (bin/orthologize.py).
    matched = int(adata.var_names.isin(gene_vocab).sum())
    min_genes = int(os.environ.get("SCLLM_MIN_HUMAN_GENES", "5000"))
    logger.info(f"{matched} of {len(gene_vocab)} vocabulary genes found in the data")
    if matched < min_genes:
        logger.error(f"Only {matched} genes match InstructCell's vocabulary (minimum {min_genes}); "
                     "map mouse genes to human orthologs first with bin/orthologize.py")
        sys.exit(1)
    # The vocabulary is mostly uppercase, so uppercase matching is tried first.
    try:
        adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=True)
        logger.info("Successfully unified genes with uppercase matching")
    except ValueError as e:
        logger.warning(f"Uppercase matching failed: {e}. Trying case-sensitive matching...")
        adata = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)
    logger.info(f"Data shape after unification: {adata.shape}")
    
    model_id = f"zjunlp/InstructCell-{args.model_variant}"
    logger.info(f"Loading model: {model_id}")
    try:
        model = InstructCell.from_pretrained(model_id, cache_dir=args.cache_dir)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Dataset-wide values from the config; if there is any, extract_metadata uses only these.
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
    
    if not dataset_metadata:
        logger.info("Using per-cell metadata from obs columns")
        if config.get('species_key'):
            logger.info(f"  - species_key: {config.get('species_key')}")
        if config.get('tissue_key'):
            logger.info(f"  - tissue_key: {config.get('tissue_key')}")
        if config.get('sequencing_method_key'):
            logger.info(f"  - sequencing_method_key: {config.get('sequencing_method_key')}")
    
    logger.info(f"Generating predictions for {len(adata)} cells...")
    predicted_cell_types = []
    
    for i in range(len(adata)):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing cell {i+1}/{len(adata)}")
        
        if sparse.issparse(adata.X):
            gene_counts = adata[i, :].X.toarray().flatten()
        else:
            gene_counts = adata[i, :].X.flatten()
        
        cell_metadata = extract_metadata(adata, i, config, dataset_metadata if dataset_metadata else None)
        
        prompt = build_prompt(cell_metadata, include_cell_types=cell_type_list)
        
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
            
            text_response = result.get("text", "")
            pred = extract_prediction(text_response)
            predicted_cell_types.append(pred)
            
        except Exception as e:
            logger.warning(f"Error predicting cell {i}: {e}")
            predicted_cell_types.append("")
    
    logger.info("Creating results DataFrame...")
    true_labels = list(adata.obs[annotation_key].values)
    
    results_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'true_labels': true_labels,
        'predicted_labels': predicted_cell_types
    })
    
    logger.info(f"Saving results to: {args.output}")
    results_df.to_csv(args.output, index=False)
    logger.info("Prediction completed successfully")


if __name__ == "__main__":
    main()

