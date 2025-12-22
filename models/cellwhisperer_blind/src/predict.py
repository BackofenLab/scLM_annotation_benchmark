"""
Computes CellWhisperer similarity scores between cells and ALL cell types from ontology.
This is a "blind" prediction mode that scores against all known cell types.

OPTIMIZED FOR MAC MPS: Uses cache=False to avoid repeated model loading

Usage:
  python predict.py --input <h5ad_path> --output <csv_path> --cache_dir <cache_dir>

Outputs:
  - CSV file with cell_id and predicted_labels columns
"""
import sys, torch, anndata as ad, pathlib
import pandas as pd
import numpy as np
import argparse
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from util import download_if_missing

def parse_args():
    parser = argparse.ArgumentParser(description='CellWhisperer cell type prediction (blind mode)')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--cache_dir', required=True, help='Path to cache directory. Model checkpoints and cell type list will be loaded from here.')
    parser.add_argument('--cell_types_file', default='all_cell_types.txt', help='Name of cell types file in cache_dir (default: all_cell_types.txt)')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
adata_path = args.input
output_csv = args.output
CACHE_DIR = pathlib.Path(args.cache_dir)
CELL_TYPES_FILE = CACHE_DIR / args.cell_types_file
# Configuration
CKPT = CACHE_DIR / "cellwhisperer_emb/cellwhisperer_jointemb_v1.ckpt"              # adjust!
URL = "https://medical-epigenomics.org/papers/schaefer2024/data/models/cellwhisperer_clip_v1.ckpt"

download_if_missing(URL, CKPT)
# ----------------- load model on GPU -------------------
# IMPORTANT: cache=False to avoid repeated GPU loading on Mac MPS
pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(CKPT, eval=True, cache=False)
model = pl_model.model  # Get the actual TranscriptomeTextDualEncoderModel

# Use MPS if available on Mac, otherwise CUDA or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Mac MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Ensure model is on the correct device
pl_model = pl_model.to(device)
model = pl_model.model

# ----------------- load data -------------------------
adata = ad.read_h5ad(adata_path)
if "counts" in adata.layers:
    adata.X = adata.layers["counts"]
else:
    raise ValueError("Counts layer not found in adata.layers")
#gene_ids = adata.var["gene_name"].tolist()
# Reduce batch size for Mac MPS to avoid memory issues
batch_size = 32 if device.type == "mps" else 128 if device.type == "cuda" else 16
print(f"Using batch size: {batch_size}")

# ----------------- load cell types from file ------------------
print(f"Loading cell types from {CELL_TYPES_FILE}...")
if not CELL_TYPES_FILE.exists():
    raise FileNotFoundError(f"Cell types file not found: {CELL_TYPES_FILE}")

with open(CELL_TYPES_FILE, 'r') as f:
    cell_type_labels = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(cell_type_labels)} cell types from ontology")

# ----------------- create embeddings ------------------
print(f"Processing {adata.n_obs} cells...")

# Get logit scale from the model (needed for scoring)
logit_scale = pl_model.model.discriminator.temperature.exp()

import time
start_time = time.time()

print(f"\n=== Computing similarity scores against {len(cell_type_labels)} cell types ===")

# Compute similarity scores between transcriptomes and ALL cell type labels
scores, grouping_keys = score_transcriptomes_vs_texts(
    transcriptome_input=adata,  # Your AnnData object
    text_list_or_text_embeds=cell_type_labels,  # All cell type labels from ontology
    logit_scale=logit_scale,  # Scale factor from the model
    model=model,  # The loaded model
    transcriptome_processor=transcriptome_processor,  # From load_cellwhisperer_model
    batch_size=batch_size,
    average_mode=None,  # Don't average, get scores for each cell
    score_norm_method="softmax"  # Normalize scores using softmax
)

print(f"Similarity scores shape: {scores.shape}")  # Should be [n_cell_types, n_cells]
print(f"Top scoring cell type for first cell: {cell_type_labels[scores[:, 0].argmax()]}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# ----------------- Save Results ---------------------------
print("\n=== SAVING RESULTS ===")

cell_ids = adata.obs.index.tolist()

# Get predicted labels (highest scoring cell type for each cell)
predicted_indices = scores.argmax(dim=0)  # Shape: [n_cells]
predicted_labels = [cell_type_labels[i] for i in predicted_indices]

# Create and save results DataFrame
results_df = pd.DataFrame({
    'cell_id': cell_ids,
    'predicted_labels': predicted_labels
})

results_df.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")

print(f"\nPrediction completed successfully!")