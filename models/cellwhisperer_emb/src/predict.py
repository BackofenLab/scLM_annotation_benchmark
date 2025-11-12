"""
Computes CellWhisperer similarity scores between cells and cell type labels
Evaluates predictions against true labels and computes ROC/AUC metrics
OPTIMIZED FOR MAC MPS: Uses cache=False to avoid repeated model loading

Usage:
  python predict.py --input <h5ad_path> --output <csv_path> --true_label_key <label_key>

Outputs:
  - CSV file with true_labels and predicted_labels columns
"""
import sys, torch, anndata as ad, pathlib
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from util import download_if_missing

def parse_args():
    parser = argparse.ArgumentParser(description='CellWhisperer cell type prediction')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to output CSV file (base name)')
    parser.add_argument('--annotation_keys', nargs='+', required=True, help='Keys in adata.obs for true cell type labels')
    parser.add_argument('--cache_dir', required = True, help='Path to cache directory. Model checkpoints will be downloaded here.')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
adata_path = args.input
output_csv_base = args.output
ANNOTATION_KEYS = args.annotation_keys
CACHE_DIR = pathlib.Path(args.cache_dir)
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
#gene_ids = adata.var["gene_name"].tolist()
# Reduce batch size for Mac MPS to avoid memory issues
batch_size = 32 if device.type == "mps" else 128 if device.type == "cuda" else 16
print(f"Using batch size: {batch_size}")

# ----------------- create embeddings ------------------
print(f"Processing {adata.n_obs} cells...")

# Get logit scale from the model (needed for scoring)
logit_scale = pl_model.model.discriminator.temperature.exp()

import time
start_time = time.time()

# Process each annotation key separately
results_per_key = {}

for annotation_key in ANNOTATION_KEYS:
    print(f"\n=== Processing annotation key: {annotation_key} ===")
    
    # Check if annotation key exists
    if annotation_key not in adata.obs.columns:
        print(f"Warning: '{annotation_key}' not found in adata.obs.columns")
        print(f"Available columns: {list(adata.obs.columns)}")
        continue
    
    # Get unique cell type labels for this annotation level
    cell_type_labels = adata.obs[annotation_key].unique().tolist()
    print(f"Computing similarity scores to {len(cell_type_labels)} cell type labels...")
    
    # Compute similarity scores between transcriptomes and cell type labels
    scores, grouping_keys = score_transcriptomes_vs_texts(
        transcriptome_input=adata,  # Your AnnData object
        text_list_or_text_embeds=cell_type_labels,  # Cell type labels
        logit_scale=logit_scale,  # Scale factor from the model
        model=model,  # The loaded model
        transcriptome_processor=transcriptome_processor,  # From load_cellwhisperer_model
        batch_size=batch_size,
        average_mode=None,  # Don't average, get scores for each cell
        score_norm_method="softmax"  # Normalize scores using softmax
    )
    
    # Store results for this annotation key
    results_per_key[annotation_key] = {
        'scores': scores,
        'cell_type_labels': cell_type_labels,
        'true_labels': adata.obs[annotation_key].values,
        'cell_id': adata.obs[annotation_key].index
    }
    
    print(f"Similarity scores shape: {scores.shape}")  # Should be [n_cell_types, n_cells]
    print(f"Top scoring cell type for first cell: {cell_type_labels[scores[:, 0].argmax()]}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# ----------------- Save Results for Each Annotation Key ---------------------------
print("\n=== SAVING RESULTS ===")

from pathlib import Path

# Get base output path info
output_path = Path(output_csv_base)
output_dir = output_path.parent
output_stem = output_path.stem
output_suffix = output_path.suffix

for annotation_key, results in results_per_key.items():
    print(f"\n--- Processing results for {annotation_key} ---")
    
    scores = results['scores']
    cell_type_labels = results['cell_type_labels']
    true_labels = results['true_labels']
    cell_id = results['cell_id']
    
    print(f"Found {len(np.unique(true_labels))} unique true labels")
    
    # Get predicted labels (highest scoring cell type for each cell)
    predicted_indices = scores.argmax(dim=0)  # Shape: [n_cells]
    predicted_labels = [cell_type_labels[i] for i in predicted_indices]
    
    # Create comparison DataFrame with only required columns
    results_df = pd.DataFrame({
        'cell_id': cell_id,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    })
    
    # Create output filename for this annotation level
    output_csv = output_dir / f"{output_stem}_{annotation_key}{output_suffix}"
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    
    # Calculate accuracy
    accuracy = (results_df['true_labels'] == results_df['predicted_labels']).mean()
    print(f"Overall accuracy: {accuracy:.3f}")
    
    # Calculate ROC curves and AUC for each cell type
    print(f"--- ROC/AUC Analysis for {annotation_key} ---")
    
    # Create binary labels for ROC analysis
    auc_scores = {}
    
    for i, cell_type in enumerate(cell_type_labels):
        # Binary classification: this cell type vs all others
        y_true_binary = (true_labels == cell_type).astype(int)
        y_scores = scores[i, :].numpy()  # Similarity scores for this cell type
        
        # Only calculate AUC if we have both positive and negative samples
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            auc_score = auc(fpr, tpr)
            auc_scores[cell_type] = auc_score
            print(f"{cell_type:15} AUC: {auc_score:.3f}")
        else:
            auc_scores[cell_type] = np.nan
            print(f"{cell_type:15} AUC: N/A (no positive/negative samples)")
    
    # Calculate mean AUC
    valid_aucs = [score for score in auc_scores.values() if not np.isnan(score)]
    if valid_aucs:
        mean_auc = np.mean(valid_aucs)
        print(f"Mean AUC for {annotation_key}: {mean_auc:.3f}")
    else:
        print(f"Mean AUC for {annotation_key}: N/A")
    
    print(f"Label distribution in {annotation_key}:")
    label_counts = pd.Series(true_labels).value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(true_labels)*100:.1f}%)")

print(f"\nPrediction completed successfully!")