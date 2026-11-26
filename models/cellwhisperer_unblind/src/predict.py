"""Cell type annotation with CellWhisperer, with the dataset's labels as candidates (unblind).

CellWhisperer embeds each cell's transcriptome and every candidate label, and each cell gets the
label with the highest similarity. The candidates are the labels in obs[--annotation_key], or a
dataset's own list in EXPECTED_TYPES_FILE. The script also prints exact-match accuracy and per-label ROC AUC for the log.

Input: preprocessed h5ad with raw counts in layers["counts"] and human Ensembl IDs in
var["ensembl_id"]. Output: CSV with cell_id, true_labels and predicted_labels. The checkpoint
is downloaded into --cache_dir on first use. Called by run.sh.
"""
import sys, torch, anndata as ad, pathlib
import os
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
    parser.add_argument('--annotation_key', required=True, help='Key in adata.obs for true cell type labels')
    parser.add_argument('--cache_dir', required = True, help='Path to cache directory. Model checkpoints will be downloaded here.')
    return parser.parse_args()

args = parse_args()
adata_path = args.input
output_csv_base = args.output
ANNOTATION_KEY = args.annotation_key
CACHE_DIR = pathlib.Path(args.cache_dir)
# CellWhisperer CLIP checkpoint from the authors' website, stored in the cache directory
CKPT = CACHE_DIR / "cellwhisperer_emb/cellwhisperer_jointemb_v1.ckpt"
URL = "https://medical-epigenomics.org/papers/schaefer2024/data/models/cellwhisperer_clip_v1.ckpt"

download_if_missing(URL, CKPT)
# cache=False avoids repeated model loading on Apple MPS.
pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(CKPT, eval=True, cache=False)
model = pl_model.model

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Mac MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

pl_model = pl_model.to(device)
model = pl_model.model

adata = ad.read_h5ad(adata_path)
if "counts" in adata.layers:
    adata.X = adata.layers["counts"]
else:
    raise ValueError("Counts layer not found in adata.layers")

# Stop if the data carry too few human genes for this human-only model, e.g. mouse data that
# were not mapped to human orthologs (bin/orthologize.py). It would otherwise see no genes and
# return one label for every cell.
MIN_HUMAN_GENES = int(os.environ.get("SCLLM_MIN_HUMAN_GENES", "5000"))
n_human = int(pd.Series(adata.var.get("ensembl_id", adata.var_names)).astype(str).str.startswith("ENSG").sum())
print(f"Human Ensembl gene IDs in the data: {n_human}")
if n_human < MIN_HUMAN_GENES:
    raise SystemExit(f"Only {n_human} human Ensembl gene IDs (minimum {MIN_HUMAN_GENES}); "
                     "map mouse genes to human orthologs first with bin/orthologize.py")

# Smaller batches on MPS than on CUDA, to fit into memory
batch_size = 32 if device.type == "mps" else 128 if device.type == "cuda" else 16
print(f"Using batch size: {batch_size}")

print(f"Processing {adata.n_obs} cells...")

logit_scale = pl_model.model.discriminator.temperature.exp()

import time
start_time = time.time()

annotation_key = ANNOTATION_KEY
print(f"\nProcessing annotation key: {annotation_key}")

if annotation_key not in adata.obs.columns:
    print(f"Error: '{annotation_key}' not found in adata.obs.columns")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

def expected_types(found):
    """The expected cell types: the names in EXPECTED_TYPES_FILE, one per line, if it is set (a
    dataset's own list in annotation mode), else the cell types found in the labels."""
    path = os.environ.get("EXPECTED_TYPES_FILE")
    if not path:
        return found
    with open(path, encoding="utf-8") as fh:
        return list(dict.fromkeys(line.strip() for line in fh if line.strip()))


# Candidates: the dataset's own labels, or its own list (expected_types)
cell_type_labels = expected_types(adata.obs[annotation_key].unique().tolist())
print(f"Computing similarity scores to {len(cell_type_labels)} cell type labels...")

# Softmax-normalized similarities, shape [n_cell_types, n_cells]
scores, grouping_keys = score_transcriptomes_vs_texts(
    transcriptome_input=adata,
    text_list_or_text_embeds=cell_type_labels,
    logit_scale=logit_scale,
    model=model,
    transcriptome_processor=transcriptome_processor,
    batch_size=batch_size,
    average_mode=None,  # one score per cell
    score_norm_method="softmax"
)

print(f"Similarity scores shape: {scores.shape}")
print(f"Top scoring cell type for first cell: {cell_type_labels[scores[:, 0].argmax()]}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

print("\nSaving results")

true_labels = adata.obs[annotation_key].values
cell_id = adata.obs[annotation_key].index

print(f"Found {len(np.unique(true_labels))} unique true labels")

predicted_indices = scores.argmax(dim=0)
predicted_labels = [cell_type_labels[i] for i in predicted_indices]

results_df = pd.DataFrame({
    'cell_id': cell_id,
    'true_labels': true_labels,
    'predicted_labels': predicted_labels
})

results_df.to_csv(output_csv_base, index=False)
print(f"Saved predictions to {output_csv_base}")

# Exact-match accuracy and per-label ROC AUC, printed for the log only
accuracy = (results_df['true_labels'] == results_df['predicted_labels']).mean()
print(f"Overall accuracy: {accuracy:.3f}")

print(f"ROC/AUC analysis for {annotation_key}")

auc_scores = {}

for i, cell_type in enumerate(cell_type_labels):
    y_true_binary = (true_labels == cell_type).astype(int)
    y_scores = scores[i, :].numpy()
    
    # AUC needs both positive and negative cells
    if len(np.unique(y_true_binary)) > 1:
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        auc_score = auc(fpr, tpr)
        auc_scores[cell_type] = auc_score
        print(f"{cell_type:15} AUC: {auc_score:.3f}")
    else:
        auc_scores[cell_type] = np.nan
        print(f"{cell_type:15} AUC: N/A (no positive/negative samples)")

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