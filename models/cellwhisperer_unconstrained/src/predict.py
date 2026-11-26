"""Cell type annotation with CellWhisperer, scored against a predefined list of cell type names (unconstrained).

CellWhisperer embeds each cell's transcriptome and every candidate name, and each cell gets the
name with the highest similarity. The candidates come from --cell_types_file, a file in the
parent directory of --cache_dir with one name per line (all_cell_types.txt: all Cell Ontology
names).

Input: preprocessed h5ad with raw counts in layers["counts"] and human Ensembl IDs in
var["ensembl_id"]. Output: CSV with cell_id, true_labels and predicted_labels. The checkpoint
is downloaded into --cache_dir on first use. Called by run.sh.
"""
import sys, torch, anndata as ad, pathlib
import os
import pandas as pd
import numpy as np
import argparse
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from util import download_if_missing

def parse_args():
    parser = argparse.ArgumentParser(description='CellWhisperer cell type prediction (unconstrained mode)')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--cache_dir', required=True, help='Path to cache directory. Model checkpoints and cell type list will be loaded from here.')
    parser.add_argument('--cell_types_file', default='all_cell_types.txt', help='Name of cell types file in cache_dir (default: all_cell_types.txt)')
    parser.add_argument('--annotation_key', required=True, help='Key in adata.obs for true cell type labels')
    return parser.parse_args()

args = parse_args()
adata_path = args.input
output_csv = args.output
CACHE_DIR = pathlib.Path(args.cache_dir)
CELL_TYPES_FILE = CACHE_DIR.parent / args.cell_types_file
ANNOTATION_KEY = args.annotation_key
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

if ANNOTATION_KEY not in adata.obs.columns:
    print(f"Error: '{ANNOTATION_KEY}' not found in adata.obs.columns")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

# Smaller batches on MPS than on CUDA, to fit into memory
batch_size = 32 if device.type == "mps" else 128 if device.type == "cuda" else 16
print(f"Using batch size: {batch_size}")

# Candidate cell types, one name per line
print(f"Loading cell types from {CELL_TYPES_FILE}...")
if not CELL_TYPES_FILE.exists():
    raise FileNotFoundError(f"Cell types file not found: {CELL_TYPES_FILE}")

with open(CELL_TYPES_FILE, 'r') as f:
    cell_type_labels = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(cell_type_labels)} cell types from ontology")

print(f"Processing {adata.n_obs} cells...")

logit_scale = pl_model.model.discriminator.temperature.exp()

import time
start_time = time.time()

print(f"\nComputing similarity scores against {len(cell_type_labels)} cell types")

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

cell_ids = adata.obs.index.tolist()
true_labels = adata.obs[ANNOTATION_KEY].values

predicted_indices = scores.argmax(dim=0)
predicted_labels = [cell_type_labels[i] for i in predicted_indices]

results_df = pd.DataFrame({
    'cell_id': cell_ids,
    'true_labels': true_labels,
    'predicted_labels': predicted_labels
})

results_df.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")

print(f"\nPrediction completed successfully!")