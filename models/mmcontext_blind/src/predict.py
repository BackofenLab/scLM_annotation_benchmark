"""
MMContext cell type prediction script (blind mode)
Computes MMContext embeddings and annotates cells with ALL cell types from ontology.
This is a "blind" prediction mode that scores against all known cell types.

Usage:
  python predict.py --input <h5ad_path> --output <csv_path> --annotation_key <label_key> --cache_dir <cache_dir> --cell_types_file <file>

Outputs:
  - CSV file with cell_id, true_labels, and predicted_labels columns
"""
import argparse
import pathlib
import sys
import os
import pandas as pd
import numpy as np
import time
from adata_hf_datasets import InitialEmbedder
from adata_hf_datasets.dataset import AnnDataSetConstructor
from adata_hf_datasets.pp.utils import ensure_log_norm
from adata_hf_datasets.pp.geneformer import pp_adata_geneformer
from mmcontext.embed.model_utils import prepare_model_and_embed, embed_labels
import anndata as ad
from sentence_transformers import SentenceTransformer
from mmcontext.eval import OmicsQueryAnnotator
import scanpy as sc

def parse_args():
    parser = argparse.ArgumentParser(description='MMContext cell type prediction (blind mode)')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--annotation_key', required=True, help='Key in adata.obs for true cell type labels')
    parser.add_argument('--cache_dir', required=True, help='Path to cache directory. Model checkpoints and cell type list will be loaded from here.')
    parser.add_argument('--cell_types_file', default='all_cell_types.txt', help='Name of cell types file in cache_dir (default: all_cell_types.txt)')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
adata_path = args.input
output_csv = args.output
ANNOTATION_KEY = args.annotation_key
CACHE_DIR = pathlib.Path(args.cache_dir)
CELL_TYPES_FILE = CACHE_DIR.parent / args.cell_types_file
MODEL_ID = "jo-mengr/mmcontext-pubmedbert-gs10k"
INITIAL_EMB = "gs10k"
EMB_KEY = f"X_{INITIAL_EMB}"

print("=== Loading data ===")
adata = ad.read_h5ad(adata_path)
print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

# Stop if the data carry too few human genes for this human-only model, e.g. mouse data that
# were not mapped to human orthologs (bin/orthologize.py). It would otherwise see no genes and
# return one label for every cell.
MIN_HUMAN_GENES = int(os.environ.get("SCLLM_MIN_HUMAN_GENES", "5000"))
n_human = int(pd.Series(adata.var.get("ensembl_id", adata.var_names)).astype(str).str.startswith("ENSG").sum())
print(f"Human Ensembl gene IDs in the data: {n_human}")
if n_human < MIN_HUMAN_GENES:
    raise SystemExit(f"Only {n_human} human Ensembl gene IDs (minimum {MIN_HUMAN_GENES}); "
                     "map mouse genes to human orthologs first with bin/orthologize.py")

# Remove trailing .1 ... (dot and numbers) in adata.var.index
#adata.var.index = adata.var.index.str.replace(r'\.\d+$', '', regex=True)

print("=== Computing embeddings ===")
embedder = InitialEmbedder(method=INITIAL_EMB)
embedder.prepare(adata)
embeddings = embedder.embed(adata)
adata.obsm[EMB_KEY] = embeddings

adata.obs["sample_idx"] = adata.obs.index
#write processed adata to h5ad
processed_path = adata_path.replace(".h5ad", "_pp.h5ad")
adata.write_h5ad(processed_path)
constructor = AnnDataSetConstructor(dataset_format="single")
constructor.add_anndata(adata, sentence_keys=["sample_idx"], adata_link=processed_path)
ds = constructor.get_dataset()

print("=== Loading MMContext model ===")
model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

print("=== Computing MMContext embeddings ===")
emb_df, path_map = prepare_model_and_embed(
    model, 
    ds, 
    index_col="sample_idx", 
    adata_download_dir=str(CACHE_DIR),
    main_col="cell_sentence_1", 
    layer_key=EMB_KEY
)

# Sort emb_df so that the sample_idx (index column) matches the order of adata.obs.index
# Convert adata.obs.index to a list to avoid pandas interpreting it as a single key
emb_df_sorted = emb_df.set_index("sample_idx").reindex(list(adata.obs.index)).reset_index()

embedding_matrix = np.vstack(emb_df_sorted["embedding"].to_numpy())
adata.obsm["mmcontext_emb"] = embedding_matrix

print("=== Creating annotator ===")
annotator = OmicsQueryAnnotator(model)

# Check if annotation key exists (needed for true_labels)
if ANNOTATION_KEY not in adata.obs.columns:
    print(f"Error: '{ANNOTATION_KEY}' not found in adata.obs.columns")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

# ----------------- load cell types from file ------------------
print(f"\n=== Loading cell types from {CELL_TYPES_FILE} ===")
if not CELL_TYPES_FILE.exists():
    raise FileNotFoundError(f"Cell types file not found: {CELL_TYPES_FILE}")

with open(CELL_TYPES_FILE, 'r') as f:
    cell_type_labels = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(cell_type_labels)} cell types from ontology")

start_time = time.time()

print(f"\n=== Computing similarity scores against {len(cell_type_labels)} cell types ===")

# Annotate cells using ALL cell types from ontology
annotator.annotate_omics_data(
    adata,
    labels=cell_type_labels,
    emb_key="mmcontext_emb",
    n_top=5,
)

# Get predictions
predicted_labels = adata.obs["best_label"].values
true_labels = adata.obs[ANNOTATION_KEY].values
cell_id = adata.obs.index.values

print(f"Prediction completed for {len(predicted_labels)} cells")
print(f"Top predicted cell type for first cell: {predicted_labels[0]}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nProcessing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# ----------------- Save Results ---------------------------
print("\n=== SAVING RESULTS ===")

print(f"Found {len(np.unique(true_labels))} unique true labels")

# Create comparison DataFrame with only required columns
results_df = pd.DataFrame({
    'cell_id': cell_id,
    'true_labels': true_labels,
    'predicted_labels': predicted_labels
})

# Save to CSV
results_df.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")

# Calculate accuracy
accuracy = (results_df['true_labels'] == results_df['predicted_labels']).mean()
print(f"Overall accuracy: {accuracy:.3f}")

# Print label distribution
print(f"Label distribution in {ANNOTATION_KEY}:")
label_counts = pd.Series(true_labels).value_counts()
for label, count in label_counts.items():
    print(f"  {label}: {count} ({count/len(true_labels)*100:.1f}%)")

print(f"\nPrediction completed successfully!")