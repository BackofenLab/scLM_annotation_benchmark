"""
MMContext cell type prediction script
Computes MMContext embeddings and annotates cells with cell type labels
Handles multiple annotation keys and saves predictions to CSV files

Usage:
  python predict.py --input <h5ad_path> --output <csv_path> --annotation_keys <key1> <key2> ... --cache_dir <cache_dir>

Outputs:
  - CSV file(s) with true_labels and predicted_labels columns (one per annotation key)
"""
import argparse
import pathlib
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
    parser = argparse.ArgumentParser(description='MMContext cell type prediction')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to output CSV file (base name)')
    parser.add_argument('--annotation_key', required=True, help='Key in adata.obs for true cell type labels')
    parser.add_argument('--cache_dir', required=True, help='Path to cache directory')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
adata_path = args.input
output_csv_base = args.output
ANNOTATION_KEY = args.annotation_key
CACHE_DIR = pathlib.Path(args.cache_dir)
MODEL_ID = "jo-mengr/mmcontext-pubmedbert-gs10k"
INITIAL_EMB = "gs10k"
EMB_KEY = f"X_{INITIAL_EMB}"

print("=== Loading data ===")
adata = ad.read_h5ad(adata_path)
print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

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

# Process annotation key
annotation_key = ANNOTATION_KEY
start_time = time.time()

print(f"\n=== Processing annotation key: {annotation_key} ===")

# Check if annotation key exists
if annotation_key not in adata.obs.columns:
    print(f"Error: '{annotation_key}' not found in adata.obs.columns")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

# Get unique cell type labels for this annotation level
unique_labels = adata.obs[annotation_key].unique()
print(f"Found {len(unique_labels)} unique cell type labels")

# Annotate cells
annotator.annotate_omics_data(
    adata,
    labels=unique_labels,
    emb_key="mmcontext_emb",
    n_top=5,
)

# Get predictions
predicted_labels = adata.obs["best_label"].values
true_labels = adata.obs[annotation_key].values
cell_id = adata.obs.index.values

print(f"Prediction completed for {len(predicted_labels)} cells")

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
results_df.to_csv(output_csv_base, index=False)
print(f"Saved predictions to {output_csv_base}")
    
# Calculate accuracy
accuracy = (results_df['true_labels'] == results_df['predicted_labels']).mean()
print(f"Overall accuracy: {accuracy:.3f}")

# Print label distribution
print(f"Label distribution in {annotation_key}:")
label_counts = pd.Series(true_labels).value_counts()
for label, count in label_counts.items():
    print(f"  {label}: {count} ({count/len(true_labels)*100:.1f}%)")

print(f"\nPrediction completed successfully!")