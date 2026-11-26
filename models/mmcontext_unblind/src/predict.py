"""Cell type annotation with mmcontext, with the dataset's labels as candidates (unblind).

The model (jo-mengr/mmcontext-pubmedbert-gs10k from the Hugging Face Hub) embeds each cell from
its gs10k initial embedding and every candidate label in a shared space, and each cell gets the
most similar label. The candidates are the labels in obs[--annotation_key].

Input: preprocessed h5ad with human Ensembl IDs in var["ensembl_id"]. Output: CSV with cell_id,
true_labels and predicted_labels; the processed data are also written to <input>_pp.h5ad.
Called by run.sh.
"""
import argparse
import pathlib
import os
import sys
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

args = parse_args()
adata_path = args.input
output_csv_base = args.output
ANNOTATION_KEY = args.annotation_key
CACHE_DIR = pathlib.Path(args.cache_dir)
MODEL_ID = "jo-mengr/mmcontext-pubmedbert-gs10k"
INITIAL_EMB = "gs10k"
EMB_KEY = f"X_{INITIAL_EMB}"

print("Loading data")
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

print("Computing embeddings")
embedder = InitialEmbedder(method=INITIAL_EMB)
embedder.prepare(adata)
embeddings = embedder.embed(adata)
adata.obsm[EMB_KEY] = embeddings

adata.obs["sample_idx"] = adata.obs.index
# AnnDataSetConstructor refers to the data by file path, so the processed data are written
# next to the input.
processed_path = adata_path.replace(".h5ad", "_pp.h5ad")
adata.write_h5ad(processed_path)
constructor = AnnDataSetConstructor(dataset_format="single")
constructor.add_anndata(adata, sentence_keys=["sample_idx"], adata_link=processed_path)
ds = constructor.get_dataset()

print("Loading MMContext model")
model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

print("Computing MMContext embeddings")
emb_df, path_map = prepare_model_and_embed(
    model, 
    ds, 
    index_col="sample_idx", 
    adata_download_dir=str(CACHE_DIR),
    main_col="cell_sentence_1", 
    layer_key=EMB_KEY
)

# Embeddings in the order of adata.obs.index (passed as a list so that pandas does not read it
# as a single key)
emb_df_sorted = emb_df.set_index("sample_idx").reindex(list(adata.obs.index)).reset_index()

embedding_matrix = np.vstack(emb_df_sorted["embedding"].to_numpy())
adata.obsm["mmcontext_emb"] = embedding_matrix

print("Creating annotator")
annotator = OmicsQueryAnnotator(model)

annotation_key = ANNOTATION_KEY
start_time = time.time()

print(f"\nProcessing annotation key: {annotation_key}")

if annotation_key not in adata.obs.columns:
    print(f"Error: '{annotation_key}' not found in adata.obs.columns")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

# Candidates: the dataset's own labels
unique_labels = adata.obs[annotation_key].unique()
print(f"Found {len(unique_labels)} unique cell type labels")

# Writes the most similar label of each cell to obs["best_label"].
annotator.annotate_omics_data(
    adata,
    labels=unique_labels,
    emb_key="mmcontext_emb",
    n_top=5,
)

predicted_labels = adata.obs["best_label"].values
true_labels = adata.obs[annotation_key].values
cell_id = adata.obs.index.values

print(f"Prediction completed for {len(predicted_labels)} cells")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nProcessing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

print("\nSaving results")

print(f"Found {len(np.unique(true_labels))} unique true labels")

results_df = pd.DataFrame({
    'cell_id': cell_id,
    'true_labels': true_labels,
    'predicted_labels': predicted_labels
})

results_df.to_csv(output_csv_base, index=False)
print(f"Saved predictions to {output_csv_base}")
    
# Exact-match accuracy, printed for the log only
accuracy = (results_df['true_labels'] == results_df['predicted_labels']).mean()
print(f"Overall accuracy: {accuracy:.3f}")

print(f"Label distribution in {annotation_key}:")
label_counts = pd.Series(true_labels).value_counts()
for label, count in label_counts.items():
    print(f"  {label}: {count} ({count/len(true_labels)*100:.1f}%)")

print(f"\nPrediction completed successfully!")