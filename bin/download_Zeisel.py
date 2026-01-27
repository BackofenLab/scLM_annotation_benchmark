# Script to download Zeisel mouse brain data and convert to h5ad
# Data available at: http://mousebrain.org/adolescent/downloads.html
# Data was accessed on 19.12.2025
# Data was published in: https://www.cell.com/cell/fulltext/S0092-8674%2818%2930789-X?utm_source=chatgpt.com

import loompy
import scanpy as sc

loom_path_read = #"path_to_data.../l5_all_rev1.loom"
loom_path_write = #"path_to_data.../l5_all_rev1.h5ad"

with loompy.connect(loom_path_read, mode="r") as ds:
    print("shape (genes x cells):", ds.shape)
    print("row attrs (genes):", list(ds.ra.keys()))
    print("col attrs (cells):", list(ds.ca.keys()))
    print("layers:", list(ds.layers.keys()))

# Read loom -> AnnData
adata = sc.read_loom(
    loom_path_read,
    sparse=True,
    cleanup=True,     # removes loom-specific cruft and harmonizes dtypes a bit
)


# The loom is genes x cells; AnnData expects obs=cells, var=genes.
# scanpy's read_loom typically returns it already oriented as cells x genes,
# but we sanity-check and transpose if needed.
if adata.n_obs == 27998 and adata.n_vars == 160796:
    adata = adata.T

# Set names from your attributes
# After read_loom, row/col attrs are usually in .var / .obs already.
# Your file contains var['Gene'] and var['Accession'], obs['CellID'].
if "CellID" in adata.obs:
    adata.obs_names = adata.obs["CellID"].astype(str)
if "Gene" in adata.var:
    adata.var_names = adata.var["Gene"].astype(str)

# Keep accession as an extra gene identifier (optional but useful)
if "Accession" in adata.var:
    adata.var["ensembl_id"] = adata.var["Accession"].astype(str)

# Make names unique (important for downstream)
adata.obs_names_make_unique()
adata.var_names_make_unique()

# Optional: ensure counts are in X (some looms store float/normalized; yours is default matrix)
# If you want to preserve raw counts explicitly:
adata.layers["counts"] = adata.X.copy()

# Save
adata.write_h5ad(loom_path_write, compression="gzip")

print(adata)