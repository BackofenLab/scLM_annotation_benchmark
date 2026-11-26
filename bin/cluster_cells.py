#!/usr/bin/env python
"""Cluster cells and find the marker genes of each cluster, for methods that annotate clusters.

The cells of --data are clustered with Leiden on the PCA of their log-normalized expression.
The marker genes of each cluster are ranked with rank_genes_groups, without ribosomal,
mitochondrial and hemoglobin genes, and named by their gene symbols. --data can hold more cells
than the cells to annotate (--cells), for example a clustering sample that contains them.

The defaults are those of the per-cluster analysis of the paper: the PCA uses all genes without
scaling. --n_top_genes restricts it to highly variable genes (selected within each batch with
--batch_key) and --scale scales the genes first, as in the usual scanpy workflow.

Writes
    <out>.clusters.csv  cell_id, cluster for the cells of --cells
    <out>.markers.csv   cluster, markers (gene symbols, best first, separated by ", ")
    <out>.json          the parameters and the number of clusters

    cluster_cells.py --data context.h5ad --cells test_set.h5ad --out HIHA_1000 --resolution 1.0
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

ENSEMBL_RE = r"^ENS[A-Z]*G\d"
PLACEHOLDERS = ("nan", "None", "", "NA", "<NA>")
EXCLUDED_PREFIXES = ("RPS", "RPL", "MT-", "HB")


def symbol_source(adata):
    """The gene names of var_names, gene_names, feature_name or gene_symbols, whichever holds the
    largest share of entries that are neither Ensembl ids nor placeholders."""
    def usable(values):
        return float((~values.str.match(ENSEMBL_RE) & ~values.isin(PLACEHOLDERS)).mean())

    candidates = [pd.Index(adata.var_names.astype(str))]
    candidates += [pd.Index(adata.var[c].astype(str))
                   for c in ("gene_names", "feature_name", "gene_symbols") if c in adata.var.columns]
    return max(candidates, key=usable)


def principal_components(adata, n_pcs, n_top_genes, batch_key, scale, seed):
    """Store the PCA of the cells in adata.obsm["X_pca"]."""
    if not n_top_genes and not scale:
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1), random_state=seed)
        return
    data = adata
    if n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat",
                                    batch_key=batch_key)
        data = adata[:, adata.var["highly_variable"]]
    data = data.copy()
    if scale:
        sc.pp.scale(data, max_value=10)
    sc.pp.pca(data, n_comps=min(n_pcs, data.n_vars - 1, data.n_obs - 1), random_state=seed)
    adata.obsm["X_pca"] = data.obsm["X_pca"]


def cluster(adata, resolution, n_pcs, n_neighbors, n_top_genes, batch_key, scale, seed):
    """Leiden clusters of the cells, as strings."""
    principal_components(adata, n_pcs, n_top_genes, batch_key, scale, seed)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, key_added="cluster", random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)
    return adata.obs["cluster"].astype(str)


def marker_genes(adata, n_markers, method):
    """The marker gene symbols of each cluster, best first."""
    symbols = pd.Index([s.upper() for s in symbol_source(adata)])
    drop = symbols.str.startswith(EXCLUDED_PREFIXES)
    ranked = adata[:, ~drop].copy() if drop.any() else adata
    sc.tl.rank_genes_groups(ranked, groupby="cluster", method=method)
    names = ranked.uns["rank_genes_groups"]["names"]
    named = ranked.copy()
    named.var_names = pd.Index(symbol_source(named)).str.upper()
    named.var_names_make_unique()
    symbol_of = dict(zip(ranked.var_names.astype(str), named.var_names.astype(str)))
    return {c: [symbol_of.get(g, g) for g in list(names[c])[:n_markers]]
            for c in names.dtype.names}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="cells to cluster (h5ad, log-normalized)")
    parser.add_argument("--cells", required=True,
                        help="cells to annotate (h5ad); they must all be in --data")
    parser.add_argument("--out", required=True, help="prefix of the output files")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--n_marker_genes", type=int, default=25)
    parser.add_argument("--n_top_genes", type=int, default=0,
                        help="highly variable genes for the PCA; 0 uses all genes")
    parser.add_argument("--batch_key", default=None,
                        help="obs column with the batch, for selecting highly variable genes")
    parser.add_argument("--scale", action="store_true", help="scale the genes before the PCA")
    parser.add_argument("--rank_method", default="wilcoxon")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = sc.read_h5ad(args.data)
    batch_key = args.batch_key if args.batch_key in adata.obs.columns else None
    clusters = cluster(adata, args.resolution, args.n_pcs, args.n_neighbors, args.n_top_genes,
                       batch_key, args.scale, args.seed)
    markers = marker_genes(adata, args.n_marker_genes, args.rank_method)

    cells = pd.Index(ad.read_h5ad(args.cells, backed="r").obs_names.astype(str))
    clusters.index = clusters.index.astype(str)
    missing = cells.difference(clusters.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} of {len(cells)} cells to annotate are not in --data")
    per_cell = clusters.loc[cells]

    out = Path(args.out)
    per_cell.rename_axis("cell_id").rename("cluster").to_frame().to_csv(
        f"{out}.clusters.csv")
    pd.DataFrame([{"cluster": c, "markers": ", ".join(genes)}
                  for c, genes in sorted(markers.items())]).to_csv(f"{out}.markers.csv",
                                                                   index=False)
    summary = {**{key: value for key, value in vars(args).items()
                  if key not in ("data", "cells", "out")},
               "clustered_cells": int(adata.n_obs), "clusters": len(markers),
               "annotated_cells": len(per_cell), "clusters_of_annotated_cells":
               int(per_cell.nunique())}
    Path(f"{out}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(markers)} clusters on {adata.n_obs} cells; the {len(per_cell)} cells to annotate "
          f"fall into {per_cell.nunique()} of them")


if __name__ == "__main__":
    main()
