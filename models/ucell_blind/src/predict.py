#!/usr/bin/env python
"""UCell-style cluster-free marker scoring for cell-type annotation.

Scores each ScTypeDB positive-marker signature per cell with the UCell rank-based
Mann-Whitney U statistic (Andreatta & Carmona, CSBJ 2021), then assigns each cell its
top-scoring cell type. The marker source is ScTypeDB — the SAME sets scType uses — so
scType (per-cluster) vs UCell (per-cell, cluster-free) isolates the clustering paradigm,
not the marker database.

UCell score for a signature of n genes in a given cell:
    rank genes by expression (rank 1 = highest); genes beyond max_rank -> max_rank+1;
    u = sum of the signature genes' ranks;   u_min = n*(n+1)/2;
    score = 1 - (u - u_min) / (n * max_rank)   in [~0, 1], higher = more enriched.

Emits the pipeline's contract: cell_id, true_labels, predicted_labels (one row per cell).
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc


def load_signatures(xlsx_path, tissue):
    """ScTypeDB -> {cell_type: [UPPERCASE positive marker symbols]} for the given tissue.

    ScTypeDB columns: tissueType, cellName, geneSymbolmore1 (positive), geneSymbolmore2
    (negative). We use the positive markers only (a clean, standard marker baseline).
    """
    db = pd.read_excel(xlsx_path)
    db = db[db["tissueType"].astype(str).str.strip().str.lower() == str(tissue).strip().lower()]
    sigs = {}
    for _, r in db.iterrows():
        ct = str(r["cellName"]).strip()
        pos = str(r.get("geneSymbolmore1", "") or "")
        genes = [g.strip().upper() for g in pos.replace(";", ",").split(",")
                 if g.strip() and g.strip().lower() != "nan"]
        if genes:
            sigs.setdefault(ct, set()).update(genes)
    return {k: sorted(v) for k, v in sigs.items()}


def ucell_scores(X, gene_upper, signatures, max_rank=1500, chunk=1000):
    """Return (cell_types, scores[n_cells, n_types]) using the UCell statistic.

    Gene symbols are matched case-insensitively (UPPERCASE both sides): exact for human,
    and an approximate ortholog match for mouse (conserved 1:1 symbols differ only in case).
    """
    n_cells, n_genes = X.shape
    col_of = defaultdict(list)
    for j, g in enumerate(gene_upper):
        col_of[g].append(j)
    sig_cols = {}
    for ct, genes in signatures.items():
        cols = []
        for g in genes:
            cols.extend(col_of.get(g, []))
        if cols:
            sig_cols[ct] = np.array(sorted(set(cols)))
    cts = [ct for ct in signatures if ct in sig_cols]
    if not cts:
        return [], np.zeros((n_cells, 0), dtype=np.float32)
    mr = min(int(max_rank), n_genes)
    scores = np.zeros((n_cells, len(cts)), dtype=np.float32)
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        block = X[start:end]
        if hasattr(block, "toarray"):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float32)
        # rank 1 = highest expression: argsort descending, then scatter ranks back.
        order = np.argsort(-block, axis=1, kind="stable")
        ranks = np.empty(order.shape, dtype=np.int32)
        rows = np.arange(block.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, n_genes + 1, dtype=np.int32)[None, :]
        capped = np.where(ranks > mr, mr + 1, ranks).astype(np.float64)
        for k, ct in enumerate(cts):
            cols = sig_cols[ct]
            n = len(cols)
            u = capped[:, cols].sum(axis=1)
            u_min = n * (n + 1) / 2.0
            scores[start:end, k] = (1.0 - (u - u_min) / (n * mr)).astype(np.float32)
    return cts, scores


def gene_symbols(adata):
    if "gene_names" in adata.var.columns:
        return adata.var["gene_names"].astype(str).values
    for c in ("feature_name", "gene_name", "gene_symbols", "symbol"):
        if c in adata.var.columns:
            return adata.var[c].astype(str).values
    return np.asarray(adata.var_names, dtype=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markers", required=True, help="ScTypeDB_full.xlsx")
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--organism", default="human")
    ap.add_argument("--annotation_key", default="cell_type")
    ap.add_argument("--max_rank", type=int, default=1500)
    a = ap.parse_args()

    adata = sc.read_h5ad(a.input)
    gene_upper = np.char.upper(gene_symbols(adata).astype(str))

    sigs = load_signatures(a.markers, a.tissue)
    if not sigs:
        raise SystemExit(f"No ScTypeDB signatures found for tissue '{a.tissue}'.")

    cts, scores = ucell_scores(adata.X, gene_upper, sigs, max_rank=a.max_rank)
    if not cts:
        raise SystemExit("No signature marker genes overlap the dataset genes.")

    best = scores.argmax(axis=1)
    best_val = scores.max(axis=1)
    preds = np.array([cts[i] for i in best], dtype=object)
    preds[best_val <= 0] = "Unknown"

    if a.annotation_key in adata.obs.columns:
        true = adata.obs[a.annotation_key].astype(str).values
    else:
        true = np.array(["NA"] * adata.n_obs, dtype=object)

    out = pd.DataFrame({
        "cell_id": np.asarray(adata.obs_names, dtype=str),
        "true_labels": true,
        "predicted_labels": preds,
    })
    out.to_csv(a.output, index=False)
    print(f"UCell: wrote {len(out)} predictions over {len(cts)} candidate types "
          f"(tissue='{a.tissue}', organism='{a.organism}') to {a.output}")


if __name__ == "__main__":
    main()
