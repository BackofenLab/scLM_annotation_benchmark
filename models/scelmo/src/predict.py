#!/usr/bin/env python3

import argparse
import pickle
import sys
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
import scanpy as sc
import re
import hnswlib


# -----------------------------
# I/O helpers
# -----------------------------

def _labels_and_array_from_obj(obj, what: str) -> Tuple[List[str], np.ndarray]:
    """Accepts dict{name->vec}, (labels,embeddings), or dict with 'labels'+'embeddings'."""
    if isinstance(obj, dict) and all(isinstance(k, (str, bytes)) for k in obj.keys()):
        labels = [k.decode() if isinstance(k, bytes) else str(k) for k in obj.keys()]
        arr = np.vstack([np.array(v, dtype=np.float32).ravel() for v in obj.values()])
        return labels, arr
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        labels, arr = obj[0], obj[1]
        return [str(x) for x in labels], np.asarray(arr, dtype=np.float32)
    if isinstance(obj, dict) and 'labels' in obj and any(k in obj for k in ('embeddings','vectors','arr','array','embs')):
        emb_key = next(k for k in ('embeddings','vectors','arr','array','embs') if k in obj)
        return [str(x) for x in obj['labels']], np.asarray(obj[emb_key], dtype=np.float32)
    raise ValueError(f"Unrecognized {what} pickle structure.")


def _load_pickle(path: str, what: str) -> Tuple[List[str], np.ndarray]:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return _labels_and_array_from_obj(obj, what)

def filter_celltypes(ct_labels, ct_emb, remove):
    """
    Remove specific labels from the cell-type embedding set.
    """
    def norm(x):
        s = x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
        s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        return s

    rem = {norm(r) for r in remove}
    keep_mask = np.array([norm(lbl) not in rem for lbl in ct_labels], dtype=bool)

    labels_f = [lbl for lbl, k in zip(ct_labels, keep_mask) if k]
    emb_f = np.asarray(ct_emb, dtype=np.float32)[keep_mask]
    return labels_f, emb_f



# -----------------------------
# Math helpers
# -----------------------------

def row_l2(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def _norm_gene_key(x) -> str:
    s = str(x).strip()
    u = s.upper()
    if u.startswith('ENSG'):
        u = u.split('.')[0]  # strip Ensembl version suffix
    return u


# -----------------------------
# Core refactored functions
# -----------------------------

def compute_cell_embeddings(Xw, gene_emb: np.ndarray) -> np.ndarray:
    # Dense-ify if sparse
    Xw = np.asarray(Xw.todense()) if hasattr(Xw, 'todense') else np.asarray(Xw)
    C = Xw @ gene_emb
    denom = Xw.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-6)
    C = C / denom
    return row_l2(C.astype(np.float32))


def build_label_index(ct_emb: np.ndarray, space: str = 'cosine', ef_construction: int = 200,
                      M: int = 16, ef: int = 200) -> hnswlib.Index:
    """Build an HNSW index over label embeddings and set ef for queries."""
    ct_emb = np.asarray(ct_emb, dtype=np.float32)
    dim = ct_emb.shape[1]
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=ct_emb.shape[0], ef_construction=ef_construction, M=M)
    index.add_items(ct_emb, np.arange(ct_emb.shape[0]))
    index.set_ef(ef)
    return index


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_h5ad', help='Path to input .h5ad')
    ap.add_argument('--output_csv', help='Path to output CSV file')
    ap.add_argument('--gene-pickle', required=True, help='Path to gene embeddings .pickle')
    ap.add_argument('--celltype-pickle', required=True, help='Path to cell-type embeddings .pickle')
    ap.add_argument('--annotation_keys', default='true_labels', help='obs column with ground-truth labels (fallback to cell_type if missing)')
    args = ap.parse_args()

    # Load artifacts
    gene_labels, gene_emb = _load_pickle(args.gene_pickle, 'gene')
    ct_labels, ct_emb = _load_pickle(args.celltype_pickle, 'celltype')
    ct_labels, ct_emb = filter_celltypes(ct_labels, ct_emb, remove=["SMC1", "SMC2", "MHC class II"]) # these are not cell types
    if gene_emb.shape[0] != len(gene_labels):
        raise ValueError('Gene labels/embeddings mismatch')
    if ct_emb.shape[0] != len(ct_labels):
        raise ValueError('Cell-type labels/embeddings mismatch')

    # Read dataset
    adata = sc.read_h5ad(args.input_h5ad)
    X = adata.raw.X if adata.raw is not None else adata.X
    var_names = np.array(adata.raw.var_names if adata.raw is not None else adata.var_names)

    # Map genes (case-insensitive, strip Ensembl version)
    vocab_index = {_norm_gene_key(g): i for i, g in enumerate(gene_labels)}
    mat_idx, emb_idx = [], []
    for j, g in enumerate(var_names):
        k = vocab_index.get(_norm_gene_key(g))
        if k is not None:
            mat_idx.append(j); emb_idx.append(k)

    print(f"[info] overlapping genes: {len(emb_idx)} / {len(var_names)}")
    if len(emb_idx) < 200:
        warnings.warn('Low gene overlap; check Ensembl-vs-symbol IDs and version suffixes.')

    # Normalize counts (Scanpy idiom) and subset to overlap
    A = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    sc.pp.normalize_total(A, target_sum=1e6)
    sc.pp.log1p(A)
    Xw = A.X[:, mat_idx]  # weights

    # Compute cell embeddings (gene embeddings row-L2 first)
    gene_sub = row_l2(gene_emb[np.array(emb_idx)])
    cell_emb = compute_cell_embeddings(Xw, gene_sub)

    # Prepare label embeddings and build index
    ct_emb = row_l2(np.asarray(ct_emb, dtype=np.float32))
    index = build_label_index(ct_emb, space='cosine', ef_construction=200, M=16, ef=200)

    # Query k=1 for each cell
    labels_idx, dists = index.knn_query(cell_emb, k=1)
    pred_idx = labels_idx.ravel()
    predicted = [ct_labels[i] for i in pred_idx]

    # Ground truth
    ann_col = args.annotation-keys if hasattr(args, 'annotation-keys') and args.annotation-keys in adata.obs else None
    if ann_col is None:
        ann_col = 'cell_type' if 'cell_type' in adata.obs else None
    if ann_col is None:
        true = ['NA'] * adata.n_obs
        warnings.warn('No annotation column found; writing NA for true_labels')
    else:
        true = adata.obs[ann_col].astype(str).tolist()

    pd.DataFrame({'predicted_labels': predicted, 'true_labels': true}).to_csv(args.output_csv, index=False)
    print(f"[ok] saved {adata.n_obs} rows to {args.output_csv}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
