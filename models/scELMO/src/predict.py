#!/usr/bin/env python3
"""
scELMo-style zero-shot annotation

Usage:
  python scelmo_min.py input.h5ad output.csv \
      --gene-pickle /path/gene_embeddings.pickle \
      --celltype-pickle /path/celltype_embeddings.pickle \
      [--annotation-keys true_labels]

Expects pickles to be either:
  - dict {name -> vector}, or
  - (labels, embeddings) tuple/list
Writes CSV with columns: predicted_labels,true_labels
"""
import argparse, pickle, sys
import re, numpy as np, pandas as pd, scanpy as sc
from scipy import sparse
from collections import Counter

def _labels_and_array_from_obj(obj):
    # dict {label->vec}
    if isinstance(obj, dict) and all(isinstance(k, (str, bytes)) for k in obj.keys()):
        labels = [k.decode() if isinstance(k, bytes) else str(k) for k in obj.keys()]
        embs = np.vstack([np.array(v, dtype=np.float32).ravel() for v in obj.values()])
        return labels, embs
    # (labels, embeddings)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        labels, embs = obj[0], obj[1]
        return [str(x) for x in labels], np.asarray(embs, dtype=np.float32)
    # dict with fields
    if isinstance(obj, dict) and "labels" in obj and any(k in obj for k in ("embeddings","vectors","arr","array","embs")):
        emb_key = next(k for k in ("embeddings","vectors","arr","array","embs") if k in obj)
        return [str(x) for x in obj["labels"]], np.asarray(obj[emb_key], dtype=np.float32)
    raise ValueError("Unrecognized pickle structure (need dict{name->vec} or (labels,embeddings)).")

def _load_pickle(path):
    with open(path, "rb") as f:
        return _labels_and_array_from_obj(pickle.load(f))

def to_dense(x): return x.A if sparse.issparse(x) else np.asarray(x)

def counts_to_log1p_cpm(X):
    if sparse.issparse(X):
        lib = np.asarray(X.sum(axis=1)).ravel()
        lib = np.maximum(lib, 1.0)
        cpm = sparse.diags(1.0 / lib) @ X * 1e6
        return cpm.log1p()
    lib = X.sum(axis=1, keepdims=True); lib = np.maximum(lib, 1.0)
    return np.log1p((X / lib) * 1e6)

def compute_cell_embeddings(X, gene_emb):
    Xw = counts_to_log1p_cpm(X)
    Xw = to_dense(Xw).astype(np.float32)
    C = Xw @ gene_emb
    sums = Xw.sum(axis=1, keepdims=True); sums = np.maximum(sums, 1e-6)
    C = C / sums
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C.astype(np.float32)

def cosine_argmax(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A @ B.T).argmax(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad")
    ap.add_argument("--output_csv")
    ap.add_argument("--gene-pickle", required=True)
    ap.add_argument("--celltype-pickle", required=True)
    ap.add_argument("--annotation-keys", default="cell_type")
    args = ap.parse_args()

    # Load pickles
    gene_labels, gene_emb = _load_pickle(args.gene_pickle)
    ct_labels, ct_emb   = _load_pickle(args.celltype_pickle)
    if gene_emb.shape[0] != len(gene_labels): raise ValueError("Gene labels/embeddings mismatch.")
    if ct_emb.shape[0]   != len(ct_labels):   raise ValueError("Cell-type labels/embeddings mismatch.")

    # Read h5ad (prefer raw if present)
    adata = sc.read_h5ad(args.input_h5ad)
    X = adata.raw.X if adata.raw is not None else adata.X
    var_names = np.array(adata.raw.var_names if adata.raw is not None else adata.var_names)

    # Overlap by case-insensitive exact match
    vocab_index = {str(g).upper(): i for i, g in enumerate(gene_labels)}
    mat_idx, emb_idx = [], []
    for j, g in enumerate(var_names):
        k = vocab_index.get(str(g).upper())
        if k is not None:
            mat_idx.append(j); emb_idx.append(k)
    if len(emb_idx) < 50:
        raise ValueError(f"Too few overlapping genes: {len(emb_idx)}")

    X_sub = X[:, mat_idx]
    gene_emb_sub = gene_emb[np.array(emb_idx)]

    # Predict
    cell_emb = compute_cell_embeddings(X_sub, gene_emb_sub)
    pred_idx = cosine_argmax(cell_emb, ct_emb)
    preds = [ct_labels[i] for i in pred_idx]

    # Truth
    if args.annotation_keys in adata.obs:
        truth = adata.obs[args.annotation_keys].astype(str).tolist()
    elif "cell_type" in adata.obs:
        truth = adata.obs["cell_type"].astype(str).tolist()
    else:
        truth = ["NA"] * adata.n_obs

    pd.DataFrame({"predicted_labels": preds, "true_labels": truth}).to_csv(args.output_csv, index=False)
    print(f"Saved {adata.n_obs} rows to {args.output_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr); sys.exit(2)
