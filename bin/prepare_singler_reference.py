#!/usr/bin/env python
"""Download a CELLxGENE Census dataset and turn it into a reference h5ad for SingleR.

The reference has the layout of the preprocessed test sets (Ensembl IDs as var_names,
log-normalized X, gene symbols in var['gene_names'], raw counts in layers['counts']), so that
models/singler_unconstrained/src/predict.R reads it like a test set and matches genes by Ensembl ID.
Unlike bin/preprocess.py, it applies no QC and no cell or gene filtering, so that no labelled
cell type is lost. Cells without a label are removed and every label is capped at
--cap_per_label cells (with a fixed seed), which keeps SingleR fast on large atlases.

    prepare_singler_reference.py --census_id ID --census_version 2025-11-08 \\
        --label_key cell_type --species human --out reference.h5ad
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make util.* importable
from util.pp_utils import (  # noqa: E402
    ensure_gene_names_column,
    ensure_raw_counts_layer,
    ensure_norm_log,
    remove_infinite_values,
    ensure_ensembl_ids,
)

# cellxgene_census can fail to import (e.g. x86 builds under Rosetta on Apple Silicon); the
# error is raised when the package is needed, so that --help works.
try:
    import cellxgene_census  # type: ignore
except Exception as e:  # pragma: no cover
    cellxgene_census = None  # type: ignore[assignment]
    _census_import_error = e

# Species of the dataset YAML, as BioMart dataset prefix for ensure_ensembl_ids. Other values
# become "hsapiens".
_SPECIES_MAP = {
    "human": "hsapiens", "homo sapiens": "hsapiens", "hsapiens": "hsapiens",
    "mouse": "mmusculus", "mus musculus": "mmusculus", "mmusculus": "mmusculus",
}


def _biomart_species(s: str) -> str:
    return _SPECIES_MAP.get(str(s).strip().lower(), "hsapiens")


def subsample_positions(labels, cap: int, seed: int):
    """Sorted positions of at most cap cells per label, drawn with the given seed; cap <= 0 keeps
    all cells."""
    if not cap or cap <= 0:
        return np.arange(len(labels))
    rng = np.random.RandomState(seed)
    keep = []
    for lab in pd.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def read_rows(elem, rows):
    """The given rows (sorted positions) of a matrix in an open h5ad file: a CSR matrix is read
    row by row and a dense array by index, so that the rest of the matrix is never loaded; other
    encodings are read whole."""
    if isinstance(elem, h5py.Dataset):
        return elem[rows]
    if elem.attrs.get("encoding-type") != "csr_matrix":
        whole = read_elem(elem)
        return whole.iloc[rows] if isinstance(whole, pd.DataFrame) else whole[rows]
    indptr = elem["indptr"][:]
    starts, ends = indptr[rows], indptr[rows + 1]
    data = np.concatenate([elem["data"][s:e] for s, e in zip(starts, ends)])
    indices = np.concatenate([elem["indices"][s:e] for s, e in zip(starts, ends)])
    indptr_new = np.concatenate([[0], np.cumsum(ends - starts)])
    return sp.csr_matrix((data, indices, indptr_new), shape=(len(rows), int(elem.attrs["shape"][1])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census_id", required=True, help="CELLxGENE Census dataset_id")
    ap.add_argument("--census_version", default="stable", help="Census version tag (default: stable)")
    ap.add_argument("--label_key", default="cell_type", help="obs column holding reference labels")
    ap.add_argument("--species", default="human", help="dataset species (human/mouse)")
    ap.add_argument("--out", required=True, help="output reference h5ad path")
    ap.add_argument("--cap_per_label", type=int, default=500,
                    help="max cells per label (0/negative = no cap)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for per-label subsampling")
    args = ap.parse_args()

    if cellxgene_census is None:
        sys.exit(f"[ERROR] cellxgene_census import failed: {_census_import_error}")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    source = out.parent / f".{out.stem}_source.h5ad"

    print(f"[INFO] Downloading Census source h5ad id={args.census_id} version={args.census_version}",
          flush=True)
    cellxgene_census.download_source_h5ad(
        str(args.census_id), to_path=str(source), census_version=str(args.census_version)
    )
    # Large atlases do not fit in memory, so the cells are chosen from obs alone and only their
    # rows are read.
    with h5py.File(source, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        print(f"[INFO] source: {len(obs)} cells x {len(var)} genes", flush=True)

        if args.label_key not in obs.columns:
            sys.exit(f"[ERROR] label_key '{args.label_key}' not in obs columns: "
                     f"{list(obs.columns)[:40]}")

        # Cells whose label is missing, empty or 'nan' are removed.
        lab = obs[args.label_key].astype(str)
        labelled = np.flatnonzero(
            (lab.notna() & (lab.str.strip() != "") & (lab.str.strip().str.lower() != "nan")).values)
        print(f"[INFO] dropped {len(obs) - len(labelled)} cells with empty/NA '{args.label_key}'",
              flush=True)

        # Subsampled before normalization to save time; normalization works per cell.
        keep = labelled[subsample_positions(lab.values[labelled], args.cap_per_label, args.seed)]
        adata = ad.AnnData(
            X=read_rows(f["X"], keep),
            obs=obs.iloc[keep],
            var=var,
            obsm={k: read_rows(f["obsm"][k], keep) for k in f.get("obsm", {})},
            layers={k: read_rows(f["layers"][k], keep) for k in f.get("layers", {})},
            uns=read_elem(f["uns"]) if "uns" in f else None,
        )
        if "raw" in f:
            adata.raw = ad.AnnData(X=read_rows(f["raw"]["X"], keep), obs=adata.obs,
                                   var=read_elem(f["raw"]["var"]))
    print(f"[INFO] after subsample(cap={args.cap_per_label}): {adata.n_obs} cells, "
          f"{adata.obs[args.label_key].astype(str).nunique()} labels", flush=True)

    # The steps of preprocess_adata without QC and filtering.
    ensure_gene_names_column(adata)                              # symbols, before the index changes
    ensure_raw_counts_layer(adata, raise_on_missing=True)        # raw counts in X and layers['counts']
    if adata.raw is not None:
        adata.raw = None
    for k in [k for k in list(adata.layers) if k != "counts"]:
        del adata.layers[k]
    ensure_norm_log(adata)                                       # normalize and log1p X
    adata = remove_infinite_values(adata)
    adata = ensure_ensembl_ids(adata, species=_biomart_species(args.species))  # Ensembl var_names
    ensure_gene_names_column(adata, "gene_names")                # keep symbols in var['gene_names']

    # Categorical and object columns of obs are written as strings, which zellkonverter's R
    # reader handles.
    for c in adata.obs.columns:
        if adata.obs[c].dtype == object or str(adata.obs[c].dtype) == "category":
            adata.obs[c] = adata.obs[c].astype(str)

    adata.write_h5ad(out)
    print(f"[INFO] wrote SingleR reference: {out} "
          f"({adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs[args.label_key].astype(str).nunique()} labels)", flush=True)

    try:
        source.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
