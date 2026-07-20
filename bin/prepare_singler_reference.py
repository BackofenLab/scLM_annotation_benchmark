#!/usr/bin/env python
"""Fetch a CELLxGENE Census dataset and normalize it into a SingleR reference h5ad.

The output is structurally identical to the pipeline's preprocessed *query* h5ads
(Ensembl ``var_names``, log1p-normalized ``X``, gene symbols in ``var['gene_names']``, a
``counts`` layer) so that ``models/singler_blind/src/predict.R`` can read it with the exact same
``zellkonverter::readH5AD(reader="R")`` call it uses for the query, relabel ``X``->``logcounts``,
and intersect genes on Ensembl IDs.

Unlike ``bin/preprocess.py`` this deliberately SKIPS QC + gene/cell filtering: a reference must keep
every labeled cell type (filtering could drop rare classes). It only fixes the counts/Ensembl/norm
conventions and subsamples per label (deterministic) to keep SingleR fast on 120k-400k-cell atlases.

Fetch primitive is the same one ``bin/download.py`` uses:
``cellxgene_census.download_source_h5ad(id, to_path, census_version)``.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make util.* importable
from util.pp_utils import (  # noqa: E402
    ensure_gene_names_column,
    ensure_raw_counts_layer,
    ensure_norm_log,
    remove_infinite_values,
    ensure_ensembl_ids,
)

# CellxGene Census native deps may fail to import (e.g. x86 under Rosetta on Apple Silicon);
# defer the error to runtime so --help still works and the message is actionable.
try:
    import cellxgene_census  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    cellxgene_census = None  # type: ignore[assignment]
    _census_import_error = e

# Dataset YAML `species` -> pybiomart species used by ensure_ensembl_ids.
_SPECIES_MAP = {
    "human": "hsapiens", "homo sapiens": "hsapiens", "hsapiens": "hsapiens",
    "mouse": "mmusculus", "mus musculus": "mmusculus", "mmusculus": "mmusculus",
}


def _biomart_species(s: str) -> str:
    return _SPECIES_MAP.get(str(s).strip().lower(), "hsapiens")


def subsample_per_label(adata, label_key: str, cap: int, seed: int):
    """Deterministically cap the number of cells per label (keeps all labels)."""
    if not cap or cap <= 0:
        return adata
    rng = np.random.RandomState(seed)
    labels = adata.obs[label_key].astype(str).values
    keep = []
    for lab in pd.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return adata[keep].copy()


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
    adata = sc.read_h5ad(source)
    print(f"[INFO] source: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

    if args.label_key not in adata.obs.columns:
        sys.exit(f"[ERROR] label_key '{args.label_key}' not in obs columns: "
                 f"{list(adata.obs.columns)[:40]}")

    # Drop cells with no usable label (NA / empty / literal 'nan').
    lab = adata.obs[args.label_key].astype(str)
    keep = lab.notna() & (lab.str.strip() != "") & (lab.str.strip().str.lower() != "nan")
    n0 = adata.n_obs
    adata = adata[keep.values].copy()
    print(f"[INFO] dropped {n0 - adata.n_obs} cells with empty/NA '{args.label_key}'", flush=True)

    # Subsample per label BEFORE normalization to save work (deterministic).
    adata = subsample_per_label(adata, args.label_key, args.cap_per_label, args.seed)
    print(f"[INFO] after subsample(cap={args.cap_per_label}): {adata.n_obs} cells, "
          f"{adata.obs[args.label_key].astype(str).nunique()} labels", flush=True)

    # --- Normalize to the query convention (mirror preprocess_adata; SKIP QC + gene/cell filters) ---
    ensure_gene_names_column(adata)                              # capture symbols before Ensembl swap
    ensure_raw_counts_layer(adata, raise_on_missing=True)        # raw counts -> X + layers['counts']
    if adata.raw is not None:
        adata.raw = None
    for k in [k for k in list(adata.layers) if k != "counts"]:
        del adata.layers[k]
    ensure_norm_log(adata, in_place=True)                        # normalize + log1p into X
    adata = remove_infinite_values(adata)
    adata = ensure_ensembl_ids(adata, species=_biomart_species(args.species))  # var_names -> Ensembl
    ensure_gene_names_column(adata, "gene_names")                # keep symbols in var['gene_names']

    # Sanitize obs for a clean h5ad round-trip (categorical/object -> str), matching what the
    # query h5ads carry so zellkonverter reader="R" reads it without dtype surprises.
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
