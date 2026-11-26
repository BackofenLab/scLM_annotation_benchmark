#!/usr/bin/env python
"""Select the pool of highly variable genes for the ``hvg`` gene variant.

Reads each dataset's clustering context (``hvg.source_dir``), drops genes by symbol prefix
(``hvg.exclude_prefixes``: ribosomal, mitochondrial, haemoglobin) and selects ``hvg.n_top_genes``
genes with ``scanpy.pp.highly_variable_genes``, per batch where the dataset's YAML names a batch
key. The flavor is ``seurat`` because the contexts hold log-normalised expression without counts,
which the ``seurat_v3`` flavors need. Writes one CSV per dataset to ``hvg.out_dir``.

Run from paper/experiments, since the data paths in config.yml are relative:
    python src/hvg.py [--datasets ...] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def dataset_batch_key(dataset, datasets_dir):
    """Batch key declared by the dataset's own YAML, or None if it has none."""
    cfg_path = Path(datasets_dir) / f"{dataset}.yml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as fh:
        return (yaml.safe_load(fh) or {}).get("batch_key")


ENSEMBL_RE = r"^ENS[A-Z]*G\d"


def gene_symbols(adata):
    """Gene symbols for an AnnData indexed by Ensembl ids.

    Taken from whichever of ``var_names``, ``gene_names``, ``feature_name`` and ``gene_symbols``
    has the largest share of usable symbols. Returns ``(symbols, column, unusable share)``.
    """
    def usable_fraction(values):
        """Share of entries that are neither Ensembl ids nor null placeholders such as "nan"."""
        missing = values.isin(("nan", "None", "", "NA", "<NA>"))
        return float((~values.str.match(ENSEMBL_RE) & ~missing).mean())

    candidates = [("var_names", pd.Index(adata.var_names.astype(str)))]
    candidates += [(c, pd.Index(adata.var[c].astype(str)))
                   for c in ("gene_names", "feature_name", "gene_symbols")
                   if c in adata.var.columns]
    best_col, best = max(candidates, key=lambda kv: usable_fraction(kv[1]))
    return best, best_col, 1.0 - usable_fraction(best)


def excluded_mask(adata, prefixes):
    """Genes whose symbol starts with one of ``prefixes``. Returns ``(mask, how, symbol_col)``.

    The mask comes from the symbols because the ``mt``, ``ribo`` and ``hb`` columns of the
    contexts are all zero for HFRA and HIHA.
    """
    symbols, symbol_col, ens_frac = gene_symbols(adata)
    upper = pd.Index([s.upper() for s in symbols])
    mask = pd.Series(False, index=range(len(upper)))
    for prefix in prefixes:
        mask |= pd.Series(upper.str.startswith(prefix.upper()), index=range(len(upper)))
    how = (f"symbol prefixes ({', '.join(prefixes)}) on var['{symbol_col}']"
           f"{f', {ens_frac:.0%} unresolved' if ens_frac > 0.01 else ''}")
    return mask.to_numpy(), how, symbol_col


def select(dataset, cfg, datasets_dir):
    hvg_cfg = cfg["hvg"]
    src = Path(hvg_cfg["source_dir"]) / f"{dataset}_context.h5ad"
    if not src.exists():
        raise SystemExit(f"{dataset}: no context h5ad at {src}")

    adata = sc.read_h5ad(src)
    n_genes_before = adata.n_vars

    drop, how, _symbol_col = excluded_mask(adata, hvg_cfg["exclude_prefixes"])
    if not drop.any():
        raise SystemExit(
            f"{dataset}: the exclusion step matched 0 of {n_genes_before} genes via {how}. "
            f"That is almost certainly a bug rather than a clean dataset -- check that the "
            f"symbol column resolved to symbols and not to Ensembl IDs.")
    adata = adata[:, ~drop].copy()

    batch_key = None
    if hvg_cfg.get("use_batch_key"):
        batch_key = dataset_batch_key(dataset, datasets_dir)
        if batch_key and batch_key not in adata.obs.columns:
            print(f"  [{dataset}] batch key '{batch_key}' absent from the context; "
                  f"selecting without it", file=sys.stderr)
            batch_key = None

    sc.pp.highly_variable_genes(
        adata,
        flavor=hvg_cfg["flavor"],
        n_top_genes=hvg_cfg["n_top_genes"],
        batch_key=batch_key,
    )
    keep = adata.var["highly_variable"].to_numpy()
    symbols, _col, _frac = gene_symbols(adata)

    # Stop if a selected gene belongs to an excluded class.
    selected_upper = pd.Index([s.upper() for s in symbols[keep]])
    leaked = {p: sorted(selected_upper[selected_upper.str.startswith(p.upper())])
              for p in hvg_cfg["exclude_prefixes"]}
    leaked = {p: g for p, g in leaked.items() if g}
    if leaked:
        raise SystemExit(f"{dataset}: excluded gene classes present in the selection: {leaked}")

    # sentences.py matches var_id (the Ensembl id); gene_symbol is for reading.
    out = pd.DataFrame({
        "var_id": list(adata.var_names[keep]),
        "gene_symbol": list(symbols[keep]),
    })
    out["dataset"] = dataset
    out["flavor"] = hvg_cfg["flavor"]
    out["n_top_genes"] = hvg_cfg["n_top_genes"]
    out["batch_key"] = batch_key or ""
    out["n_cells_context"] = adata.n_obs
    out["n_genes_before_exclusion"] = n_genes_before
    out["n_genes_excluded"] = int(drop.sum())
    out["exclusion_method"] = how
    return out


def out_path(dataset, cfg):
    hvg_cfg = cfg["hvg"]
    suffix = "_batch" if hvg_cfg.get("use_batch_key") else ""
    name = f"{dataset}_hvg_{hvg_cfg['flavor']}_{hvg_cfg['n_top_genes']}{suffix}.csv"
    return Path(hvg_cfg["out_dir"]) / name


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yml"))
    p.add_argument("--datasets", nargs="+", help="override the dataset list in the config")
    p.add_argument("--datasets-dir", default=str(REPO_ROOT / "datasets"),
                   help="where the per-dataset YAMLs live, for the batch key")
    p.add_argument("--force", action="store_true", help="recompute even if the CSV exists")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    Path(cfg["hvg"]["out_dir"]).mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        dest = out_path(dataset, cfg)
        if dest.exists() and not args.force:
            print(f"[skip] {dataset}: {dest} exists (--force to recompute)")
            continue
        print(f"[run ] {dataset}")
        table = select(dataset, cfg, args.datasets_dir)
        table.to_csv(dest, index=False)
        row = table.iloc[0]
        print(f"[done] {dataset}: {len(table)} genes -> {dest}\n"
              f"       {row['n_genes_excluded']} of {row['n_genes_before_exclusion']} genes "
              f"excluded via {row['exclusion_method']}; {row['n_cells_context']:,} context cells; "
              f"batch_key={row['batch_key'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
