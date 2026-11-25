#!/usr/bin/env python
"""Produce the highly-variable gene pool used by the gene-list ablation.

Run once per dataset; the ablation then consumes only the small CSV this writes, so the study is
reproducible without any of the heavy inputs.

Why this reads the CONTEXT h5ad and uses flavor="seurat"
-------------------------------------------------------
``seurat_v3``/``seurat_v3_paper`` model the mean-variance relationship on RAW COUNTS. Checked
against the actual files rather than assumed:

* ``cache/datasets/context/<ds>_context.h5ad`` stores only a log1p ``X`` -- no ``layers``, no
  ``raw`` -- so the v3 flavors are unavailable there.
* ``cache/datasets/preprocessed/`` is EMPTY: the pipeline ran the Zenodo fast-path, so the full
  preprocessed data was never materialised locally.
* ``<ds>_sub10000_random.h5ad`` would have counts at natural composition, but the Zenodo manifest
  states those files are not on the record.
* ``<ds>_sub10000_stratified.h5ad`` does have counts, but stratified sampling over-represents rare
  types, which is exactly the composition the mean-variance fit is sensitive to.

So the choice is 100k cells at near-natural composition with ``seurat``, or 10k cells at distorted
composition with ``seurat_v3_paper``. Cell count and composition matter more here than reproducing
Seurat's variance-stabilising transform, because the goal is to drop uninformative genes rather
than to match Seurat exactly. Two consequences to state in the write-up rather than hide: ``seurat``
selects on normalised dispersion, and the contexts are floored-proportional rather than a natural
sample.

The pipeline itself computes no HVGs -- its only scanpy preprocessing calls are ``normalize_total``,
``log1p``, ``calculate_qc_metrics``, ``filter_cells`` and ``filter_genes``. The ``highly_variable``
column that appears in the HIHA subset is inherited from the source atlas and flags all 25,234
genes, so it carries no information. Hence this script.
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
    """Gene symbols for an AnnData whose ``var_names`` are Ensembl IDs.

    Picks whichever candidate column carries the FEWEST Ensembl-looking entries, rather than taking
    the first that exists. Checked against the four contexts and the fixed-order approach fails on
    two of them: HIHA's ``gene_names`` holds Ensembl IDs (the symbols are in ``feature_name`` /
    ``gene_symbols``), and HFRA's ``gene_names`` is a mixture of symbols and Ensembl IDs.
    """
    def usable_fraction(values):
        """Share of entries that are real symbols: not Ensembl IDs and not null placeholders.

        Both halves are required. Scoring on Ensembl content alone picks HIHA's ``gene_symbols``,
        which is 0% Ensembl only because it is almost entirely the string ``nan`` -- it contains no
        ribosomal genes at all, while ``gene_names`` carries the real 108.
        """
        missing = values.isin(("nan", "None", "", "NA", "<NA>"))
        return float((~values.str.match(ENSEMBL_RE) & ~missing).mean())

    candidates = [("var_names", pd.Index(adata.var_names.astype(str)))]
    candidates += [(c, pd.Index(adata.var[c].astype(str)))
                   for c in ("gene_names", "feature_name", "gene_symbols")
                   if c in adata.var.columns]
    best_col, best = max(candidates, key=lambda kv: usable_fraction(kv[1]))
    return best, best_col, 1.0 - usable_fraction(best)


def excluded_mask(adata, prefixes):
    """Genes to drop BEFORE selection: ribosomal, mitochondrial, haemoglobin.

    Derived here from resolved symbols rather than from the ``mt`` / ``ribo`` / ``hb`` columns that
    preprocessing writes, because those columns are not trustworthy across these four datasets:
    HFRA has all three at zero, HIHA likewise but carries a ``mito`` column flagging 24,011 of
    25,234 genes, and human_pancreas reports zero mitochondrial genes. Only Zeisel looks sane.
    Recomputing from symbols is deterministic and auditable; the flags are neither.

    Returns ``(mask, how, symbol_col)``.
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

    # Verify the exclusion actually held through selection. Cheap, and it fails loudly rather than
    # letting a ribosomal gene reach a cell sentence and quietly undermine the whole ablation.
    selected_upper = pd.Index([s.upper() for s in symbols[keep]])
    leaked = {p: sorted(selected_upper[selected_upper.str.startswith(p.upper())])
              for p in hvg_cfg["exclude_prefixes"]}
    leaked = {p: g for p, g in leaked.items() if g}
    if leaked:
        raise SystemExit(f"{dataset}: excluded gene classes present in the selection: {leaked}")

    # Both identifiers are stored: `var_id` is what the h5ad is subset by, `gene_symbol` is what
    # ends up in the cell sentence. Keeping only one would force the consumer to re-resolve.
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
