#!/usr/bin/env python
"""Cluster each dataset's context and write the prompt files of the cluster arms.

Leiden clustering of the clustering context (``cluster.source_dir``) at every resolution of
``cluster.resolutions``, then the marker genes of each cluster (``rank_genes_groups``, without
ribosomal, mitochondrial and haemoglobin genes). ``bin/build_context.py`` puts every eval cell into
the context, so each eval cell gets the prompt of its cluster. Writes the assignments, markers and
``cluster_summary.csv`` to ``results/clusters/`` and the prompt files of ``A4_llm_percluster`` to
``results/prompts/``. With ``--study lists`` it reads the stored clusters at ``lists.resolution``
and writes the prompt files of the list study's cluster arm.

Run from paper/experiments, since the data paths in config.yml are relative:
    python src/cluster.py
    python src/cluster.py --study lists
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def eval_cell_ids(dataset, cfg):
    """Cell ids of the eval subset."""
    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    if not src.exists():
        raise SystemExit(f"{dataset}: no eval subset at {src}")
    return pd.Index(ad.read_h5ad(src, backed="r").obs_names.astype(str))


def cluster_context(dataset, cfg, resolution, seed=0):
    """Leiden clusters of the context and the marker genes of each cluster.

    Returns ``(assignments, markers, info)``: ``assignments`` is a Series of cluster ids indexed by
    cell id, ``markers`` maps each cluster id to its marker gene symbols, highest ranked first.
    """
    ccfg = cfg["cluster"]
    src = Path(ccfg["source_dir"]) / f"{dataset}_context.h5ad"
    if not src.exists():
        raise SystemExit(f"{dataset}: no context h5ad at {src}")

    adata = sc.read_h5ad(src)
    # The contexts are log-normalised already, so PCA runs on X.
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1), random_state=seed)
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, key_added="cluster",
                 random_state=seed, flavor="igraph", n_iterations=2, directed=False)

    # The clustering uses all genes; the markers exclude the gene classes the HVG pool drops.
    from hvg import excluded_mask
    drop, _how, _col = excluded_mask(adata, ["RPS", "RPL", "MT-", "HB"])
    ranked_on = adata[:, ~drop].copy() if drop.any() else adata
    sc.tl.rank_genes_groups(ranked_on, groupby="cluster", method=ccfg["rank_method"])
    names = ranked_on.uns["rank_genes_groups"]["names"]
    adata = ranked_on

    # The markers are Ensembl ids; the prompts name gene symbols.
    from sentences import resolve_symbols
    symbol_of = dict(zip(adata.var_names.astype(str),
                         resolve_symbols(adata.copy())[0].var_names.astype(str)))

    n_markers = ccfg["n_marker_genes"]
    markers = {c: [symbol_of.get(g, g) for g in list(names[c])[:n_markers]]
               for c in names.dtype.names}

    assignments = pd.Series(adata.obs["cluster"].astype(str).values,
                            index=adata.obs_names.astype(str), name="cluster")
    info = {"dataset": dataset, "resolution": resolution, "n_clusters": len(markers),
            "n_context_cells": int(adata.n_obs)}
    return assignments, markers, info


def restrict_to_eval(assignments, dataset, cfg):
    """The cluster of each eval cell; stops if an eval cell is missing from the context."""
    ids = eval_cell_ids(dataset, cfg)
    missing = ids.difference(assignments.index)
    if len(missing):
        raise SystemExit(
            f"{dataset}: {len(missing)} of {len(ids)} eval cells are absent from the context. "
            f"build_context.py is supposed to seed the context with every eval cell, so this "
            f"means the context and the subset are out of sync -- do not score this arm.")
    return assignments.loc[ids]


def materialize_cluster_prompts(dataset, cfg, resolution, per_eval, markers, out_dir,
                                arm="A4_llm_percluster", modes=None, candidates=None):
    """Write the prompt files of a cluster arm: one row per eval cell, with its cluster's prompt.

    The rows are per cell, as for the per-cell arms; the runner sends each distinct prompt once.
    The resolution takes the gene-variant field of the file name (``res0.5``, ``res1.0``).
    ``modes`` and ``candidates(mode, seed)`` serve the imperfect-list study, as in
    ``sentences.materialize``.
    """
    import numpy as np
    from prompts import build_prompt, candidate_list

    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    ds_cfg = _dataset_cfg(dataset)
    truth_key = ds_cfg.get("annotation_key", "cell_type")
    obs = ad.read_h5ad(src, backed="r").obs
    labels = sorted(obs[truth_key].astype(str).unique())
    tissue, species = ds_cfg.get("tissue"), (ds_cfg.get("species") or ds_cfg.get("organism"))
    override = ds_cfg.get("context_phrase")

    written = []
    for mode_name, mode in (modes or cfg["modes"]).items():
        # A cluster prompt consists of its marker genes, so a mode without genes is skipped.
        if not mode.get("genes"):
            continue
        for seed in cfg["repeats"]["seeds"]:
            rng = np.random.default_rng(seed)
            listed = candidates(mode_name, seed) if candidates else labels
            # One list order per cluster, so all cells of a cluster share one prompt.
            per_cluster = {c: (candidate_list(listed, rng) if mode.get("cell_types") else None)
                           for c in sorted(markers)}
            rows = []
            for cell_id, cluster in per_eval.items():
                rows.append({
                    "cell_id": cell_id,
                    "true_labels": str(obs.loc[cell_id, truth_key]),
                    "cluster": cluster,
                    "prompt": build_prompt(
                        "marker_genes", mode,
                        num_genes=str(cfg["cluster"]["n_marker_genes"]),
                        marker_genes=markers.get(cluster, []),
                        unique_cell_types=per_cluster.get(cluster),
                        tissue=tissue, species=species, context_override=override),
                })
            name = f"{dataset}__{arm}__{mode_name}__res{resolution}__seed{seed}.csv"
            pd.DataFrame(rows).to_csv(Path(out_dir) / name, index=False)
            written.append(name)
    return written


def _dataset_cfg(dataset):
    with open(REPO_ROOT / "datasets" / f"{dataset}.yml") as fh:
        return yaml.safe_load(fh) or {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yml"))
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--resolutions", nargs="+", type=float)
    p.add_argument("--out-dir", default=None,
                   help="where cluster assignments and markers are written "
                        "(default: results/clusters)")
    p.add_argument("--study", default="ablation",
                   help="the study tree (see `studies` in config). The candidate-list study reuses "
                        "the clusters stored in --out-dir instead of clustering again")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    resolutions = args.resolutions or cfg["cluster"]["resolutions"]
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir or (root / "results" / "clusters"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = root / cfg["studies"][args.study] / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    if args.study == "lists":
        from lists import prompt_args
        resolution = cfg["lists"]["resolution"]
        for dataset in datasets:
            study = prompt_args(cfg, prompts_dir.parent, dataset)
            arm = next(name for name, spec in study.pop("arms").items()
                       if spec["granularity"] == "cluster")
            tag = out_dir / f"{dataset}_res{resolution}"
            # Cluster ids as strings, since the lists are drawn in the order of sorted(markers).
            per_eval = pd.read_csv(f"{tag}_assignments.csv", index_col=0, dtype=str)["cluster"]
            markers = {c: genes.split(", ") for c, genes in
                       pd.read_csv(f"{tag}_markers.csv", dtype=str).itertuples(index=False)}
            written = materialize_cluster_prompts(dataset, cfg, resolution, per_eval, markers,
                                                  prompts_dir, arm=arm, **study)
            print(f"[done] {dataset} res={resolution}: {len(written)} prompt files -> {prompts_dir}")
        return 0

    rows = []
    for dataset in datasets:
        for resolution in resolutions:
            assignments, markers, info = cluster_context(dataset, cfg, resolution)
            per_eval = restrict_to_eval(assignments, dataset, cfg)

            tag = f"{dataset}_res{resolution}"
            per_eval.rename("cluster").to_frame().to_csv(out_dir / f"{tag}_assignments.csv")
            pd.DataFrame([{"cluster": c, "markers": ", ".join(g)}
                          for c, g in sorted(markers.items())]).to_csv(
                out_dir / f"{tag}_markers.csv", index=False)

            written = materialize_cluster_prompts(dataset, cfg, resolution, per_eval, markers,
                                                  prompts_dir)

            info["n_clusters_in_eval"] = int(per_eval.nunique())
            info["calls_per_cell_arm"] = len(per_eval)
            info["prompt_files"] = len(written)
            rows.append(info)
            print(f"[done] {dataset} res={resolution}: {info['n_clusters']} clusters on "
                  f"{info['n_context_cells']:,} context cells, "
                  f"{info['n_clusters_in_eval']} of them present among the {len(per_eval)} eval "
                  f"cells -> {info['n_clusters_in_eval']} LLM calls instead of {len(per_eval)}")
    pd.DataFrame(rows).to_csv(out_dir / "cluster_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
