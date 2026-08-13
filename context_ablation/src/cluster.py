#!/usr/bin/env python
"""Cluster the context, derive per-cluster marker genes, and map cluster labels back to eval cells.

This is arm A4. It deliberately mirrors ``models/scsa_blind/src/predict.py``, which already does
cluster -> ``rank_genes_groups`` -> per-cell mapping on the same context object: the arm is then
methodologically identical to the cluster-level baselines except for how a cluster gets its label,
which is what makes the comparison to them clean.

Why the CONTEXT rather than the 1,000-cell eval subset
------------------------------------------------------
Clustering 1,000 stratified cells would produce clusters shaped by the stratification rather than
by the data, and rare types would have too few cells to separate at all. ``build_context.py``
guarantees the context CONTAINS every eval cell, so clusters are learned on ~100k cells and the
labels are then read off for exactly the cells the other arms annotate. Same eval set, same
scoring, only the granularity differs.

Two resolutions are run on purpose. Resolution sensitivity is not a nuisance parameter here: the
Introduction's argument against cluster-level annotation is that it is "only as fine-grained as the
clustering that preceded it", so measuring how much the answer moves with resolution IS the claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def eval_cell_ids(dataset, cfg):
    """Barcodes of the cells every arm annotates, so A4 scores exactly the same set."""
    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    if not src.exists():
        raise SystemExit(f"{dataset}: no eval subset at {src}")
    return pd.Index(ad.read_h5ad(src, backed="r").obs_names.astype(str))


def cluster_context(dataset, cfg, resolution, seed=0):
    """Leiden clusters on the context, plus marker genes per cluster.

    Returns ``(assignments, markers, info)`` where ``assignments`` is a Series indexed by cell id
    and ``markers`` maps cluster id -> ordered gene symbols.
    """
    ccfg = cfg["cluster"]
    src = Path(ccfg["source_dir"]) / f"{dataset}_context.h5ad"
    if not src.exists():
        raise SystemExit(f"{dataset}: no context h5ad at {src}")

    adata = sc.read_h5ad(src)
    # The contexts store a log1p X and nothing else -- no counts layer, no raw -- so the standard
    # neighbourhood pipeline runs directly on X. Verified against all four files.
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1), random_state=seed)
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, key_added="cluster",
                 random_state=seed, flavor="igraph", n_iterations=2, directed=False)

    # Marker detection excludes ribosomal, mitochondrial and haemoglobin genes -- the same classes
    # the HVG arm drops. CLUSTERING still uses every gene, matching what scType and SCSA do; only
    # the marker LIST is filtered, which is also what a careful analyst reports.
    #
    # Not cosmetic: on the unfiltered run, 16 of 284 clusters had marker lists more than half
    # ribosomal, and HIHA's cluster 0 was ribosomal top to bottom. Handing an annotator a cluster
    # described only by RPL/RPS genes measures the gene selection, not the annotator.
    from hvg import excluded_mask
    drop, _how, _col = excluded_mask(adata, ["RPS", "RPL", "MT-", "HB"])
    ranked_on = adata[:, ~drop].copy() if drop.any() else adata
    sc.tl.rank_genes_groups(ranked_on, groupby="cluster", method=ccfg["rank_method"])
    names = ranked_on.uns["rank_genes_groups"]["names"]
    adata = ranked_on

    # Marker genes are reported as var_names (Ensembl ids here), so resolve to symbols -- the
    # prompt has to name genes the model can recognise.
    from sentences import resolve_symbols  # local import: shares one resolution rule
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
    """Cluster label per EVAL cell, with the containment guarantee checked rather than assumed."""
    ids = eval_cell_ids(dataset, cfg)
    missing = ids.difference(assignments.index)
    if len(missing):
        raise SystemExit(
            f"{dataset}: {len(missing)} of {len(ids)} eval cells are absent from the context. "
            f"build_context.py is supposed to seed the context with every eval cell, so this "
            f"means the context and the subset are out of sync -- do not score this arm.")
    return assignments.loc[ids]


def materialize_cluster_prompts(dataset, cfg, resolution, per_eval, markers, out_dir):
    """Write A4's prompts: ONE ROW PER EVAL CELL, carrying that cell's cluster prompt.

    Per cell rather than per cluster so the output drops straight into the same scoring path as
    every other arm -- predictions must be per cell to compare against per-cell truth. The cost of
    the redundancy is nil: the runner's cache is keyed on prompt text, so the ~13-59 distinct
    cluster prompts are called once each and the remaining ~950 rows are cache hits.

    Resolution rides in the `variant` slot of the filename. The five-field naming has no separate
    place for it, and it is genuinely a variant of the arm: aggregate.py then carries `res0.5` /
    `res1.0` through as its own column, which is what the two-resolution panel needs.
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
    for mode_name, mode in cfg["modes"].items():
        # M5 withholds the gene list, and a cluster prompt is nothing BUT the gene list, so the
        # mode has no meaning for this arm.
        if not mode.get("genes"):
            continue
        for seed in cfg["repeats"]["seeds"]:
            rng = np.random.default_rng(seed)
            # One shuffle per CLUSTER, not per cell: every cell in a cluster must receive the
            # identical prompt or the cache cannot collapse them and the arm costs 30x more.
            per_cluster = {c: (candidate_list(labels, rng) if mode.get("cell_types") else None)
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
            name = (f"{dataset}__A4_llm_percluster__{mode_name}__res{resolution}__"
                    f"seed{seed}.csv")
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
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    resolutions = args.resolutions or cfg["cluster"]["resolutions"]
    out_dir = Path(args.out_dir or (Path(__file__).resolve().parents[1] / "results" / "clusters"))
    out_dir.mkdir(parents=True, exist_ok=True)

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

            prompts_dir = (Path(__file__).resolve().parents[1] / cfg["execution"]["prompts_dir"])
            prompts_dir.mkdir(parents=True, exist_ok=True)
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
