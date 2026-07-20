#!/usr/bin/env python
"""
Build a clustering CONTEXT for the cluster-level annotators (scType, SCSA).

These methods cluster the cells, assign a type per cluster, and propagate it back to cells.
On a 1000-cell eval subset that caps them hard: a rare type has too few cells to form its own
cluster, so it can never receive a correct label. This script builds a larger context that
the methods cluster instead, then they score and emit only the fixed eval cells.

Design (see also stratified_subset in subset.py, whose floored-proportional logic this mirrors):

1. Containment seed. The context MUST contain every cell of both fixed eval subsets
   (sub1000, sub10000) so their labels can be read straight off the context clustering. The
   union of the two subset h5ads is taken verbatim as the seed.

2. Floored-proportional fill up to `max_size` (default 100k; the whole dataset if smaller).
   A type is clusterable on a k-NN graph only above ~2*sqrt(n/k) cells (modularity resolution
   limit); at n=100k that is ~140-165 for the k used here, so each type is first topped up to
   `min_cells_per_type` (default 150) from the not-yet-selected full-data cells. The remaining
   budget is then distributed in proportion to each type's natural abundance, so the bulk of the
   context still reflects real composition. A type with fewer than the floor in the full data
   contributes all of its cells (structure cannot be manufactured) -- an intrinsic limit that is
   reported, not worked around.

Emits <dataset>_context.h5ad plus a QC CSV (per-type: full / seed / context counts). Sampling is
seeded (RANDOM_SEED) so the context is reproducible.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
CONTEXT_MAX_SIZE = 100_000
CONTEXT_MIN_CELLS_PER_TYPE = 150


def slim_context(adata: anndata.AnnData) -> anndata.AnnData:
    """Drop everything the cluster-level annotators don't read, to shrink the context for
    caching / Zenodo transfer. scType and SCSA cluster on ``X`` (the log1p-normalized expression)
    and read only ``obs`` (labels) + ``var`` (gene names); raw counts, embeddings, neighbor graphs
    and plotting metadata are dead weight (often ~half the file). ``X``/``obs``/``var`` are left
    exactly as-is, so the correctly-transformed matrix the models consume is unchanged and their
    predictions are identical. Mutates and returns ``adata``."""
    for attr in ("layers", "obsm", "varm", "obsp", "varp"):
        store = getattr(adata, attr)
        for key in list(store.keys()):
            del store[key]
    if adata.raw is not None:
        adata.raw = None
    adata.uns.clear()
    return adata


def build_context(
    full: anndata.AnnData,
    seed_ids: list[str],
    annotation_key: str,
    max_size: int = CONTEXT_MAX_SIZE,
    min_cells_per_type: int = CONTEXT_MIN_CELLS_PER_TYPE,
    lean: bool = True,
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Return (context AnnData, QC DataFrame).

    Parameters
    ----------
    full : AnnData
        Full preprocessed dataset (the pool to sample from).
    seed_ids : list[str]
        Cell ids that MUST appear in the context (union of the eval subsets). Guarantees
        containment; ids not present in ``full`` are reported and skipped.
    annotation_key : str
        Column in ``full.obs`` holding cell-type labels.
    max_size : int
        Cap on the context size; the whole dataset is used if it has fewer cells.
    min_cells_per_type : int
        Per-type floor the fill aims for (clustering resolution limit).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    obs_names = full.obs_names.astype(str)
    labels = full.obs[annotation_key].astype(str)
    pos_by_id = {cid: i for i, cid in enumerate(obs_names)}

    # --- Containment seed ---
    seed_set = set(seed_ids)
    present = [c for c in seed_ids if c in pos_by_id]
    missing = seed_set - set(present)
    if missing:
        logger.warning(
            "%d/%d seed (eval) cells are absent from the full dataset and cannot be included "
            "in the context; containment will be incomplete for those cells.",
            len(missing), len(seed_set),
        )
    selected = set(present)

    n_full = full.n_obs
    effective_size = min(max_size, n_full)
    if len(selected) > effective_size:
        # The eval subsets alone already exceed the cap: keep them all (containment wins).
        logger.warning(
            "Seed (%d cells) exceeds max_size (%d); using the seed as the context.",
            len(selected), effective_size,
        )
        effective_size = len(selected)

    full_counts = labels.value_counts()
    # Positions of the not-yet-selected cells, grouped by type, in a deterministic order.
    avail_by_type: dict[str, list[int]] = {}
    sel_pos = {pos_by_id[c] for c in selected}
    for t in full_counts.index:
        pos = np.where(labels.values == t)[0]
        rest = [int(p) for p in pos if int(p) not in sel_pos]
        rng.shuffle(rest)
        avail_by_type[str(t)] = rest

    seed_labels = labels.iloc[[pos_by_id[c] for c in selected]] if selected else labels.iloc[[]]
    seed_counts = seed_labels.value_counts()

    def take(t: str, n: int) -> None:
        n = max(0, min(n, len(avail_by_type[t])))
        for _ in range(n):
            selected.add(obs_names[avail_by_type[t].pop()])

    # --- Floor pass: bring every type up to the per-type minimum where the pool allows.
    # Rarest-first, so that when the budget is tight the floor guarantee goes to the types that
    # actually need it (abundant types clear the floor via the proportional fill regardless).
    # At the real scale (max_size >> n_types * floor) the order is a no-op. ---
    for t in full_counts.sort_values(ascending=True).index:
        t = str(t)
        need = min_cells_per_type - int(seed_counts.get(t, 0))
        if need > 0:
            take(t, min(need, effective_size - len(selected)))
        if len(selected) >= effective_size:
            break

    # --- Proportional fill: distribute the remaining budget by natural abundance. ---
    remaining = effective_size - len(selected)
    if remaining > 0:
        headroom = pd.Series(
            {str(t): len(avail_by_type[str(t)]) for t in full_counts.index}
        )
        headroom = headroom[headroom > 0]
        if headroom.sum() > 0:
            weights = full_counts.reindex(headroom.index).astype(float)
            alloc = (weights / weights.sum() * remaining)
            alloc = np.minimum(alloc.round().astype(int), headroom.astype(int))
            for t, n in alloc.items():
                take(str(t), int(n))
            # Any rounding shortfall: top up from whatever pool still has cells.
            leftover = effective_size - len(selected)
            for t in headroom.index:
                if leftover <= 0:
                    break
                before = len(selected)
                take(str(t), leftover)
                leftover -= len(selected) - before

    context = full[[pos_by_id[c] for c in obs_names if c in selected]].copy()
    if lean:
        slim_context(context)

    ctx_counts = context.obs[annotation_key].astype(str).value_counts()
    qc_rows = []
    for t in sorted(full_counts.index.astype(str)):
        qc_rows.append(
            {
                "cell_type": t,
                "full_count": int(full_counts.get(t, 0)),
                "seed_count": int(seed_counts.get(t, 0)),
                "context_count": int(ctx_counts.get(t, 0)),
                "below_floor_in_full": int(full_counts.get(t, 0)) < min_cells_per_type,
            }
        )
    qc_df = pd.DataFrame(qc_rows)
    return context, qc_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="Full preprocessed h5ad (sampling pool).")
    ap.add_argument("--subsets", required=True, nargs="+",
                    help="Fixed eval subset h5ads whose cells must be contained (e.g. sub1000, sub10000).")
    ap.add_argument("--output", required=True, help="Output context h5ad path.")
    ap.add_argument("--annotation_key", default="cell_type")
    ap.add_argument("--max_size", type=int, default=CONTEXT_MAX_SIZE)
    ap.add_argument("--min_cells_per_type", type=int, default=CONTEXT_MIN_CELLS_PER_TYPE)
    ap.add_argument("--qc_csv", default=None, help="Optional per-type QC CSV path.")
    ap.add_argument("--keep_full", action="store_true",
                    help="Keep raw counts / embeddings / uns in the context (default: slim to the "
                         "log-norm X + obs + var that scType/SCSA actually read, ~halving the file).")
    a = ap.parse_args()

    logger.info("Reading full dataset: %s", a.full)
    full = anndata.read_h5ad(a.full)
    if a.annotation_key not in full.obs.columns:
        raise SystemExit(f"annotation_key '{a.annotation_key}' not in full.obs "
                         f"(have: {list(full.obs.columns)})")

    seed_ids: list[str] = []
    for s in a.subsets:
        ids = anndata.read_h5ad(s, backed="r").obs_names.astype(str).tolist()
        seed_ids.extend(ids)
        logger.info("Seed subset %s: %d cells", s, len(ids))
    # De-duplicate while preserving order.
    seed_ids = list(dict.fromkeys(seed_ids))

    context, qc_df = build_context(
        full, seed_ids, a.annotation_key,
        max_size=a.max_size, min_cells_per_type=a.min_cells_per_type,
        lean=not a.keep_full,
    )

    # Containment assertion: every seed cell present in the full data must be in the context.
    present_seed = {c for c in seed_ids if c in set(full.obs_names.astype(str))}
    ctx_ids = set(context.obs_names.astype(str))
    missing = present_seed - ctx_ids
    if missing:
        raise SystemExit(f"Containment broken: {len(missing)} eval cells missing from context.")

    n_below = int(qc_df["below_floor_in_full"].sum())
    logger.info(
        "Context: %d cells, %d types (%d type(s) below the %d-cell floor even in the full data "
        "-> may not form a cluster).",
        context.n_obs, (qc_df["context_count"] > 0).sum(), n_below, a.min_cells_per_type,
    )
    logger.info("Containment OK: all %d in-pool eval cells are in the context.", len(present_seed))

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    context.write(a.output)
    logger.info("Wrote context: %s", a.output)
    if a.qc_csv:
        Path(a.qc_csv).parent.mkdir(parents=True, exist_ok=True)
        qc_df.to_csv(a.qc_csv, index=False)
        logger.info("Wrote context QC: %s", a.qc_csv)


if __name__ == "__main__":
    main()
