#!/usr/bin/env python
"""Build the clustering context of the methods that annotate clusters (scType, SCSA).

These methods cluster the cells and give every cell the label of its cluster. In a test set of
1,000 cells, a rare cell type has too few cells to form a cluster of its own. The methods
therefore cluster a larger sample, the context, and report labels only for the test cells.

The context contains every cell of the test sets (--subsets, e.g. those of 1,000 and 10,000
cells). It is filled up to --max_size cells (default 100,000, or the whole dataset if smaller)
with other cells of the dataset: first every cell type up to --min_cells_per_type cells
(default 150), rarest type first, then in proportion to the type frequencies of the dataset.
A type with fewer cells in the whole dataset contributes all of them. The default of 150
follows the resolution limit of modularity clustering, about 2*sqrt(n/k) cells on a k-nearest
neighbour graph, which is 141 to 163 cells for n = 100,000 and the k of scType (20) and SCSA
(15). Sampling uses a fixed seed (RANDOM_SEED).

Writes the context h5ad and, with --qc_csv, a CSV of the counts per type in the dataset, the
test sets and the context.

    build_context.py --full pp.h5ad --subsets sub1000.h5ad sub10000.h5ad --output context.h5ad
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
    """Remove layers, obsm, varm, obsp, varp, raw and uns from adata, in place, and return it.

    scType and SCSA read only X (log-normalized expression), obs and var, which are kept, so
    their results do not change and the file becomes smaller."""
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
    """Return the context and a table of the counts per type.

    seed_ids are the test cells, which the context contains; ids missing from full are logged
    and skipped. max_size caps the context size and min_cells_per_type is the number of cells per
    type that the fill aims for. With lean, the context is reduced by slim_context.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    obs_names = full.obs_names.astype(str)
    labels = full.obs[annotation_key].astype(str)
    pos_by_id = {cid: i for i, cid in enumerate(obs_names)}

    # Test cells
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
        # The test cells alone exceed max_size: keep all of them.
        logger.warning(
            "Seed (%d cells) exceeds max_size (%d); using the seed as the context.",
            len(selected), effective_size,
        )
        effective_size = len(selected)

    full_counts = labels.value_counts()
    # Positions of the cells not yet selected, per type, shuffled with the fixed seed.
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

    # Every type up to min_cells_per_type cells where the dataset has them. Rarest types first, so
    # that they get these cells when max_size is too small for all types.
    for t in full_counts.sort_values(ascending=True).index:
        t = str(t)
        need = min_cells_per_type - int(seed_counts.get(t, 0))
        if need > 0:
            take(t, min(need, effective_size - len(selected)))
        if len(selected) >= effective_size:
            break

    # The remaining cells in proportion to the type frequencies of the dataset.
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
            # Cells missing after rounding come from the types that still have cells.
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

    # Every test cell found in the dataset must be in the context.
    present_seed = {c for c in seed_ids if c in set(full.obs_names.astype(str))}
    ctx_ids = set(context.obs_names.astype(str))
    missing = present_seed - ctx_ids
    if missing:
        raise SystemExit(f"Containment broken: {len(missing)} eval cells missing from context.")

    n_below = int(qc_df["below_floor_in_full"].sum())
    logger.info(
        "Context: %d cells, %d types (%d type(s) have fewer than %d cells in the full data "
        "and may not form a cluster).",
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
