#!/usr/bin/env python
"""Draw a test set of cells from a preprocessed dataset.

The stratified strategy samples every cell type in proportion to its frequency, with at least
--min_cells_per_type cells per type, and drops the types with fewer cells. The random strategy
samples cells uniformly. If --subset_size is not below the number of cells, the whole dataset is
kept. Writes the subset h5ad and, in --qc_dir, a CSV and a PDF that compare the cell type counts
of the dataset and the subset.

    subset.py --input pp.h5ad --output sub.h5ad --annotation_key cell_type --subset_size 1000 \\
        --dataset NAME --qc_dir qc --strategy stratified
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MIN_CELLS_PER_TYPE = 20


def set_pub_style():
    sns.set_context("paper", font_scale=1)
    sns.set_style("white")
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def stratified_subset(
    adata: anndata.AnnData,
    annotation_key: str,
    subset_size: int,
    min_cells_per_type: int = MIN_CELLS_PER_TYPE,
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Sample about subset_size cells in proportion to the cell type frequencies, with at least
    min_cells_per_type cells per type; types with fewer cells are dropped. Returns the subset and
    a table of the counts and proportions per type."""
    type_counts = adata.obs[annotation_key].value_counts()
    total_cells = type_counts.sum()

    kept_types = type_counts[type_counts >= min_cells_per_type]
    dropped_types = type_counts[type_counts < min_cells_per_type]

    if len(kept_types) == 0:
        raise ValueError(
            f"No cell types have >= {min_cells_per_type} cells. "
            f"Cannot create subset. Largest type has {type_counts.max()} cells."
        )

    kept_total = kept_types.sum()
    effective_size = min(subset_size, kept_total)
    raw_targets = (kept_types / kept_total * effective_size).astype(float)

    targets = raw_targets.clip(lower=min_cells_per_type).astype(float)

    while targets.sum() > effective_size:
        over = targets.sum() - effective_size
        above_min = targets[targets > min_cells_per_type]
        if len(above_min) == 0:
            break
        reduction = (above_min / above_min.sum() * over).clip(
            upper=above_min - min_cells_per_type
        )
        targets[above_min.index] -= reduction
        targets = targets.clip(lower=min_cells_per_type)

    while targets.sum() < effective_size:
        deficit = effective_size - targets.sum()
        headroom = kept_types - targets
        headroom = headroom[headroom > 0]
        if len(headroom) == 0:
            break
        addition = (headroom / headroom.sum() * deficit).clip(upper=headroom)
        targets[headroom.index] += addition

    targets = targets.round().astype(int)
    targets = targets.clip(upper=kept_types)

    selected_indices = []
    for cell_type, target_n in targets.items():
        type_mask = adata.obs[annotation_key] == cell_type
        type_indices = adata.obs.index[type_mask].tolist()
        n = min(int(target_n), len(type_indices))
        selected_indices.extend(random.sample(type_indices, n))

    subset = adata[selected_indices].copy()

    subset_counts = subset.obs[annotation_key].value_counts()
    qc_rows = []
    for cell_type in sorted(set(type_counts.index)):
        orig_count = int(type_counts.get(cell_type, 0))
        sub_count = int(subset_counts.get(cell_type, 0))
        is_dropped = cell_type in dropped_types.index
        reason = ""
        if is_dropped:
            reason = f"below minimum (had {orig_count}, need {min_cells_per_type})"
        qc_rows.append(
            {
                "cell_type": cell_type,
                "original_count": orig_count,
                "original_proportion": orig_count / total_cells,
                "subset_count": sub_count,
                "subset_proportion": sub_count / len(subset) if len(subset) > 0 else 0.0,
                "dropped": is_dropped,
                "drop_reason": reason,
            }
        )

    qc_df = pd.DataFrame(qc_rows)
    return subset, qc_df


def random_subset(
    adata: anndata.AnnData,
    annotation_key: str,
    subset_size: int,
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Sample subset_size cells uniformly at random. Rare types can be missing from the subset,
    since min_cells_per_type does not apply. Returns the subset and a table of the counts and
    proportions per type."""
    type_counts = adata.obs[annotation_key].value_counts()
    total_cells = int(type_counts.sum())

    n = min(subset_size, adata.n_obs)
    rng = np.random.default_rng(RANDOM_SEED)
    selected_positions = rng.choice(adata.n_obs, size=n, replace=False)

    subset = adata[selected_positions].copy()

    subset_counts = subset.obs[annotation_key].value_counts()
    qc_rows = []
    for cell_type in sorted(set(type_counts.index)):
        orig_count = int(type_counts.get(cell_type, 0))
        sub_count = int(subset_counts.get(cell_type, 0))
        qc_rows.append(
            {
                "cell_type": cell_type,
                "original_count": orig_count,
                "original_proportion": orig_count / total_cells,
                "subset_count": sub_count,
                "subset_proportion": sub_count / len(subset) if len(subset) > 0 else 0.0,
                "dropped": False,
                "drop_reason": "",
            }
        )
    return subset, pd.DataFrame(qc_rows)


def plot_subset_qc(qc_df: pd.DataFrame, dataset: str, subset_size: int, min_cells_per_type: int,
                   pdf_path: Path):
    """Write bar plots of the cell type counts and proportions in the dataset and the subset, and
    of the types dropped for having fewer than min_cells_per_type cells, to pdf_path."""
    set_pub_style()

    kept = qc_df[~qc_df["dropped"]].copy()
    dropped = qc_df[qc_df["dropped"]].copy()

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(max(8, len(kept) * 0.4), 5))
        x = np.arange(len(kept))
        width = 0.35
        ax.bar(
            x - width / 2,
            kept["original_count"],
            width,
            label="Original",
            color="#4878d0",
        )
        ax.bar(
            x + width / 2,
            kept["subset_count"],
            width,
            label=f"Subset (n={subset_size})",
            color="#ee854a",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(kept["cell_type"], rotation=45, ha="right")
        ax.set_ylabel("Cell count")
        ax.set_title(f"{dataset} - Cell type counts: original vs. subset")
        ax.legend()
        sns.despine()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, len(kept) * 0.4), 5))
        ax.bar(
            x - width / 2,
            kept["original_proportion"],
            width,
            label="Original",
            color="#4878d0",
        )
        ax.bar(
            x + width / 2,
            kept["subset_proportion"],
            width,
            label=f"Subset (n={subset_size})",
            color="#ee854a",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(kept["cell_type"], rotation=45, ha="right")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{dataset} - Cell type proportions: original vs. subset")
        ax.legend()
        sns.despine()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if len(dropped) > 0:
            fig, ax = plt.subplots(figsize=(max(6, len(dropped) * 0.5), 4))
            ax.barh(dropped["cell_type"], dropped["original_count"], color="#e15759")
            ax.axvline(
                x=min_cells_per_type,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"Minimum threshold ({min_cells_per_type})",
            )
            ax.set_xlabel("Original cell count")
            ax.set_title(f"{dataset} - Dropped cell types (below minimum)")
            ax.legend()
            sns.despine()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parser = argparse.ArgumentParser(
        description="Stratified subset sampling for scRNA-seq data"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input preprocessed h5ad file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output subset h5ad file"
    )
    parser.add_argument(
        "--annotation_key",
        type=str,
        required=True,
        help="Cell type annotation column",
    )
    parser.add_argument(
        "--subset_size", type=int, required=True, help="Target number of cells"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (for plot titles)"
    )
    parser.add_argument(
        "--qc_dir", type=Path, required=True, help="Directory for QC outputs"
    )
    parser.add_argument(
        "--min_cells_per_type",
        type=int,
        default=MIN_CELLS_PER_TYPE,
        help=f"Minimum cells per type (default: {MIN_CELLS_PER_TYPE})",
    )
    parser.add_argument(
        "--strategy",
        choices=["stratified", "random"],
        required=True,
        help="Subsampling strategy: 'stratified' preserves per-type proportions; 'random' is uniform over cells",
    )
    args = parser.parse_args()

    logger.info("Reading input: %s", args.input)
    # Read into memory: with the anndata version of the environment, subsetting a backed sparse
    # file raises an AttributeError in _validate_indices and loses X. The dataset must fit in RAM.
    adata = anndata.read_h5ad(args.input)
    logger.info("Input shape: %s cells x %s genes", adata.n_obs, adata.n_vars)

    default_min = MIN_CELLS_PER_TYPE
    if args.strategy == "random" and args.min_cells_per_type != default_min:
        logger.info(
            "random mode: ignoring --min_cells_per_type=%s",
            args.min_cells_per_type,
        )

    if args.subset_size >= adata.n_obs:
        logger.warning(
            "subset_size (%s) >= n_obs (%s), using the full dataset",
            args.subset_size,
            adata.n_obs,
        )
        subset = adata.copy()
        type_counts = adata.obs[args.annotation_key].value_counts()
        total = int(type_counts.sum())
        qc_rows = []
        for cell_type in sorted(type_counts.index):
            count = int(type_counts[cell_type])
            qc_rows.append(
                {
                    "cell_type": cell_type,
                    "original_count": count,
                    "original_proportion": count / total,
                    "subset_count": count,
                    "subset_proportion": count / total,
                    "dropped": False,
                    "drop_reason": "",
                }
            )
        qc_df = pd.DataFrame(qc_rows)
    elif args.strategy == "random":
        subset, qc_df = random_subset(adata, args.annotation_key, args.subset_size)
    else:
        subset, qc_df = stratified_subset(
            adata, args.annotation_key, args.subset_size, args.min_cells_per_type
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(args.output, compression="gzip")
    logger.info("Subset of %s cells written to %s", subset.n_obs, args.output)

    args.qc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.qc_dir / f"cell_type_distribution_{args.strategy}.csv"
    qc_df.to_csv(csv_path, index=False)
    logger.info("QC CSV written: %s", csv_path)

    pdf_path = args.qc_dir / f"subset_qc_plots_{args.strategy}.pdf"
    plot_subset_qc(qc_df, args.dataset, args.subset_size, args.min_cells_per_type, pdf_path)
    logger.info("QC plots written: %s", pdf_path)

    n_kept = (~qc_df["dropped"]).sum()
    n_dropped = qc_df["dropped"].sum()
    logger.info("Summary: %s cell types kept, %s dropped", n_kept, n_dropped)
    if n_dropped > 0:
        for _, row in qc_df[qc_df["dropped"]].iterrows():
            logger.info("  Dropped: %s (%s)", row["cell_type"], row["drop_reason"])


if __name__ == "__main__":
    main()
