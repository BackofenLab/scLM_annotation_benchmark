#!/usr/bin/env python
"""
Main preprocessing script for single-cell RNA-seq data.

This script performs a streamlined preprocessing pipeline:
1. Finds and sets raw counts in adata.X and adata.layers["counts"]
2. Performs quality control filtering
3. Filters genes and cells based on expression thresholds
4. Ensures log normalization
5. Removes infinite values
6. Generates a summary of the preprocessed data

Configuration can be provided via a YAML dataset config file (as used in Nextflow main.nf).
"""

import argparse
import logging
from pathlib import Path
import yaml
import scanpy as sc
from anndata import AnnData

from util.pp_utils import (
    ensure_raw_counts_layer,
    ensure_ensembl_ids,
    pp_quality_control,
    filter_genes_and_cells,
    ensure_norm_log,
    remove_infinite_values,
    summarize_adata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path | None) -> dict:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : Path, optional
        Path to the YAML configuration file. If None, returns empty dict.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None or not config_path.exists():
        logger.info("No config file provided or file does not exist. Using defaults.")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config or {}


def preprocess_adata(
    adata: AnnData,
    *,
    count_layer_key: str | None = None,
    ensembl_col: str | None = None,
    gene_names_col: str | None = None,
    species: str = "hsapiens",
    nmads_main: int = 5,
    nmads_mt: int = 3,
    pct_counts_mt_threshold: float = 8.0,
    percent_top: list[int] | None = None,
    log1p_for_qc: bool = True,
    min_cells: int = 10,
    min_genes: int = 200,
    summary_path: str | Path | None = None,
) -> AnnData:
    """
    Preprocess an AnnData object with quality control, filtering, and normalization.

    This function applies a streamlined preprocessing pipeline:
    1. Ensures raw counts are properly stored in adata.X and adata.layers["counts"]
    2. Ensures Ensembl IDs are present and set as index
    3. Performs quality control filtering (outlier detection based on QC metrics)
    4. Filters genes and cells based on expression thresholds
    5. Ensures normalization and log transformation of adata.X (normalize first, then log-transform)
    6. Removes infinite values
    7. Optionally generates a summary of the preprocessed data

    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess.
    count_layer_key : str, optional
        Key in adata.layers containing raw counts. If None, defaults to "counts".
        This can be overridden by dataset config.
    ensembl_col : str, optional
        Column name containing Ensembl IDs. If None, searches in:
        "ensembl_id", "feature_id", "gene_id", or index.
    gene_names_col : str, optional
        Column name containing gene names/symbols for mapping. If None, searches in:
        index (if not Ensembl IDs), "gene_names", or "feature_names".
    species : str, optional
        Species name for pybiomart query. Default is "hsapiens".
    nmads_main : int, optional
        Number of MADs for main QC outlier detection. Default is 5.
    nmads_mt : int, optional
        Number of MADs for mitochondrial outlier detection. Default is 3.
    pct_counts_mt_threshold : float, optional
        Absolute threshold for % mitochondrial counts. Default is 8.0.
    percent_top : list[int], optional
        List of top gene counts for QC metrics. Default is [20].
    log1p_for_qc : bool, optional
        Whether to compute log1p values for QC metrics. Default is True.
    min_cells : int, optional
        Minimum cells per gene for filtering. Default is 10.
    min_genes : int, optional
        Minimum genes per cell for filtering. Default is 200.
    summary_path : str | Path, optional
        Path to write the summary file. If None, no summary is generated.

    Returns
    -------
    AnnData
        The preprocessed AnnData object.
    """
    logger.info("Starting preprocessing pipeline.")
    logger.info(f"Input shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 1: Ensure raw counts are properly stored
    logger.info("Step 1: Finding and setting raw counts...")
    ensure_raw_counts_layer(adata, raw_layer_key=count_layer_key, raise_on_missing=True)
    logger.info("Raw counts set in adata.X and adata.layers['counts']")

    # Step 2: Ensure Ensembl IDs
    logger.info("Step 2: Ensuring Ensembl IDs are present...")
    adata = ensure_ensembl_ids(
        adata,
        ensembl_col=ensembl_col,
        gene_names_col=gene_names_col,
        species=species,
        raise_on_failure=True,
    )
    logger.info(f"After Ensembl ID mapping: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 3: Quality control
    logger.info("Step 3: Performing quality control...")
    adata = pp_quality_control(
        adata,
        nmads_main=nmads_main,
        nmads_mt=nmads_mt,
        pct_counts_mt_threshold=pct_counts_mt_threshold,
        percent_top=percent_top,
        log1p_for_qc=log1p_for_qc,
    )
    logger.info(f"After QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 4: Filter genes and cells
    logger.info("Step 4: Filtering genes and cells...")
    adata = filter_genes_and_cells(adata, min_cells=min_cells, min_genes=min_genes)
    logger.info(f"After filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 5: Ensure normalization and log transformation
    logger.info("Step 5: Ensuring normalization and log transformation...")
    ensure_norm_log(adata, in_place=True)
    logger.info("Normalization and log transformation complete")

    # Step 6: Remove infinite values
    logger.info("Step 6: Removing infinite values...")
    adata = remove_infinite_values(adata)
    logger.info(f"After removing infinite values: {adata.n_obs} cells × {adata.n_vars} genes")

    # Step 7: Generate summary
    if summary_path is not None:
        logger.info("Step 7: Generating summary...")
        summarize_adata(adata, out_path=str(summary_path))
        logger.info(f"Summary written to {summary_path}")

    logger.info("Preprocessing complete!")
    logger.info(f"Final shape: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def main():
    """Main entry point for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess single-cell RNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  preprocess.py --input data.h5ad --output processed.h5ad

  # With dataset config file
  preprocess.py --input data.h5ad --output processed.h5ad --config datasets/demo.yml

  # Custom parameters
  preprocess.py --input data.h5ad --output processed.h5ad \\
      --min_cells 5 --min_genes 100 --nmads_main 3
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input AnnData file (.h5ad or .zarr)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output AnnData file (.h5ad or .zarr)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to dataset config YAML file (as used in Nextflow main.nf)",
    )
    parser.add_argument(
        "--count_layer_key",
        type=str,
        default=None,
        help="Key in adata.layers containing raw counts (default: 'counts' or from config)",
    )
    parser.add_argument(
        "--ensembl_col",
        type=str,
        default=None,
        help="Column name containing Ensembl IDs (default: searches in 'ensembl_id', 'feature_id', 'gene_id', or index)",
    )
    parser.add_argument(
        "--gene_names_col",
        type=str,
        default=None,
        help="Column name containing gene names for mapping (default: searches in index, 'gene_names', or 'feature_names')",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="hsapiens",
        help="Species name for pybiomart query (default: 'hsapiens')",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to write preprocessing summary text file (optional)",
    )

    # QC parameters
    parser.add_argument(
        "--nmads_main",
        type=int,
        default=5,
        help="Number of MADs for main QC outlier detection (default: 5)",
    )
    parser.add_argument(
        "--nmads_mt",
        type=int,
        default=3,
        help="Number of MADs for mitochondrial outlier detection (default: 3)",
    )
    parser.add_argument(
        "--pct_counts_mt_threshold",
        type=float,
        default=8.0,
        help="Absolute threshold for %% mitochondrial counts (default: 8.0)",
    )
    parser.add_argument(
        "--percent_top",
        type=int,
        nargs="+",
        default=None,
        help="Top gene counts for QC metrics (default: [20])",
    )
    parser.add_argument(
        "--log1p_for_qc",
        action="store_true",
        default=True,
        help="Compute log1p values for QC metrics (default: True)",
    )
    parser.add_argument(
        "--no_log1p_for_qc",
        dest="log1p_for_qc",
        action="store_false",
        help="Do not compute log1p values for QC metrics",
    )

    # Filtering parameters
    parser.add_argument(
        "--min_cells",
        type=int,
        default=1,
        help="Minimum cells per gene for filtering (default: 10)",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=1,
        help="Minimum genes per cell for filtering (default: 200)",
    )

    args = parser.parse_args()

    # Load configuration from file if provided
    config = load_config(args.config)

    # Override with config file values if present
    # Config file can contain a 'preprocessing' section with these keys
    if "preprocessing" in config:
        prep_config = config["preprocessing"]
        count_layer_key = prep_config.get("count_layer_key", args.count_layer_key)
        ensembl_col = prep_config.get("ensembl_col", args.ensembl_col)
        gene_names_col = prep_config.get("gene_names_col", args.gene_names_col)
        species = prep_config.get("species", args.species)
        nmads_main = prep_config.get("nmads_main", args.nmads_main)
        nmads_mt = prep_config.get("nmads_mt", args.nmads_mt)
        pct_counts_mt_threshold = prep_config.get(
            "pct_counts_mt_threshold", args.pct_counts_mt_threshold
        )
        percent_top = prep_config.get("percent_top", args.percent_top)
        log1p_for_qc = prep_config.get("log1p_for_qc", args.log1p_for_qc)
        min_cells = prep_config.get("min_cells", args.min_cells)
        min_genes = prep_config.get("min_genes", args.min_genes)
    else:
        count_layer_key = args.count_layer_key
        ensembl_col = args.ensembl_col
        gene_names_col = args.gene_names_col
        species = args.species
        nmads_main = args.nmads_main
        nmads_mt = args.nmads_mt
        pct_counts_mt_threshold = args.pct_counts_mt_threshold
        percent_top = args.percent_top
        log1p_for_qc = args.log1p_for_qc
        min_cells = args.min_cells
        min_genes = args.min_genes

    # Handle percent_top default
    if percent_top is None:
        percent_top = [20]

    # Determine summary path
    summary_path = args.summary
    if summary_path is None:
        # Default to output directory with .txt extension
        summary_path = args.output.parent / f"{args.output.stem}_summary.txt"

    # Read input file
    logger.info(f"Reading input file: {args.input}")
    if args.input.suffix == ".zarr":
        adata = sc.read_zarr(args.input)
    else:
        adata = sc.read_h5ad(args.input)

    # Preprocess
    adata_processed = preprocess_adata(
        adata,
        count_layer_key=count_layer_key,
        ensembl_col=ensembl_col,
        gene_names_col=gene_names_col,
        species=species,
        nmads_main=nmads_main,
        nmads_mt=nmads_mt,
        pct_counts_mt_threshold=pct_counts_mt_threshold,
        percent_top=percent_top,
        log1p_for_qc=log1p_for_qc,
        min_cells=min_cells,
        min_genes=min_genes,
        summary_path=summary_path,
    )

    # Write output
    logger.info(f"Writing output to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".zarr":
        adata_processed.write_zarr(args.output)
    else:
        adata_processed.write_h5ad(args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()

