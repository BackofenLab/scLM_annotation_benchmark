#!/usr/bin/env python
"""Preprocess a single-cell RNA-seq dataset.

Puts the raw counts in X and layers["counts"], removes outlier cells by QC metrics, filters
cells and genes, log-normalizes X, removes genes with non-finite values and sets Ensembl IDs as
var_names (see preprocess_adata). Optionally maps the cell type labels to Cell Ontology terms.
With --labels_optional, data without a label column get one with a placeholder label.
Settings in the `preprocessing` section of the dataset YAML (--config) override the command
line. Writes the processed .h5ad or .zarr file, a text summary and QC plots in
<cache_dir>/plots.

    preprocess.py --input raw.h5ad --output pp.h5ad --config datasets/NAME.yml --cache_dir cache
"""

import argparse
import logging
from pathlib import Path
import yaml
import scanpy as sc
from anndata import AnnData
import pandas as pd

from util.cli import boolean
from util.pp_utils import (
    ensure_raw_counts_layer,
    ensure_gene_names_column,
    ensure_ensembl_ids,
    pp_quality_control,
    filter_genes_and_cells,
    ensure_norm_log,
    remove_infinite_values,
    summarize_adata,
    plot_preprocessing_qc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# The label of every cell in data without labels (annotation mode).
PLACEHOLDER_LABEL = "unlabeled"


def load_config(config_path: Path | None) -> dict:
    """Return the YAML config at config_path, or {} if there is none."""
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
    min_cells: int = 10,
    min_genes: int = 200,
    summary_path: str | Path | None = None,
    qc_plots_path: str | Path | None = None,
) -> AnnData:
    """Preprocess adata and return the result.

    The steps are: raw counts into X and layers["counts"] (adata.raw and the other layers are
    dropped), removal of QC outliers (pp_quality_control), cell and gene filtering, normalization
    and log1p, removal of genes with non-finite values and Ensembl IDs as var_names. A text
    summary goes to summary_path and the QC plots next to qc_plots_path, if given.
    """
    logger.info("Starting preprocessing pipeline.")
    logger.info(f"Input shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # Numbers before filtering, shown in the QC plots.
    initial_cells = adata.n_obs
    initial_genes = adata.n_vars

    qc_plots_dir = Path(qc_plots_path).parent if qc_plots_path else None
    # File name prefix of the QC plots, e.g. "HIHA_pp_qc_plots".
    qc_plots_prefix = Path(qc_plots_path).stem if qc_plots_path else None

    # Later steps read the gene symbols from adata.var["gene_names"].
    used_gene_names_source = ensure_gene_names_column(adata, gene_names_col)
    logger.info(
        f"Using '{used_gene_names_source}' as gene names source; ensured `adata.var['gene_names']`."
    )

    logger.info("Step 1: Finding and setting raw counts...")
    ensure_raw_counts_layer(adata, raw_layer_key=count_layer_key, raise_on_missing=True)
    logger.info("Raw counts set in adata.X and adata.layers['counts']")

    # adata.raw and all layers except "counts" are not needed and take memory.
    if adata.raw is not None:
        logger.info("Removing adata.raw to free memory (counts already extracted)")
        adata.raw = None

    layers_to_remove = [k for k in adata.layers.keys() if k != "counts"]
    if layers_to_remove:
        logger.info(
            f"Removing {len(layers_to_remove)} unnecessary layer(s): {', '.join(layers_to_remove)}"
        )
        for layer_key in layers_to_remove:
            del adata.layers[layer_key]

    logger.info("Step 2: Performing quality control...")
    adata = pp_quality_control(
        adata,
        nmads_main=nmads_main,
        nmads_mt=nmads_mt,
        pct_counts_mt_threshold=pct_counts_mt_threshold,
        percent_top=percent_top,
        plots_dir=qc_plots_dir,
        plots_prefix=qc_plots_prefix,
    )
    logger.info(f"After QC: {adata.n_obs} cells × {adata.n_vars} genes")

    logger.info("Step 3: Filtering genes and cells...")
    adata = filter_genes_and_cells(adata, min_cells=min_cells, min_genes=min_genes)
    logger.info(f"After filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    logger.info("Step 4: Ensuring normalization and log transformation...")
    ensure_norm_log(adata)
    logger.info("Normalization and log transformation complete")

    logger.info("Step 5: Removing infinite values...")
    adata = remove_infinite_values(adata)
    logger.info(f"After removing infinite values: {adata.n_obs} cells × {adata.n_vars} genes")

    logger.info("Step 6: Ensuring Ensembl IDs are present...")
    adata = ensure_ensembl_ids(
        adata,
        ensembl_col=ensembl_col,
        gene_names_col=gene_names_col,
        species=species,
        raise_on_failure=True,
    )
    logger.info(f"After Ensembl ID mapping: {adata.n_obs} cells × {adata.n_vars} genes")

    # The Ensembl ID mapping can change var and its index.
    ensure_gene_names_column(adata, "gene_names")

    if summary_path is not None:
        logger.info("Step 7: Generating summary...")
        summarize_adata(adata, out_path=str(summary_path))
        logger.info(f"Summary written to {summary_path}")

    logger.info("Generating QC plots after preprocessing...")
    after_png = qc_plots_dir / f"{qc_plots_prefix}_after.png" if qc_plots_dir else None
    plot_preprocessing_qc(
        adata,
        page_title="QC Summary After Preprocessing",
        initial_cells=initial_cells,
        initial_genes=initial_genes,
        save_path=after_png,
    )

    # The plots before and after preprocessing as a two-page PDF.
    if qc_plots_dir:
        before_png = qc_plots_dir / f"{qc_plots_prefix}_before.png"
        pdf_path = qc_plots_dir / f"{qc_plots_prefix}.pdf"
        from PIL import Image
        pages = []
        for png in [before_png, after_png]:
            if png and png.exists():
                pages.append(Image.open(png).convert("RGB"))
        if pages:
            pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=300)
            logger.info(f"Combined QC PDF saved to {pdf_path}")
        logger.info(f"QC plots saved to {qc_plots_dir}")

    logger.info("Preprocessing complete.")
    logger.info(f"Final shape: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess single-cell RNA-seq data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  preprocess.py --input data.h5ad --output processed.h5ad
  preprocess.py --input data.h5ad --output processed.h5ad --config datasets/toy.yml
  preprocess.py --input data.h5ad --output processed.h5ad --min_cells 5 --min_genes 100
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
        "--organism",
        type=str,
        default="Homo sapiens",
        help="Organism name to store in adata.obs['organism'] (default: 'Homo sapiens', can be overridden by config)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to write preprocessing summary text file (optional)",
    )

    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("cache"),
        help="Folder for the QC plots (written to <cache_dir>/plots/)",
    )

    parser.add_argument(
        "--mappings",
        type=Path,
        default=None,
        help="Mapping table of cell type labels to Cell Ontology terms (required with --apply_ontology_mapping true)",
    )
    parser.add_argument(
        "--annotation_key",
        type=str,
        default=None,
        help="Cell type annotation column (required if --apply_ontology_mapping is true)",
    )
    parser.add_argument(
        "--apply_ontology_mapping",
        type=boolean,
        default=False,
        help="Apply ontology mapping to adata.obs[annotation_key] (default: false)",
    )
    parser.add_argument(
        "--append_new_labels",
        type=boolean,
        default=True,
        help="Add labels that the mapping table does not hold as unreviewed rows (default: true)",
    )
    parser.add_argument(
        "--labels_optional",
        type=boolean,
        default=False,
        help=f"Without an annotation_key column, give every cell the label '{PLACEHOLDER_LABEL}' "
             "instead of stopping (default: false)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (used as origin tag in ontology mapping warnings)",
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

    # Filtering parameters
    parser.add_argument(
        "--min_cells",
        type=int,
        default=10,
        help="Minimum cells per gene for filtering (default: 10)",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Minimum genes per cell for filtering (default: 200)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Values in the config's 'preprocessing' section take precedence over the command line.
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
        min_cells = args.min_cells
        min_genes = args.min_genes

    # Organism of the dataset, stored in adata.obs["organism"] for the models that read it (e.g.
    # Cell2Sentence). A dataset config can set it, as Zeisel's does for mouse.
    organism = config.get("organism", args.organism)

    if percent_top is None:
        percent_top = [20]

    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.parent / f"{args.output.stem}_summary.txt"

    # Base name of the QC plot files.
    output_filename = args.output.stem
    if output_filename.endswith('.h5ad'):
        output_filename = output_filename[:-5]
    elif output_filename.endswith('.zarr'):
        output_filename = output_filename[:-5]
    
    qc_plots_path = args.cache_dir / "plots" / f"{output_filename}_qc_plots"
    qc_plots_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input file: {args.input}")
    if args.input.suffix == ".zarr":
        adata = sc.read_zarr(args.input)
    else:
        adata = sc.read_h5ad(args.input)

    if "organism" in adata.obs.columns:
        n_mismatch = int((adata.obs["organism"].astype(str) != str(organism)).sum())
        if n_mismatch > 0:
            logger.info(
                "Overriding adata.obs['organism'] to '%s' (was mixed; %d/%d mismatched).",
                organism,
                n_mismatch,
                adata.n_obs,
            )
    else:
        logger.info("Setting adata.obs['organism'] to '%s'.", organism)
    adata.obs["organism"] = str(organism)

    if args.annotation_key and args.annotation_key not in adata.obs.columns:
        if not args.labels_optional:
            raise SystemExit(f"The label column '{args.annotation_key}' (annotation_key) is not in "
                             f"obs, which has: {', '.join(map(str, adata.obs.columns))}")
        logger.info("No label column '%s': every cell gets the label '%s'", args.annotation_key,
                    PLACEHOLDER_LABEL)
        adata.obs[args.annotation_key] = PLACEHOLDER_LABEL
        args.apply_ontology_mapping = False

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
        min_cells=min_cells,
        min_genes=min_genes,
        summary_path=summary_path,
        qc_plots_path=qc_plots_path,
    )

    logger.info(f"Writing output to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Arrow-backed columns, categories and index to object dtype, since anndata
        cannot write Arrow arrays to HDF5."""
        df = df.copy()
        for col in df.columns:
            series = df[col]
            if hasattr(series.dtype, 'categories'):
                cat_arr = series.cat.categories
                cat_type = type(cat_arr.array).__name__ if hasattr(cat_arr, 'array') else ""
                if "Arrow" in cat_type or "arrow" in str(cat_arr.dtype).lower():
                    new_cats = cat_arr.astype(str).astype(object)
                    df[col] = pd.Categorical(series, categories=new_cats, ordered=series.cat.ordered)
            else:
                arr = series.array if hasattr(series, 'array') else series
                arr_type = type(arr).__name__
                dtype_str = str(series.dtype).lower()
                if "Arrow" in arr_type or "arrow" in dtype_str or "pyarrow" in dtype_str:
                    df[col] = series.astype(str).astype(object)
        idx = df.index
        idx_arr = idx.array if hasattr(idx, 'array') else idx
        idx_type = type(idx_arr).__name__
        if "Arrow" in idx_type or "arrow" in str(idx.dtype).lower():
            df.index = idx.astype(str).astype(object)
        return df
    
    adata_processed.obs = _sanitize_for_h5ad(adata_processed.obs)
    adata_processed.var = _sanitize_for_h5ad(adata_processed.var)

    if args.apply_ontology_mapping:
        if not args.annotation_key or not args.dataset or not args.mappings:
            raise SystemExit(
                "--apply_ontology_mapping true requires --annotation_key, --dataset and --mappings"
            )
        logger.info("Mapping the labels in '%s' to Cell Ontology terms", args.annotation_key)
        from util.ontology_mapping import map_adata_labels
        adata_processed = map_adata_labels(
            adata_processed, args.annotation_key, args.mappings, args.cache_dir,
            origin=args.dataset, append=args.append_new_labels)
        # The ontology mapping adds columns to obs.
        adata_processed.obs = _sanitize_for_h5ad(adata_processed.obs)
        adata_processed.var = _sanitize_for_h5ad(adata_processed.var)
    elif args.annotation_key:
        # Without mapping, the labels are restored from an original_<annotation_key> column if
        # the input has one.
        original_key = f"original_{args.annotation_key}"
        if original_key in adata_processed.obs.columns:
            logger.info("Restoring '%s' from '%s' backup column...", args.annotation_key, original_key)
            adata_processed.obs[args.annotation_key] = adata_processed.obs[original_key].copy()

    if args.output.suffix == ".zarr":
        adata_processed.write_zarr(args.output)
    else:
        adata_processed.write_h5ad(args.output, compression="gzip")

    logger.info("Done.")


if __name__ == "__main__":
    main()

