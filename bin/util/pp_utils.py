"""Preprocessing functions for single-cell RNA-seq data in AnnData objects: raw count detection,
quality control, filtering, normalization, Ensembl ID mapping, a text summary and QC plots."""

import gc
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData
from scipy.stats import median_abs_deviation
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)

# Offline tables of gene names and Ensembl IDs, used when BioMart cannot be reached.
_DEFAULT_ENSEMBL_CACHE_DIR = Path(__file__).resolve().parents[2] / "resources" / "ensembl"


def _load_ensembl_cache(cache_dir: Path = _DEFAULT_ENSEMBL_CACHE_DIR) -> pd.DataFrame | None:
    """Load the offline table of gene names and Ensembl IDs, or None if there is none.

    Reads the last file '<dataset>_ensembl_<YYMMDD>.csv' in cache_dir by name (columns gene_name
    and ensembl_id) and returns it with BioMart's column names "Gene name" and "Gene stable ID".
    The species is not checked.
    """
    if not cache_dir.exists():
        return None

    # With equal prefixes, the last file by name has the latest date.
    candidates = sorted(cache_dir.glob("*_ensembl_*.csv"))
    if not candidates:
        return None

    cache_path = candidates[-1]
    logger.info(f"Loading cached ENSEMBL mapping from {cache_path}")
    df = pd.read_csv(cache_path)
    if not {"gene_name", "ensembl_id"}.issubset(df.columns):
        logger.warning(
            f"Cache file {cache_path} is missing 'gene_name'/'ensembl_id' columns; ignoring."
        )
        return None
    return (
        df.dropna(subset=["gene_name", "ensembl_id"])
          .rename(columns={"gene_name": "Gene name", "ensembl_id": "Gene stable ID"})
    )


def ensure_gene_names_column(adata: AnnData, gene_names_col: str | None = None) -> str:
    """Write the gene symbols to adata.var['gene_names'] and return the name of their source.

    The source is gene_names_col if given ("var_names" for the index), else the first of the
    columns gene_names, feature_names and feature_name, else the index if it looks like gene
    symbols. Values are stripped and empty ones become NA. Raises ValueError if there is no
    source or it holds no names.
    """
    def _var_names_look_like_gene_names(var_names: pd.Index) -> bool:
        if var_names is None or len(var_names) == 0:
            return False
        s = pd.Series(var_names.astype(str))
        s = s.replace({"": pd.NA}).dropna()
        if s.empty:
            return False

        frac_ens = s.str.match(r"^ENS[A-Z]*\d+", na=False).mean()
        frac_symbol_like = s.str.match(r"^[A-Za-z][A-Za-z0-9_.-]*$", na=False).mean()
        frac_all_numeric = s.str.match(r"^\d+$", na=False).mean()

        # The thresholds are strict; if the index is rejected, --gene_names_col can name a
        # column with the symbols.
        if frac_ens >= 0.5:
            return False
        if frac_all_numeric >= 0.5:
            return False
        if frac_symbol_like < 0.5:
            return False
        return True

    # Sources in order of priority, listed in the error message.
    candidates: list[str] = []
    if gene_names_col:
        candidates.append(gene_names_col)
    candidates.extend(["gene_names", "feature_names", "feature_name", "var_names"])

    chosen: str | None = None
    s: pd.Series | None = None

    if gene_names_col == "var_names":
        if not _var_names_look_like_gene_names(adata.var_names):
            raise ValueError(
                "Requested gene names source 'var_names', but `adata.var_names` does not look like gene names "
                "(e.g., appears Ensembl-like or not symbol-like). Please provide `--gene_names_col` pointing to "
                "a column in `adata.var` that contains gene symbols."
            )
        chosen = "var_names"
        s = pd.Series(adata.var_names.astype(str), index=adata.var.index)
    else:
        if gene_names_col and gene_names_col in adata.var.columns:
            chosen = gene_names_col
            s = adata.var[chosen]
        else:
            for col in ["gene_names", "feature_names", "feature_name"]:
                if col in adata.var.columns:
                    chosen = col
                    s = adata.var[col]
                    break
            if chosen is None and _var_names_look_like_gene_names(adata.var_names):
                chosen = "var_names"
                s = pd.Series(adata.var_names.astype(str), index=adata.var.index)

    if chosen is None or s is None:
        raise ValueError(
            "Could not determine gene names source. "
            f"Tried (in order): {', '.join(candidates)}. "
            f"Available var columns: {', '.join(map(str, adata.var.columns))}. "
            "If gene names are stored in the var index, you can also pass `--gene_names_col var_names`."
        )

    non_null = s.dropna()
    if non_null.empty:
        raise ValueError(
            f"Gene names column '{chosen}' contains only missing values; cannot proceed."
        )

    non_empty = non_null.astype(str).str.strip()
    if (non_empty == "").all():
        raise ValueError(
            f"Gene names column '{chosen}' contains only empty/whitespace values; cannot proceed."
        )

    out = s.astype(object).copy()
    mask = s.notna()
    out.loc[mask] = s.loc[mask].astype(str).str.strip()
    out.loc[out == ""] = pd.NA
    if out.dropna().empty:
        raise ValueError(
            f"After cleaning, gene names column '{chosen}' contains only missing values; cannot proceed."
        )

    adata.var["gene_names"] = out
    return chosen


def is_raw_counts(
    X,
    min_val: float = -1e-6,
    max_noninteger_fraction: float = 0.01,
    check_sparsity: bool = True,
) -> bool:
    """Return True if X looks like raw counts: no value below min_val (slightly below 0 to allow
    for rounding errors) and at most max_noninteger_fraction non-integer entries.

    check_sparsity has no effect.
    """
    if sp.issparse(X):
        data = X.data  # stored values, no copy
        if len(data) == 0:
            return True  # all zeros

        # Implicit zeros are non-negative integers, so only the stored values are checked.
        if data.min() < min_val:
            return False

        # Fraction of all entries, as in the dense branch.
        n_total = X.shape[0] * X.shape[1]
        n_noninteger = np.sum(np.abs(data - np.round(data)) > 1e-6)
        fraction_noninteger = n_noninteger / n_total

        if fraction_noninteger > max_noninteger_fraction:
            return False

        return True
    else:
        if np.min(X) < min_val:
            return False

        diffs = np.abs(X - np.round(X))
        noninteger_mask = diffs > 1e-6
        fraction_noninteger = np.mean(noninteger_mask)

        if fraction_noninteger > max_noninteger_fraction:
            return False

        return True


def is_log_transformed(X, tol: float = 1e-3) -> bool:
    """Return True if X looks log-transformed: no value below -tol and a maximum below 100."""
    if sp.issparse(X):
        if X.nnz == 0:
            return False  # all zeros
        # Only stored values can be negative.
        if X.data.min() < -tol:
            return False
        max_val = X.data.max()
    else:
        if np.min(X) < -tol:
            return False
        max_val = np.max(X)
    return max_val < 100


def is_normalized(X, axis: int = 1, var_threshold: float = 0.2) -> bool:
    """Return True if X looks normalized: the sums along axis (per cell by default) have a
    coefficient of variation (std/mean) below var_threshold."""
    sums = np.asarray(X.sum(axis=axis)).ravel()
    mean_sum = sums.mean()
    std_sum = sums.std()

    if mean_sum == 0:
        logger.warning(
            "Row sums are zero. Data might be empty or already transformed in a different way."
        )
        return False

    ratio = std_sum / mean_sum
    logger.debug("Row sum ratio: std/mean=%.3f, threshold=%.3f", ratio, var_threshold)
    return ratio < var_threshold


def ensure_raw_counts_layer(
    adata: AnnData,
    raw_layer_key: str | None = None,
    raise_on_missing: bool = True,
) -> None:
    """Put the raw counts in adata.layers['counts'] and adata.X, in place, as int32 in the layer.

    The counts are taken from the first of: the layer raw_layer_key (default "counts"),
    adata.raw.X, adata.X and an existing 'counts' layer, where all but the first must pass
    is_raw_counts. If none does, an error is logged and, with raise_on_missing, a ValueError
    raised.
    """
    if raw_layer_key is None:
        raw_layer_key = "counts"

    # 1) The given layer, used without checking.
    if raw_layer_key in adata.layers:
        logger.info("Using layer '%s' for raw counts", raw_layer_key)
        if raw_layer_key != "counts":
            adata.layers["counts"] = adata.layers[raw_layer_key]
        adata.X = adata.layers["counts"].copy()
    # 2) adata.raw
    elif adata.raw is not None and is_raw_counts(adata.raw.X):
        logger.info("Detected raw counts in adata.raw.X; saving to layer 'counts'")
        # adata.raw can have more genes than adata.
        if adata.raw.n_vars > adata.n_vars:
            common_vars = adata.var_names.intersection(adata.raw.var_names)
            if len(common_vars) == 0:
                logger.warning(
                    "No common variables found between adata.var_names and adata.raw.var_names. "
                    "Cannot use adata.raw.X for raw counts."
                )
            elif len(common_vars) < adata.n_vars:
                logger.warning(
                    f"Only {len(common_vars)} out of {adata.n_vars} variables found in adata.raw. "
                    "Cannot use adata.raw.X for raw counts due to incomplete overlap."
                )
            else:
                # All genes of adata are in adata.raw: take their columns.
                raw_var_to_idx = {
                    var: idx for idx, var in enumerate(adata.raw.var_names)
                }
                var_indices = [raw_var_to_idx[var] for var in adata.var_names]
                adata.layers["counts"] = adata.raw.X[:, var_indices].copy()
                adata.X = adata.layers["counts"].copy()
        else:
            # Assumes the same genes in the same order as adata.
            adata.layers["counts"] = adata.raw.X.copy()
            adata.X = adata.layers["counts"].copy()
    # 3) adata.X
    elif is_raw_counts(adata.X):
        logger.info("Detected raw counts in adata.X; saving to layer 'counts'")
        adata.layers["counts"] = adata.X.copy()
    # 4) An existing 'counts' layer
    elif "counts" in adata.layers and is_raw_counts(adata.layers["counts"]):
        logger.info("Using existing 'counts' layer for raw counts")
        adata.X = adata.layers["counts"].copy()
    else:
        # Layers to suggest in the log.
        available_layers = list(adata.layers.keys()) if hasattr(adata, "layers") and adata.layers else []
        
        msg = (
            "Could not find raw counts: "
            f"no layer '{raw_layer_key}', adata.X and adata.raw.X not raw, and no valid 'counts' layer."
        )
        logger.error(msg)
        
        if available_layers:
            logger.info(f"Available layers in adata.layers: {', '.join(available_layers)}")
            logger.info(
                "If any of these layers contain raw counts, please set 'count_layer_key' in the "
                "preprocessing section of your dataset YAML config file. For example:\n"
                "  preprocessing:\n"
                f"    count_layer_key: \"{available_layers[0]}\""
            )
        else:
            logger.info("No layers found in adata.layers.")
        
        if raise_on_missing:
            raise ValueError(msg)
        return

    if "counts" in adata.layers and adata.layers["counts"].dtype != np.int32:
        logger.info(
            "Converting adata.layers['counts'] from %s to int32",
            adata.layers["counts"].dtype,
        )
        adata.layers["counts"] = adata.layers["counts"].astype(np.int32)

    # The int32 cast above truncates non-integer values, so this check only finds negative ones.
    if "counts" in adata.layers and not is_raw_counts(adata.layers["counts"]):
        logger.error(
            "Layer 'counts' does not appear to contain integer raw counts; "
            "downstream steps may fail."
        )


def ensure_norm_log(
    adata: AnnData, in_place: bool = True, var_threshold: float = 0.2
) -> None:
    """Normalize adata.X to 10,000 counts per cell and apply log1p, in place, and cast it to
    float32 if it is not a float matrix.

    Both checks run on X before normalization; a step is skipped if is_normalized
    (with var_threshold) or is_log_transformed finds it done. in_place has no effect.
    """
    logger.info("Checking if data in adata.X appears normalized and log-transformed.")
    already_norm = is_normalized(adata.X, var_threshold=var_threshold)
    already_log = is_log_transformed(adata.X)

    if not already_norm:
        logger.info(
            "Data does not appear to be normalized. Applying sc.pp.normalize_total() in place."
        )
        sc.pp.normalize_total(adata, target_sum=1e4)
    else:
        logger.info("Data already appears to be normalized.")

    if not already_log:
        logger.info(
            "Data does not appear to be log-transformed. Applying sc.pp.log1p() in place."
        )
        sc.pp.log1p(adata)
    else:
        logger.info("Data already appears to be log-transformed.")

    if not np.issubdtype(adata.X.dtype, np.floating):
        logger.info("Casting adata.X to float32 to ensure compatibility.")
        adata.X = adata.X.astype(np.float32)


def pp_quality_control(
    adata: AnnData,
    nmads_main: int = 5,
    nmads_mt: int = 3,
    pct_counts_mt_threshold: float = 8.0,
    percent_top: list[int] | None = None,
    plots_dir: Path | None = None,
    plots_prefix: str | None = None,
) -> AnnData:
    """Remove outlier cells by QC metrics and return the filtered copy.

    A cell is removed if its total counts, its number of genes or its percentage of counts in
    the top percent_top[0] genes lies more than nmads_main MADs from the median, or if its
    percentage of mitochondrial counts lies more than nmads_mt MADs from the median or above
    pct_counts_mt_threshold. The QC plot before filtering is saved in plots_dir if plots_dir and
    plots_prefix are given.
    """
    logger.info("Starting quality control checks.")

    # Gene sets are found by symbol in var_names; with Ensembl IDs as var_names they are empty
    # and no cell is removed for its mitochondrial counts.
    logger.info("Labeling mitochondrial, ribosomal, and hemoglobin genes in adata.var")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.upper().str.contains(r"^HB[^P]")

    if percent_top is None:
        percent_top = [20]
    # percent_top values must not exceed n_vars.
    for p_top in percent_top:
        if p_top > adata.n_vars:
            logger.warning(
                f"max(percent_top)={max(percent_top)} cannot be greater than n_vars={adata.n_vars}. "
                "Setting to n_vars - 1."
            )
            percent_top = [adata.n_vars - 1]
            break

    logger.info(
        "Calculating QC metrics with percent_top=%s for gene sets [mt, ribo, hb].",
        percent_top,
    )
    # log1p=False: no log1p_* metrics are computed here. If the input obs already has
    # log1p_total_counts, the outlier test below uses those columns.
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=percent_top,
        log1p=False,
    )

    logger.info("Generating QC plots before preprocessing...")
    save_path = plots_dir / f"{plots_prefix}_before.png" if plots_dir and plots_prefix else None
    plot_preprocessing_qc(
        adata,
        page_title="QC Summary Before Preprocessing",
        save_path=save_path,
    )

    def is_outlier(metric_values: np.ndarray, nmads: int) -> pd.Series:
        """Mask of the values more than nmads MADs (unscaled) from the median."""
        M = metric_values
        med = np.median(M)
        mad = median_abs_deviation(M)
        lower_bound = med - nmads * mad
        upper_bound = med + nmads * mad
        outlier_mask = (M < lower_bound) | (M > upper_bound)
        return outlier_mask

    logger.info(
        "Flagging outliers for total counts, number of genes, and %% in top genes."
    )
    if "log1p_total_counts" in adata.obs.columns:
        total_counts_metric = adata.obs["log1p_total_counts"].values
        n_genes_metric = adata.obs["log1p_n_genes_by_counts"].values
    else:
        total_counts_metric = adata.obs["total_counts"].values
        n_genes_metric = adata.obs["n_genes_by_counts"].values
    
    outlier_main = is_outlier(total_counts_metric, nmads_main) | is_outlier(n_genes_metric, nmads_main)
    # e.g. 'pct_counts_in_top_20_genes' if percent_top=[20].
    if f"pct_counts_in_top_{percent_top[0]}_genes" in adata.obs.columns:
        outlier_main |= is_outlier(
            adata.obs[f"pct_counts_in_top_{percent_top[0]}_genes"].values, nmads_main
        )

    logger.info("Flagging outliers for mitochondrial fraction.")
    mt_metric = adata.obs["pct_counts_mt"].values
    outlier_mt = is_outlier(mt_metric, nmads_mt) | (mt_metric > pct_counts_mt_threshold)

    adata.obs["outlier"] = outlier_main
    adata.obs["mt_outlier"] = outlier_mt

    n_before = adata.n_obs
    keep_mask = ~(adata.obs["outlier"] | adata.obs["mt_outlier"])
    adata = adata[keep_mask].copy()
    n_after = adata.n_obs
    del keep_mask

    logger.info(
        "Filtered out %d cells as outliers (main or mt). Remaining cells: %d.",
        n_before - n_after,
        n_after,
    )

    logger.info("QC filtering complete.")
    
    gc.collect()
    
    return adata


def filter_genes_and_cells(
    adata: AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
) -> AnnData:
    """Make the cell names unique and remove cells with fewer than min_genes genes, then genes
    found in fewer than min_cells cells. Raises ValueError if no cell is left."""
    logger.info("Filtering genes and cells.")
    logger.info("Making cell names unique.")
    adata.obs_names_make_unique()

    logger.info(f"Filtering cells expressing fewer than {min_genes} genes...")
    sc.pp.filter_cells(adata, min_genes=min_genes)

    logger.info(f"Filtering genes expressed in fewer than {min_cells} cells...")
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if adata.n_obs == 0:
        logger.error("No cells left after filtering. Exiting.")
        raise ValueError("No cells left after filtering. Exiting.")

    logger.info(f"After filtering: {adata.n_obs} cells, {adata.n_vars} genes.")
    
    gc.collect()
    
    return adata


def remove_infinite_values(adata: AnnData) -> AnnData:
    """Remove the genes with non-finite values (inf or NaN) in adata.X."""
    logger.info("Checking for infinite values in adata.X.")
    
    if sp.issparse(adata.X):
        if not np.isfinite(adata.X.data).all():
            finite_mask = np.ones(adata.n_vars, dtype=bool)
            # Assumes no explicitly stored zeros: nonzero() skips them, X.data does not.
            row_indices, col_indices = adata.X.nonzero()
            non_finite_mask = ~np.isfinite(adata.X.data)
            bad_cols = np.unique(col_indices[non_finite_mask])
            finite_mask[bad_cols] = False
        else:
            finite_mask = np.ones(adata.n_vars, dtype=bool)
    else:
        finite_mask = np.isfinite(adata.X).all(axis=0)
    
    n_bad = np.count_nonzero(~finite_mask)
    if n_bad > 0:
        logger.warning(
            "Dropping %d genes that contain infinite values.", n_bad
        )
        adata = adata[:, finite_mask].copy()
    else:
        logger.info("No infinite values found.")

    return adata


def ensure_ensembl_ids(
    adata: AnnData,
    ensembl_col: str | None = None,
    gene_names_col: str | None = None,
    species: str = "hsapiens",
    raise_on_failure: bool = True,
    ensembl_id_threshold: float = 0.5,
) -> AnnData:
    """Set Ensembl gene IDs as var_names and in var['ensembl_id'] and return the AnnData.

    The IDs come from the index or from a column (ensembl_col, else ensembl_id, feature_id or
    gene_id) in which at least ensembl_id_threshold of the values start with "ENS". Otherwise
    gene symbols are mapped with Ensembl BioMart for species ("hsapiens", "mmusculus"), or with
    the offline table in resources/ensembl if no BioMart host answers. Version suffixes are
    removed, and genes without an ID and all but the first gene of each ID are dropped. On
    failure, a ValueError is raised if raise_on_failure, else adata is returned without the
    mapping.
    """
    logger.info("Ensuring Ensembl IDs are present in the dataset.")

    # Step 1: Ensembl IDs in the index
    index_values = adata.var_names.astype(str)
    index_ensembl_count = pd.Series(index_values).str.contains(r"^ENS", na=False).sum()
    index_ensembl_fraction = index_ensembl_count / len(index_values) if len(index_values) > 0 else 0.0
    logger.info(
        f"Found {index_ensembl_count} Ensembl IDs in index (out of {len(index_values)} genes, "
        f"{index_ensembl_fraction:.1%})"
    )

    # Step 2: Ensembl IDs in the columns
    ensembl_cols_to_check = []
    if ensembl_col:
        ensembl_cols_to_check = [ensembl_col]
    else:
        ensembl_cols_to_check = ["ensembl_id", "feature_id", "gene_id"]

    found_ensembl_col = None
    ensembl_col_count = 0
    ensembl_col_fraction = 0.0
    for col in ensembl_cols_to_check:
        if col in adata.var.columns:
            col_values = adata.var[col].astype(str)
            col_ensembl_count = pd.Series(col_values).str.contains(r"^ENS", na=False).sum()
            col_ensembl_fraction = col_ensembl_count / len(col_values) if len(col_values) > 0 else 0.0
            logger.info(
                f"Found {col_ensembl_count} Ensembl IDs in column '{col}' "
                f"({col_ensembl_fraction:.1%})"
            )
            if col_ensembl_count > ensembl_col_count:
                ensembl_col_count = col_ensembl_count
                ensembl_col_fraction = col_ensembl_fraction
                found_ensembl_col = col

    # An index below the threshold holds gene names, which are saved before it is replaced.
    index_contains_gene_names = index_ensembl_fraction < ensembl_id_threshold
    
    # Step 3: Source of the Ensembl IDs, if one reaches the threshold
    if (index_ensembl_fraction >= ensembl_id_threshold and 
        index_ensembl_fraction >= ensembl_col_fraction):
        logger.info(
            f"Using index as source of Ensembl IDs "
            f"({index_ensembl_fraction:.1%} >= {ensembl_id_threshold:.1%} threshold)."
        )
        adata.var["ensembl_id"] = index_values
        ensembl_ids = index_values.to_series()
    elif (found_ensembl_col and 
          ensembl_col_fraction >= ensembl_id_threshold):
        logger.info(
            f"Using column '{found_ensembl_col}' as source of Ensembl IDs "
            f"({ensembl_col_fraction:.1%} >= {ensembl_id_threshold:.1%} threshold)."
        )
        adata.var["ensembl_id"] = adata.var[found_ensembl_col].astype(str)
        ensembl_ids = pd.Series(adata.var["ensembl_id"].values, index=adata.var.index)
        if index_contains_gene_names:
            logger.info("Saving gene names from index to adata.var['gene_names'] before overwriting with Ensembl IDs.")
            adata.var["gene_names"] = index_values.values
    else:
        # Step 4: Map gene names to Ensembl IDs
        if index_ensembl_count > 0 or ensembl_col_count > 0:
            logger.warning(
                f"Found some Ensembl IDs (index: {index_ensembl_fraction:.1%}, "
                f"columns: {ensembl_col_fraction:.1%}) but below threshold "
                f"({ensembl_id_threshold:.1%}). Proceeding to map gene names."
            )
        logger.info("No Ensembl IDs found above threshold. Attempting to map gene names to Ensembl IDs using pybiomart.")

        gene_names_cols_to_check = []
        if gene_names_col:
            gene_names_cols_to_check = [gene_names_col]
        else:
            gene_names_cols_to_check = ["gene_names", "feature_names"]

        found_gene_names_col = None
        gene_names = None

        # Always true in this branch, so the gene names come from the index and gene_names_col
        # is not used.
        if index_ensembl_fraction < ensembl_id_threshold:
            logger.info("Using index as source of gene names.")
            gene_names = index_values.copy()
            found_gene_names_col = "index"
            logger.info("Saving gene names from index to adata.var['gene_names'] before overwriting with Ensembl IDs.")
            adata.var["gene_names"] = index_values.values
        else:
            for col in gene_names_cols_to_check:
                if col in adata.var.columns:
                    logger.info(f"Found column '{col}' with gene names.")
                    gene_names = adata.var[col].astype(str)
                    found_gene_names_col = col
                    break

        if gene_names is None:
            error_msg = (
                "Could not find gene names for mapping to Ensembl IDs. "
                "Please provide gene names in one of the following:\n"
                "- Index (if not Ensembl IDs)\n"
                "- Column 'gene_names' or 'feature_names'\n"
                "- Or specify 'gene_names_col' in dataset config YAML"
            )
            logger.error(error_msg)
            if raise_on_failure:
                raise ValueError(error_msg)
            return adata

        logger.info(f"Using '{found_gene_names_col}' as source of gene names for mapping.")
        if isinstance(gene_names, pd.Index):
            gene_names = pd.Series(gene_names)
        logger.info(f"Example gene name: {gene_names.iloc[0] if len(gene_names) > 0 else 'N/A'}")

        # BioMart mirrors, tried in this order.
        import pybiomart
        ensembl_hosts = [
            "http://www.ensembl.org",
            "http://useast.ensembl.org",
            "http://uswest.ensembl.org",
            "http://asia.ensembl.org",
        ]
        logger.info(f"Fetching Ensembl IDs from pybiomart for species '{species}'. This may take a moment...")

        biomart_df = None
        last_error = None
        for host in ensembl_hosts:
            try:
                logger.info(f"Trying BioMart host: {host}")
                dataset = pybiomart.Dataset(
                    name=f"{species}_gene_ensembl", host=host
                )
                result = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
                # BioMart sometimes returns an error page without raising an error; the parsed
                # result then lacks these columns, and the next host is tried.
                if not {"Gene name", "Gene stable ID"}.issubset(result.columns):
                    raise ValueError(
                        f"unexpected columns {list(result.columns)} "
                        "(expected 'Gene name', 'Gene stable ID')"
                    )
                biomart_df = result
                logger.info(f"Successfully connected to {host}")
                break
            except Exception as e:
                biomart_df = None
                last_error = e
                logger.warning(f"BioMart host {host} failed: {e}")
                continue

        if biomart_df is None:
            logger.warning(
                f"All Ensembl BioMart hosts failed (last error: {last_error}). "
                f"Trying offline fallback in {_DEFAULT_ENSEMBL_CACHE_DIR}..."
            )
            biomart_df = _load_ensembl_cache()
            if biomart_df is None:
                error_msg = (
                    f"All Ensembl BioMart hosts failed and no offline ENSEMBL cache "
                    f"was found under {_DEFAULT_ENSEMBL_CACHE_DIR}. Last error: {last_error}\n"
                    "Please check:\n"
                    "- Network connection to Ensembl\n"
                    "- Species name is correct (e.g., 'hsapiens' or 'mmusculus')\n"
                    "- Set 'gene_names_col' in dataset config YAML to specify the column with gene names\n"
                    "- Or set 'ensembl_col' in dataset config YAML if Ensembl IDs already exist"
                )
                logger.error(error_msg)
                if raise_on_failure:
                    raise ValueError(error_msg)
                return adata
            logger.info("Using cached ENSEMBL mapping as BioMart fallback.")

        try:
            biomart_df = biomart_df.drop_duplicates(subset="Gene name").set_index("Gene name")

            ensembl_ids = []
            mapped_count = 0
            for symbol in gene_names:
                if symbol in biomart_df.index:
                    ensembl_ids.append(biomart_df.loc[symbol, "Gene stable ID"])
                    mapped_count += 1
                else:
                    ensembl_ids.append("")

            logger.info(f"Mapped {mapped_count} out of {len(gene_names)} genes to Ensembl IDs.")

            valid_count = sum(1 for eid in ensembl_ids if eid and eid.startswith("ENS"))
            if valid_count == 0:
                error_msg = (
                    "Failed to map any gene names to Ensembl IDs. "
                    "Possible reasons:\n"
                    "- Gene names don't match Ensembl gene symbols\n"
                    f"- Wrong species specified (current: '{species}')\n"
                    "- Network issues connecting to Ensembl\n"
                    "Please check:\n"
                    "- Set 'species' in dataset config YAML (e.g., 'hsapiens' or 'mmusculus')\n"
                    "- Set 'gene_names_col' in dataset config YAML to specify the column with gene names\n"
                    "- Or set 'ensembl_col' in dataset config YAML if Ensembl IDs already exist"
                )
                logger.error(error_msg)
                if raise_on_failure:
                    raise ValueError(error_msg)
                return adata

            adata.var["ensembl_id"] = pd.Series(ensembl_ids, index=adata.var_names)

        except Exception as e:
            error_msg = (
                f"Error processing BioMart results: {e}\n"
                "Please check:\n"
                "- Species name is correct (e.g., 'hsapiens' or 'mmusculus')\n"
                "- Set 'gene_names_col' in dataset config YAML to specify the column with gene names\n"
                "- Or set 'ensembl_col' in dataset config YAML if Ensembl IDs already exist"
            )
            logger.error(error_msg)
            if raise_on_failure:
                raise ValueError(error_msg)
            return adata

        ensembl_ids = pd.Series(ensembl_ids, index=adata.var_names)

    # Step 5: Clean up the Ensembl IDs and set them as index
    logger.info("Cleaning up Ensembl IDs and setting as index...")

    if pd.Series(ensembl_ids).str.contains(r"\.").any():
        logger.info("Removing version numbers from Ensembl IDs (everything after '.').")
        ensembl_ids = pd.Series(ensembl_ids).str.split(".").str[0]

    valid_mask = pd.Series(ensembl_ids).str.startswith("ENS", na=False)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(
            f"Removing {invalid_count} genes with invalid Ensembl IDs "
            "(not starting with 'ENS' or empty)."
        )
        adata = adata[:, valid_mask].copy()
        ensembl_ids = ensembl_ids[valid_mask]

    adata.var["ensembl_id"] = ensembl_ids.values

    is_dup = pd.Series(ensembl_ids).duplicated(keep="first")
    dup_count = is_dup.sum()
    if dup_count > 0:
        logger.warning(
            f"Found {dup_count} genes with duplicate Ensembl IDs; "
            "keeping first occurrence and removing duplicates."
        )
        adata = adata[:, ~is_dup].copy()
        ensembl_ids = ensembl_ids[~is_dup]

    adata.var.index = ensembl_ids.values
    logger.info(f"Set Ensembl IDs as index. Final shape: {adata.n_obs} cells × {adata.n_vars} genes")

    gc.collect()

    return adata

def _infer_X_type(adata):
    """Guess what adata.X holds (raw counts, log-normalized, scaled) for the summary."""
    X = adata.X
    if X.size == 0:
        return "unknown (empty)", {"n": 0, "min": None, "max": None, "int_like": None, "hints": []}
    
    vmin, vmax = float(X.min()), float(X.max())
    
    if sp.issparse(X):
        # Stored values only; total_n counts all entries.
        vals_for_check = X.data
        total_n = X.shape[0] * X.shape[1]
    else:
        # At most 100,000 values, sampled with numpy's global random state.
        max_sample_size = 100000
        if X.size > max_sample_size:
            flat_indices = np.random.choice(X.size, size=max_sample_size, replace=False)
            X_flat = np.asarray(X).ravel()
            vals_for_check = X_flat[flat_indices]
            del X_flat
        else:
            vals_for_check = np.asarray(X).ravel()
        total_n = X.size

    if vals_for_check.size == 0:
        int_like = False
    else:
        finite_mask = np.isfinite(vals_for_check)
        if not np.all(finite_mask):
            int_like = False
        else:
            rounded = np.round(vals_for_check)
            int_like = np.allclose(vals_for_check, rounded, atol=1e-6, rtol=1e-6)
    nonneg = vmin >= 0

    if not nonneg:
        meaning = "likely centered/scaled (negative values present)"
    elif int_like:
        meaning = "likely raw counts (non-negative integers)"
    elif vmax < 10:
        meaning = "likely log-normalized (e.g., log1p)"
    elif vmax > 20:
        meaning = "likely normalized (linear scale, not log)"
    else:
        meaning = "unclear (between heuristics)"

    hints = []
    if "log1p" in getattr(adata, "uns", {}): hints.append("uns['log1p'] present")
    lyr = set(getattr(adata, "layers", {}).keys())
    if {"counts","raw"} & lyr: hints.append(f"layers: {sorted({'counts','raw'}&lyr)}")
    if {"log1p","lognorm","normalized"} & lyr: hints.append(f"layers: {sorted({'log1p','lognorm','normalized'}&lyr)}")

    return meaning, {"n": int(total_n), "min": vmin, "max": vmax, "int_like": bool(int_like), "hints": hints}

def summarize_adata(adata, out_path="summary.txt"):
    """Write a text summary of adata (X, raw, layers, obs, var, uns) to out_path."""
    L=[]; A=L.append
    A("# AnnData summary")
    A(f"shape: {adata.n_obs:,} × {adata.n_vars:,}")
    A(f"obs_names: {'unique' if adata.obs_names.is_unique else 'non-unique'}; "
      f"var_names: {'unique' if adata.var_names.is_unique else 'non-unique'}")

    meaning, stats = _infer_X_type(adata)
    A("\n## X (inference)")
    A(f"semantic guess: {meaning}")
    A(f"n_values={stats['n']}, min={stats['min']}, max={stats['max']}, int_like={stats['int_like']}")
    if stats["hints"]: A("hints: " + " | ".join(stats["hints"]))
    A(f"X sparse: {sp.issparse(adata.X)}")

    A("\n## .raw")
    if adata.raw is not None:
        A(f"raw.shape: {adata.raw.n_obs:,} × {adata.raw.n_vars:,}")
        A("raw.var columns: " + ", ".join(list(adata.raw.var.columns)[:20]))
    else:
        A("None")

    A("\n## layers")
    try: A(", ".join(list(adata.layers.keys())) or "(none)")
    except Exception: A("(unavailable)")
    
    if "counts" in adata.layers:
        counts_layer = adata.layers["counts"]
        counts_min = float(counts_layer.min())
        counts_max = float(counts_layer.max())
        counts_dtype = str(counts_layer.dtype)
        A(f"\n### counts layer")
        A(f"dtype: {counts_dtype}")
        A(f"min: {counts_min}, max: {counts_max}")
        A(f"sparse: {sp.issparse(counts_layer)}")

    A("\n## HVGs")
    A(f"n_hvg: {int(adata.var['highly_variable'].sum()):,}" if "highly_variable" in adata.var.columns else "none")

    A("\n## preprocessing")
    if "neighbors" in adata.uns: A(f"neighbors: {adata.uns['neighbors'].get('params',{})}")
    if "pca" in adata.uns:
        vr = adata.uns["pca"].get("variance_ratio"); A(f"PCA: n_comps={None if vr is None else len(vr)}")
    emb = [k for k in getattr(adata,"obsm",{}).keys() if k.startswith("X_")]
    A("embeddings: " + (", ".join(emb) if emb else "(none)"))
    obsp = list(getattr(adata,"obsp",{}).keys())
    A("obsp: " + (", ".join(obsp) if obsp else "(none)"))

    A("\n## obs")
    A(f"n_columns: {adata.obs.shape[1]}")
    A("columns: " + ", ".join(list(adata.obs.columns)[:40]))

    A("\n## var")
    A(f"n_columns: {adata.var.shape[1]}")
    A("columns: " + ", ".join(list(adata.var.columns)[:40]))

    A("\n## uns keys")
    try:
        k = list(adata.uns.keys()); A(", ".join(k[:60]) if k else "(none)")
    except Exception: A("(unavailable)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))
    print(f"Wrote summary to {out_path.resolve()}")

def plot_preprocessing_qc(
    adata,
    page_title="QC Summary Preprocessing",
    initial_cells: int | None = None,
    initial_genes: int | None = None,
    save_path: Path | str | None = None,
):
    """Plot the QC metrics of adata on one page and save it as a PNG (300 dpi) if save_path is
    given. The summary panel shows the share of cells and genes removed since initial_cells and
    initial_genes."""

    sns.set_context("paper", font_scale=1)
    sns.set_style("white")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
        }
    )

    fig = plt.figure(figsize=(18, 12))
    gs_main = fig.add_gridspec(2, 1, hspace=0.4)
    
    # Top row: two single violins and a wider panel with the three percentage violins.
    gs_top = gs_main[0].subgridspec(1, 3, wspace=0.3, width_ratios=[0.22, 0.22, 0.56])
    
    gs_bottom = gs_main[1].subgridspec(1, 3, wspace=0.3, width_ratios=[0.35, 0.35, 0.30])
    
    fig.suptitle(page_title, fontsize=16, fontweight="bold")

    # matplotlib's ax.violinplot instead of sc.pl.violin: once scanpy is imported,
    # seaborn.violinplot, which sc.pl.violin calls, segfaults on macOS 26 in recent conda
    # environments.

    # Top row
    # Plot 1: violin plot of total_counts
    ax = fig.add_subplot(gs_top[0, 0])
    if "total_counts" in adata.obs.columns:
        ax.violinplot([adata.obs["total_counts"].to_numpy()], showmedians=True)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.set_title("total_counts")
    else:
        ax.text(0.5, 0.5, "total_counts\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("total_counts")

    # Plot 2: violin plot of n_genes_by_counts
    ax = fig.add_subplot(gs_top[0, 1])
    if "n_genes_by_counts" in adata.obs.columns:
        ax.violinplot([adata.obs["n_genes_by_counts"].to_numpy()], showmedians=True)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.set_title("n_genes_by_counts")
    else:
        ax.text(0.5, 0.5, "n_genes_by_counts\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("n_genes_by_counts")

    # Plot 3: violin plot of pct_counts_mt, pct_counts_ribo, pct_counts_hb
    ax = fig.add_subplot(gs_top[0, 2])
    _pct_keys = ("pct_counts_mt", "pct_counts_ribo", "pct_counts_hb")
    if all(k in adata.obs.columns for k in _pct_keys):
        _pct_data = [adata.obs[k].to_numpy() for k in _pct_keys]
        ax.violinplot(_pct_data, showmedians=True)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["mt", "ribo", "hb"])
        ax.set_title("Mito %, Ribo %, HB %")
    else:
        ax.text(0.5, 0.5, "pct_counts_mt or pct_counts_ribo or pct_counts_hb\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Mito %, Ribo %, HB %")

    # Bottom row
    # Plot 4: scatter plot of total_counts and n_genes_by_counts
    ax = fig.add_subplot(gs_bottom[0, 0])
    if "total_counts" in adata.obs.columns and "n_genes_by_counts" in adata.obs.columns:
        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", color="pct_counts_mt", ax=ax, show=False, color_map="magma_r")
        if ax.collections and getattr(ax.collections[0], "colorbar", None) is not None:
            ax.collections[0].colorbar.remove()
        # Add a compact color legend
        if ax.collections:
            sm = ax.collections[0]
            cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("pct_counts_mt", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Scatter plot\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Counts vs Genes Detected")

    # Plot 5: scatter plot of total_counts and pct_counts_in_top_20_genes
    ax = fig.add_subplot(gs_bottom[0, 1])
    if "total_counts" in adata.obs.columns and "pct_counts_in_top_20_genes" in adata.obs.columns:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_in_top_20_genes", ax=ax, show=False)
    else:
        ax.text(0.5, 0.5, "Scatter plot\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("total_counts vs pct_counts_in_top_20_genes")
    
    # Plot 6: summary statistics
    ax = fig.add_subplot(gs_bottom[0, 2])
    ax.axis("off")
    
    # Build summary text with filtering info if available
    summary_lines = [
        "Dataset Summary:",
        "---------------------",
        f"Cells: {adata.n_obs:,}",
        f"Genes: {adata.n_vars:,}",
    ]
    
    # Calculate and display filtering info if initial counts differ from current
    summary_lines.append("")
    if initial_cells is not None and initial_cells != adata.n_obs:
        cells_filtered_pct = ((initial_cells - adata.n_obs) / initial_cells * 100)
        summary_lines.append(f"Cells filtered: {cells_filtered_pct:.1f}%")
    
    if initial_genes is not None and initial_genes != adata.n_vars:
        genes_filtered_pct = ((initial_genes - adata.n_vars) / initial_genes * 100)
        summary_lines.append(f"Genes filtered: {genes_filtered_pct:.1f}%")
    
    summary_lines.extend([
        "",
        "Cell QC:",
        f" - Total counts (median): {adata.obs.get('total_counts', pd.Series([0])).median():.0f}",
        f" - Genes/cell (median): {adata.obs.get('n_genes_by_counts', pd.Series([0])).median():.0f}",
        f" - MT % (median): {adata.obs.get('pct_counts_mt', pd.Series([0])).median():.2f}%",
        f" - Ribo % (median): {adata.obs.get('pct_counts_ribo', pd.Series([0])).median():.2f}%",
        f" - HB % (median): {adata.obs.get('pct_counts_hb', pd.Series([0])).median():.2f}%",
    ])
    
    summary_text = "\n".join(summary_lines)
    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.3, edgecolor="black", linewidth=0.5),
    )

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
