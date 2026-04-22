"""
Preprocessing utility functions for single-cell RNA-seq data.

This module provides core utility functions for preprocessing AnnData objects,
including raw count detection, quality control, filtering, and normalization.
"""

import gc
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData
from scipy.stats import median_abs_deviation
import pybiomart
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


def is_raw_counts(
    X,
    min_val: float = -1e-6,
    max_noninteger_fraction: float = 0.01,
    check_sparsity: bool = True,
) -> bool:
    """
    Check if a matrix `X` likely contains raw (integer) single-cell counts.

    Parameters
    ----------
    X : np.ndarray or sparse matrix
        The expression matrix to check. Rows are cells, columns are genes.
    min_val : float, optional
        Minimum allowed value (just under 0 to account for floating point noise).
        If we detect values significantly below 0, we assume it's not raw counts.
    max_noninteger_fraction : float, optional
        Maximum fraction of entries that can be non-integer before we say
        it's not raw counts. Default is 0.01 (1%).
    check_sparsity : bool, optional
        Whether to expect a decent fraction of zeros for typical scRNA data.
        Default is True.

    Returns
    -------
    bool
        True if `X` likely contains raw (integer) counts, False otherwise.
    """
    if sp.issparse(X):
        data = X.data  # flat array of non-zero values, no copy
        if len(data) == 0:
            return True  # all zeros are valid integer counts

        # 1. Check for negative values (implicit zeros are non-negative)
        if data.min() < min_val:
            return False

        # 2. Check fraction of non-integer values
        # Only non-zero values can be non-integer; implicit zeros are integers.
        # Compute fraction relative to total elements for consistent semantics.
        n_total = X.shape[0] * X.shape[1]
        n_noninteger = np.sum(np.abs(data - np.round(data)) > 1e-6)
        fraction_noninteger = n_noninteger / n_total

        if fraction_noninteger > max_noninteger_fraction:
            return False

        return True
    else:
        # Dense path (unchanged)
        if np.min(X) < min_val:
            return False

        diffs = np.abs(X - np.round(X))
        noninteger_mask = diffs > 1e-6
        fraction_noninteger = np.mean(noninteger_mask)

        if fraction_noninteger > max_noninteger_fraction:
            return False

        return True


def is_log_transformed(X, tol: float = 1e-3) -> bool:
    """
    Check if the data is likely log-transformed using a naive heuristic.

    Parameters
    ----------
    X : np.ndarray or sparse matrix
        The expression matrix to check.
    tol : float, optional
        Tolerance for negative values. If any value is below -tol, we say
        it's not log-transformed. Default is 1e-3.

    Returns
    -------
    bool
        True if the data appears log-transformed, False otherwise.
    """
    if sp.issparse(X):
        if X.nnz == 0:
            return False  # all zeros — not log-transformed data
        # Only stored values can be negative; implicit zeros are >= -tol
        if X.data.min() < -tol:
            return False
        max_val = X.data.max()
    else:
        if np.min(X) < -tol:
            return False
        max_val = np.max(X)
    return max_val < 100


def is_normalized(X, axis: int = 1, var_threshold: float = 0.2) -> bool:
    """
    Check if the data is likely normalized by testing row-sum consistency.

    Parameters
    ----------
    X : np.ndarray or sparse matrix
        The expression matrix to check. By default, rows are cells and columns are genes.
    axis : int, optional
        The axis along which to sum. Default is 1 (summing each row/cell).
    var_threshold : float, optional
        The cutoff for the ratio (std/mean) of row sums.
        If std/mean < var_threshold, we declare the data normalized. Default is 0.2.

    Returns
    -------
    bool
        True if the data appears normalized, False otherwise.
    """
    # .sum(axis=) works on both sparse and dense — returns a small dense vector
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
    """
    Guarantee that `adata.X` and `adata.layers['counts']` contain the raw count matrix.

    This function searches for raw counts in the following order:
    1. If `raw_layer_key` is provided and exists in `adata.layers`, use that layer.
    2. Else if `adata.raw.X` contains raw counts, use that.
    3. Else if `adata.X` itself appears to be raw counts, store a copy as `adata.layers['counts']`.
    4. Else if a `'counts'` layer already exists and is valid, use that.
    5. Otherwise log an error (and optionally raise).

    Parameters
    ----------
    adata : AnnData
        AnnData to check/modify in place.
    raw_layer_key : str, optional
        Optional key in `adata.layers` from which to source raw counts.
        If None, defaults to "counts".
    raise_on_missing : bool, optional
        If True, raises ValueError when no valid raw counts are found.
        Default is True.

    Raises
    ------
    ValueError
        If `raise_on_missing` is True and no valid raw counts could be located.
    """
    # Default to "counts" if raw_layer_key is None
    if raw_layer_key is None:
        raw_layer_key = "counts"

    # 1) Prefer the user-specified layer
    if raw_layer_key in adata.layers:
        logger.info("Using layer '%s' for raw counts", raw_layer_key)
        if raw_layer_key != "counts":
            adata.layers["counts"] = adata.layers[raw_layer_key]
        adata.X = adata.layers["counts"].copy()
    # 2) Detect if raw contains raw counts
    elif adata.raw is not None and is_raw_counts(adata.raw.X):
        logger.info("Detected raw counts in adata.raw.X; saving to layer 'counts'")
        # Handle case where adata.raw might have more variables than adata
        if adata.raw.n_vars > adata.n_vars:
            # Find intersection of variable names between raw and current adata
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
                # All current variables are present in raw, subset accordingly
                raw_var_to_idx = {
                    var: idx for idx, var in enumerate(adata.raw.var_names)
                }
                var_indices = [raw_var_to_idx[var] for var in adata.var_names]
                adata.layers["counts"] = adata.raw.X[:, var_indices].copy()
                adata.X = adata.layers["counts"].copy()
        else:
            # Standard case: raw has same or fewer variables
            adata.layers["counts"] = adata.raw.X.copy()
            adata.X = adata.layers["counts"].copy()
    # 3) Detect if X is raw counts
    elif is_raw_counts(adata.X):
        logger.info("Detected raw counts in adata.X; saving to layer 'counts'")
        adata.layers["counts"] = adata.X.copy()
    # 4) Fall back to an existing 'counts' layer
    elif "counts" in adata.layers and is_raw_counts(adata.layers["counts"]):
        logger.info("Using existing 'counts' layer for raw counts")
        adata.X = adata.layers["counts"].copy()
    else:
        # List available layers for helpful error message
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

    # Convert to int32 if not already in that format
    if "counts" in adata.layers and adata.layers["counts"].dtype != np.int32:
        logger.info(
            "Converting adata.layers['counts'] from %s to int32",
            adata.layers["counts"].dtype,
        )
        adata.layers["counts"] = adata.layers["counts"].astype(np.int32)

    # Re-validate that 'counts' really contains integer counts
    if "counts" in adata.layers and not is_raw_counts(adata.layers["counts"]):
        logger.error(
            "Layer 'counts' does not appear to contain integer raw counts; "
            "downstream steps may fail."
        )


def ensure_norm_log(
    adata: AnnData, in_place: bool = True, var_threshold: float = 0.2
) -> None:
    """
    Ensures that `adata.X` is normalized and then log-transformed.
    Applies normalization first (sc.pp.normalize_total), then log transformation (sc.pp.log1p).

    Parameters
    ----------
    adata : AnnData
        The AnnData object whose .X matrix will be checked/preprocessed.
    in_place : bool, optional
        If True, modifies `adata.X` in place. Default is True.
    var_threshold : float, optional
        Threshold used by `is_normalized` to decide if data appears normalized.
        Default is 0.2.

    Notes
    -----
    The checks used here are naive heuristics. Adjust them for your data as needed.
    The order of operations is: normalize first, then log-transform.
    """
    logger.info("Checking if data in adata.X appears normalized and log-transformed.")
    already_norm = is_normalized(adata.X, var_threshold=var_threshold)
    already_log = is_log_transformed(adata.X)

    # Step 1: Normalize first (on raw counts)
    if not already_norm:
        logger.info(
            "Data does not appear to be normalized. Applying sc.pp.normalize_total() in place."
        )
        sc.pp.normalize_total(adata, target_sum=1e4)  # modifies adata.X in place
    else:
        logger.info("Data already appears to be normalized.")

    # Step 2: Then log-transform
    if not already_log:
        logger.info(
            "Data does not appear to be log-transformed. Applying sc.pp.log1p() in place."
        )
        sc.pp.log1p(adata)  # modifies adata.X in place
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
    """
    Perform quality control filtering on single-cell RNA-seq data based on various metrics.

    This function:
    1. Labels certain genes (mitochondrial, ribosomal, hemoglobin) in `adata.var`.
    2. Calculates QC metrics for these gene sets (e.g., % mitochondrial reads).
    3. Identifies outlier cells using a median absolute deviation (MAD) heuristic
       on metrics like total counts, number of genes, % counts in top genes, etc.
    4. Filters out low-quality cells (e.g., outliers, high % mt).

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing single-cell data.
    nmads_main : int, optional
        The number of MADs to define outliers for main QC metrics (total counts,
        number of genes, etc.). Default is 5.
    nmads_mt : int, optional
        The number of MADs to define outliers for mitochondrial percent. Default is 3.
    pct_counts_mt_threshold : float, optional
        An absolute threshold for % mitochondrial counts. Cells above this are flagged
        as outliers. Default is 8.0 (i.e., 8%).
    percent_top : list of int, optional
        A list of top gene counts for sc.pp.calculate_qc_metrics (e.g., [20, 50, 100]).
        If None, defaults to [20].

    Returns
    -------
    AnnData
        A filtered AnnData object with low-quality cells removed.
    """
    logger.info("Starting quality control checks.")

    # 1. Label relevant gene categories in adata.var
    logger.info("Labeling mitochondrial, ribosomal, and hemoglobin genes in adata.var")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.upper().str.contains(r"^HB[^P]")

    # 2. Calculate QC metrics
    if percent_top is None:
        percent_top = [20]
    # max of percent top can't be greater than n_vars
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
    # Use log1p=False since data is already log-transformed
    # We'll gracefully check for log1p metrics if they exist (from previous steps)
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

    # 3. Define a helper function for outlier detection using median absolute deviation
    def is_outlier(metric_values: np.ndarray, nmads: int) -> pd.Series:
        """
        Returns a boolean mask where True indicates the cell is an outlier
        based on the given metric array and nmads.
        """
        M = metric_values
        med = np.median(M)
        mad = median_abs_deviation(M)
        lower_bound = med - nmads * mad
        upper_bound = med + nmads * mad
        outlier_mask = (M < lower_bound) | (M > upper_bound)
        return outlier_mask

    # 4. Main QC outliers: total counts, number of genes, % counts in top X genes
    logger.info(
        "Flagging outliers for total counts, number of genes, and %% in top genes."
    )
    # Use log1p metrics if available, otherwise use regular metrics
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

    # 5. Mitochondrial outliers
    logger.info("Flagging outliers for mitochondrial fraction.")
    mt_metric = adata.obs["pct_counts_mt"].values
    outlier_mt = is_outlier(mt_metric, nmads_mt) | (mt_metric > pct_counts_mt_threshold)

    # Combine all outliers
    adata.obs["outlier"] = outlier_main
    adata.obs["mt_outlier"] = outlier_mt

    # 6. Filter out the outliers
    n_before = adata.n_obs
    # Create boolean mask first to avoid intermediate copies
    keep_mask = ~(adata.obs["outlier"] | adata.obs["mt_outlier"])
    adata = adata[keep_mask].copy()
    n_after = adata.n_obs
    # Explicitly delete the mask to free memory
    del keep_mask

    logger.info(
        "Filtered out %d cells as outliers (main or mt). Remaining cells: %d.",
        n_before - n_after,
        n_after,
    )

    logger.info("QC filtering complete.")
    
    # Force garbage collection after QC to free memory
    gc.collect()
    
    return adata


def filter_genes_and_cells(
    adata: AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
) -> AnnData:
    """
    Filter genes and cells based on expression thresholds.

    This function:
    1. Makes gene and cell names unique.
    2. Filters out cells expressing fewer than `min_genes` genes.
    3. Filters out genes expressed in fewer than `min_cells` cells.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to filter.
    min_cells : int, optional
        Minimum number of cells in which a gene must be expressed to keep that gene.
        Default is 10.
    min_genes : int, optional
        Minimum number of genes a cell must express to keep that cell.
        Default is 200.

    Returns
    -------
    AnnData
        The filtered AnnData object.
    """
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
    
    # Force garbage collection after filtering to free memory
    gc.collect()
    
    return adata


def remove_infinite_values(adata: AnnData) -> AnnData:
    """
    Remove genes that contain infinite values from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to process.

    Returns
    -------
    AnnData
        The AnnData object with genes containing infinite values removed.
    """
    logger.info("Checking for infinite values in adata.X.")
    
    # For sparse matrices, check columns without converting to dense
    if sp.issparse(adata.X):
        # Check non-zero values for infinities first (fast check)
        if not np.isfinite(adata.X.data).all():
            # If any infinities found, identify which genes (columns) contain them
            finite_mask = np.ones(adata.n_vars, dtype=bool)
            # Get column indices for non-finite values
            row_indices, col_indices = adata.X.nonzero()
            non_finite_mask = ~np.isfinite(adata.X.data)
            bad_cols = np.unique(col_indices[non_finite_mask])
            finite_mask[bad_cols] = False
        else:
            # All non-zero values are finite, so all genes are finite
            finite_mask = np.ones(adata.n_vars, dtype=bool)
    else:
        # Dense matrix: check all values column-wise
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
    """
    Ensure that Ensembl IDs are present in the AnnData object and set as index.

    This function:
    1. Checks for Ensembl IDs in the index or columns ("ensembl_id", "feature_id", "gene_id")
    2. If not found, maps gene names to Ensembl IDs using pybiomart
    3. Stores Ensembl IDs in "ensembl_id" column
    4. Sets Ensembl IDs as the var index
    5. Cleans up (removes version numbers, invalid IDs, duplicates, empty values)

    Parameters
    ----------
    adata : AnnData
        The AnnData object to process.
    ensembl_col : str, optional
        Column name containing Ensembl IDs. If None, searches in:
        "ensembl_id", "feature_id", "gene_id", or index.
    gene_names_col : str, optional
        Column name containing gene names/symbols for mapping. If None, searches in:
        index (if not Ensembl IDs), "gene_names", or "feature_names".
    species : str, optional
        Species name for pybiomart query. Default is "hsapiens".
    raise_on_failure : bool, optional
        If True, raises ValueError when Ensembl IDs cannot be obtained.
        Default is True.
    ensembl_id_threshold : float, optional
        Minimum fraction of Ensembl IDs required to consider a source valid.
        Default is 0.5 (50%). If a source has fewer than this fraction, the
        function will attempt to map gene names to Ensembl IDs instead.

    Returns
    -------
    AnnData
        The AnnData object with Ensembl IDs set as index.

    Raises
    ------
    ValueError
        If `raise_on_failure` is True and Ensembl IDs cannot be obtained.
    """
    logger.info("Ensuring Ensembl IDs are present in the dataset.")

    # Step 1: Check for Ensembl IDs in index
    index_values = adata.var_names.astype(str)
    index_ensembl_count = pd.Series(index_values).str.contains(r"^ENS", na=False).sum()
    index_ensembl_fraction = index_ensembl_count / len(index_values) if len(index_values) > 0 else 0.0
    logger.info(
        f"Found {index_ensembl_count} Ensembl IDs in index (out of {len(index_values)} genes, "
        f"{index_ensembl_fraction:.1%})"
    )

    # Step 2: Check for Ensembl IDs in columns
    ensembl_cols_to_check = []
    if ensembl_col:
        ensembl_cols_to_check = [ensembl_col]
    else:
        # Default columns to check
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

    # Track if index contains gene names (to save before overwriting)
    index_contains_gene_names = index_ensembl_fraction < ensembl_id_threshold
    
    # Step 3: Determine best source of Ensembl IDs (only if threshold is met)
    if (index_ensembl_fraction >= ensembl_id_threshold and 
        index_ensembl_fraction >= ensembl_col_fraction):
        logger.info(
            f"Using index as source of Ensembl IDs "
            f"({index_ensembl_fraction:.1%} >= {ensembl_id_threshold:.1%} threshold)."
        )
        # Store in ensembl_id column
        adata.var["ensembl_id"] = index_values
        # Index already has Ensembl IDs; convert to Series for downstream use
        ensembl_ids = index_values.to_series()
    elif (found_ensembl_col and 
          ensembl_col_fraction >= ensembl_id_threshold):
        logger.info(
            f"Using column '{found_ensembl_col}' as source of Ensembl IDs "
            f"({ensembl_col_fraction:.1%} >= {ensembl_id_threshold:.1%} threshold)."
        )
        # Store in ensembl_id column
        adata.var["ensembl_id"] = adata.var[found_ensembl_col].astype(str)
        # Use values to avoid unnecessary copy
        ensembl_ids = pd.Series(adata.var["ensembl_id"].values, index=adata.var.index)
        # If index contains gene names, save them before overwriting
        if index_contains_gene_names:
            logger.info("Saving gene names from index to adata.var['gene_names'] before overwriting with Ensembl IDs.")
            adata.var["gene_names"] = index_values.values
    else:
        # Step 4: Need to map gene names to Ensembl IDs
        if index_ensembl_count > 0 or ensembl_col_count > 0:
            logger.warning(
                f"Found some Ensembl IDs (index: {index_ensembl_fraction:.1%}, "
                f"columns: {ensembl_col_fraction:.1%}) but below threshold "
                f"({ensembl_id_threshold:.1%}). Proceeding to map gene names."
            )
        logger.info("No Ensembl IDs found above threshold. Attempting to map gene names to Ensembl IDs using pybiomart.")

        # Find gene names column
        gene_names_cols_to_check = []
        if gene_names_col:
            gene_names_cols_to_check = [gene_names_col]
        else:
            # Default columns to check
            gene_names_cols_to_check = ["gene_names", "feature_names"]

        found_gene_names_col = None
        gene_names = None

        # First check if index contains gene names (not Ensembl IDs above threshold)
        if index_ensembl_fraction < ensembl_id_threshold:
            logger.info("Using index as source of gene names.")
            gene_names = index_values.copy()
            found_gene_names_col = "index"
            # Save gene names before they get overwritten
            logger.info("Saving gene names from index to adata.var['gene_names'] before overwriting with Ensembl IDs.")
            adata.var["gene_names"] = index_values.values
        else:
            # Check columns
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
        # Convert to Series if it's an Index for consistent handling
        if isinstance(gene_names, pd.Index):
            gene_names = pd.Series(gene_names)
        logger.info(f"Example gene name: {gene_names.iloc[0] if len(gene_names) > 0 else 'N/A'}")

        # Map gene names to Ensembl IDs using pybiomart
        logger.info(f"Fetching Ensembl IDs from pybiomart for species '{species}'. This may take a moment...")
        try:
            dataset = pybiomart.Dataset(
                name=f"{species}_gene_ensembl", host="http://www.ensembl.org"
            )
            biomart_df = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])

            # Drop duplicates so that each gene symbol maps to exactly one Ensembl ID
            biomart_df = biomart_df.drop_duplicates(subset="Gene name").set_index("Gene name")

            # Map gene symbols to Ensembl IDs
            ensembl_ids = []
            mapped_count = 0
            for symbol in gene_names:
                if symbol in biomart_df.index:
                    ensembl_ids.append(biomart_df.loc[symbol, "Gene stable ID"])
                    mapped_count += 1
                else:
                    ensembl_ids.append("")

            logger.info(f"Mapped {mapped_count} out of {len(gene_names)} genes to Ensembl IDs.")

            # Check if we got valid Ensembl IDs
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

            # Store in ensembl_id column
            adata.var["ensembl_id"] = pd.Series(ensembl_ids, index=adata.var_names)

        except Exception as e:
            error_msg = (
                f"Error fetching Ensembl IDs from pybiomart: {e}\n"
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

        ensembl_ids = pd.Series(ensembl_ids, index=adata.var_names)

    # Step 5: Clean up Ensembl IDs and set as index
    logger.info("Cleaning up Ensembl IDs and setting as index...")

    # Remove version numbers (everything after the dot)
    if pd.Series(ensembl_ids).str.contains(r"\.").any():
        logger.info("Removing version numbers from Ensembl IDs (everything after '.').")
        ensembl_ids = pd.Series(ensembl_ids).str.split(".").str[0]

    # Remove invalid IDs (not starting with "ENS" or empty)
    valid_mask = pd.Series(ensembl_ids).str.startswith("ENS", na=False)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(
            f"Removing {invalid_count} genes with invalid Ensembl IDs "
            "(not starting with 'ENS' or empty)."
        )
        adata = adata[:, valid_mask].copy()
        ensembl_ids = ensembl_ids[valid_mask]

    # Ensure ensembl_id column is updated
    adata.var["ensembl_id"] = ensembl_ids.values

    # Remove duplicates (keep first occurrence)
    is_dup = pd.Series(ensembl_ids).duplicated(keep="first")
    dup_count = is_dup.sum()
    if dup_count > 0:
        logger.warning(
            f"Found {dup_count} genes with duplicate Ensembl IDs; "
            "keeping first occurrence and removing duplicates."
        )
        adata = adata[:, ~is_dup].copy()
        ensembl_ids = ensembl_ids[~is_dup]
        # gene_names column will be automatically filtered by the adata filtering above

    # Set Ensembl IDs as index
    adata.var.index = ensembl_ids.values
    logger.info(f"Set Ensembl IDs as index. Final shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # Force garbage collection after Ensembl ID mapping to free memory
    gc.collect()

    return adata

def _infer_X_type(adata):
    X = adata.X
    if X.size == 0:
        return "unknown (empty)", {"n": 0, "min": None, "max": None, "int_like": None, "hints": []}
    
    # Compute min/max on full matrix (matches X.min() and X.max())
    vmin, vmax = float(X.min()), float(X.max())
    
    if sp.issparse(X):
        # For sparse matrices, check all non-zero values for int_like
        vals_for_check = X.data
        total_n = X.shape[0] * X.shape[1]  # Total elements including zeros
    else:
        # For dense matrices, sample values to avoid creating full copy
        # Sample up to 100k values for int_like check (sufficient for statistical check)
        max_sample_size = 100000
        if X.size > max_sample_size:
            # Random sampling without replacement - use ravel() but only on sampled indices
            # This is more memory efficient than raveling the entire array
            flat_indices = np.random.choice(X.size, size=max_sample_size, replace=False)
            X_flat = np.asarray(X).ravel()
            vals_for_check = X_flat[flat_indices]
            # Allow garbage collection of full flattened array
            del X_flat
        else:
            # Small enough to check all values
            vals_for_check = np.asarray(X).ravel()
        total_n = X.size

    if vals_for_check.size == 0:
        int_like = False
    else:
        # Memory-efficient check: avoid creating rounded copy
        # Check if values are close to integers without creating intermediate arrays
        finite_mask = np.isfinite(vals_for_check)
        if not np.all(finite_mask):
            int_like = False
        else:
            # Check integer-likeness by comparing differences to rounded values
            # Use in-place operations where possible
            rounded = np.round(vals_for_check)
            # Use absolute difference with small tolerance
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
    
    # Report counts layer if present
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
    print(f"[OK] Wrote summary to {out_path.resolve()}")

def plot_preprocessing_qc(
    adata,
    page_title="QC Summary Preprocessing",
    initial_cells: int | None = None,
    initial_genes: int | None = None,
    save_path: Path | str | None = None,
):
    """
    Generate QC plots on a single page (3x3 grid) and save as high-resolution PNG.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object
    page_title : str, optional
        Title displayed on the page.
    initial_cells : int, optional
        Initial number of cells before filtering (for display in Plot 6).
    initial_genes : int, optional
        Initial number of genes before filtering (for display in Plot 6).
    save_path : Path or str, optional
        Path to save the PNG file. If None, the plot is not saved.
    """

    # Set publication style
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
    # Main grid with 2 rows
    gs_main = fig.add_gridspec(2, 1, hspace=0.4)
    
    # Top row: 22%, 22%, 56% layout. 5% for 3 %mt, %ribo, %hb violin plots
    gs_top = gs_main[0].subgridspec(1, 3, wspace=0.3, width_ratios=[0.22, 0.22, 0.56])
    
    # Bottom row: 35%, 35%, 30% layout
    gs_bottom = gs_main[1].subgridspec(1, 3, wspace=0.3, width_ratios=[0.35, 0.35, 0.30])
    
    fig.suptitle(page_title, fontsize=16, fontweight="bold")

    # Top row plots
    # Plot 1: violin plot of total_counts
    ax = fig.add_subplot(gs_top[0, 0])
    if "total_counts" in adata.obs.columns:
        sc.pl.violin(adata, keys="total_counts", ax=ax, show=False)
        ax.set_title("total_counts")
    else:
        ax.text(0.5, 0.5, "total_counts\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("total_counts")

    # Plot 2: violin plot of n_genes_by_counts
    ax = fig.add_subplot(gs_top[0, 1])
    if "n_genes_by_counts" in adata.obs.columns:
        sc.pl.violin(adata, keys="n_genes_by_counts", ax=ax, show=False)
        ax.set_title("n_genes_by_counts")
    else:
        ax.text(0.5, 0.5, "n_genes_by_counts\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("n_genes_by_counts")

    # Plot 3: violin plot of pct_counts_mt, pct_counts_ribo, pct_counts_hb
    ax = fig.add_subplot(gs_top[0, 2])
    if "pct_counts_mt" in adata.obs.columns and "pct_counts_ribo" in adata.obs.columns and "pct_counts_hb" in adata.obs.columns:
        sc.pl.violin(adata, keys=["pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"], ax=ax, show=False)
        ax.set_title("Mito %, Ribo %, HB %")
    else:
        ax.text(0.5, 0.5, "pct_counts_mt or pct_counts_ribo or pct_counts_hb\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Mito %, Ribo %, HB %")

    # Bottom row plots
    # Plot 4: scatter plot of total_counts vs n_genes_by_counts
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

    # Plot 5: scatter plot of total_counts vs pct_counts_in_top_20_genes
    ax = fig.add_subplot(gs_bottom[0, 1])
    if "total_counts" in adata.obs.columns and "pct_counts_in_top_20_genes" in adata.obs.columns:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_in_top_20_genes", ax=ax, show=False)
    else:
        ax.text(0.5, 0.5, "Scatter plot\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("total_counts vs pct_counts_in_top_20_genes")
    
    # ax = fig.add_subplot(gs_bottom[0, 1])
    # if "counts" in adata.layers:
    #     sc.pl.highest_expr_genes(adata, n_top=20, layer="counts", ax=ax, show=False)
    #     ax.set_title("Top 20 Expressed Genes")
    # else:
    #     ax.text(0.5, 0.5, "Top genes\nnot available", ha="center", va="center", transform=ax.transAxes)
    #     ax.set_title("Top Expressed Genes")

    # Plot 6: Data summary statistics
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

# usage:
# summarize_adata(adata, "summary.txt")
