# Preprocessing Documentation

This document describes the preprocessing pipeline for single-cell RNA-seq data implemented in `bin/pp/preprocess.py` and the utility functions in `bin/pp/pp_utils.py`.

## Overview

The preprocessing pipeline performs a streamlined set of operations to prepare single-cell RNA-seq data for downstream analysis. The pipeline focuses on:

1. **Raw count detection and storage**: Finding raw counts in the AnnData object and ensuring they are properly stored
2. **Ensembl ID mapping**: Ensuring Ensembl IDs are present and set as the gene index
3. **Quality control**: Filtering out low-quality cells based on QC metrics
4. **Gene and cell filtering**: Removing genes and cells that don't meet expression thresholds
5. **Normalization and log transformation**: Ensuring normalized and log-transformed data in `adata.X` (normalize first, then log-transform)
6. **Data cleaning**: Removing infinite values
7. **Summary generation**: Creating a text summary of the preprocessed data

## Pipeline Steps

### Step 1: Raw Count Detection and Storage

**Function**: `ensure_raw_counts_layer()`

This step ensures that raw counts are properly stored in both `adata.X` and `adata.layers["counts"]`. The function searches for raw counts in the following order:

1. **User-specified layer**: If `count_layer_key` is provided and exists in `adata.layers`, that layer is used
2. **adata.raw.X**: If `adata.raw` exists and contains raw counts, those are used
3. **adata.X**: If `adata.X` itself appears to be raw counts, a copy is stored in `adata.layers["counts"]`
4. **Existing "counts" layer**: If a "counts" layer already exists and is valid, it is used

Raw counts are detected using heuristics:
- Non-negative values (with small tolerance for floating-point errors)
- Mostly integer values (at least 99% should be integers)
- Typical sparsity patterns of single-cell data

**Configuration**:
- `count_layer_key` (str, optional): Key in `adata.layers` containing raw counts. Defaults to `"counts"` if not specified. Can be set via dataset config file or command-line argument.

### Step 2: Ensembl ID Mapping

**Function**: `ensure_ensembl_ids()`

This step ensures that Ensembl IDs are present in the dataset and sets them as the gene index. The function:

1. **Checks for existing Ensembl IDs**:
   - First checks the index (if IDs start with "ENS")
   - Then checks columns: "ensembl_id", "feature_id", "gene_id" (or user-specified `ensembl_col`)
   - Uses the source with the most Ensembl IDs

2. **Maps gene names to Ensembl IDs** (if not found):
   - Searches for gene names in:
     - Index (if not Ensembl IDs)
     - Columns: "gene_names", "feature_names" (or user-specified `gene_names_col`)
   - Uses pybiomart to map gene symbols to Ensembl IDs
   - Requires network connection to Ensembl

3. **Cleans up Ensembl IDs**:
   - Removes version numbers (everything after ".")
   - Removes invalid IDs (not starting with "ENS" or empty)
   - Removes duplicates (keeps first occurrence)
   - Sets Ensembl IDs as the `adata.var` index

4. **Stores Ensembl IDs**: Always stores Ensembl IDs in `adata.var["ensembl_id"]` column

**Configuration**:
- `ensembl_col` (str, optional): Column name containing Ensembl IDs. If None, searches in "ensembl_id", "feature_id", "gene_id", or index.
- `gene_names_col` (str, optional): Column name containing gene names/symbols for mapping. If None, searches in index (if not Ensembl IDs), "gene_names", or "feature_names".
- `species` (str, default="hsapiens"): Species name for pybiomart query. Common values: "hsapiens" (human), "mmusculus" (mouse).

**Error Handling**:
If Ensembl IDs cannot be obtained, the function raises a `ValueError` with instructions to:
- Set `ensembl_col` in dataset config YAML if Ensembl IDs already exist
- Set `gene_names_col` in dataset config YAML to specify the column with gene names
- Set `species` in dataset config YAML (e.g., "hsapiens" or "mmusculus")
- Check network connection to Ensembl

### Step 3: Quality Control

**Function**: `pp_quality_control()`

This step identifies and filters out low-quality cells using median absolute deviation (MAD) based outlier detection.

**QC Metrics Calculated**:
- Total counts per cell
- Number of genes expressed per cell
- Percentage of counts in top genes (e.g., top 20 genes)
- Percentage of mitochondrial reads
- Percentage of ribosomal reads
- Percentage of hemoglobin reads

**Outlier Detection**:
- **Main outliers**: Cells that are outliers in total counts, number of genes, or percentage in top genes (using `nmads_main` MADs)
- **Mitochondrial outliers**: Cells that are outliers in mitochondrial percentage (using `nmads_mt` MADs) OR exceed the absolute threshold `pct_counts_mt_threshold`

**Configuration**:
- `nmads_main` (int, default=5): Number of MADs for main QC outlier detection
- `nmads_mt` (int, default=3): Number of MADs for mitochondrial outlier detection
- `pct_counts_mt_threshold` (float, default=8.0): Absolute threshold for % mitochondrial counts (cells above this are filtered)
- `percent_top` (list[int], default=[20]): List of top gene counts for QC metrics calculation
- `log1p_for_qc` (bool, default=True): Whether to compute log1p-transformed values for QC metrics

### Step 4: Gene and Cell Filtering

**Function**: `filter_genes_and_cells()`

This step filters out genes and cells that don't meet minimum expression thresholds.

**Filtering Criteria**:
- **Genes**: Remove genes expressed in fewer than `min_cells` cells
- **Cells**: Remove cells expressing fewer than `min_genes` genes

**Configuration**:
- `min_cells` (int, default=10): Minimum number of cells in which a gene must be expressed
- `min_genes` (int, default=200): Minimum number of genes a cell must express

### Step 5: Normalization and Log Transformation

**Function**: `ensure_norm_log()`

This step ensures that `adata.X` is normalized first, then log-transformed. The function checks if the data is already normalized and log-transformed using heuristics:

- **Normalization check**:
  - Row sums are relatively consistent (std/mean < threshold)
- **Log transformation check**: 
  - No negative values (with tolerance)
  - Maximum value typically < 100 for log1p-transformed scRNA-seq data

**Order of operations**:
1. If the data is not normalized, `sc.pp.normalize_total()` is applied with `target_sum=1e4` (normalizes on raw counts)
2. If the data is not log-transformed, `sc.pp.log1p()` is applied

**Configuration**:
- `var_threshold` (float, default=0.2): Threshold for normalized check (std/mean of row sums)

### Step 6: Remove Infinite Values

**Function**: `remove_infinite_values()`

This step removes genes that contain infinite values (inf or -inf) from the dataset. Genes with any infinite values in any cell are removed.

### Step 7: Summary Generation

**Function**: `summarize_adata()` (from `generate_summary.py`)

This step generates a text summary of the preprocessed AnnData object, including:
- Shape (number of cells and genes)
- Data type inference (raw counts, log-normalized, etc.)
- Information about `.raw`, `.layers`, and `.obs`/`.var` columns
- Preprocessing status (HVGs, PCA, embeddings, etc.)

**Configuration**:
- `summary_path` (str | Path, optional): Path to write the summary file. If not provided, defaults to `{output_stem}_summary.txt` in the output directory.

## Usage

### Command-Line Interface

```bash
# Basic usage
python -m bin.pp.preprocess --input data.h5ad --output processed.h5ad

# With dataset config file
python -m bin.pp.preprocess --input data.h5ad --output processed.h5ad --config datasets/demo.yml

# Custom parameters
python -m bin.pp.preprocess \
    --input data.h5ad \
    --output processed.h5ad \
    --min_cells 5 \
    --min_genes 100 \
    --nmads_main 3 \
    --summary summary.txt
```

### Python API

```python
from bin.pp.preprocess import preprocess_adata
import scanpy as sc

# Load data
adata = sc.read_h5ad("data.h5ad")

# Preprocess
adata_processed = preprocess_adata(
    adata,
    count_layer_key="counts",
    min_cells=10,
    min_genes=200,
    nmads_main=5,
    nmads_mt=3,
    summary_path="summary.txt"
)

# Save
adata_processed.write_h5ad("processed.h5ad")
```

## Configuration via Dataset Config File

The preprocessing script can read configuration from a YAML dataset config file (as used in Nextflow `main.nf`). Add a `preprocessing` section to your dataset config:

```yaml
name: demo
url: https://example.com/data.h5ad
format: h5ad
batch_key: method
annotation_keys: [cell_type]

preprocessing:
  count_layer_key: "counts"  # or another layer name
  ensembl_col: "ensembl_id"  # optional: column with Ensembl IDs
  gene_names_col: "gene_names"  # optional: column with gene names for mapping
  species: "hsapiens"  # species for pybiomart (hsapiens, mmusculus, etc.)
  min_cells: 10
  min_genes: 200
  nmads_main: 5
  nmads_mt: 3
  pct_counts_mt_threshold: 8.0
  percent_top: [20, 50, 100]
  log1p_for_qc: true
```

Command-line arguments override config file values.

## Configuration Options Summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `count_layer_key` | str | `"counts"` | Key in `adata.layers` containing raw counts |
| `ensembl_col` | str | None | Column name containing Ensembl IDs (searches multiple defaults if None) |
| `gene_names_col` | str | None | Column name containing gene names for mapping (searches multiple defaults if None) |
| `species` | str | `"hsapiens"` | Species name for pybiomart query (e.g., "hsapiens", "mmusculus") |
| `nmads_main` | int | 5 | Number of MADs for main QC outlier detection |
| `nmads_mt` | int | 3 | Number of MADs for mitochondrial outlier detection |
| `pct_counts_mt_threshold` | float | 8.0 | Absolute threshold for % mitochondrial counts |
| `percent_top` | list[int] | `[20]` | Top gene counts for QC metrics |
| `log1p_for_qc` | bool | True | Compute log1p values for QC metrics |
| `min_cells` | int | 10 | Minimum cells per gene for filtering |
| `min_genes` | int | 200 | Minimum genes per cell for filtering |
| `summary_path` | str\|Path | None | Path to write summary file (auto-generated if None) |

## Output

The preprocessing script produces:

1. **Processed AnnData file**: The preprocessed data saved in H5AD or Zarr format
2. **Summary file**: A text summary of the preprocessed data (optional, auto-generated by default)

The processed AnnData object will have:
- `adata.X`: Log-normalized expression matrix
- `adata.layers["counts"]`: Raw count matrix
- `adata.obs`: Cell metadata with QC metrics and outlier flags
- `adata.var`: Gene metadata with gene category labels (mt, ribo, hb) and Ensembl IDs
- `adata.var.index`: Ensembl gene IDs (cleaned, deduplicated)
- `adata.var["ensembl_id"]`: Ensembl gene IDs column

## Notes

- **No highly variable gene selection**: The current pipeline does not perform HVG selection. This can be added downstream if needed.
- **No low-frequency category consolidation**: Unlike some preprocessing pipelines, this one does not consolidate or remove low-frequency categories in metadata columns.
- **No extensive backed object handling**: The pipeline assumes data fits in memory.
- **Raw counts preservation**: Raw counts are always preserved in `adata.layers["counts"]` even after normalization.

## Utility Functions

The `pp_utils.py` module provides standalone utility functions that can be used independently:

- `is_raw_counts()`: Check if a matrix contains raw counts
- `is_log_transformed()`: Check if data is log-transformed
- `is_normalized()`: Check if data is normalized
- `ensure_raw_counts_layer()`: Find and set raw counts
- `ensure_ensembl_ids()`: Ensure Ensembl IDs are present and set as index
- `ensure_norm_log()`: Ensure normalization and log transformation (normalize first, then log-transform)
- `pp_quality_control()`: Perform QC filtering
- `filter_genes_and_cells()`: Filter genes and cells
- `remove_infinite_values()`: Remove infinite values

See the function docstrings for detailed usage information.

