# SCSA: Single-Cell RNA-seq Annotation Method

## Overview

SCSA (Single-Cell RNA-seq Annotation) is an automated tool for annotating cell types in single-cell RNA-seq data. It uses a score-based annotation model that combines differentially expressed genes (markers) with confidence levels from curated cell marker databases.

## How SCSA Works

### 1. **Input: Cluster-Level Marker Genes**

SCSA requires **cluster-level marker genes** as input, not individual cell data. The input CSV file contains:

- **Gene names** per cluster (which genes are markers for each cluster)
- **Log fold changes** (how much more expressed in this cluster vs others)
- **Statistical scores** (t-test scores or similar)
- **P-values** (statistical significance)

**Example input format:**
```
gene,0_n,0_l,0_s,0_p,1_n,1_l,1_s,1_p,...
0,LTB,2.42,30.60,3.6e-159,LYZ,6.24,33.96,1.0e-252,...
1,IL32,2.55,30.04,2.0e-151,S100A9,7.35,33.65,3.7e-248,...
```

Where:
- `0_n`, `1_n`: Gene names for clusters 0, 1
- `0_l`, `1_l`: Log fold changes
- `0_s`, `1_s`: Statistical scores
- `0_p`, `1_p`: P-values

### 2. **Marker Gene Filtering**

For each cluster, SCSA filters marker genes based on your thresholds:

- **Fold change threshold** (`-f`): Minimum log fold change (e.g., 1.5)
- **P-value threshold** (`-p`): Maximum p-value (e.g., 0.01)

Only genes meeting both criteria are used for annotation.

### 3. **Database Matching**

SCSA matches the filtered marker genes against curated databases:

- **CellMarker**: Database of cell type-specific markers
- **CancerSEA**: Cancer cell state markers (optional)

For each cell type in the database, SCSA:
1. Finds overlapping markers between cluster markers and cell type markers
2. Creates a marker-gene matrix
3. Calculates expression-weighted scores

### 4. **Z-Score Calculation**

SCSA calculates a **Z-score** for each cell type:

1. **Marker matrix**: Creates a matrix matching cluster markers to database cell type markers
2. **Expression weighting**: Weights markers by their expression levels (log fold changes)
3. **Z-score normalization**: Normalizes scores across all cell types to identify the best match

**Higher Z-score = Better match** between cluster markers and cell type markers

### 5. **Output: Cluster-Level Predictions**

SCSA outputs the **top-scoring cell types** for each cluster:

```
Cell Type                    Z-score    Cluster
Natural killer cell         5.070      4
B cell                      5.916      6
Mesenchymal stem cell       3.852      5
```

## Key Concepts

### Why Cluster-Level, Not Cell-Level?

SCSA works at the **cluster level** because:

1. **Marker detection** requires comparing groups of cells (cluster vs rest)
2. **Statistical power** comes from aggregating signals across many cells
3. **Robustness** - individual cells are noisy, clusters are more stable

### Mapping to Individual Cells

To get cell-level predictions, you map cluster predictions back to cells:

- All cells in cluster 4 → "Natural killer cell"
- All cells in cluster 6 → "B cell"
- etc.

This is done automatically by the `run_scsa_pipeline.py` script.

## Workflow

### Complete Pipeline

```
1. Single-cell data (h5ad file)
   ↓
2. Clustering (Louvain/Leiden)
   ↓
3. Marker gene detection (per cluster)
   ↓
4. Create CSV with markers
   ↓
5. SCSA annotation
   ↓
6. Map predictions to cells
   ↓
7. Output: cell_id, true_labels, predicted_labels
```

### Step-by-Step Process

1. **Preprocessing** (scanpy):
   - Find highly variable genes
   - Scale data
   - Compute PCA
   - Build neighborhood graph
   - Cluster cells (Louvain)

2. **Marker Detection**:
   - For each cluster, find genes differentially expressed vs all other cells
   - Compute log fold changes, scores, and p-values
   - **Critical**: Must be computed on normalized (not scaled) data for valid log fold changes

3. **SCSA Annotation**:
   - Filter markers by fold change and p-value thresholds
   - Match against CellMarker/CancerSEA database
   - Calculate Z-scores for each cell type
   - Rank cell types by Z-score

4. **Cell-Level Mapping**:
   - Assign cluster prediction to all cells in that cluster
   - Combine with true labels (if available)

## Important Notes

### Log Fold Changes Are Critical

- SCSA uses log fold changes to filter genes (`-f` threshold)
- **Must be computed on normalized (log-transformed) data**, NOT scaled data
- Scaling (z-score normalization) breaks log fold change calculations
- Missing log fold changes → SCSA can't filter properly → wrong predictions

### Why Different Results Than Expected?

Even with correct preprocessing, you might get different results than published examples because:

1. **Different clustering**: Different scanpy versions or random seeds → different clusters
2. **Different markers**: Different clusters → different marker genes
3. **Different preprocessing**: Subtle differences in normalization/scaling
4. **Database updates**: CellMarker database may have been updated

**This is normal** - what matters is that SCSA works correctly on YOUR data.

### Z-Score Interpretation

- **Z-score > 3**: Strong match (high confidence)
- **Z-score 1-3**: Moderate match (reasonable confidence)
- **Z-score < 1**: Weak match (low confidence)
- **Negative Z-score**: Poor match

## Usage

### Quick Start

```bash
# 1. Clone repository
./prepare.sh

# 2. Set up environment (optional)
cd SCSA
python3 -m venv scsa
source scsa/bin/activate
pip install -r requirements.txt

# 3. Run pipeline
./run.sh input.h5ad output.csv -t cell_type
```

### Input Requirements

- **h5ad file**: AnnData format with single-cell expression data
- **True labels column**: Column name in `adata.obs` containing true cell type labels (for comparison)

### Output Format

CSV file with three columns:
- `cell_id`: Cell identifier from `adata.obs.index`
- `true_labels`: True cell type labels
- `predicted_labels`: SCSA predicted cell type

## References

- **Original Paper**: Cao Y, Wang X and Peng G (2020) SCSA: A Cell Type Annotation Tool for Single-Cell RNA-seq Data. Front. Genet. 11:490. doi: [10.3389/fgene.2020.00490](https://doi.org/10.3389/fgene.2020.00490)
- **Repository**: https://github.com/bioinfo-ibms-pumc/SCSA
- **CellMarker Database**: http://xteam.xbio.top/CellMarker/

## Troubleshooting

### Missing Log Fold Changes

**Problem**: CSV has NaN values for log fold changes

**Solution**: Ensure markers are computed on normalized (not scaled) data. The `run_scsa_pipeline.py` script handles this automatically.

### No Predictions

**Problem**: SCSA returns "Unknown" or no predictions

**Possible causes**:
- No marker genes meet the thresholds (try lowering `-f` or `-p`)
- Markers don't match database (check gene ID format - use `-E` for gene symbols)
- Wrong species/tissue filter (check `-g` and `-k` options)

### Different Clusters Than Expected

**Problem**: Get 9 clusters instead of 8 (or different number)

**Explanation**: This is normal - clustering depends on:
- Scanpy version
- Random seed
- Resolution parameter
- Preprocessing steps

Different clusters → different markers → different predictions, but predictions should still be valid.

