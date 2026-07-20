#!/usr/bin/env python3
"""
Complete SCSA pipeline: Create CSV, run SCSA, and map predictions to cells.

This script:
1. Creates a CSV file from an h5ad file (scanpy format)
2. Runs SCSA.py on the CSV
3. Maps cluster-level predictions back to individual cells
4. Outputs a CSV with cell_id, true_labels, and predicted_labels
"""

import sys
import os
import subprocess
import argparse
import pandas as pd
import anndata as ad
import scanpy as sc

GENE_NAME_COLUMNS = (
    "gene_names",
    "gene_name",
    "feature_name",
    "gene_symbol",
    "symbol",
    "genes",
)


def set_gene_index_from_available_columns(adata):
    """
    Set adata.var_names to the best available gene-name column.

    Falls back to existing var_names if none of the known columns are present.
    Empty values are replaced with the original var_names.
    """
    fallback_names = pd.Series(adata.var_names.astype(str), index=adata.var.index)

    for col in GENE_NAME_COLUMNS:
        if col not in adata.var.columns:
            continue

        gene_names = pd.Series(adata.var[col], index=adata.var.index).astype("string")
        gene_names = gene_names.fillna("").str.strip()
        is_invalid = gene_names.eq("") | gene_names.str.lower().isin({"nan", "none"})
        resolved_names = gene_names.mask(is_invalid, fallback_names).astype(str)

        adata.var_names = pd.Index(resolved_names)
        if not adata.var_names.is_unique:
            adata.var_names_make_unique()

        print(
            f"  Using gene names from adata.var['{col}'] "
            f"({(~is_invalid).sum()}/{len(gene_names)} populated)"
        )
        return col

    adata.var_names = pd.Index(adata.var_names.astype(str))
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    print("  No gene-name column found in adata.var; using existing var_names")
    return None


def create_scanpy_csv(h5ad_path, output_csv, cluster_col='louvain', 
                      resolution=1.0, n_top_genes=2000, 
                      n_neighbors=15, n_pcs=50, random_state=0):
    """
    Create scanpy-compatible CSV from h5ad file.
    
    Parameters:
    -----------
    h5ad_path : str
        Path to input h5ad file
    output_csv : str
        Path to output CSV file
    cluster_col : str
        Name of cluster column to use/create
    resolution : float
        Resolution for Louvain clustering (if clustering needed)
    n_top_genes : int
        Number of highly variable genes
    n_neighbors : int
        Number of neighbors for neighborhood graph
    n_pcs : int
        Number of principal components
    random_state : int
        Random seed
    """
    print(f"Step 1: Reading h5ad file: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    
    # Check if clustering already exists
    needs_clustering = cluster_col not in adata.obs.columns

    # Use the most suitable available gene-name column for marker reporting.
    set_gene_index_from_available_columns(adata)
    
    if needs_clustering:
        print(f"  Finding {n_top_genes} highly variable genes...")
        # flavor='seurat' (dispersion-based) is correct for the log1p-normalized X this pipeline
        # provides. 'seurat_v3' expects RAW counts and only warns on log-normalized input, silently
        # picking a suboptimal HVG set; the pipeline's contexts are also counts-free (slimmed), so
        # seurat_v3 could never get the data it wants.
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()
        
        # Keep a copy of normalized data for marker detection (before scaling)
        adata_for_markers = adata.copy()
        
        print("  Scaling...")
        sc.pp.scale(adata, max_value=10)
        
        print(f"  Computing PCA ({n_pcs} components)...")
        sc.tl.pca(adata, svd_solver="arpack", n_comps=n_pcs)
        
        print(f"  Building neighborhood graph (k={n_neighbors})...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
        
        print(f"  Running Louvain clustering (resolution={resolution})...")
        sc.tl.louvain(adata, resolution=resolution, key_added=cluster_col, random_state=random_state)
        
        # Copy cluster assignments to the normalized data for marker detection
        adata_for_markers.obs[cluster_col] = adata.obs[cluster_col]
    else:
        print(f"  Using existing cluster column: '{cluster_col}'")
        # For marker detection, use normalized (not scaled) data
        # If data is already scaled, we need to work with what we have
        adata_for_markers = adata.copy()
        # Try to get raw data if available
        if 'log1p' in adata.layers:
            adata_for_markers.X = adata.layers['log1p'].copy()
        elif adata.raw is not None:
            adata_for_markers = adata.raw.to_adata()
            adata_for_markers.obs[cluster_col] = adata.obs[cluster_col]
            set_gene_index_from_available_columns(adata_for_markers)
    
    print(f"  Found {len(adata.obs[cluster_col].unique())} clusters")
    
    print("  Computing marker genes per cluster (on normalized data)...")
    # Use normalized (not scaled) data for marker detection to get proper log fold changes
    sc.tl.rank_genes_groups(adata_for_markers, groupby=cluster_col, method="t-test")
    
    print(f"  Creating CSV: {output_csv}")
    result = adata_for_markers.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    dat = pd.DataFrame({
        group + '_' + key[:1]: result[key][group]
        for group in groups
        for key in ['names', 'logfoldchanges', 'scores', 'pvals']
    })
    
    # Filter out rows where log fold changes are missing (critical for SCSA)
    # Keep at least the top genes even if some have missing logFC
    initial_rows = len(dat)
    for group in groups:
        lfc_col = group + '_l'
        if lfc_col in dat.columns:
            # For each cluster, keep genes with valid logFC or top genes by score
            valid_mask = dat[lfc_col].notna()
            if valid_mask.sum() < 100:  # If too few valid, keep top 100 by score
                score_col = group + '_s'
                if score_col in dat.columns:
                    top_genes = dat.nlargest(100, score_col).index
                    valid_mask = dat.index.isin(top_genes) | valid_mask
            # For now, keep all genes but warn about missing logFC
            missing_count = (~valid_mask).sum()
            if missing_count > 0:
                print(f"    WARNING: {missing_count} genes in cluster {group} have missing log fold changes")
    
    dat.to_csv(output_csv, index=True)
    print(f"  ✓ CSV created with {len(dat)} genes and {len(dat.columns)} columns")
    
    # Verify log fold changes
    lfc_cols = [c for c in dat.columns if c.endswith('_l')]
    if lfc_cols:
        total_cells = len(dat) * len(lfc_cols)
        valid_lfc = sum(dat[col].notna().sum() for col in lfc_cols)
        pct_valid = 100 * valid_lfc / total_cells
        print(f"  Log fold changes: {valid_lfc}/{total_cells} valid ({pct_valid:.1f}%)")
        if pct_valid < 50:
            print(f"  WARNING: Less than 50% of log fold changes are valid!")
            print(f"  This may cause SCSA to miss important marker genes.")
    
    return adata, cluster_col


def run_scsa(csv_path, db_path, output_path, source='scanpy', 
             foldchange=1.5, pvalue=0.01, gensymbol=True, 
             species='Human', tissue='All', celltype='normal', 
             target='cellmarker', outfmt='txt', python_cmd=None):
    """
    Run SCSA.py on the CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV
    db_path : str
        Path to SCSA database
    output_path : str
        Path to SCSA output file
    source : str
        Source format (scanpy, seurat, scran, cellranger)
    foldchange : float
        Fold change threshold
    pvalue : float
        P-value threshold
    gensymbol : bool
        Use gene symbols instead of Ensembl IDs
    species : str
        Species (Human, Mouse)
    tissue : str
        Tissue type
    celltype : str
        Cell type (normal, cancer)
    target : str
        Target database (cellmarker, cancersea)
    outfmt : str
        Output format (txt, ms-excel)
    python_cmd : str, optional
        Python executable to use (default: python3)
    """
    print(f"\nStep 2: Running SCSA on {csv_path}")
    
    # Determine Python executable
    if python_cmd is None:
        # Try to use same Python as current process
        python_cmd = sys.executable
    
    # Use 'SCSA.py' directly - run.sh already changes to SCSA directory
    cmd = [
        python_cmd, 'SCSA.py',
        '-d', db_path,
        '-i', csv_path,
        '-s', source,
        '-o', output_path,
        '-f', str(foldchange),
        '-p', str(pvalue),
        '-g', species,
        '-k', tissue,
        '-T', celltype,
        '-t', target,
        '-m', outfmt
    ]
    
    if gensymbol:
        cmd.append('-E')
    
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ ERROR: SCSA failed!")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    
    # Check if output was actually created and has content
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            print(f"  WARNING: SCSA output file is empty: {output_path}")
            print(f"  This usually means SCSA found no matching cell types.")
            print(f"  SCSA stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
            print(f"  SCSA stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
        else:
            print(f"  ✓ SCSA completed successfully (output file size: {file_size} bytes)")
    else:
        print(f"  WARNING: SCSA output file was not created: {output_path}")
        print(f"  SCSA stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
        print(f"  SCSA stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
    
    return output_path


def map_to_cells(scsa_result_file, adata, cluster_col, true_label_col=None):
    """
    Map SCSA cluster-level predictions to individual cells.
    
    Parameters:
    -----------
    scsa_result_file : str
        Path to SCSA output file
    adata : AnnData
        AnnData object with cell information
    cluster_col : str
        Name of cluster column
    true_label_col : str, optional
        Name of column containing true labels (if available)
    
    Returns:
    --------
    pd.DataFrame with cell_id, true_labels, predicted_labels
    """
    print(f"\nStep 3: Mapping predictions to cells")
    
    # Check if file exists and is not empty
    if not os.path.exists(scsa_result_file):
        print(f"  ERROR: SCSA result file not found: {scsa_result_file}")
        sys.exit(1)
    
    if os.path.getsize(scsa_result_file) == 0:
        print(f"  ERROR: SCSA result file is empty: {scsa_result_file}")
        sys.exit(1)
    
    # Read SCSA results
    try:
        scsa_results = pd.read_csv(scsa_result_file, sep='\t')
    except Exception as e:
        print(f"  ERROR: Failed to read SCSA results file: {e}")
        print(f"  File path: {scsa_result_file}")
        # Try to show first few lines for debugging
        try:
            with open(scsa_result_file, 'r') as f:
                lines = f.readlines()[:10]
                print(f"  First 10 lines of file:")
                for i, line in enumerate(lines, 1):
                    print(f"    {i}: {line.strip()}")
        except:
            pass
        sys.exit(1)
    
    # Check if file has expected columns
    expected_cols = ['Cell Type', 'Z-score', 'Cluster']
    if not all(col in scsa_results.columns for col in expected_cols):
        print(f"  ERROR: SCSA results file missing expected columns")
        print(f"  Expected columns: {expected_cols}")
        print(f"  Found columns: {list(scsa_results.columns)}")
        print(f"  File path: {scsa_result_file}")
        print(f"  First few rows:")
        print(scsa_results.head())
        sys.exit(1)
    
    # Check if file is empty (only header)
    if len(scsa_results) == 0:
        print(f"  WARNING: SCSA results file contains no data (only header)")
        print(f"  File path: {scsa_result_file}")
        print(f"  This usually means SCSA found no matching cell types for any clusters")
        cluster_predictions = {}
    else:
        # Get the top prediction per cluster (highest Z-score)
        cluster_predictions = {}
        for cluster in scsa_results['Cluster'].unique():
            cluster_data = scsa_results[scsa_results['Cluster'] == cluster]
            # Convert Z-score to numeric, handling any non-numeric values
            cluster_data['Z-score'] = pd.to_numeric(cluster_data['Z-score'], errors='coerce')
            # Drop rows with NaN Z-scores
            cluster_data = cluster_data.dropna(subset=['Z-score'])
            if len(cluster_data) > 0:
                top_prediction = cluster_data.loc[cluster_data['Z-score'].idxmax()]
                cluster_predictions[str(cluster)] = top_prediction['Cell Type']
    
    print(f"  Found predictions for {len(cluster_predictions)} clusters")
    if len(cluster_predictions) > 0:
        print(f"  Clusters with predictions: {sorted(cluster_predictions.keys())}")
    
    # Map predictions to cells
    cell_data = []
    unmapped_clusters = set()
    
    for cell_idx, cluster in enumerate(adata.obs[cluster_col]):
        cluster_str = str(cluster)
        cell_id = adata.obs.index[cell_idx]
        
        # Get true label (required)
        if true_label_col not in adata.obs.columns:
            print(f"\nERROR: Column '{true_label_col}' not found in adata.obs")
            print(f"Available columns: {list(adata.obs.columns)}")
            sys.exit(1)
        true_label = adata.obs[true_label_col].iloc[cell_idx]
        
        # Get predicted label
        if cluster_str in cluster_predictions:
            predicted_label = cluster_predictions[cluster_str]
        else:
            unmapped_clusters.add(cluster_str)
            predicted_label = 'Unknown'
        
        cell_data.append({
            'cell_id': cell_id,
            'true_labels': true_label,
            'predicted_labels': predicted_label
        })
    
    if unmapped_clusters:
        print(f"  WARNING: {len(unmapped_clusters)} clusters not found in SCSA results:")
        print(f"    {sorted(unmapped_clusters)}")
    
    result_df = pd.DataFrame(cell_data)
    print(f"  ✓ Mapped predictions to {len(result_df)} cells")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Complete SCSA pipeline: Create CSV, run SCSA, map to cells"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input h5ad file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to output CSV file (base name, will append annotation key)'
    )
    parser.add_argument(
        '-d', '--db',
        default='whole.db',
        help='Path to SCSA database (default: whole.db)'
    )
    parser.add_argument(
        '--annotation_key',
        required=True,
        help='Key in adata.obs for true cell type labels'
    )
    parser.add_argument(
        '--cluster-col',
        default='louvain',
        help='Cluster column name (default: louvain)'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=1.0,
        help='Resolution for Louvain clustering (default: 1.0)'
    )
    parser.add_argument(
        '-f', '--foldchange',
        type=float,
        default=1.5,
        help='Fold change threshold (default: 1.5)'
    )
    parser.add_argument(
        '-p', '--pvalue',
        type=float,
        default=0.01,
        help='P-value threshold (default: 0.01)'
    )
    parser.add_argument(
        '-g', '--species',
        default='Human',
        help='Species (default: Human)'
    )
    parser.add_argument(
        '-k', '--tissue',
        default='All',
        help='Tissue type (default: All)'
    )
    parser.add_argument(
        '--no-gensymbol',
        action='store_true',
        help='Use Ensembl IDs instead of gene symbols'
    )
    parser.add_argument(
        '--temp-dir',
        default=None,
        help='Directory for temporary files (default: same as output)'
    )
    
    args = parser.parse_args()
    
    # Set up temporary file paths
    if args.temp_dir is None:
        temp_dir = os.path.dirname(args.output) or '.'
    else:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    # Set up output path info (similar to cellwhisperer)
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    output_stem = os.path.splitext(os.path.basename(output_path))[0]
    output_suffix = os.path.splitext(output_path)[1] or '.csv'
    
    # Optional clustering context: when CONTEXT_H5AD points to a larger (up to 100k) h5ad that
    # CONTAINS the eval subset, cluster + rank markers + SCSA-score on that context (so rare cell
    # types can form their own cluster), then keep only the eval-subset cells. Without it, cluster
    # the eval subset itself (original behavior). The scored set is always exactly the eval subset.
    context_h5ad = os.environ.get("CONTEXT_H5AD", "")
    use_context = bool(context_h5ad) and os.path.exists(context_h5ad)
    cluster_input = context_h5ad if use_context else args.input
    if use_context:
        print(f"SCSA: clustering context '{context_h5ad}', scoring eval cells from '{args.input}'.")

    base_name = os.path.splitext(os.path.basename(cluster_input))[0]
    temp_csv = os.path.join(temp_dir, f"{base_name}_markers.csv")
    temp_scsa_output = os.path.join(temp_dir, f"{base_name}_scsa_result.txt")

    try:
        # Step 1: Create CSV from h5ad (only needs to be done once)
        print("="*60)
        print("STEP 1: Creating marker gene CSV from h5ad")
        print("="*60)
        adata, cluster_col = create_scanpy_csv(
            cluster_input, temp_csv,
            cluster_col=args.cluster_col,
            resolution=args.resolution
        )
        
        # Step 2: Run SCSA (only needs to be done once)
        print("\n" + "="*60)
        print("STEP 2: Running SCSA annotation")
        print("="*60)
        run_scsa(
            temp_csv, args.db, temp_scsa_output,
            source='scanpy',
            foldchange=args.foldchange,
            pvalue=args.pvalue,
            gensymbol=not args.no_gensymbol,
            species=args.species,
            tissue=args.tissue
        )
        
        # Step 3: Map to cells
        print("\n" + "="*60)
        print("STEP 3: Mapping predictions to cells")
        print("="*60)
        
        annotation_key = args.annotation_key
        print(f"\n--- Processing annotation key: {annotation_key} ---")
        
        # Check if annotation key exists
        if annotation_key not in adata.obs.columns:
            print(f"Error: '{annotation_key}' not found in adata.obs.columns")
            print(f"Available columns: {list(adata.obs.columns)}")
            sys.exit(1)
        
        # Map to cells using this annotation key (over the clustering source)
        result_df = map_to_cells(
            temp_scsa_output, adata, cluster_col,
            true_label_col=annotation_key
        )

        # In context mode, keep only the eval-subset cells, propagating their context cluster's
        # label. The context is union-constructed to contain every eval cell.
        if use_context:
            import anndata as _ad
            eval_ids = _ad.read_h5ad(args.input, backed="r").obs_names.astype(str).tolist()
            result_df['cell_id'] = result_df['cell_id'].astype(str)
            n_ctx = len(result_df)
            missing = set(eval_ids) - set(result_df['cell_id'])
            if missing:
                print(f"  WARNING: {len(missing)}/{len(eval_ids)} eval cells absent from the "
                      f"clustering context -> Unknown.")
            result_df = (result_df.drop_duplicates('cell_id').set_index('cell_id')
                                  .reindex(eval_ids).reset_index())
            result_df['predicted_labels'] = result_df['predicted_labels'].fillna('Unknown')
            print(f"  Context filter: {n_ctx} context cells -> {len(result_df)} eval cells")

        # Save final output
        print(f"\nSaving results to {args.output}")
        result_df.to_csv(args.output, index=False)
        print(f"  ✓ Saved {len(result_df)} cell predictions")
        
        # Print summary
        print(f"\nSummary for {annotation_key}:")
        print(f"  Total cells: {len(result_df)}")
        print(f"  Unique true labels: {len(result_df['true_labels'].unique())}")
        print(f"  Unique predicted labels: {len(result_df['predicted_labels'].unique())}")
        
        # Calculate accuracy
        accuracy = (result_df['true_labels'] == result_df['predicted_labels']).mean()
        print(f"  Overall accuracy: {accuracy:.3f}")
        
        # Keep temporary files for inspection (don't delete them)
        print(f"\nTemporary files kept for inspection:")
        if os.path.exists(temp_csv):
            print(f"  Marker genes CSV: {temp_csv}")
        if os.path.exists(temp_scsa_output):
            print(f"  SCSA results: {temp_scsa_output}")
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Output files saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Pipeline failed!")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
