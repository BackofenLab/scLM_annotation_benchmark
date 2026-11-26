#!/usr/bin/env python3
"""Annotate the cells of a test set with SCSA.

Clusters the cells (Louvain), writes the marker genes of every cluster (scanpy t-test) to a CSV
in SCSA's scanpy format, runs SCSA.py on it and gives every cell the cell type with the highest
Z-score of its cluster. If CONTEXT_H5AD names a clustering context that contains the test cells
(bin/build_context.py), the context is clustered and only the test cells are reported. If
LABEL_CONSTRAINT names a label table, only the candidates that count for the true cell types
compete (bin/util/label_constraint.py). Writes a CSV with cell_id, true_labels and
predicted_labels; the marker CSV and the SCSA result are kept next to it.

run.sh calls this script from the SCSA checkout, since SCSA.py is called by its relative path.
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
    """Set var_names to the first column of GENE_NAME_COLUMNS in adata.var and return its name,
    or None if there is none. Empty names keep the original var_name."""
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
    """Cluster the cells of h5ad_path unless obs has cluster_col, write the marker genes of every
    cluster in SCSA's scanpy format to output_csv and return the AnnData and cluster_col.
    
    Clustering: n_top_genes highly variable genes, scaling, n_pcs principal components, a
    neighbour graph with n_neighbors and Louvain at the given resolution."""
    print(f"Step 1: Reading h5ad file: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    
    needs_clustering = cluster_col not in adata.obs.columns

    # SCSA matches the markers by gene symbol.
    set_gene_index_from_available_columns(adata)
    
    if needs_clustering:
        print(f"  Finding {n_top_genes} highly variable genes...")
        # flavor='seurat' works on log-normalized X. 'seurat_v3' needs raw counts, which the
        # clustering contexts do not contain.
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()
        
        # The marker genes are computed on the unscaled data.
        adata_for_markers = adata.copy()
        
        print("  Scaling...")
        sc.pp.scale(adata, max_value=10)
        
        print(f"  Computing PCA ({n_pcs} components)...")
        sc.tl.pca(adata, svd_solver="arpack", n_comps=n_pcs)
        
        print(f"  Building neighborhood graph (k={n_neighbors})...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
        
        print(f"  Running Louvain clustering (resolution={resolution})...")
        sc.tl.louvain(adata, resolution=resolution, key_added=cluster_col, random_state=random_state)
        
        adata_for_markers.obs[cluster_col] = adata.obs[cluster_col]
    else:
        print(f"  Using existing cluster column: '{cluster_col}'")
        # Existing clusters: the marker genes come from the 'log1p' layer or adata.raw if
        # present, else from X as it is.
        adata_for_markers = adata.copy()
        if 'log1p' in adata.layers:
            adata_for_markers.X = adata.layers['log1p'].copy()
        elif adata.raw is not None:
            adata_for_markers = adata.raw.to_adata()
            adata_for_markers.obs[cluster_col] = adata.obs[cluster_col]
            set_gene_index_from_available_columns(adata_for_markers)
    
    print(f"  Found {len(adata.obs[cluster_col].unique())} clusters")
    
    print("  Computing marker genes per cluster (on normalized data)...")
    # SCSA filters the markers by log fold change, which needs unscaled data.
    sc.tl.rank_genes_groups(adata_for_markers, groupby=cluster_col, method="t-test")
    
    print(f"  Creating CSV: {output_csv}")
    result = adata_for_markers.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    dat = pd.DataFrame({
        group + '_' + key[:1]: result[key][group]
        for group in groups
        for key in ['names', 'logfoldchanges', 'scores', 'pvals']
    })
    
    # Warns about clusters with missing log fold changes; no gene is removed, so valid_mask
    # only sets the count in the warning.
    for group in groups:
        lfc_col = group + '_l'
        if lfc_col in dat.columns:
            valid_mask = dat[lfc_col].notna()
            if valid_mask.sum() < 100:
                score_col = group + '_s'
                if score_col in dat.columns:
                    top_genes = dat.nlargest(100, score_col).index
                    valid_mask = dat.index.isin(top_genes) | valid_mask
            missing_count = (~valid_mask).sum()
            if missing_count > 0:
                print(f"    WARNING: {missing_count} genes in cluster {group} have missing log fold changes")
    
    dat.to_csv(output_csv, index=True)
    print(f"  CSV created with {len(dat)} genes and {len(dat.columns)} columns")
    
    # Share of valid log fold changes over all genes and clusters
    lfc_cols = [c for c in dat.columns if c.endswith('_l')]
    if lfc_cols:
        total_cells = len(dat) * len(lfc_cols)
        valid_lfc = sum(dat[col].notna().sum() for col in lfc_cols)
        pct_valid = 100 * valid_lfc / total_cells
        print(f"  Log fold changes: {valid_lfc}/{total_cells} valid ({pct_valid:.1f}%)")
        if pct_valid < 50:
            print("  WARNING: Less than 50% of log fold changes are valid.")
            print("  This may cause SCSA to miss important marker genes.")
    
    return adata, cluster_col


def run_scsa(csv_path, db_path, output_path, source='scanpy', 
             foldchange=1.5, pvalue=0.01, gensymbol=True, 
             species='Human', tissue='All', celltype='normal', 
             target='cellmarker', outfmt='txt', python_cmd=None):
    """Run SCSA.py on the marker CSV, writing its result to output_path, and exit if SCSA fails.
    
    The arguments are SCSA's options -s, -f, -p, -E (gene symbols), -g, -k, -T, -t and -m;
    python_cmd defaults to the interpreter of this script."""
    print(f"\nStep 2: Running SCSA on {csv_path}")
    
    if python_cmd is None:
        python_cmd = sys.executable
    
    # Relative path: run.sh starts this script in the SCSA checkout.
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
        print("  ERROR: SCSA failed.")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    
    # SCSA can exit without error and still write no result.
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            print(f"  WARNING: SCSA output file is empty: {output_path}")
            print("  This usually means SCSA found no matching cell types.")
            print(f"  SCSA stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
            print(f"  SCSA stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
        else:
            print(f"  SCSA completed (output file size: {file_size} bytes)")
    else:
        print(f"  WARNING: SCSA output file was not created: {output_path}")
        print(f"  SCSA stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
        print(f"  SCSA stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
    
    return output_path


def map_to_cells(scsa_result_file, adata, cluster_col, true_label_col=None, kept=None):
    """Give every cell of adata the cell type with the highest Z-score of its cluster, or
    'Unknown', and return cell_id, true_labels (from true_label_col) and predicted_labels.
    
    With kept, a dict of SCSA cell types to listed types, only the cell types in kept compete and
    the listed type is reported."""
    print("\nStep 3: Mapping predictions to cells")
    
    if not os.path.exists(scsa_result_file):
        print(f"  ERROR: SCSA result file not found: {scsa_result_file}")
        sys.exit(1)
    
    if os.path.getsize(scsa_result_file) == 0:
        print(f"  ERROR: SCSA result file is empty: {scsa_result_file}")
        sys.exit(1)
    
    try:
        scsa_results = pd.read_csv(scsa_result_file, sep='\t')
    except Exception as e:
        print(f"  ERROR: Failed to read SCSA results file: {e}")
        print(f"  File path: {scsa_result_file}")
        try:
            with open(scsa_result_file, 'r') as f:
                lines = f.readlines()[:10]
                print("  First 10 lines of file:")
                for i, line in enumerate(lines, 1):
                    print(f"    {i}: {line.strip()}")
        except:
            pass
        sys.exit(1)
    
    expected_cols = ['Cell Type', 'Z-score', 'Cluster']
    if not all(col in scsa_results.columns for col in expected_cols):
        print("  ERROR: SCSA results file missing expected columns")
        print(f"  Expected columns: {expected_cols}")
        print(f"  Found columns: {list(scsa_results.columns)}")
        print(f"  File path: {scsa_result_file}")
        print("  First few rows:")
        print(scsa_results.head())
        sys.exit(1)
    
    # Constrained mode: only the candidates in ``kept`` compete for a cluster, answered as the listed
    # cell type they count for.
    if kept is not None:
        scsa_results = scsa_results[scsa_results['Cell Type'].isin(kept)]
        scsa_results = scsa_results.assign(**{'Cell Type': scsa_results['Cell Type'].map(kept)})

    # No rows: an empty SCSA result, or no candidate left in constrained mode.
    if len(scsa_results) == 0:
        print("  WARNING: SCSA results file contains no data (only header)")
        print(f"  File path: {scsa_result_file}")
        print("  This usually means SCSA found no matching cell types for any clusters")
        cluster_predictions = {}
    else:
        # Top cell type per cluster (highest Z-score); non-numeric Z-scores are ignored.
        cluster_predictions = {}
        for cluster in scsa_results['Cluster'].unique():
            cluster_data = scsa_results[scsa_results['Cluster'] == cluster]
            cluster_data['Z-score'] = pd.to_numeric(cluster_data['Z-score'], errors='coerce')
            cluster_data = cluster_data.dropna(subset=['Z-score'])
            if len(cluster_data) > 0:
                top_prediction = cluster_data.loc[cluster_data['Z-score'].idxmax()]
                cluster_predictions[str(cluster)] = top_prediction['Cell Type']
    
    print(f"  Found predictions for {len(cluster_predictions)} clusters")
    if len(cluster_predictions) > 0:
        print(f"  Clusters with predictions: {sorted(cluster_predictions.keys())}")
    
    cell_data = []
    unmapped_clusters = set()
    
    for cell_idx, cluster in enumerate(adata.obs[cluster_col]):
        cluster_str = str(cluster)
        cell_id = adata.obs.index[cell_idx]
        
        if true_label_col not in adata.obs.columns:
            print(f"\nERROR: Column '{true_label_col}' not found in adata.obs")
            print(f"Available columns: {list(adata.obs.columns)}")
            sys.exit(1)
        true_label = adata.obs[true_label_col].iloc[cell_idx]
        
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
    print(f"  Mapped predictions to {len(result_df)} cells")
    
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
    
    # Directory of the marker CSV and the SCSA result
    if args.temp_dir is None:
        temp_dir = os.path.dirname(args.output) or '.'
    else:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    
    # With CONTEXT_H5AD, a larger sample that contains the test cells (bin/build_context.py) is
    # clustered and annotated, so that rare cell types can form their own cluster; only the test
    # cells are reported. Without it, the test set itself is clustered.
    context_h5ad = os.environ.get("CONTEXT_H5AD", "")
    use_context = bool(context_h5ad) and os.path.exists(context_h5ad)
    cluster_input = context_h5ad if use_context else args.input
    if use_context:
        print(f"SCSA: clustering context '{context_h5ad}', scoring eval cells from '{args.input}'.")

    base_name = os.path.splitext(os.path.basename(cluster_input))[0]
    temp_csv = os.path.join(temp_dir, f"{base_name}_markers.csv")
    temp_scsa_output = os.path.join(temp_dir, f"{base_name}_scsa_result.txt")

    try:
        print("="*60)
        print("STEP 1: Creating marker gene CSV from h5ad")
        print("="*60)
        adata, cluster_col = create_scanpy_csv(
            cluster_input, temp_csv,
            cluster_col=args.cluster_col,
            resolution=args.resolution
        )
        
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
        
        print("\n" + "="*60)
        print("STEP 3: Mapping predictions to cells")
        print("="*60)
        
        annotation_key = args.annotation_key
        print(f"\nAnnotation key: {annotation_key}")
        
        if annotation_key not in adata.obs.columns:
            print(f"Error: '{annotation_key}' not found in adata.obs.columns")
            print(f"Available columns: {list(adata.obs.columns)}")
            sys.exit(1)
        
        # Constrained mode (LABEL_CONSTRAINT): only the candidates that count for one of the true
        # cell types compete (bin/util/label_constraint.py). SCSA ranks a cluster's candidates by
        # their own markers, so this picks what an SCSA run on the kept types alone picks.
        kept = None
        constraint = os.environ.get("LABEL_CONSTRAINT")
        if constraint:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "..", "..", "..", "bin"))
            from util.label_constraint import constrain
            truth = (ad.read_h5ad(args.input, backed="r").obs if use_context else adata.obs)
            candidates = pd.read_csv(temp_scsa_output, sep='\t')['Cell Type'].unique()
            kept = constrain(candidates, constraint, "SCSA", truth[annotation_key].astype(str))
            if not kept:
                sys.exit("no SCSA candidate counts for the listed cell types")

        # Labels for all clustered cells
        result_df = map_to_cells(
            temp_scsa_output, adata, cluster_col,
            true_label_col=annotation_key, kept=kept
        )

        # Context mode: keep only the test cells, each with the label of its context cluster.
        # The context contains every test cell (bin/build_context.py).
        if use_context:
            import anndata as _ad
            eval_obs = _ad.read_h5ad(args.input, backed="r").obs
            eval_ids = eval_obs.index.astype(str).tolist()
            result_df['cell_id'] = result_df['cell_id'].astype(str)
            n_ctx = len(result_df)
            missing = set(eval_ids) - set(result_df['cell_id'])
            if missing:
                print(f"  WARNING: {len(missing)}/{len(eval_ids)} eval cells absent from the "
                      f"clustering context; they are labeled Unknown.")
            result_df = (result_df.drop_duplicates('cell_id').set_index('cell_id')
                                  .reindex(eval_ids).reset_index())
            result_df['predicted_labels'] = result_df['predicted_labels'].fillna('Unknown')
            # True labels come from the test set, as for every other method; the labels of the
            # context can name a type differently.
            result_df['true_labels'] = eval_obs[annotation_key].to_numpy()
            print(f"  Context filter: {n_ctx} context cells, {len(result_df)} eval cells kept")

        if kept is not None:
            result_df['reachable'] = result_df['true_labels'].astype(str).isin(set(kept.values()))

        print(f"\nSaving results to {args.output}")
        result_df.to_csv(args.output, index=False)
        print(f"  Saved {len(result_df)} cell predictions")
        
        print(f"\nSummary for {annotation_key}:")
        print(f"  Total cells: {len(result_df)}")
        print(f"  Unique true labels: {len(result_df['true_labels'].unique())}")
        print(f"  Unique predicted labels: {len(result_df['predicted_labels'].unique())}")
        
        # Exact string matches only; the evaluation maps labels to the Cell Ontology.
        accuracy = (result_df['true_labels'] == result_df['predicted_labels']).mean()
        print(f"  Overall accuracy: {accuracy:.3f}")
        
        print("\nTemporary files kept for inspection:")
        if os.path.exists(temp_csv):
            print(f"  Marker genes CSV: {temp_csv}")
        if os.path.exists(temp_scsa_output):
            print(f"  SCSA results: {temp_scsa_output}")
        
        print("\nSCSA annotation completed.")
        print(f"  Output files saved to: {output_dir}")
        
    except Exception as e:
        print("\nERROR: SCSA annotation failed.")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
