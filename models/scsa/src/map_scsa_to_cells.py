#!/usr/bin/env python3
"""
Map SCSA cluster-level predictions to individual cells.

SCSA provides cell type predictions at the cluster level. This script maps
those predictions back to individual cells based on their cluster assignments.
"""

import pandas as pd
import anndata as ad
import argparse
import sys

def map_scsa_to_cells(scsa_result_file, anndata_file, output_file=None):
    """
    Map SCSA cluster-level predictions to individual cells.
    
    Parameters:
    -----------
    scsa_result_file : str
        Path to SCSA output file (e.g., result.txt)
    anndata_file : str
        Path to AnnData file containing cell-level cluster assignments
    output_file : str, optional
        Path to output file. If None, prints to stdout.
    """
    # Read SCSA results
    scsa_results = pd.read_csv(scsa_result_file, sep='\t')
    
    # Get the top prediction per cluster (highest Z-score)
    cluster_predictions = {}
    for cluster in scsa_results['Cluster'].unique():
        cluster_data = scsa_results[scsa_results['Cluster'] == cluster]
        # Get the cell type with highest Z-score for this cluster
        top_prediction = cluster_data.loc[cluster_data['Z-score'].idxmax()]
        cluster_predictions[str(cluster)] = {
            'cell_type': top_prediction['Cell Type'],
            'z_score': top_prediction['Z-score']
        }
    
    print(f"Found predictions for {len(cluster_predictions)} clusters:")
    for cluster, pred in cluster_predictions.items():
        print(f"  Cluster {cluster}: {pred['cell_type']} (Z-score: {pred['z_score']:.3f})")
    
    # Read AnnData file
    print(f"\nReading AnnData file: {anndata_file}")
    adata = ad.read_h5ad(anndata_file, backed='r')
    adata = adata.to_memory()
    
    # Check for cluster column (louvain is the default from scanpy)
    cluster_col = None
    for col in ['louvain', 'leiden', 'cluster', 'clusters']:
        if col in adata.obs.columns:
            cluster_col = col
            break
    
    if cluster_col is None:
        print("\nERROR: Could not find cluster assignment column in AnnData.")
        print(f"Available columns: {list(adata.obs.columns)}")
        print("\nPlease ensure your AnnData has a cluster column named:")
        print("  - 'louvain' (default from scanpy)")
        print("  - 'leiden' (alternative clustering)")
        print("  - 'cluster' or 'clusters'")
        sys.exit(1)
    
    print(f"Using cluster column: '{cluster_col}'")
    print(f"Number of cells: {adata.n_obs}")
    
    # Map predictions to cells
    cell_predictions = []
    unmapped_clusters = set()
    
    for cell_idx, cluster in enumerate(adata.obs[cluster_col]):
        cluster_str = str(cluster)
        if cluster_str in cluster_predictions:
            pred = cluster_predictions[cluster_str]
            cell_predictions.append({
                'cell_id': adata.obs.index[cell_idx],
                'cluster': cluster_str,
                'predicted_cell_type': pred['cell_type'],
                'z_score': pred['z_score']
            })
        else:
            unmapped_clusters.add(cluster_str)
            cell_predictions.append({
                'cell_id': adata.obs.index[cell_idx],
                'cluster': cluster_str,
                'predicted_cell_type': 'Unknown',
                'z_score': None
            })
    
    if unmapped_clusters:
        print(f"\nWARNING: {len(unmapped_clusters)} clusters in AnnData not found in SCSA results:")
        print(f"  {sorted(unmapped_clusters)}")
        print("  These cells will be labeled as 'Unknown'")
    
    # Create output DataFrame
    cell_df = pd.DataFrame(cell_predictions)
    
    # Add to AnnData if possible
    adata.obs['scsa_predicted_cell_type'] = cell_df['predicted_cell_type'].values
    adata.obs['scsa_z_score'] = cell_df['z_score'].values
    
    # Save or print results
    if output_file:
        if output_file.endswith('.h5ad'):
            print(f"\nSaving AnnData with predictions to: {output_file}")
            adata.write(output_file)
        else:
            print(f"\nSaving cell-level predictions to: {output_file}")
            cell_df.to_csv(output_file, sep='\t', index=False)
    else:
        print("\nCell-level predictions (first 20 rows):")
        print(cell_df.head(20).to_string(index=False))
        print(f"\n... ({len(cell_df)} total cells)")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total cells: {len(cell_df)}")
    print(f"\nPredicted cell types:")
    type_counts = cell_df['predicted_cell_type'].value_counts()
    for cell_type, count in type_counts.items():
        pct = 100 * count / len(cell_df)
        print(f"  {cell_type}: {count} ({pct:.1f}%)")
    
    return cell_df, adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map SCSA cluster-level predictions to individual cells"
    )
    parser.add_argument(
        '-s', '--scsa-result',
        required=True,
        help='Path to SCSA output file (e.g., result.txt)'
    )
    parser.add_argument(
        '-a', '--anndata',
        required=True,
        help='Path to AnnData file (.h5ad) containing cell cluster assignments'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output file path. If .h5ad, saves AnnData. Otherwise saves TSV.'
    )
    
    args = parser.parse_args()
    
    map_scsa_to_cells(args.scsa_result, args.anndata, args.output)

