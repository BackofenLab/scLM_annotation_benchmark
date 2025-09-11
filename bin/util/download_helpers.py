"""
Helper functions for the download script to apply ontology mapping.
"""
from pathlib import Path
from typing import List
import anndata as ad

from .ontology_helpers import apply_ontology_mapping_to_adata


def apply_ontology_mapping_to_dataset(
    adata: ad.AnnData,
    annotation_keys: List[str],
    cache_dir: Path,
    verbose: bool = True
) -> ad.AnnData:
    """
    Apply ontology mapping to a dataset during the download process.
    
    This is a simple wrapper around apply_ontology_mapping_to_adata that's
    specifically designed for use in the download script.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to process
    annotation_keys : List[str]
        List of annotation column names to map to ontology terms
    cache_dir : Path
        Cache directory for ontology files
    verbose : bool, default True
        If True, print progress information
        
    Returns
    -------
    ad.AnnData
        AnnData object with ontology-mapped annotations
    """
    if verbose:
        print(f"[INFO] Applying ontology mapping to annotation keys: {annotation_keys}")
    
    adata = apply_ontology_mapping_to_adata(
        adata=adata,
        annotation_keys=annotation_keys,
        cache_dir=cache_dir,
        force_remapping=False,
        verbose=verbose
    )
    
    if verbose:
        print("[INFO] Ontology mapping completed successfully")
    
    return adata
