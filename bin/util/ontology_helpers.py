"""
Helper functions for ontology-based cell type mapping and AnnData processing.

This module provides reusable functions for:
1. Checking which cell types can be mapped to ontology terms
2. Creating/updating ontology mapping CSV files
3. Applying ontology mappings to AnnData objects
"""
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import anndata as ad
import pronto

from .cell_ontology_similarity import build_label_index, map_string_to_term, download_cl_ontology


def check_ontology_mapping_coverage(
    cell_types: List[str],
    ontology_path: str
) -> Tuple[Set[str], Set[str], Dict]:
    """
    Check which cell types can be mapped to ontology terms.
    
    Parameters
    ----------
    cell_types : List[str]
        List of cell type names to check
    ontology_path : str
        Path to the ontology file (OWL/JSON)
        
    Returns
    -------
    Tuple[Set[str], Set[str], Dict]
        - Set of mapped cell types
        - Set of unmapped cell types  
        - Label index dictionary for future use
    """
    try:
        ont = pronto.Ontology(ontology_path)
        label_index = build_label_index(ont)
    except Exception as e:
        raise RuntimeError(f"Failed to load ontology from {ontology_path}: {e}")
    
    mapped_types = set()
    unmapped_types = set()
    
    for cell_type in cell_types:
        if map_string_to_term(cell_type, label_index) is not None:
            mapped_types.add(cell_type)
        else:
            unmapped_types.add(cell_type)
    
    return mapped_types, unmapped_types, label_index


def create_or_update_ontology_mappings(
    unmapped_cell_types: List[str],
    cache_dir: Path,
    force_remapping: bool = False,
    origin: str = "unknown"
) -> Dict[str, str]:
    """
    Create or update ontology mappings for unmapped cell types.
    
    Parameters
    ----------
    unmapped_cell_types : List[str]
        List of cell types that need mapping
    cache_dir : Path
        Cache directory for ontology files and mappings
    force_remapping : bool, default False
        If True, remap all labels even if they already exist
    origin : str, default "unknown"
        Origin of the cell type labels (e.g., dataset name or model name)
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping original cell type -> chosen_match
        
    Raises
    ------
    RuntimeError
        If ontology mapping creation fails
    """
    if not unmapped_cell_types:
        return {}
    
    try:
        # Import here to avoid circular imports
        from .ontology_mapper import map_cell_types_to_ontology, load_ontology_mappings
        
        print(f"Creating ontology mappings for {len(unmapped_cell_types)} unmapped cell types...")
        
        # Create/update the mapping CSV
        mapping_csv_path = map_cell_types_to_ontology(
            predicted_labels=unmapped_cell_types,
            cache_dir=cache_dir,
            force_remapping=force_remapping,
            origin=origin
        )
        
        # Load the mappings
        mappings = load_ontology_mappings(cache_dir)  # CSV is stored in parent directory
        if mappings is None:
            raise RuntimeError("Failed to load ontology mappings from CSV file")
        return mappings
        
    except Exception as e:
        raise RuntimeError(f"Failed to create ontology mappings: {e}")


def apply_ontology_mappings_to_cell_types(
    cell_types: List[str],
    unmapped_types: Set[str],
    mappings: Dict[str, str],
    label_index: Dict,
    verbose: bool = True
) -> Tuple[List[str], Dict[str, str]]:
    """
    Apply ontology mappings to replace unmapped cell types.
    
    Parameters
    ----------
    cell_types : List[str]
        Original list of cell type names
    unmapped_types : Set[str]
        Set of cell types that couldn't be mapped directly
    mappings : Dict[str, str]
        Dictionary mapping original -> chosen_match from CSV
    label_index : Dict
        Ontology label index for validation
    verbose : bool, default True
        If True, print mapping results
        
    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        - List of updated cell type names
        - Dictionary of successful replacements (original -> new)
    """
    updated_cell_types = cell_types.copy()
    successful_replacements = {}
    
    if verbose:
        print("Applying ontology mappings...")
    
    for i, cell_type in enumerate(cell_types):
        if cell_type in unmapped_types and cell_type in mappings:
            chosen_match = mappings[cell_type].strip()
            
            if chosen_match and chosen_match != cell_type:
                # Verify the chosen_match can be mapped to ontology
                if map_string_to_term(chosen_match, label_index) is not None:
                    updated_cell_types[i] = chosen_match
                    successful_replacements[cell_type] = chosen_match
                else:
                    if verbose:
                        print(f"  Failed: '{cell_type}' -> '{chosen_match}' (not in ontology)")
            else:
                if verbose:
                    if not chosen_match:
                        print(f"  Failed: '{cell_type}' (no chosen_match available)")
                    else:
                        print(f"  Failed: '{cell_type}' (chosen_match same as original)")
    
    if verbose and successful_replacements:
        print(f"Successfully mapped {len(successful_replacements)} cell types")
    
    return updated_cell_types, successful_replacements


def apply_ontology_mapping_to_cell_type_list(
    cell_types: List[str],
    cache_dir: Path,
    force_remapping: bool = False,
    verbose: bool = True,
    origin: str = "unknown",
    label_index: Optional[Dict] = None,
    discover_only: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Apply ontology mapping to a list of cell type labels.
    
    This is the core mapping function that can be used independently of AnnData.
    
    Parameters
    ----------
    cell_types : List[str]
        List of cell type labels to map
    cache_dir : Path
        Cache directory for ontology files
    force_remapping : bool, default False
        If True, remap all labels even if they already exist in CSV
    verbose : bool, default True
        If True, print progress information
    origin : str, default "unknown"
        Origin of the cell type labels (e.g., dataset name or model name)
        
    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        - List of mapped cell type labels
        - Dictionary of successful replacements (original -> new)
    """
    if verbose:
        print(f"Applying ontology mapping to {len(set(cell_types))} unique cell types...")
    
    # Download ontology if needed
    try:
        cl_file_path = download_cl_ontology(cache_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to download ontology: {e}")
    
    # Get unique cell types for processing
    unique_cell_types = list(set(cell_types))

    # Check ontology mapping coverage. If a preloaded label_index is supplied (e.g. by
    # map_predictions.py, which parses the OWL once per process), reuse it instead of calling
    # check_ontology_mapping_coverage, which would re-parse the 63MB OWL on every call.
    try:
        if label_index is None:
            mapped_types, unmapped_types, label_index = check_ontology_mapping_coverage(
                cell_types=unique_cell_types,
                ontology_path=str(cl_file_path)
            )
        else:
            mapped_types = {c for c in unique_cell_types
                            if map_string_to_term(c, label_index) is not None}
            unmapped_types = set(unique_cell_types) - mapped_types
    except Exception as e:
        raise RuntimeError(f"Failed to check ontology coverage: {e}")
    
    if verbose:
        print(f"  Total unique cell types: {len(unique_cell_types)}")
        print(f"  Directly mapped: {len(mapped_types)}")
        print(f"  Need mapping: {len(unmapped_types)}")
    
    # Skip if all types are already mapped
    if not unmapped_types:
        if verbose:
            print("  All cell types are already mapped to ontology")
        return cell_types, {}
    
    # Create/update ontology mappings for unmapped types. This is the discovery + append
    # side-effect: every not-yet-in-CSV, non-CL label is written as reviewed=False for human
    # review. It runs regardless of `discover_only` so novel labels always surface.
    mappings = create_or_update_ontology_mappings(
        unmapped_cell_types=list(unmapped_types),
        cache_dir=cache_dir,
        force_remapping=force_remapping,
        origin=origin
    )

    if mappings is None:
        raise RuntimeError("Failed to create ontology mappings")

    # Discovery-only: append novel labels for review but do NOT rewrite the labels. Used for
    # datasets with apply_ontology_mapping=false (e.g. cellxgene sources that already carry CL
    # terms), where a marker method's free-text labels should still be surfaced for curation
    # without altering the scored labels. Return the inputs unchanged.
    if discover_only:
        if verbose:
            print(f"  discover_only: appended {len(unmapped_types)} novel label(s) for "
                  "review; labels left unchanged.")
        return cell_types, {}

    # Apply mappings to the full cell type list
    updated_cell_types, replacements = apply_ontology_mappings_to_cell_types(
        cell_types=cell_types,
        unmapped_types=unmapped_types,
        mappings=mappings,
        label_index=label_index,
        verbose=verbose
    )
    
    return updated_cell_types, replacements


def apply_ontology_mapping_to_adata(
    adata: ad.AnnData,
    annotation_key: str,
    cache_dir: Path,
    force_remapping: bool = False,
    verbose: bool = True,
    origin: str = "unknown"
) -> ad.AnnData:
    """
    Apply ontology mapping to AnnData object annotation columns.
    
    This is a wrapper around apply_ontology_mapping_to_cell_type_list that handles
    AnnData-specific logic like backup columns and column restoration.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to modify
    annotation_key : str
        Annotation column names to process
    cache_dir : Path
        Cache directory for ontology files
    force_remapping : bool, default False
        If True, remap all labels even if they already exist in CSV
    verbose : bool, default True
        If True, print progress information
    origin : str, default "unknown"
        Origin of the cell type labels (e.g., dataset name or model name)
        
    Returns
    -------
    ad.AnnData
        Modified AnnData object with updated annotations
    """
    if verbose:
        print(f"Applying ontology mapping to annotation columns...")
    
    # Process each annotation column
    key = annotation_key
    if key not in adata.obs.columns:
        if verbose:
            print(f"Warning: Column '{key}' not found in adata.obs")
    
    # Check if original_{key} column exists and restore from it
    original_key = f"original_{key}"
    if original_key in adata.obs.columns:
        if verbose:
            print(f"\nRestoring '{key}' from '{original_key}' before reprocessing...")
        adata.obs[key] = adata.obs[original_key].copy()
    
    if verbose:
        print(f"\nProcessing annotation column: {key}")
    
    # Get the cell type list
    original_annotations = adata.obs[key].tolist()
    
    # Apply ontology mapping using the core function
    try:
        updated_annotations, replacements = apply_ontology_mapping_to_cell_type_list(
            cell_types=original_annotations,
            cache_dir=cache_dir,
            force_remapping=force_remapping,
            verbose=verbose,
            origin=origin
        )
    except Exception as e:
        raise RuntimeError(f"Failed to apply ontology mapping to '{key}': {e}")
    
    # Update the AnnData object if there were successful mappings
    if replacements:
        # Store original annotations (only if not already exists)
        if original_key not in adata.obs.columns:
            adata.obs[original_key] = original_annotations.copy()
            if verbose:
                print(f"  Created backup column: '{original_key}'")
        else:
            if verbose:
                print(f"  Using existing backup column: '{original_key}'")
        
        # Update with mapped annotations
        adata.obs[key] = updated_annotations
        
        if verbose:
            print(f"  Updated column: '{key}' with {len(replacements)} mapped cell types")
    else:
        if verbose:
            print(f"  No successful mappings for '{key}'")
    
    return adata
