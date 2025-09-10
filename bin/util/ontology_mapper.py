"""
Ontology-based cell type mapping utilities using omicverse.

This module provides functions to map predicted cell type labels to 
standardized ontology terms using the CellOntologyMapper from omicverse.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Import the download function from cell_ontology_similarity
from .cell_ontology_similarity import download_cl_ontology



def map_cell_types_to_ontology(
    predicted_labels: List[str],
    cache_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    force_remapping: bool = False
) -> Path:
    """
    Map predicted cell type labels to ontology terms using omicverse CellOntologyMapper.
    
    Creates or updates a CSV file with mappings that can be manually edited by users.
    
    Parameters
    ----------
    predicted_labels : List[str]
        List of unique predicted cell type labels to map
    cache_dir : Path
        Directory to store the ontology files and mapping results
    model_name : str, default "all-MiniLM-L6-v2"
        Name of the model to use for semantic matching
    force_remapping : bool, default False
        If True, remap all labels even if they already exist in the CSV
        
    Returns
    -------
    Path
        Path to the CSV file containing the mappings
        
    Raises
    ------
    ImportError
        If omicverse is not installed
    RuntimeError
        If ontology download or mapping fails
    """
    try:
        import omicverse as ov
    except ImportError:
        raise ImportError(
            "omicverse is required for ontology mapping. "
            "Install it with: pip install omicverse or include it into the dependencies of the nextflow process"
        )
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Path for the mapping CSV file
    mapping_csv_path = cache_dir / "cell_type_ontology_mappings.csv"
    
    # Load existing mappings if they exist
    existing_mappings = {}
    if mapping_csv_path.exists() and not force_remapping:
        try:
            existing_df = pd.read_csv(mapping_csv_path, sep=';')
            existing_mappings = dict(zip(existing_df['predicted_label'], existing_df.index))
            print(f"Loaded {len(existing_mappings)} existing mappings from {mapping_csv_path}")
        except Exception as e:
            print(f"Warning: Could not load existing mappings: {e}")
            existing_mappings = {}
    
    # Filter out labels that already have mappings (unless force_remapping is True)
    if not force_remapping and existing_mappings:
        labels_to_map = [label for label in predicted_labels if label not in existing_mappings]
        if len(labels_to_map) < len(predicted_labels):
            print(f"Skipping {len(predicted_labels) - len(labels_to_map)} already mapped labels")
    else:
        labels_to_map = predicted_labels
    
    # If no new labels to map, return existing CSV path
    if not labels_to_map:
        print("No new labels to map. Returning existing mappings.")
        return mapping_csv_path
    
    print(f"Mapping {len(labels_to_map)} cell type labels to ontology terms...")
    
    try:
        # Download Cell Ontology using our existing download function
        print("Downloading Cell Ontology...")
        cl_file_path = download_cl_ontology(
            cache_dir=cache_dir, 
            cl_url="https://purl.obolibrary.org/obo/cl/cl-base.json"
        )
        
        # Initialize the CellOntologyMapper
        print(f"Initializing CellOntologyMapper with model: {model_name}")
        mapper = ov.single.CellOntologyMapper(
            cl_obo_file=str(cl_file_path),
            model_name=model_name,
            local_model_dir=str(cache_dir)
        )
        
        # Map the cell types
        print("Performing ontology mapping...")
        mapping_results = mapper.map_cells(labels_to_map)
        
    except Exception as e:
        raise RuntimeError(f"Failed to map cell types: {e}")
    
    # Process the mapping results
    new_mappings = []
    for label in labels_to_map:
        if label in mapping_results:
            result = mapping_results[label]
            
            # Extract best match
            best_match = result.get("best_match", "")
            
            # Extract top 3 matches
            top3_matches = result.get("top3_matches", [])
            top3_labels = []
            for match in top3_matches:
                if isinstance(match, dict) and "label" in match:
                    top3_labels.append(match["label"])
            
            # Create mapping entry
            mapping_entry = {
                "predicted_label": label,
                "chosen_match": best_match,  # Default to best_match, user can edit
                "best_match": best_match,
                "top3_matches": "; ".join(top3_labels)  # Join with semicolon for alternative matches
            }
            new_mappings.append(mapping_entry)
        else:
            # Handle case where mapping failed
            mapping_entry = {
                "predicted_label": label,
                "chosen_match": "",
                "best_match": "",
                "top3_matches": ""
            }
            new_mappings.append(mapping_entry)
    
    # Create DataFrame for new mappings
    new_df = pd.DataFrame(new_mappings)
    
    # Combine with existing mappings if they exist
    if mapping_csv_path.exists() and not force_remapping:
        try:
            existing_df = pd.read_csv(mapping_csv_path, sep=';')
            # Append new mappings to existing ones
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not merge with existing mappings: {e}")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # Remove duplicates (keep first occurrence)
    combined_df = combined_df.drop_duplicates(subset=['predicted_label'], keep='first')
    
    # Sort by predicted_label for easier manual editing
    combined_df = combined_df.sort_values('predicted_label').reset_index(drop=True)
    
    # Save to CSV with semicolon delimiter
    combined_df.to_csv(mapping_csv_path, sep=';', index=False)
    
    print(f"Ontology mappings saved to: {mapping_csv_path}")
    print(f"Total mappings in file: {len(combined_df)}")
    print(f"New mappings added: {len(new_mappings)}")
    
    return mapping_csv_path


def load_ontology_mappings(cache_dir: Path) -> Optional[Dict[str, str]]:
    """
    Load ontology mappings from the CSV file.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing the mapping CSV file
        
    Returns
    -------
    Optional[Dict[str, str]]
        Dictionary mapping predicted_label -> chosen_match, or None if file doesn't exist
    """
    mapping_csv_path = cache_dir / "cell_type_ontology_mappings.csv"
    
    if not mapping_csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(mapping_csv_path, sep=';')
        # Create mapping from predicted_label to chosen_match
        mappings = dict(zip(df['predicted_label'], df['chosen_match']))
        return mappings
    except Exception as e:
        print(f"Warning: Could not load ontology mappings: {e}")
        return None


def apply_ontology_mappings(
    predicted_labels: List[str], 
    cache_dir: Path,
    fallback_to_original: bool = True
) -> List[str]:
    """
    Apply ontology mappings to a list of predicted labels.
    
    Parameters
    ----------
    predicted_labels : List[str]
        List of predicted cell type labels
    cache_dir : Path
        Directory containing the mapping CSV file
    fallback_to_original : bool, default True
        If True, use original label when no mapping is found
        
    Returns
    -------
    List[str]
        List of mapped labels (or original labels if no mapping found and fallback_to_original=True)
    """
    mappings = load_ontology_mappings(cache_dir)
    
    if mappings is None:
        if fallback_to_original:
            return predicted_labels
        else:
            return [""] * len(predicted_labels)
    
    mapped_labels = []
    for label in predicted_labels:
        mapped_label = mappings.get(label, "")
        if not mapped_label and fallback_to_original:
            mapped_label = label
        mapped_labels.append(mapped_label)
    
    return mapped_labels
