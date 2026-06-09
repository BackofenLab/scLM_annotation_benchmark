"""
Ontology-based cell type mapping utilities using omicverse.

This module provides functions to map predicted cell type labels to 
standardized ontology terms using the CellOntologyMapper from omicverse.
"""
import os
import sys
from pathlib import Path
from typing import Any, Optional
import pandas as pd

# Import the download function from cell_ontology_similarity
from .cell_ontology_similarity import download_cl_ontology
from .label_normalization import canonicalize_empty_labels


def _read_mapping_csv(mapping_csv_path: Path) -> pd.DataFrame:
    """Read ``cell_type_ontology_mappings.csv`` robustly.

    Uses ``keep_default_na=False`` so blank cells and the literal text ``"nan"``
    are NOT silently converted to float ``NaN`` -- pandas' default ``na_values``
    include ``"nan"``, which previously destroyed the ``predicted_label`` of
    empty-prediction rows on every read/write round-trip (so manual reviews of
    those rows never matched). The ``predicted_label`` column is then
    canonicalized so any legacy blank/``"nan"`` key lines up with the
    ``EMPTY_PREDICTION_LABEL`` sentinel that ``normalize_predicted_labels``
    emits for empty predictions.

    Parameters
    ----------
    mapping_csv_path : Path
        Path to the semicolon-delimited mapping CSV.

    Returns
    -------
    pd.DataFrame
        The mapping table with a stable, string ``predicted_label`` column.
    """
    df = pd.read_csv(mapping_csv_path, sep=';', keep_default_na=False)
    if 'predicted_label' in df.columns:
        df['predicted_label'] = canonicalize_empty_labels(df['predicted_label'])
    return df


def map_cell_types_to_ontology(
    predicted_labels: list[str],
    cache_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    force_remapping: bool = False,
    origin: str = "unknown"
) -> Path:
    """
    Map predicted cell type labels to ontology terms using omicverse CellOntologyMapper.
    
    Creates or updates a CSV file with mappings that can be manually edited by users.
    
    Parameters
    ----------
    predicted_labels : list[str]
        list of unique predicted cell type labels to map
    cache_dir : Path
        Directory to store the ontology files and mapping results
    model_name : str, default "all-MiniLM-L6-v2"
        Name of the model to use for semantic matching
    force_remapping : bool, default False
        If True, remap all labels even if they already exist in the CSV
    origin : str, default "unknown"
        Origin of the cell type labels (e.g., dataset name or model name)
        
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
    except ImportError as e:
        error_msg = str(e)

        # Keep the classic guidance for a truly missing package.
        if "No module named 'omicverse'" in error_msg:
            raise ImportError(
                "omicverse is required for ontology mapping. "
                "Install it with: pip install omicverse or include it into the dependencies of the nextflow process"
            ) from e

        # Surface binary/linker errors explicitly (common in conda + system libstdc++ conflicts).
        raise ImportError(
            "Failed to import omicverse due to a runtime dependency/linking error. "
            "If you see a CXXABI/libstdc++ message, ensure the conda env libraries are preferred "
            "(for example: export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}\"). "
            f"Original error: {error_msg}"
        ) from e
    
    cache_dir = Path(cache_dir) 
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Path for the mapping CSV file
    mapping_csv_path = cache_dir / "cell_type_ontology_mappings.csv"
    
    # Load existing mappings if they exist
    existing_mappings = {}
    if mapping_csv_path.exists() and not force_remapping:
        try:
            existing_df = _read_mapping_csv(mapping_csv_path)
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
            
            # Create mapping entry
            mapping_entry = {
                "reviewed": False,  # New mappings need to be reviewed
                "origin": origin,
                "predicted_label": label,
                "chosen_match": best_match,  # Default to best_match, user can edit
                "best_match": best_match
            }
            new_mappings.append(mapping_entry)
        else:
            # Handle case where mapping failed
            mapping_entry = {
                "reviewed": False,  # New mappings need to be reviewed
                "origin": origin,
                "predicted_label": label,
                "chosen_match": "",
                "best_match": ""
            }
            new_mappings.append(mapping_entry)
    
    # Create DataFrame for new mappings
    new_df = pd.DataFrame(new_mappings)
    
    # Combine with existing mappings if they exist
    if mapping_csv_path.exists() and not force_remapping:
        try:
            existing_df = _read_mapping_csv(mapping_csv_path)
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
    
    # Reorder columns to ensure correct order
    desired_columns = ['reviewed', 'origin', 'predicted_label', 'chosen_match', 'best_match']
    # Only include columns that exist in the dataframe
    column_order = [col for col in desired_columns if col in combined_df.columns]
    # Add any extra columns that might exist but weren't in our desired list
    extra_columns = [col for col in combined_df.columns if col not in column_order]
    column_order.extend(extra_columns)
    combined_df = combined_df[column_order]
    
    # Save to CSV with semicolon delimiter
    combined_df.to_csv(mapping_csv_path, sep=';', index=False)
    
    print(f"Ontology mappings saved to: {mapping_csv_path}")
    print(f"Total mappings in file: {len(combined_df)}")
    print(f"New mappings added: {len(new_mappings)}")
    
    # Create warning file for new mappings
    if new_mappings:
        warning_file_path = cache_dir / "ONTOLOGY_MAPPING_WARNING.txt"
        try:
            with open(warning_file_path, 'w') as f:
                f.write("NEW CELL TYPE MAPPINGS DETECTED\n")
                f.write("=" * 50 + "\n")
                f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of new mappings: {len(new_mappings)}\n")
                f.write(f"Mapping file: {mapping_csv_path}\n\n")
                f.write("New mappings added:\n")
                for mapping in new_mappings:
                    predicted = mapping['predicted_label']
                    chosen = mapping['chosen_match']
                    if chosen:
                        f.write(f"  '{predicted}' -> '{chosen}'\n")
                    else:
                        f.write(f"  '{predicted}' -> [NO MAPPING FOUND]\n")
                f.write("\nACTION REQUIRED:\n")
                f.write("1. Review the mappings in the CSV file\n")
                f.write("2. Update 'chosen_match' column if needed\n")
                f.write("3. Set 'reviewed' column to 'True' after verification\n")
                f.write("4. Delete this warning file when done\n")
            
            print(f"Warning: {len(new_mappings)} new cell type mappings created. Check {warning_file_path}")
            
        except Exception as e:
            print(f"Could not create warning file: {e}")
    
    return mapping_csv_path


def load_ontology_mappings(cache_dir: Path) -> Optional[dict[str, str]]:
    """
    Load ontology mappings from the CSV file.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing the mapping CSV file
        
    Returns
    -------
    Optional[dict[str, str]]
        Dictionary mapping predicted_label -> chosen_match, or None if file doesn't exist
    """
    # CSV is stored in cache directory
    cache_dir = Path(cache_dir)
    mapping_csv_path = cache_dir / "cell_type_ontology_mappings.csv"
    
    if not mapping_csv_path.exists():
        return None
    
    try:
        df = _read_mapping_csv(mapping_csv_path)
        # Create mapping from predicted_label to chosen_match
        mappings = dict(zip(df['predicted_label'], df['chosen_match']))
        return mappings
    except Exception as e:
        print(f"Warning: Could not load ontology mappings: {e}")
        return None


def load_ontology_mappings_with_review_status(cache_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load ontology mappings from the CSV file with review status.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing the mapping CSV file
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with columns: predicted_label, chosen_match, reviewed, or None if file doesn't exist
    """
    # CSV is stored in cache directory
    cache_dir = Path(cache_dir)
    mapping_csv_path = cache_dir / "cell_type_ontology_mappings.csv"
    
    if not mapping_csv_path.exists():
        return None
    
    try:
        df = _read_mapping_csv(mapping_csv_path)
        return df
    except Exception as e:
        print(f"Warning: Could not load ontology mappings with review status: {e}")
        return None


def filter_predictions_by_review_status(
    df: pd.DataFrame,
    pred_col: str,
    cache_dir: Path,
    max_unreviewed_ratio: float = 0.1,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Filter predictions to exclude those with unreviewed ontology mappings.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictions
    pred_col : str
        Column name containing predicted labels
    cache_dir : Path
        Directory containing the mapping CSV file
    max_unreviewed_ratio : float, default 0.1
        Maximum acceptable ratio of unreviewed predictions (0.0-1.0)
    verbose : bool, default True
        If True, print detailed information about filtering
        
    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - Filtered DataFrame with only reviewed predictions
        - Statistics dictionary with filtering information
        
    Raises
    ------
    RuntimeError
        If the ratio of unreviewed predictions exceeds the threshold
    """
    if verbose:
        print(f"Filtering predictions by ontology mapping review status...")
    
    # Load mappings with review status
    mappings_df = load_ontology_mappings_with_review_status(cache_dir)
    
    if mappings_df is None:
        if verbose:
            print("  No ontology mappings found. Keeping all predictions.")
        return df, {
            "total_predictions": len(df),
            "filtered_predictions": len(df),
            "unreviewed_predictions": 0,
            "unreviewed_ratio": 0.0,
            "unreviewed_labels": [],
            "filtered": False
        }
    
    # Get unique predicted labels
    unique_labels = df[pred_col].unique()
    
    # Check review status for each unique label
    reviewed_labels = set()
    unreviewed_labels = set()
    unmapped_labels = set()
    
    for label in unique_labels:
        mapping_row = mappings_df[mappings_df['predicted_label'] == label]
        if len(mapping_row) == 0:
            unmapped_labels.add(label)
        else:
            reviewed_value = mapping_row.iloc[0]['reviewed']
            # Handle both boolean and string values
            reviewed = reviewed_value if isinstance(reviewed_value, bool) else str(reviewed_value).lower() in ['true', '1', 'yes']
            if reviewed:
                reviewed_labels.add(label)
            else:
                unreviewed_labels.add(label)
    
    # Count predictions by category
    total_predictions = len(df)
    unreviewed_predictions = len(df[df[pred_col].isin(unreviewed_labels)])
    unreviewed_ratio = unreviewed_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Create statistics
    stats = {
        "total_predictions": total_predictions,
        "unreviewed_predictions": unreviewed_predictions,
        "unreviewed_ratio": unreviewed_ratio,
        "unreviewed_labels": list(unreviewed_labels),
        "unmapped_labels": list(unmapped_labels),
        "reviewed_labels": list(reviewed_labels),
        "filtered": False
    }
    
    if verbose:
        print(f"  Total predictions: {total_predictions}")
        print(f"  Reviewed labels: {len(reviewed_labels)}")
        print(f"  Unreviewed labels: {len(unreviewed_labels)}")
        print(f"  Unmapped labels: {len(unmapped_labels)}")
        print(f"  Unreviewed predictions: {unreviewed_predictions} ({unreviewed_ratio:.1%})")
    
    # Check if unreviewed ratio exceeds threshold
    if unreviewed_ratio > max_unreviewed_ratio:
        error_msg = (
            f"Too many unreviewed predictions: {unreviewed_ratio:.1%} > {max_unreviewed_ratio:.1%} "
            f"({unreviewed_predictions}/{total_predictions} predictions)\n"
            f"Unreviewed labels: {sorted(unreviewed_labels)}\n"
            f"Please review the ontology mappings in the CSV file and set 'reviewed' to 'True' "
            f"for verified mappings, or increase the --max-unreviewed-ratio threshold."
        )
        raise RuntimeError(error_msg)
    
    # Filter out unreviewed predictions
    if unreviewed_labels:
        filtered_df = df[~df[pred_col].isin(unreviewed_labels)].copy()
        filtered_predictions = len(filtered_df)
        stats["filtered_predictions"] = filtered_predictions
        stats["filtered"] = True
        
        if verbose:
            print(f"  Filtered out {unreviewed_predictions} unreviewed predictions")
            print(f"  Remaining predictions: {filtered_predictions}")
            print(f"  Unreviewed labels excluded: {sorted(unreviewed_labels)}")
    else:
        filtered_df = df.copy()
        stats["filtered_predictions"] = total_predictions
        if verbose:
            print("  No unreviewed predictions found. Keeping all predictions.")
    
    return filtered_df, stats


def apply_ontology_mappings(
    predicted_labels: list[str], 
    cache_dir: Path,
    fallback_to_original: bool = True
) -> list[str]:
    """
    Apply ontology mappings to a list of predicted labels.
    
    Parameters
    ----------
    predicted_labels : list[str]
        list of predicted cell type labels
    cache_dir : Path
        Directory containing the mapping CSV file
    fallback_to_original : bool, default True
        If True, use original label when no mapping is found
        
    Returns
    -------
    list[str]
        list of mapped labels (or original labels if no mapping found and fallback_to_original=True)
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
