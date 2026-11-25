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
from .cell_ontology_similarity import CL_FILE, CL_URL, download_cl_ontology
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


def _resolve_best_matches(
    labels: list[str],
    cl_file_path: Path,
    cache_dir: Path,
    model_name: str,
) -> dict[str, dict]:
    """Return ``{label: {"best_match": <CL term name or "">}}`` for each label.

    Tries omicverse's semantic ``CellOntologyMapper`` first (rich fuzzy matching). If
    omicverse is unavailable or fails to import/run, falls back to an omicverse-free
    lookup: labels that match a CL term name or CL-id exactly, or else an EXACT synonym, get
    that term's canonical name as ``best_match``; labels with no hit get an empty
    ``best_match``. Either way these are suggestions for manual review, never scores.
    """
    try:
        import omicverse as ov
        print(f"Initializing CellOntologyMapper with model: {model_name}")
        mapper = ov.single.CellOntologyMapper(
            cl_obo_file=str(cl_file_path),
            model_name=model_name,
            local_model_dir=str(cache_dir),
        )
        print("Performing ontology mapping (omicverse semantic matcher)...")
        return mapper.map_cells(labels)
    except Exception as e:
        print(
            f"omicverse mapping unavailable ({e}); falling back to Cell-Ontology index "
            "lookup for best-match suggestions (labels still appended for review)."
        )

    # Omicverse-free fallback. `cl_file_path` is the OBO-Graph JSON omicverse consumes; pronto
    # reads the OBO file of the same pinned release.
    import pronto
    from .cell_ontology_similarity import build_label_index, build_synonym_index, map_string_to_term

    ont = pronto.Ontology(str(download_cl_ontology(cache_dir)))
    names, synonyms = build_label_index(ont), build_synonym_index(ont)
    results: dict[str, dict] = {}
    for label in labels:
        term = map_string_to_term(label, names)
        if term is None:
            term = map_string_to_term(label, synonyms)
        results[label] = {"best_match": term.name if term is not None else ""}
    return results


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

    Notes
    -----
    Best-match suggestions are computed by omicverse's semantic ``CellOntologyMapper``
    when the package imports cleanly; otherwise an omicverse-free fallback resolves each
    label against the Cell Ontology label index (exact name / synonym / CL-id match) and
    leaves ``best_match`` blank for labels with no direct hit. Either way every new label
    is appended to the CSV as ``reviewed=False`` for human review -- the append never
    depends on omicverse being installed.

    Raises
    ------
    RuntimeError
        If ontology download or best-match resolution fails
    """
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
        cl_file_path = download_cl_ontology(cache_dir, cl_url=CL_URL.removesuffix(".obo") + ".json",
                                            filename=CL_FILE.removesuffix(".obo") + ".json")

        # Resolve a best-match suggestion for each label. Prefer omicverse's semantic
        # matcher; degrade to an omicverse-free index lookup if the package is missing or
        # fails to import (e.g. x86 llvmlite under Rosetta). The append below runs either way.
        mapping_results = _resolve_best_matches(
            labels_to_map, cl_file_path, cache_dir, model_name
        )
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


