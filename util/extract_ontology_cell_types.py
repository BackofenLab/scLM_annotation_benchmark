#!/usr/bin/env python
"""
Extract cell type names from Cell Ontology (CL) database.

This script parses cl-base.json or cl-base.owl and extracts all cell type entries
with their IDs, labels, synonyms, and definitions.

Usage:
    python util/extract_ontology_cell_types.py --input cache/cl-base.json --output cache/cell_types_from_ontology.csv
    python util/extract_ontology_cell_types.py --input cache/cl-base.owl --output cache/cell_types_from_ontology.csv
    
    # Alternative: Use the JSON parser directly (faster, no dependencies)
    python util/extract_ontology_cell_types.py --input cache/cl-base.json --output cache/cell_types.csv --parser json
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def extract_cl_id(uri: str) -> Optional[str]:
    """
    Extract the CL ID from a URI.
    
    E.g., "http://purl.obolibrary.org/obo/CL_0000037" -> "CL:0000037"
    """
    if not uri:
        return None
    
    # Handle various URI formats
    parts = uri.split("/")
    last_part = parts[-1]
    
    # Check if it's a CL term (Cell Ontology)
    if last_part.startswith("CL_"):
        # Convert CL_XXXXXXX to CL:XXXXXXX format
        return last_part.replace("_", ":")
    
    return None


def extract_from_json(json_path: Path) -> List[Dict]:
    """
    Extract cell type information directly from cl-base.json (OBO Graph JSON format).
    
    This parser doesn't require external dependencies like pronto.
    
    Parameters
    ----------
    json_path : Path
        Path to the cl-base.json file
        
    Returns
    -------
    List[Dict]
        List of dictionaries containing cell type information
    """
    print(f"Loading JSON file: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cell_types = []
    
    # Navigate to nodes in the OBO Graph JSON structure
    if "graphs" not in data or len(data["graphs"]) == 0:
        raise ValueError("Invalid OBO Graph JSON format: missing 'graphs' key")
    
    nodes = data["graphs"][0].get("nodes", [])
    print(f"Found {len(nodes)} total nodes in ontology")
    
    for node in nodes:
        node_id = node.get("id", "")
        cl_id = extract_cl_id(node_id)
        
        # Only process CL (Cell Ontology) terms
        if not cl_id:
            continue
        
        label = node.get("lbl", "")
        node_type = node.get("type", "")
        
        # Skip non-CLASS nodes
        if node_type != "CLASS":
            continue
        
        # Extract metadata
        meta = node.get("meta", {})
        
        # Check if obsolete
        is_obsolete = meta.get("deprecated", False)
        if is_obsolete:
            # Include obsolete flag in output
            pass
        
        # Also check label for obsolete marker
        if label and label.lower().startswith("obsolete"):
            is_obsolete = True
        
        # Extract definition
        definition = ""
        if "definition" in meta:
            definition = meta["definition"].get("val", "")
        
        # Extract synonyms
        synonyms = []
        for syn in meta.get("synonyms", []):
            syn_val = syn.get("val", "")
            syn_type = syn.get("pred", "")  # e.g., "hasExactSynonym", "hasRelatedSynonym"
            if syn_val:
                synonyms.append(f"{syn_val} ({syn_type})" if syn_type else syn_val)
        
        # Extract subsets (useful for filtering)
        subsets = []
        for subset in meta.get("subsets", []):
            # Extract subset name from URI
            subset_name = subset.split("#")[-1] if "#" in subset else subset.split("/")[-1]
            subsets.append(subset_name)
        
        cell_types.append({
            "cl_id": cl_id,
            "label": label,
            "definition": definition[:500] if definition else "",  # Truncate long definitions
            "synonyms": "; ".join(synonyms),
            "subsets": "; ".join(subsets),
            "is_obsolete": is_obsolete,
            "uri": node_id
        })
    
    return cell_types


def extract_from_pronto(ontology_path: Path) -> List[Dict]:
    """
    Extract cell type information using pronto library.
    
    Works with both OWL and JSON ontology files.
    
    Parameters
    ----------
    ontology_path : Path
        Path to the ontology file (OWL or JSON)
        
    Returns
    -------
    List[Dict]
        List of dictionaries containing cell type information
    """
    try:
        import pronto
    except ImportError:
        raise ImportError(
            "pronto library is required for OWL parsing. "
            "Install it with: pip install pronto\n"
            "Or use --parser json for JSON files (no dependencies required)."
        )
    
    print(f"Loading ontology with pronto: {ontology_path}")
    ont = pronto.Ontology(str(ontology_path))
    
    cell_types = []
    
    for term in ont.terms():
        # Only include CL terms
        if not term.id.startswith("CL:"):
            continue
        
        # Extract synonyms with their types
        synonyms = []
        for syn in term.synonyms:
            # Get synonym text
            syn_text = getattr(syn, "description", None) or getattr(syn, "desc", None)
            if not syn_text:
                try:
                    syn_text = str(syn)
                except Exception:
                    continue
            
            # Get synonym type
            syn_scope = getattr(syn, "scope", "")
            if syn_text:
                synonyms.append(f"{syn_text} ({syn_scope})" if syn_scope else syn_text)
        
        # Extract subsets
        subsets = []
        for subset in term.subsets:
            subsets.append(str(subset))
        
        # Get definition
        definition = ""
        if term.definition:
            definition = str(term.definition)[:500]  # Truncate long definitions
        
        cell_types.append({
            "cl_id": term.id,
            "label": term.name or "",
            "definition": definition,
            "synonyms": "; ".join(synonyms),
            "subsets": "; ".join(subsets),
            "is_obsolete": term.obsolete,
            "uri": f"http://purl.obolibrary.org/obo/{term.id.replace(':', '_')}"
        })
    
    return cell_types


def save_to_csv(cell_types: List[Dict], output_path: Path) -> None:
    """Save cell types to a CSV file."""
    if not cell_types:
        print("Warning: No cell types to save")
        return
    
    # Sort by CL ID for consistent output
    cell_types_sorted = sorted(cell_types, key=lambda x: x["cl_id"])
    
    fieldnames = ["cl_id", "label", "definition", "synonyms", "subsets", "is_obsolete", "uri"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cell_types_sorted)
    
    print(f"Saved {len(cell_types_sorted)} cell types to: {output_path}")


def save_to_json(cell_types: List[Dict], output_path: Path) -> None:
    """Save cell types to a JSON file."""
    if not cell_types:
        print("Warning: No cell types to save")
        return
    
    # Sort by CL ID for consistent output
    cell_types_sorted = sorted(cell_types, key=lambda x: x["cl_id"])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cell_types_sorted, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(cell_types_sorted)} cell types to: {output_path}")


def get_label_list(cell_types: List[Dict], include_synonyms: bool = True, exclude_obsolete: bool = True) -> List[str]:
    """
    Get a simple list of all cell type labels (and optionally synonyms).
    
    Parameters
    ----------
    cell_types : List[Dict]
        List of cell type dictionaries
    include_synonyms : bool, default True
        Whether to include synonyms in the list
    exclude_obsolete : bool, default True
        Whether to exclude obsolete terms
        
    Returns
    -------
    List[str]
        List of cell type names
    """
    labels = set()
    
    for ct in cell_types:
        if exclude_obsolete and ct.get("is_obsolete", False):
            continue
        
        # Add main label
        if ct.get("label"):
            labels.add(ct["label"])
        
        # Add synonyms
        if include_synonyms and ct.get("synonyms"):
            for syn in ct["synonyms"].split("; "):
                # Extract just the synonym text (remove the type annotation)
                syn_text = syn.rsplit(" (", 1)[0] if " (" in syn else syn
                if syn_text:
                    labels.add(syn_text)
    
    return sorted(labels)


def main():
    parser = argparse.ArgumentParser(
        description="Extract cell type names from Cell Ontology database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from JSON (no dependencies required)
  python util/extract_ontology_cell_types.py --input cache/cl-base.json
  
  # Extract from OWL (requires pronto library)
  python util/extract_ontology_cell_types.py --input cache/cl-base.owl --parser pronto
  
  # Save to specific output file
  python util/extract_ontology_cell_types.py --input cache/cl-base.json --output my_cell_types.csv
  
  # Output as JSON instead of CSV
  python util/extract_ontology_cell_types.py --input cache/cl-base.json --format json
  
  # Get just the label list (simple text file)
  python util/extract_ontology_cell_types.py --input cache/cl-base.json --labels-only
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to ontology file (cl-base.json or cl-base.owl)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: derived from input filename)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "--parser", "-p",
        choices=["auto", "json", "pronto"],
        default="auto",
        help="Parser to use. 'auto' will use JSON parser for .json files, pronto for .owl files"
    )
    
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Output only a simple list of cell type labels (one per line)"
    )
    
    parser.add_argument(
        "--include-synonyms",
        action="store_true",
        default=False,
        help="When using --labels-only, also include synonyms"
    )
    
    parser.add_argument(
        "--include-obsolete",
        action="store_true",
        default=False,
        help="Include obsolete terms in output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Determine parser
    parser_type = args.parser
    if parser_type == "auto":
        if args.input.suffix.lower() == ".json":
            parser_type = "json"
        else:
            parser_type = "pronto"
    
    # Extract cell types
    print(f"Extracting cell types using {parser_type} parser...")
    
    if parser_type == "json":
        cell_types = extract_from_json(args.input)
    else:
        cell_types = extract_from_pronto(args.input)
    
    # Filter obsolete if requested
    if not args.include_obsolete:
        original_count = len(cell_types)
        cell_types = [ct for ct in cell_types if not ct.get("is_obsolete", False)]
        filtered_count = original_count - len(cell_types)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} obsolete terms")
    
    print(f"Extracted {len(cell_types)} cell types from Cell Ontology")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.labels_only:
            output_path = args.input.parent / "cell_type_labels.txt"
        else:
            output_path = args.input.parent / f"cell_types_extracted.{args.format}"
    
    # Save output
    if args.labels_only:
        labels = get_label_list(
            cell_types, 
            include_synonyms=args.include_synonyms,
            exclude_obsolete=not args.include_obsolete
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(labels))
        print(f"Saved {len(labels)} unique labels to: {output_path}")
    elif args.format == "json":
        save_to_json(cell_types, output_path)
    else:
        save_to_csv(cell_types, output_path)
    
    # Print some statistics
    print("\n--- Statistics ---")
    print(f"Total CL terms: {len(cell_types)}")
    
    labeled_count = sum(1 for ct in cell_types if ct.get("label"))
    print(f"Terms with labels: {labeled_count}")
    
    with_synonyms = sum(1 for ct in cell_types if ct.get("synonyms"))
    print(f"Terms with synonyms: {with_synonyms}")
    
    with_definition = sum(1 for ct in cell_types if ct.get("definition"))
    print(f"Terms with definitions: {with_definition}")
    
    # Show subset distribution
    subset_counts = {}
    for ct in cell_types:
        for subset in ct.get("subsets", "").split("; "):
            if subset:
                subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    if subset_counts:
        print(f"\nSubset distribution (top 10):")
        for subset, count in sorted(subset_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {subset}: {count}")
    
    return 0


if __name__ == "__main__":
    exit(main())

