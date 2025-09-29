#!/usr/bin/env python
import argparse
import sys
import yaml
import pandas as pd
import anndata
import warnings

def read_mapping_file(mapping_file):
    mapping_df = pd.read_csv(mapping_file, dtype=str, sep=";").fillna("")

    # Build reverse lookup maps (lowercased)
    reverse_maps = {
        "chosen_match": {},
        "predicted_label": {},
        "best_match": {},
        "top3_matches": {}
    }

    for _, row in mapping_df.iterrows():
        # Map values to chosen_match (case-insensitive keys)
        chosen = row["chosen_match"].strip()

        for col in ["chosen_match", "predicted_label", "best_match"]:
            key = row[col].strip().lower()
            if key:
                reverse_maps[col][key] = chosen

        # Handle top3_matches specially (comma-separated list)
        for item in row["top3_matches"].split(","):
            key = item.strip().lower()
            if key:
                reverse_maps["top3_matches"][key] = chosen
    return reverse_maps

def map_cell_type(cell_type, reverse_maps):
    key = cell_type.strip().lower()

    for col in ["chosen_match", "predicted_label", "best_match", "top3_matches"]:
        if col in reverse_maps and key in reverse_maps[col]:
            return reverse_maps[col][key].strip().title()

    warnings.warn(f"Unmapped cell type: '{cell_type}'")
    return cell_type.strip().title()


def main(args):
    all_cell_types = set()
    try:
        reverse_maps = read_mapping_file(args.mapping_file)
    except FileNotFoundError:
        warnings.warn(f"No cell type mapping file found, using original names")
        reverse_maps = {}

    for h5ad, cfg in zip(args.h5ads, args.cfgs):
        with open(cfg) as f:
            yml = yaml.safe_load(f)
        keys = yml.get('annotation_keys')
        adata = anndata.read_h5ad(h5ad)

        found = False
        obs_keys = { k.lower() : k for k in adata.obs.columns }
        for key in keys:
            lower = key.lower()
            if lower in obs_keys:
                matched_key = obs_keys[lower]
                raw_cell_types = adata.obs[matched_key].astype(str).unique()
                mapped = {map_cell_type(ct, reverse_maps) for ct in raw_cell_types}
                all_cell_types.update(mapped)
                found = True
                break
        if not found:
            raise ValueError(f"No annotation keys found for {h5ad}")

    df = pd.DataFrame(sorted(all_cell_types))
    df.to_csv(args.out, index=False, header=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ads', nargs='+', help='Path to all input .h5ad')
    ap.add_argument('--cfgs', nargs='+', help='Path to the corresponding config for the annotation keys')
    ap.add_argument('--out', help='Path to output CSV file')
    ap.add_argument('--mapping_file', help='Path to cell type ontology mapping file')
    args = ap.parse_args()
    main(args)
