#!/usr/bin/env python
import argparse
import sys
import yaml
import pandas as pd
import anndata

def main(args):
    all_cell_types = set()

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
                cell_types = adata.obs[matched_key].astype(str).str.strip().str.title().unique()
                all_cell_types.update(cell_types)
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
    args = ap.parse_args()
    main(args)
