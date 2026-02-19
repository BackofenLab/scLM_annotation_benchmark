#!/usr/bin/env python
import argparse
import yaml
import pandas as pd
import anndata
import warnings
import util.ontology_helpers as oh


def main(args):
    if len(args.h5ads) != len(args.cfgs):
        raise ValueError(
            f"Expected the same number of --h5ads and --cfgs, got "
            f"{len(args.h5ads)} and {len(args.cfgs)}"
        )

    all_cell_types = set()

    for h5ad, cfg in zip(args.h5ads, args.cfgs):
        with open(cfg) as f:
            yml = yaml.safe_load(f)
        keys = yml.get("annotation_keys") or []
        if not keys:
            warnings.warn(f"No annotation keys configured in {cfg}; skipping.")
            continue

        adata = anndata.read_h5ad(h5ad)
        missing_keys = [key for key in keys if key not in adata.obs.columns]
        if missing_keys:
            warnings.warn(
                f"Missing annotation keys in {h5ad}: {missing_keys}. Skipping this dataset."
            )
            continue

        mapped_adata = adata
        if yml.get("apply_ontology_mapping", True):
            try:
                mapped_adata = oh.apply_ontology_mapping_to_adata(
                    adata,
                    cache_dir=".",
                    annotation_keys=keys
                )
            except Exception as e:
                warnings.warn(
                    f"Ontology mapping failed for {h5ad} ({e}). "
                    "Continuing with original annotations."
                )

        cell_types = [ct.title() for ct in mapped_adata.obs[keys].values.flatten()]
        all_cell_types.update(cell_types)

    df = pd.DataFrame(sorted(all_cell_types))
    df.to_csv(args.out, index=False, header=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ads', nargs='+', help='Path to all input .h5ad')
    ap.add_argument('--cfgs', nargs='+', help='Path to the corresponding config for the annotation keys')
    ap.add_argument('--out', help='Path to output CSV file')
    args = ap.parse_args()
    main(args)
