#!/usr/bin/env python
"""
Standalone ontology mapping script for the Nextflow pipeline.

Applies ontology mapping to cell type annotations in an AnnData file.
Extracted from download.py to run as a separate Nextflow process.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import anndata


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Apply ontology mapping to cell type annotations"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input h5ad file")
    parser.add_argument("--output", type=Path, required=True, help="Output h5ad file")
    parser.add_argument(
        "--annotation_key",
        type=str,
        required=True,
        help="Cell type annotation column",
    )
    parser.add_argument(
        "--cache_dir", type=Path, required=True, help="Cache directory for ontology files"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (for origin tracking)"
    )
    parser.add_argument(
        "--apply",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Whether to apply mapping (if false, passes through unchanged)",
    )
    args = parser.parse_args()

    logger.info("Reading input: %s", args.input)
    adata = anndata.read_h5ad(args.input)
    logger.info("Shape: %s cells x %s genes", adata.n_obs, adata.n_vars)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.apply:
        from util.download_helpers import apply_ontology_mapping_to_dataset

        logger.info("Applying ontology mapping to '%s'...", args.annotation_key)
        adata = apply_ontology_mapping_to_dataset(
            adata=adata,
            annotation_key=args.annotation_key,
            cache_dir=args.cache_dir,
            verbose=True,
            origin=args.dataset,
        )
        adata.write_h5ad(args.output)
        logger.info("Mapped output written: %s", args.output)
    else:
        key = args.annotation_key
        original_key = f"original_{key}"
        if original_key in adata.obs.columns:
            logger.info("Restoring '%s' from '%s' backup column...", key, original_key)
            adata.obs[key] = adata.obs[original_key].copy()
            adata.write_h5ad(args.output)
        else:
            logger.info("No mapping applied - copying input to output")
            try:
                os.link(args.input, args.output)
            except OSError:
                shutil.copy2(args.input, args.output)

    logger.info("Done")


if __name__ == "__main__":
    main()
