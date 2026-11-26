#!/usr/bin/env python3
"""Random baseline: give every cell a cell type of the test set, drawn uniformly or with the
frequencies of the types (stratified), and write cell_id, true_labels and predicted_labels to a
CSV.

    predict.py --input test.h5ad --output predictions.csv --annotation_key cell_type
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def predict(adata, annotation_key, strategy, seed):
    if annotation_key not in adata.obs.columns:
        logger.error(f"'{annotation_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        sys.exit(1)

    true_labels = adata.obs[annotation_key].astype(str).values
    unique_types, counts = np.unique(true_labels, return_counts=True)
    if len(unique_types) == 0:
        logger.error(f"No unique cell types found in adata.obs['{annotation_key}']")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    if strategy == "uniform":
        predicted = rng.choice(unique_types, size=len(true_labels))
    elif strategy == "stratified":
        p = counts / counts.sum()
        predicted = rng.choice(unique_types, size=len(true_labels), p=p)
    else:
        logger.error(f"Unknown strategy: {strategy!r} (expected 'uniform' or 'stratified')")
        sys.exit(1)

    return pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "true_labels": true_labels,
            "predicted_labels": predicted,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Random classifier prediction")
    parser.add_argument("--input", "-i", required=True, help="Input h5ad path")
    parser.add_argument("--output", "-o", required=True, help="Output predictions.csv path")
    parser.add_argument("--annotation_key", default="cell_type",
                        help="Column in adata.obs with true labels")
    parser.add_argument("--strategy", default="stratified",
                        choices=["uniform", "stratified"],
                        help="Sampling strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info(f"Loading {args.input}")
    adata = ad.read_h5ad(args.input)
    logger.info(f"Loaded {adata.shape}; running strategy={args.strategy} seed={args.seed}")

    df = predict(adata, args.annotation_key, args.strategy, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} predictions to {out}")


if __name__ == "__main__":
    main()
