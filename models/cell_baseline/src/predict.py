#!/usr/bin/env python3
"""Constant 'cell' baseline: assigns the CL root label (CL:0000000) to every cell.

Used as a lower-bound reference for ontology-aware similarity metrics (Rada, Lin):
every Cell Ontology term descends from 'cell', so any reasonable model should
score above this baseline.
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def predict(adata, annotation_key):
    if annotation_key not in adata.obs.columns:
        logger.error(f"'{annotation_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        sys.exit(1)

    true_labels = adata.obs[annotation_key].astype(str).values
    predicted = ["cell"] * len(true_labels)

    return pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "true_labels": true_labels,
            "predicted_labels": predicted,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Constant 'cell' baseline classifier")
    parser.add_argument("--input", "-i", required=True, help="Input h5ad path")
    parser.add_argument("--output", "-o", required=True, help="Output predictions.csv path")
    parser.add_argument("--annotation_key", default="cell_type",
                        help="Column in adata.obs with true labels")
    args = parser.parse_args()

    logger.info(f"Loading {args.input}")
    adata = ad.read_h5ad(args.input)
    logger.info(f"Loaded {adata.shape}; assigning constant label 'cell' to all rows")

    df = predict(adata, args.annotation_key)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} predictions to {out}")


if __name__ == "__main__":
    main()
