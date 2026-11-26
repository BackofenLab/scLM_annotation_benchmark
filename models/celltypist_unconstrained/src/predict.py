#!/usr/bin/env python3
"""Annotate the cells of a test set with CellTypist.

Normalizes the raw counts (layers["counts"]) to 10,000 per cell and applies log1p, as
CellTypist requires, and runs the model given by --model, which is downloaded from the
CellTypist models or the Organ Atlas. Writes a CSV with cell_id, true_labels and
predicted_labels. If LABEL_CONSTRAINT names a label table, only the labels that count for the
true cell types are allowed (bin/util/label_constraint.py).

    predict.py --input test.h5ad --output predictions.csv --model Immune_All_High.pkl \\
        --annotation_key cell_type
"""

import argparse
import os
import pandas as pd
import scanpy as sc
import celltypist
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

def load_data(data_path):
    """Read an h5ad file; exit on failure."""
    logger.info(f"Loading data from: {data_path}")
    try:
        adata = sc.read_h5ad(data_path)
        logger.info(f"Loaded data with shape: {adata.shape}")
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

def _download_model(model: str) -> None:
    """Download a CellTypist model, falling back to the Organ Atlas URL if needed."""
    try:
        celltypist.models.download_models(model=model)
    except (ValueError, SystemExit):
        # Not in the CellTypist model list: the Organ Atlas URL contains the organ, the third
        # part of the model name.
        import urllib.request
        from pathlib import Path

        stem = model.removesuffix(".pkl")
        parts = stem.split("_", 2)
        if len(parts) < 3:
            raise ValueError(
                f"Cannot derive organ from model name '{model}'. "
                "Expected pattern: <Descriptor>_<Species>_<Organ>.pkl"
            )
        organ = parts[2]
        organ_atlas_url = (
            f"https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/"
            f"{organ}/models/{model}"
        )

        models_dir = Path(celltypist.models.models_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / model
        if not dest.exists():
            logger.info(f"Model not in standard list; downloading from Organ Atlas: {organ_atlas_url}")
            urllib.request.urlretrieve(organ_atlas_url, dest)
        else:
            logger.info(f"Organ Atlas model already cached: {dest}")


def run_celltypist_prediction(adata, model="Immune_All_High.pkl"):
    """Run CellTypist with the given model on adata and return its AnnotationResult; exit on
    failure."""
    logger.info(f"Running CellTypist prediction with model: {model}")
    try:
        # CellTypist matches genes by symbol in var_names.
        gene_symbol_cols = ["gene_names", "feature_name", "gene_symbols", "gene_name", "symbol"]
        for col in gene_symbol_cols:
            if col in adata.var.columns:
                logger.info(f"Swapping var.index to gene symbols from '{col}' column")
                adata.var_names = adata.var[col].astype(str).values
                adata.var_names_make_unique()
                break
        
        _download_model(model)
        model_obj = celltypist.models.Model.load(model)
        predictions = celltypist.annotate(adata, model=model_obj)
        logger.info("Prediction completed successfully")
        return predictions
    except Exception as e:
        logger.error(f"Failed to run CellTypist prediction: {e}")
        sys.exit(1)

def create_results_dataframe_for_key(adata, predictions, label_key):
    """Return a DataFrame of cell_id, true_labels (from label_key) and predicted_labels, or None
    if adata.obs has no column label_key."""
    logger.info(f"Creating results DataFrame for annotation key: {label_key}")
    
    predicted_labels = predictions.predicted_labels["predicted_labels"].values.tolist()
    predicted_idx = predictions.predicted_labels.index
    
    if label_key not in adata.obs.columns:
        logger.error(f"'{label_key}' not found in adata.obs")
        logger.error(f"Available keys: {list(adata.obs.columns)}")
        return None
    
    true_labels = list(adata.obs[label_key][predicted_idx].values)
    
    results_df = pd.DataFrame({
        'cell_id': predicted_idx,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    })
    
    logger.info(f"Created results DataFrame for {label_key} with {len(results_df)} rows")
    return results_df

def save_results(results_df, output_path):
    """Write the results to a CSV file; exit on failure."""
    logger.info(f"Saving results to: {output_path}")
    try:
        results_df.to_csv(output_path, index=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

def preprocess_data(adata):
    """Normalize to 10,000 counts per cell and apply log1p, as CellTypist requires; exit on
    failure."""
    logger.info("Preprocessing data")
    try:
        # X of the test set is already log-normalized, so this starts again from the raw counts.
        if "counts" in adata.layers:
            adata = adata.copy()
            adata.X = adata.layers["counts"]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        sys.exit(1)



        
def main():
    # Default input: the small test file util/test_adata.h5ad of the repository.
    default_input = str(Path(__file__).resolve().parents[3] / "util" / "test_adata.h5ad")
    default_output = str(Path.cwd() / "celltypist_predictions.csv")

    parser = argparse.ArgumentParser(description="CellTypist prediction for single-cell data")
    parser.add_argument("--input", "-i", default=default_input,
                       help=f"Input h5ad file path (default: {default_input})")
    parser.add_argument("--output", "-o", default=default_output,
                       help=f"Output CSV file path base name (default: {default_output})")
    parser.add_argument("--model", "-m", default="Immune_All_High.pkl", 
                       help="CellTypist model to use (default: Immune_All_High.pkl)")
    parser.add_argument("--annotation_key", default="cell_ontology_class",
                       help="Key in adata.obs containing true labels (default: cell_ontology_class)")
    
    args = parser.parse_args()
    
    adata = load_data(args.input)
    adata = preprocess_data(adata)

    model_to_use = args.model

    predictions = run_celltypist_prediction(adata, model_to_use)
    
    annotation_key = args.annotation_key
    logger.info(f"Processing annotation key: {annotation_key}")
    
    results_df = create_results_dataframe_for_key(adata, predictions, annotation_key)

    # Constrained mode (LABEL_CONSTRAINT): the best match among the labels that count for one of
    # the true cell types (bin/util/label_constraint.py), answered as that type. Each label has its
    # own linear score, so this is the model's best match with the other labels removed.
    constraint = os.environ.get("LABEL_CONSTRAINT")
    if constraint and results_df is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bin"))
        from util.label_constraint import constrain
        scores = predictions.decision_matrix
        kept = constrain(scores.columns, constraint, "CellTypist", results_df["true_labels"])
        if not kept:
            sys.exit("no CellTypist label counts for the listed cell types")
        best = scores.loc[results_df["cell_id"], list(kept)].idxmax(axis=1)
        results_df["predicted_labels"] = best.map(kept).to_numpy()
        results_df["reachable"] = results_df["true_labels"].isin(set(kept.values()))

    if results_df is not None:
        print(f"Saving results to: {args.output}")
        save_results(results_df, args.output)
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.error("Failed to create results DataFrame")
        sys.exit(1)
    
    logger.info("CellTypist prediction pipeline completed successfully")

if __name__ == "__main__":
    main()

