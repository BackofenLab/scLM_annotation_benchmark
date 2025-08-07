import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
import argparse
import json
import sys

from util.plots import UMAPCellPlotter  # Replace with correct import if needed

# -- Normalization utilities --
def normalize(text):
    return str(text).lower().replace("-", " ").replace(",", "").replace(".", "").strip()

def clean_celltype_L1(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    clean_pred_col = f"clean_{pred_col}"

    # Normalize all unique true labels once
    normalized_true_labels = {
        normalize(label): label
        for label in df[true_col].unique()
    }

    def match_true_label(pred_label):
        norm_pred = normalize(pred_label)
        for norm_true, original_true in normalized_true_labels.items():
            if norm_true in norm_pred or norm_pred in norm_true:
                return original_true
        return "Unknown"  

    df[clean_pred_col] = df[pred_col].apply(match_true_label)

    print(df.head())
    return df

def run_eval(df, true_col, pred_col, output_dir):
    
    true_labels = df[true_col]
    pred_labels = df[f"clean_{pred_col}"]

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)

    # Set up plotter
    plotter = UMAPCellPlotter()

    # Generate confusion matrix
    output_plot_path = output_dir / f"{true_col}_confusion_matrix.pdf"
    title = f"Confusion Matrix {true_col} (Balanced Accuracy = {balanced_acc:.2f})"
    plotter.plot_cm(
        true_labels=true_labels,
        pred_labels=pred_labels,
        output_path=output_plot_path,
        title=title,
        normalize=True
    )

    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Confusion matrix saved to: {output_plot_path}")
    
    return balanced_acc

def main(args):
    
    df = pd.read_csv(args.pred)
    output_dir = Path(args.out)
    
    results = {}
    
    if 'true_L1' in df.columns:
        
        true_col = 'true_L1'
        pred_col = 'predicted_cell_type'

        df_clean = clean_celltype_L1(df, true_col, pred_col)
        
        balanced_acc = run_eval(df_clean, true_col, pred_col, output_dir)
        
        results[true_col] = {
            "balanced_accuracy": round(balanced_acc, 4)
        }

    # Save results to JSON
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to: {eval_path}")
            

# -- CLI Interface --
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate label predictions")

    parser.add_argument("--pred", required=True, type=str, help="CSV file with true and predicted labels")
    parser.add_argument("--out", required=True, type=str, help="Path to results folder")

    args = parser.parse_args()
    main(args)
