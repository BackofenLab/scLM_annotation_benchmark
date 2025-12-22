#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone debug script for LangCell zero-shot prediction.

This script allows step-by-step debugging of the LangCell prediction pipeline.
You can set breakpoints in this file or in predict.py and step through the code.

Before running:
1. Create and activate the conda environment (see instructions below)
2. Run prepare.sh to download model weights (if not already done)

Conda environment setup:
    conda env create -f /Users/<user>/repos/sc_language_foundation_models/envs/models/langcell-zeroshot.yml
    conda activate langcell-zeroshot-env

Run prepare.sh (from the src directory):
    cd /Users/<user>/repos/sc_language_foundation_models/models/langcell-zeroshot/src
    bash prepare.sh

Then run this debug script:
    python debug_predict.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================================

# Default input h5ad file
DEFAULT_INPUT_H5AD = Path("/Users/<user>/repos/sc_language_foundation_models/cache/HIHA_L1_sub200.h5ad")

# Model path (the src directory where data/ckpt should exist after running prepare.sh)
# This is equivalent to what run.sh does with MODEL_PATH=$(realpath "$0")
SRC_DIR = Path(__file__).resolve().parent
MODEL_PATH = SRC_DIR / "run.sh"  # predict.py uses model_path.parent to get src dir

# Output path
DEFAULT_OUTPUT = SRC_DIR / "debug_predictions.csv"

# Annotation key to use
ANNOTATION_KEY = "AIFI_L1"  # Change this if your h5ad uses a different key

# Other settings
SEED = 42
BATCH_SIZE = 32
ALPHA = 0.1  # Fusion weight for SIM logits

# ============================================================================
# Setup paths and imports
# ============================================================================

# Add src directory to path so we can import from predict.py and utils.py
sys.path.insert(0, str(SRC_DIR))

# Now import from predict.py
from predict import (
    set_seed,
    load_model_assets,
    setup_dataframe,
    load_dataset,
    gettextfromname,
    text_encode,
    cell_encode,
    ctm,
    predict,
    create_results_dataframe,
    Pooler,
)

# Import utils
from utils import BertModel as MedBertModel
from utils import (
    LangCellDataCollatorForCellClassification as DataCollatorForCellClassification,
)
from utils import LangCellTranscriptomeTokenizer

# ============================================================================
# Create args namespace (simulating argparse)
# ============================================================================

def create_args():
    """Create an args namespace matching what parse_args() would return."""
    return Namespace(
        input=DEFAULT_INPUT_H5AD,
        output=DEFAULT_OUTPUT,
        model_path=MODEL_PATH,
        seed=SEED,
        batchsize=BATCH_SIZE,
        alpha=ALPHA,
        device=None,
        annotation_keys=[ANNOTATION_KEY],
    )


# ============================================================================
# Main debug function
# ============================================================================

def main():
    """
    Main debug function - run the prediction pipeline step by step.
    
    Set breakpoints anywhere in this function or in the imported functions
    to debug the pipeline.
    """
    
    print("=" * 80)
    print("LangCell Zero-Shot Debug Script")
    print("=" * 80)
    
    # Check if model files exist
    data_dir = SRC_DIR / "data"
    ckpt_dir = data_dir / "ckpt"
    
    if not ckpt_dir.exists():
        print(f"\n❌ ERROR: Model checkpoint directory not found at: {ckpt_dir}")
        print("\nPlease run prepare.sh first to download model weights:")
        print(f"    cd {SRC_DIR}")
        print("    bash prepare.sh")
        return
    
    print(f"\n✅ Model checkpoint directory found: {ckpt_dir}")
    
    # Check if input file exists
    if not DEFAULT_INPUT_H5AD.exists():
        print(f"\n❌ ERROR: Input h5ad file not found at: {DEFAULT_INPUT_H5AD}")
        print("Please update DEFAULT_INPUT_H5AD in this script to point to a valid h5ad file.")
        return
    
    print(f"✅ Input file found: {DEFAULT_INPUT_H5AD}")
    
    # Create args
    args = create_args()
    print(f"\n📋 Configuration:")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Model path: {args.model_path}")
    print(f"   Annotation key: {args.annotation_keys}")
    print(f"   Batch size: {args.batchsize}")
    print(f"   Alpha: {args.alpha}")
    
    # Set device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Set seed for reproducibility
    print(f"\n🎲 Setting random seed: {args.seed}")
    set_seed(args.seed)
    
    # ========================================================================
    # STEP 1: Load model assets
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading model assets...")
    print("=" * 80)
    
    model_path_parent = args.model_path.parent  # This is the src directory
    
    # SET BREAKPOINT HERE to step into load_model_assets
    model, tokenizer, text_encoder, ctm_head = load_model_assets(model_path_parent, device)
    
    print("✅ Model assets loaded successfully!")
    print(f"   - Cell encoder type: {type(model).__name__}")
    print(f"   - Text encoder type: {type(text_encoder).__name__}")
    print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # ========================================================================
    # STEP 2: Load and prepare dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Loading and preparing dataset...")
    print("=" * 80)
    
    annotation_key = args.annotation_keys[0]
    print(f"   Using annotation key: '{annotation_key}'")
    
    # SET BREAKPOINT HERE to step into load_dataset
    (
        tokenized_dataset,
        testdataset,
        texts,
        text_embs,
        dataloader,
        types,
        cell_indices
    ) = load_dataset(tokenizer, text_encoder, annotation_key, args)
    
    print("✅ Dataset loaded successfully!")
    print(f"   - Number of cells: {len(tokenized_dataset)}")
    print(f"   - Number of cell types: {len(types)}")
    print(f"   - Cell types: {types[:5]}..." if len(types) > 5 else f"   - Cell types: {types}")
    print(f"   - Text embeddings shape: {text_embs.shape}")
    
    # ========================================================================
    # STEP 3: Inspect text descriptions (helpful for debugging)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Cell type text descriptions")
    print("=" * 80)
    
    for i, (cell_type, text_desc) in enumerate(zip(types[:3], texts[:3])):
        print(f"\n   [{i}] {cell_type}:")
        print(f"       {text_desc[:100]}..." if len(text_desc) > 100 else f"       {text_desc}")
    
    if len(types) > 3:
        print(f"\n   ... and {len(types) - 3} more cell types")
    
    # ========================================================================
    # STEP 4: Run prediction
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Running prediction...")
    print("=" * 80)
    
    # SET BREAKPOINT HERE to step into predict
    pred_df = predict(
        model,
        tokenizer,
        text_encoder,
        ctm_head,
        tokenized_dataset,
        testdataset,
        texts,
        text_embs,
        dataloader,
        types,
        cell_indices,
        args,
    )
    
    print("\n✅ Prediction completed!")
    print(f"   - Results shape: {pred_df.shape}")
    print(f"   - Columns: {list(pred_df.columns)}")
    
    # ========================================================================
    # STEP 5: Save results
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Saving results...")
    print("=" * 80)
    
    output_csv = args.output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=True)
    
    print(f"✅ Results saved to: {output_csv}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Show some example predictions
    print("\n📊 Sample predictions (first 10 rows):")
    print(pred_df.head(10).to_string())
    
    # Calculate accuracy if true labels are available
    if 'true_labels' in pred_df.columns and 'predicted_labels' in pred_df.columns:
        accuracy = (pred_df['true_labels'] == pred_df['predicted_labels']).mean()
        print(f"\n📈 Overall accuracy: {accuracy:.2%}")
    
    print("\n✅ Debug script completed successfully!")
    
    return pred_df


if __name__ == "__main__":
    # You can set a breakpoint here and step through
    result_df = main()

