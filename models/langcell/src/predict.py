#!/usr/bin/env python3
"""
LangCell prediction script for single-cell data annotation.
"""

import os

import argparse
import pandas as pd
import scanpy as sc
from pathlib import Path
import sys
import logging

def main():
    print("Inside langcell prediction script")
    """
    Main function to run LangCell prediction pipeline.
    """
    # Get default input path relative to this script
    default_input = str(Path(__file__).resolve().parents[3] / "util" / "test_adata.h5ad")
    #default_output = str(Path.cwd() / "langcell_predictions.csv")

    parser = argparse.ArgumentParser(description="Langcell prediction for single-cell data")
    parser.add_argument("--input", "-i", default=default_input,
                       help=f"Input h5ad file path (default: {default_input})")
    
    args = parser.parse_args()
    
    print(args)

if __name__ == "__main__":
    main()

