#!/usr/bin/env python
"""Write the label constraint of one evaluation subset: which labels count for which of its true
cell types in a method's constrained mode (``util.label_constraint``).

Usage, in the evaluation environment:
    constrain_labels.py --input <subset.h5ad> --config <dataset.yml> --cache_dir <cache> --out <tsv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_ontology_context, load_review_status  # noqa: E402
from util.cell_ontology_similarity import download_cl_ontology  # noqa: E402
from util.label_constraint import closure, write  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="evaluation subset (h5ad)")
    p.add_argument("--config", required=True, help="dataset YAML (annotation_key)")
    p.add_argument("--cache_dir", required=True,
                   help="holds the Cell Ontology release and cell_type_ontology_mappings.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    key = (yaml.safe_load(Path(args.config).read_text()) or {}).get("annotation_key", "cell_type")
    listed = sorted(ad.read_h5ad(args.input, backed="r").obs[key].astype(str).unique())
    label_index, taxonomy = load_ontology_context(download_cl_ontology(Path(args.cache_dir)))
    _reviewed, review_map = load_review_status(Path(args.cache_dir))
    table = closure(listed, label_index, taxonomy, review_map)
    write(table, args.out)
    print(f"{args.out}: {len(table)} labels count for {len(set(table.values()))} of the "
          f"{len(listed)} listed cell types")


if __name__ == "__main__":
    main()
