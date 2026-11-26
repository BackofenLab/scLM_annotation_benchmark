#!/usr/bin/env python
"""Write the label constraint of one test set: which labels count for which of its expected cell
types in the constrained setting of a baseline (``util/label_constraint.py``). The expected cell
types are those of the test set's labels, or the names in --expected, one per line.

    constrain_labels.py --input <test_set.h5ad> --config <dataset.yml> \
        --mappings <cell_type_ontology_mappings.csv> --cache_dir <cache> --out <constraint.tsv> \
        [--expected <expected_cell_types.txt>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.cell_ontology_similarity import load_ontology_context  # noqa: E402
from util.label_constraint import closure, write  # noqa: E402
from util.ontology_mapping import load_review_status  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="test set (h5ad)")
    p.add_argument("--config", required=True, help="dataset configuration (annotation_key)")
    p.add_argument("--mappings", required=True, help="the mapping table")
    p.add_argument("--cache_dir", default="cache", help="where the Cell Ontology release is stored")
    p.add_argument("--out", required=True)
    p.add_argument("--expected", default=None,
                   help="the expected cell types, one name per line; default: those of the labels")
    args = p.parse_args()
    if args.expected:
        listed = sorted({line.strip() for line in Path(args.expected).read_text().splitlines()
                         if line.strip()})
    else:
        key = (yaml.safe_load(Path(args.config).read_text()) or {}).get("annotation_key", "cell_type")
        listed = sorted(ad.read_h5ad(args.input, backed="r").obs[key].astype(str).unique())
    label_index, taxonomy = load_ontology_context(Path(args.cache_dir))
    _reviewed, review_map = load_review_status(args.mappings)
    table = closure(listed, label_index, taxonomy, review_map)
    write(table, args.out)
    print(f"{args.out}: {len(table)} labels count for {len(set(table.values()))} of the "
          f"{len(listed)} listed cell types")


if __name__ == "__main__":
    main()
