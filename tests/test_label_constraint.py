"""Checks for the label constraint of the constrained baselines (bin/util/label_constraint.py and
its R copy): a label counts for the nearest listed type at or above it and never for one below it,
and the table survives the round trip through its file.

Run with pytest, or directly in the evaluation environment: python tests/test_label_constraint.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bin"), str(ROOT / "tests")]
from util.label_constraint import closure, constrain, write  # noqa: E402
from util.label_normalization import label_key  # noqa: E402
from test_scoring import TOY, pinned_ontology  # noqa: E402

# TOY: root -> A, B; A -> C, F; B -> C; C -> D, E. Each term has a name and an id.
NAMES = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta", "E": "epsilon", "F": "phi",
         "root": "cell"}
TERMS = {t: SimpleNamespace(id=t, name=n) for t, n in NAMES.items()}
TOY_INDEX = {label_key(k): term for term in TERMS.values() for k in (term.id, term.name)}


def toy_table(listed, review_map=None):
    return closure(listed, TOY_INDEX, TOY, review_map or {})


def test_a_label_counts_for_its_nearest_listed_type_and_never_below_it():
    table = toy_table(["alpha", "gamma"])
    assert table["delta"] == table["d"] == "gamma"      # C is one step above D, A two
    assert table["phi"] == "alpha" and table["gamma"] == "gamma"
    assert "beta" not in table                          # above gamma: never stretched down
    assert "cell" not in table and "root" not in table


def test_reviewed_mappings_count_but_a_term_name_wins():
    table = toy_table(["gamma"], {"my delta": "delta", "alpha": "delta", "junk": "cell"})
    assert table["my delta"] == "gamma"
    assert "alpha" not in table                         # alpha IS a term (coarser), whatever the review says
    assert "junk" not in table                          # mapped to the root, which never counts


def test_the_root_is_never_an_anchor():
    assert toy_table(["cell"]) == {}


def test_the_table_survives_its_file():
    table = {'na': "NA", 'a "quoted" label': "x", "α cell": "y", "#hash": "z", "two  spaces": "w"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.tsv"
        write({label_key(k): v for k, v in table.items()}, path)
        kept = constrain(list(table), path, "test", ["NA", "x", "y", "z", "w", "missing"])
    assert kept == table


def test_closure_matches_a_brute_force_check_on_the_cell_ontology():
    ont = pinned_ontology()
    if ont is None:
        return
    from evaluate import load_review_status
    from util.cell_ontology_similarity import CellTaxonomy, build_label_index
    from util.label_constraint import cl_ids
    taxonomy, label_index = CellTaxonomy.from_ontology(ont), build_label_index(ont)
    _reviewed, review_map = load_review_status(ROOT / "cache")
    listed = ["T cell", "CD4-positive, alpha-beta T cell", "monocyte", "type B pancreatic cell", "delta"]
    table = closure(listed, label_index, taxonomy, review_map)
    ids = dict(zip(listed, cl_ids(listed, label_index, review_map)))
    labels = list(label_index)[:3000] + list(review_map)
    for label, term in zip(labels, cl_ids(labels, label_index, review_map)):
        above = [(nx.shortest_path_length(taxonomy.graph, t, term), t, name)
                 for name, t in ids.items()
                 if term in taxonomy and t in taxonomy and nx.has_path(taxonomy.graph, t, term)
                 and taxonomy.is_cell_type(t)]
        assert table.get(label_key(label)) == (min(above)[2] if above else None), label


def test_the_r_copy_agrees():
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda or subprocess.run([conda, "run", "-n", "singler", "true"],
                                   capture_output=True).returncode:
        return                                          # no R environment here
    labels = ["Delta cells", "alpha", "Gamma", "a \"quoted\" one", "unknown"]
    table = {"delta cells": "delta", "gamma": "gamma", 'a "quoted" one': "q"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.tsv"
        write(table, path)
        script = (f'source("{ROOT / "bin" / "util" / "label_constraint.R"}"); '
                  f'k <- constrain(c({", ".join(map(repr, labels))}), "{path}", "test", c("delta", "q")); '
                  'cat(paste(names(k), k, sep="\\t"), sep="\\n")')
        rscript = 'R_LIBS_USER="$CONDA_PREFIX/lib/R/library" "$CONDA_PREFIX/bin/Rscript" -e "$0"'
        out = subprocess.run([conda, "run", "-n", "singler", "bash", "-c", rscript, script],
                             capture_output=True, text=True, check=True).stdout
        kept_r = dict(line.split("\t") for line in out.splitlines() if line)
        assert kept_r == constrain(labels, path, "test", ["delta", "q"])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
