"""Restrict a method's labels to a list of expected cell types: the constrained mode of a method
whose labels come from a fixed vocabulary (a reference, a marker database, a trained classifier).

The list holds the dataset's true cell types as raw labels. A label counts for a listed type when
its Cell Ontology term is that type or lies below it, and is then answered as the listed type's raw
name: the nearest one if several lie above (fewest is_a steps, ties by term id). A coarser label
never counts, since that would stretch a broad answer down to a specific type, and a label without
a term counts for nothing.

``closure`` writes the rule out once per list, as a table over every string the scorer resolves
(Cell Ontology names and ids and the keys of the reviewed mappings), so it needs no method's
vocabulary and runs before the methods. A method keeps the labels the table contains
(``constrain``, mirrored for the R wrappers in ``label_constraint.R``) and warns about the listed
types none of them reaches.

Only pandas is imported at module level, so every method's environment can use ``constrain``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .label_normalization import label_key

COLUMNS = ["label", "listed"]


def cl_ids(labels, label_index, review_map):
    """The Cell Ontology id of each label as the scorer resolves it (a term name or id, else a
    reviewed mapping), or None."""
    from .cell_ontology_similarity import map_string_to_term
    from .ontology_mapping import canonical_names
    names, _unresolved = canonical_names(pd.Series(list(labels), dtype=str), label_index,
                                         review_map)
    return [getattr(map_string_to_term(n, label_index), "id", None) for n in names]


def closure(listed, label_index, taxonomy, review_map):
    """``{label key: listed name}`` for every label the scorer resolves to a term at or below a
    listed type."""
    ids = cl_ids(listed, label_index, review_map)
    name_of = dict(zip(ids, listed))
    nearest = {}
    for t in {t for t in ids if taxonomy.is_cell_type(t)}:
        for term, steps in taxonomy.below(t).items():
            nearest[term] = min(nearest.get(term, (steps, t)), (steps, t))
    labels = [*label_index, *review_map]
    return {label_key(label): name_of[nearest[t][1]]
            for label, t in zip(labels, cl_ids(labels, label_index, review_map)) if t in nearest}


def write(table, path):
    pd.DataFrame(sorted(table.items()), columns=COLUMNS).to_csv(path, sep="\t", index=False)


def constrain(labels, path, method, truth):
    """``{label: listed name}`` for the method's labels that the table at ``path`` contains. Warns
    on stderr about the listed types (``truth``) that none of them reaches."""
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    lookup = dict(zip(table["label"], table["listed"]))
    kept = {label: lookup[label_key(label)] for label in labels if label_key(label) in lookup}
    missing = sorted(set(truth) - set(kept.values()))
    if missing:
        print(f"WARNING {method} ({Path(path).name}): {len(missing)} of {len(set(truth))} listed "
              f"cell types have no matching label: {'; '.join(missing)}", file=sys.stderr)
    return kept
