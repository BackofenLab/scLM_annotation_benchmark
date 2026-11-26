#!/usr/bin/env python
"""Resolve the unreviewed rows of the study's mapping table by fixed rules.

A predicted label maps to a Cell Ontology term if, after normalisation (``normalise``), it equals
the term's name or one of its exact or related synonyms, or an entry of ``CURATED``; ``resolve``
lists the further rules. A label that names one term after a fixed lead-in (``LEAD_IN``, as in
"the cell type is: pancreatic a cell") is resolved without the lead-in. Terms are not picked out
of longer answers such as lists of guesses. Every other label maps to ``cell``, the ontology root,
which is scored as wrong, as the benchmark's table does for gene symbols. Unmapped labels would
instead lower a run's scoring coverage in ``bin/evaluate.py``.

Only rows with ``reviewed == False`` change; ``--revisit`` adds the rows mapped to ``cell``.
``--seed-from-predictions`` first adds the labels of the predictions that the table lacks.

    python src/review_mappings.py --seed-from-predictions [DIR] [--apply]
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from score import MAPPINGS

ROOT_TERM = "cell"          # CL:0000000, the ontology root: "some cell, unspecified"


#: Greek letters the models use where CL spells the word out (`CD8-positive, αβ T cell`).
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "µ": "mu"}

#: Equivalences that the string rules cannot derive, such as other names or abbreviations of a
#: term (`th17 cell` for `T-helper 17 cell`).
CURATED = {
    "cd56-dim natural killer cell, human": "CD16-positive, CD56-dim natural killer cell, human",
    "delta cell of pancreas": "pancreatic D cell",          # the delta cell is the D cell
    "mature gamma-delta t cell, human": "mature gamma-delta T cell",
    "retinal pigment epithelium cell": "retinal pigment epithelial cell",
    "th17 cell": "T-helper 17 cell",
    "th1 cell": "T-helper 1 cell",
    "t helper 17 cell": "T-helper 17 cell",
    "retinal off bipolar cell": "OFF-bipolar cell",
    "granule cell of the dentate gyrus": "dentate gyrus granule cell",
    "m müller glial cell": "Mueller cell",
    "pancreatic polypeptide": "pancreatic PP cell",
    "pancreatic pol": "pancreatic PP cell",
    # Generalisations to a parent term; they match only a true label that is the parent itself.
    "retinal microglial cell": "microglial cell",
    "retinal microglial": "microglial cell",
    "cortical gabaergic interneuron": "cortical interneuron",
}


def normalise(label):
    """Lower-case key for matching: Greek letters spelled out, diacritics removed, "+" read as
    "positive", and every run of other characters than letters and digits made one space."""
    s = str(label)
    for glyph, word in GREEK.items():
        s = s.replace(glyph, f" {word} ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))    # naïve becomes naive
    s = s.lower().replace("+", " positive ")                     # CD8+ becomes cd8 positive
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cl_index(owl_path):
    """The CL term name for each normalised name and exact or related synonym of a CL term that
    is not obsolete."""
    import pronto
    onto = pronto.Ontology(str(owl_path))
    index = {}
    for term in onto.terms():
        if not term.id.startswith("CL:") or term.obsolete:
            continue
        if term.name:
            index.setdefault(normalise(term.name), term.name)
        for syn in term.synonyms:
            if syn.scope in ("EXACT", "RELATED"):
                index.setdefault(normalise(syn.description), term.name)
    return index


MIN_PREFIX_RATIO = 0.5   # share of the term's key that a completed prefix must cover

#: The lead-in before a single label in answers of base Gemma (A2), on the normalised text.
LEAD_IN = re.compile(r"^the cell type (?:of (?:the|this) gene (?:expression )?profile )?is ")


_CURATED_NORM = None  # built once in review(), keyed on the normalised label


def resolve(label, index, keys):
    """The CL term name for one label, or None.

    The rules in order: a curated equivalence; a term name or synonym, also without a CL id,
    leading article or list number, or with " cell" or " neuron" added; and the completion of a
    cut-off label of at least six characters that starts the key of exactly one term and covers
    at least ``MIN_PREFIX_RATIO`` of it (``'cd141-positive myeloid dendritic'``). Fuzzy matching
    is not used, since it can turn one marker or subtype into another.
    """
    n = LEAD_IN.sub("", normalise(label))
    if not n:
        return None
    # Curated equivalences come first.
    if n in _CURATED_NORM:
        return _CURATED_NORM[n]
    if n in index:                                          # already a term or synonym
        return index[n]
    stripped = re.sub(r"^cl \d+ ", "", n)                   # 'cl:0000125 astrocyte'
    if stripped != n and stripped in index:
        return index[stripped]
    # Without a leading article, list number or "cl" (`a paneth cell`, `[cl: retinal pigment
    # epithelium]`); the rest must match a term as a whole.
    bare = re.sub(r"^(?:an?|the|cl)\b\s*", "", n)
    bare = re.sub(r"^\d+\s*", "", bare).strip()
    if bare and bare != n and bare in index:
        return index[bare]
    for suffix in (" cell", " neuron"):                     # 'pancreatic ductal' + ' cell'
        if (n + suffix) in index:
            return index[n + suffix]
    if len(n) >= 6:                                         # unique-prefix completion
        matches = [k for k in keys if k.startswith(n)]
        terms = {index[k] for k in matches}
        if len(terms) == 1:
            shortest = min(matches, key=len)
            if len(n) / len(shortest) >= MIN_PREFIX_RATIO:
                return terms.pop()
    return None


def seed_from_predictions(frame, predictions_root):
    """Add a row with ``reviewed = False`` for every label of the predictions under
    ``predictions_root`` that the table lacks, so the labels can be reviewed before scoring.
    Existing rows stay as they are."""
    # In lower case, as map_predictions.py stores the labels.
    known = {s.lower() for s in frame["predicted_label"].astype(str)}
    found = {}
    for path in sorted(Path(predictions_root).glob("*/*/*/*/*/predictions.csv")):
        arm = path.parents[4].name
        dataset = path.parents[3].name
        for label in pd.read_csv(path)["predicted_labels"].astype(str).unique():
            key = label.lower()
            if key not in known:
                found.setdefault(key, f"{arm}_{dataset}")
    if not found:
        print("  no new labels in predictions")
        return frame
    print(f"  seeding {len(found):,} label(s) seen in predictions but absent from the mappings file")
    additions = pd.DataFrame([{"reviewed": False, "origin": origin, "predicted_label": label,
                               "chosen_match": None, "best_match": None}
                              for label, origin in found.items()])
    return pd.concat([frame, additions], ignore_index=True)


def review(mappings_csv, owl_path, apply_changes=False, revisit=False, predictions_root=None):
    # Labels such as "none" or "n/a" must stay strings: read as NaN, they would be written back
    # empty, and the label would lose its row.
    frame = pd.read_csv(mappings_csv, sep=";", keep_default_na=False)
    if predictions_root:
        frame = seed_from_predictions(frame, predictions_root)
    index = cl_index(owl_path)
    keys = list(index)
    global _CURATED_NORM
    _CURATED_NORM = {normalise(k): v for k, v in CURATED.items()}
    print(f"CL index: {len(index):,} normalised names and synonyms")

    todo = frame["reviewed"] == False  # noqa: E712
    if revisit:
        # Rows mapped to the root are decided again; rows with a specific term stay as they are.
        todo = todo | (frame["chosen_match"] == ROOT_TERM)
    print(f"rows: {len(frame):,} total, {todo.sum():,} to (re)consider")
    if not todo.any():
        print("nothing to review")
        return frame, 0, 0

    resolved = frame.loc[todo, "predicted_label"].map(lambda x: resolve(x, index, keys))
    matched = resolved.notna()
    frame.loc[todo, "chosen_match"] = resolved.fillna(ROOT_TERM)
    frame.loc[todo, "reviewed"] = True

    print(f"\n  matched to a specific CL term : {matched.sum():,}")
    print(f"  sent to '{ROOT_TERM}' as nonsense : {(~matched).sum():,}")

    origin = frame.loc[todo, "origin"].astype(str).str.extract(r"^(A\d)")[0].fillna("bench")
    per_arm = pd.DataFrame({"origin": origin, "matched": matched.values})
    print("\n  by arm:")
    summary = per_arm.groupby("origin")["matched"].agg(total="size", matched="sum")
    summary["pct"] = (100 * summary["matched"] / summary["total"]).round(1)
    print(summary.to_string().replace("\n", "\n    "))

    if apply_changes:
        frame.to_csv(mappings_csv, sep=";", index=False)
        print(f"\nwritten: {mappings_csv}")
    else:
        print("\ndry run; pass --apply to write")
    return frame, int(matched.sum()), int((~matched).sum())


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mappings", default=str(root / "cache" / MAPPINGS))
    p.add_argument("--ontology", default=str(root / "cache" / "cl-basic_2026-06-08.obo"),
                   help="the benchmark's pinned Cell Ontology release (CL_FILE in "
                        "bin/util/cell_ontology_similarity.py)")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--seed-from-predictions", nargs="?", const=str(root / "results" / "predictions"),
                   metavar="DIR", help="add labels seen in the predictions under DIR (default: "
                                       "results/predictions) that are not yet in the file")
    p.add_argument("--revisit", action="store_true",
                   help="also reconsider rows previously sent to the root term")
    args = p.parse_args(argv)
    review(Path(args.mappings), Path(args.ontology), apply_changes=args.apply,
           revisit=args.revisit,
           predictions_root=args.seed_from_predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
