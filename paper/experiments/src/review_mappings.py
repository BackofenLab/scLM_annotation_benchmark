#!/usr/bin/env python
"""Resolve unreviewed ontology mappings, under one explicit rule.

    A predicted label maps to a Cell Ontology term if -- after normalisation -- it IS that term
    (or one of its exact/related synonyms). Everything else maps to `cell`, the ontology root,
    and is thereby scored as wrong.

Normalisation lowercases, strips punctuation and collapses whitespace, so `'( retinal cone cell)'`,
`'* astrocyte'` and `'<retinal ganglion cell>'` resolve to their terms. Those are unambiguous
answers wearing formatting noise, and refusing them would penalise a model for punctuation.

WHAT IS DELIBERATELY NOT DONE, and why it matters more than what is:

Substring extraction. A CL term appears somewhere inside ~38% of the base model's answers, so
pulling it out would look like a large rescue. It is not one. Those answers are typically ranked
lists of guesses --

    "('hypendymal cell', 'enteric neuron', 'vascular leptomeningeal cell (mmus)', ...)"
    "(1) delta (2) pancreatic stellate cell (3) pancreatic ductal cell (4) mac"

-- and choosing a term out of a list means scoring a decision the model never made. It would inflate
exactly the arm whose weakness is the finding: a base model that cannot commit to one label has not
annotated the cell, and the study should say so rather than pick for it. The prompt asks for a bare
label in four numbered rules; ignoring all four is a failure to answer, not a formatting quibble.

The one exception is a single label behind a fixed lead-in. Decoded greedily, the base model answers
`the cell type is: pancreatic a cell` and three variants of that phrase (LEAD_IN). The lead-in is
removed and the rest must then resolve as a whole under the rules below, so the model's own
commitment is scored and a list behind a lead-in still goes to `cell`. This is an extra curation
step, and the Methods state it.

Mapping to `cell` rather than leaving the label unmapped is what makes that failure MEASURABLE.
evaluate.py gates a run on scoring coverage, so wholly unmapped output yields NaN -- the arm simply
vanishes from the results table. Sending it to the root term keeps it in the table and scores it as
wrong, which is the honest treatment: we know the answer was not the true cell type.

This mirrors the benchmark's own convention, where reviewed gene-symbol labels (`0610012g03rik`)
already carry `chosen_match = cell`, so the two scales stay comparable.

Re-runnable: only rows with `reviewed == False` are touched, so it can be run again after new
predictions add new labels.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT_TERM = "cell"          # CL:0000000, the ontology root: "some cell, unspecified"


#: Greek letters the models use where CL spells the word out (`CD8-positive, αβ T cell`).
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "µ": "mu"}

#: Equivalences no string rule can derive, each checked against the ontology by hand. These are
#: naming conventions, not guesses: `delta cell of pancreas` and `pancreatic D cell` are the same
#: cell under two vocabularies, and `Th17` is the universal abbreviation for `T-helper 17 cell`.
#: Kept as an explicit table so every judgement call is reviewable in one place rather than encoded
#: in a similarity threshold. Volumes are the number of predictions each covers in A1 + A4.
CURATED = {
    "cd56-dim natural killer cell, human": "CD16-positive, CD56-dim natural killer cell, human",
    "delta cell of pancreas": "pancreatic D cell",          # delta cell == D cell
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
    # Generalisations to a parent term. Legitimate because the predicted cell IS one of these: a
    # retinal microglial cell is a microglial cell, and cortical interneurons are GABAergic. The
    # assignment can only match a truth label that is itself the parent, so it cannot manufacture a
    # correct answer for a cell of some other type.
    "retinal microglial cell": "microglial cell",
    "retinal microglial": "microglial cell",
    "cortical gabaergic interneuron": "cortical interneuron",
}


def normalise(label):
    """Canonical form for matching: diacritics, Greek letters and punctuation all removed.

    Aggressive on purpose. The models emit `naïve`, non-breaking hyphens (U+2011), narrow no-break
    spaces (U+202F) and `CD8+` where CL writes `CD8-positive`, and an earlier, gentler version of
    this turned `naïve` into `na ve` -- destroying the word it was meant to preserve. Punctuation is
    collapsed to spaces on BOTH sides, so `CD8-positive, alpha-beta T cell` and
    `cd8‑positive alpha‑beta t cell` land on the same key.
    """
    s = str(label)
    for glyph, word in GREEK.items():
        s = s.replace(glyph, f" {word} ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))    # naïve -> naive
    s = s.lower().replace("+", " positive ")                     # CD8+ -> CD8 positive
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cl_index(owl_path):
    """Normalised name/synonym -> canonical CL term name.

    Synonyms are included because the ontology's preferred label is often not what anyone writes:
    `retinal cone cell` is the CL name, but models say `cone photoreceptor`. Scoring the latter as
    nonsense would measure vocabulary, not biology. Obsolete terms are skipped -- mapping onto a
    deprecated term would be worse than not mapping.
    """
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


MIN_PREFIX_RATIO = 0.5

#: The sentence the base model wraps one label in, on the normalised text (module docstring).
LEAD_IN = re.compile(r"^the cell type (?:of (?:the|this) gene (?:expression )?profile )?is ")


_CURATED_NORM = None  # built once in review(), keyed on the normalised label


def resolve(label, index, keys):
    """Best CL term for one label, or None. Rules ordered from certain to merely safe.

    Truncation is the dominant failure of the API arms -- `'cd141-positive myeloid dendritic'`,
    `'central memory cd8-positive, alpha-beta'` -- so a label that is a PREFIX of exactly one CL
    term is completed. Uniqueness does the work: `'cd8-positive, alpha-beta'` prefixes three
    different terms and is refused, because guessing between them would invent a subtype.

    MIN_PREFIX_RATIO guards the other failure of prefix matching, which uniqueness cannot catch:
    `'fibrocyte'` prefixes exactly one term, `'fibrocyte of adventitia of ureter'`, and completing
    it would turn a generic answer into a specific anatomical claim the model never made. Requiring
    at least half the term to be present distinguishes a truncation from a coincidence.

    Fuzzy matching is NOT used at any threshold. At 0.90 it rewrote biology --
    `'a enteroendocrine cell'` -> `'P enteroendocrine cell'`, and
    `'cd151-positive...'` -> `'CD16-positive...'` -- changing which marker or subtype was named.
    That can convert a wrong answer into a right one, which is the worst error a benchmark can make.
    """
    n = LEAD_IN.sub("", normalise(label))
    if not n:
        return None
    # Hand-checked equivalences win over every derived rule: they encode naming conventions the
    # string rules cannot see, and a later rule change must not quietly override a human decision.
    if n in _CURATED_NORM:
        return _CURATED_NORM[n]
    if n in index:                                          # already a term or synonym
        return index[n]
    stripped = re.sub(r"^cl \d+ ", "", n)                   # 'cl:0000125 astrocyte'
    if stripped != n and stripped in index:
        return index[stripped]
    # Leading articles, list numbering and CL wrappers: `a paneth cell`, `a. cd141-positive
    # myeloid dendritic cell`, `[cl: retinal pigment epithelium]`. These decorate a term without
    # changing it, so removing them is safe -- unlike substring extraction, the ENTIRE remainder
    # must still match a term exactly, so nothing is being picked out of a longer answer.
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
    """Add rows for labels that appear in predictions but not yet in the mappings file.

    Normally map_predictions.py discovers new labels during scoring, which forces the sequence
    score -> review -> score again: the first pass exists only to find out what needs reviewing.
    Seeding here collapses that to one pass, which matters because a scoring pass is ~35 minutes
    and the label set changes every time the parser is corrected.

    Labels are added with `reviewed = False` so the normal rules decide them; nothing is resolved
    here. Existing rows are never touched.
    """
    # LOWERCASED, because map_predictions.py stores labels lowercased. Seeding in the original case
    # silently defeats the whole point: the scorer looks up `cell-nd3 rdh8`, finds no row for it
    # (ours says `Cell-ND3 RDH8`), and appends its own copy as unreviewed -- which is exactly the
    # second review pass the seeding exists to avoid. Observed as 240 unreviewed rows appearing
    # mid-run after the file had been reported clean.
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

    todo = frame["reviewed"] == False  # noqa: E712 - the column holds real booleans
    if revisit:
        # Rows already sent to the root are reconsidered when the rules improve. Rows carrying a
        # SPECIFIC term are never touched: those were either matched exactly or curated by hand, and
        # a later rule change must not silently overwrite a human decision.
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
    p.add_argument("--mappings", default=str(root / "cache" / "cell_type_ontology_mappings.csv"))
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
