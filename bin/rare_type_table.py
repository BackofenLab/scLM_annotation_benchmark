#!/usr/bin/env python
"""Build the per-class rare-type outcome table that Results R3 and its figure read.

One row per (method, dataset, true cell type) with three things the aggregate snapshot cannot
express:

* ``orig_prevalence`` -- the cell type's share of the FULL dataset, not of the evaluation subset.
  This distinction is the reason the table exists. The subsets are stratified with a 20-cell
  floor, so 54 of the 87 scored cell types sit at exactly that floor and the subset's own
  prevalence is mostly an artifact of the sampler. True prevalence spans four orders of magnitude
  (HIHA alone spans 15,000x, compressed to 5x in the subset), so it is the only variable on which
  a rarity analysis means anything.
* ``recall`` -- the method's recall for that cell type.
* ``emitted`` -- whether the method ever produced that label anywhere in the dataset, which
  separates two failures that recall alone conflates: never having the label in the output
  vocabulary at all, versus producing it but assigning it to the wrong cells.

Labels are canonicalized to Cell Ontology identity on both sides before joining, because the
subset QC files record the dataset's raw labels while the predictions carry mapped ones.

Needs ``pronto`` and the cached CL ontology, so run it in the evaluation environment:
    conda run -n evaluate python bin/rare_type_table.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_model_comparison import (  # noqa: E402
    EXCLUDED_BASES, categorize_model, derive_mode, group_name, model_base,
)

DEFAULT_OUT = "cache/rare_type_outcomes_1000_stratified.csv"


def canonicalizer(ontology_path, mappings_path):
    """Return a function mapping any label string to its canonical CL name.

    Mirrors evaluate.py's ``_canonicalize_to_cl``: resolve against the ontology first, fall back
    to a reviewed manual mapping, and otherwise keep the lowercased string so two distinct classes
    are never silently merged.
    """
    import pronto  # local import: only this script needs it
    from util.cell_ontology_similarity import build_label_index, map_string_to_term

    index = build_label_index(pronto.Ontology(str(ontology_path)))
    reviewed = {}
    if Path(mappings_path).exists():
        table = pd.read_csv(mappings_path, sep=";")
        reviewed = {str(r.predicted_label).strip().lower(): str(r.chosen_match)
                    for r in table.itertuples()
                    if str(r.reviewed) == "True" and pd.notna(r.chosen_match)}

    def canon(label):
        text = str(label).strip()
        term = map_string_to_term(text, index)
        if term is None and text.lower() in reviewed:
            term = map_string_to_term(reviewed[text.lower()], index)
        return term.name.lower() if term is not None else text.lower()

    return canon


def true_prevalence(canon, qc_glob):
    """Cell-type share of the FULL dataset, summed over raw labels that share a CL identity."""
    frames = [pd.read_csv(f).assign(dataset=f.split("/")[2]) for f in glob.glob(qc_glob)]
    if not frames:
        raise SystemExit(f"No subset QC files matched {qc_glob}")
    qc = pd.concat(frames)
    qc = qc[~qc["dropped"].fillna(False)].copy()
    qc["k"] = [canon(x) for x in qc["cell_type"]]
    return qc.groupby(["dataset", "k"], as_index=False).agg(
        orig_prevalence=("original_proportion", "sum"),
        orig_count=("original_count", "sum"))


def outcomes(canon, pred_glob):
    """Per (method, dataset, class) recall and whether the label was ever emitted."""
    rows = []
    for path in glob.glob(pred_glob):
        parts = path.split("/")
        model_dir, dataset = parts[3], parts[4]
        frame = pd.read_csv(path)
        if not {"true_labels", "predicted_labels"} <= set(frame.columns):
            continue
        if "kept" in frame.columns:            # scoring excludes review-filtered rows
            frame = frame[frame["kept"] != False]
        slug = os.path.basename(path).replace("_mapped.csv", "")
        # OpenRouter runs pack every LLM into one directory, one file per model slug.
        method = slug.split("--")[-1] if "openrouter" in model_dir else model_base(model_dir)
        truth = frame["true_labels"].map(canon)
        pred = frame["predicted_labels"].map(canon)
        emitted = set(pred)
        for label, idx in truth.groupby(truth).groups.items():
            hit = pred.loc[idx] == label
            rows.append(dict(method=method, mode=derive_mode(model_dir), dataset=dataset,
                             k=label, n_cells=len(idx), recall=float(hit.mean()),
                             emitted=bool(label in emitted)))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions",
                   default="cache/models/predictions/*/*/1000_stratified/*_mapped.csv")
    p.add_argument("--subset-qc",
                   default="cache/subset_qc/*/sub1000_stratified/cell_type_distribution_stratified.csv")
    p.add_argument("--ontology", default=None,
                   help="Cell Ontology file (default: the pinned release in cache/)")
    p.add_argument("--mappings", default="cache/cell_type_ontology_mappings.csv")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    from util.cell_ontology_similarity import download_cl_ontology
    canon = canonicalizer(args.ontology or download_cl_ontology(Path("cache")), args.mappings)
    prevalence = true_prevalence(canon, args.subset_qc)
    frame = outcomes(canon, args.predictions)

    merged = frame.merge(prevalence, on=["dataset", "k"], how="left")
    unmatched = merged["orig_prevalence"].isna()
    if unmatched.any():
        names = merged.loc[unmatched, ["dataset", "k"]].drop_duplicates()
        print(f"WARNING: {unmatched.sum()} rows ({unmatched.mean():.1%}) have no true prevalence "
              f"and are dropped, covering {len(names)} labels:", file=sys.stderr)
        print(names.to_string(index=False), file=sys.stderr)
    merged = merged[~unmatched].copy()

    merged["family"] = merged["method"].map(lambda m: categorize_model(f"{m}_blind"))
    merged = merged[~merged["method"].isin(EXCLUDED_BASES)]
    merged = merged[merged["family"].isin(["LLM", "scLLM", "classical"])]
    merged["block"] = [group_name(f"{m}_{mode}") for m, mode in zip(merged["method"], merged["mode"])]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    classes = merged.drop_duplicates(["dataset", "k"])
    print(f"{len(merged)} rows | {merged['method'].nunique()} methods | "
          f"{len(classes)} cell types | prevalence "
          f"{classes['orig_prevalence'].min():.2e} to {classes['orig_prevalence'].max():.3f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
