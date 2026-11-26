#!/usr/bin/env python
"""Imperfect candidate lists: draw them, and score the study that uses them.

The benchmark's constrained setting hands a model the exact list of the dataset's cell types, which
no user has. This study replaces it with imperfect lists, one condition each (``lists`` in
config.yml), all in mode M2:

    E     the true types, the benchmark's own list (identical prompts, asked again)
    EU    E, with the option to answer "uncertain"
    L1    relatives added for half of the true types: a parent or a child (mixed granularity)
    L2-k  k absent types added: children of a true type's parent that are no true type, nor an
          ancestor or descendant of one
    L3    a quarter of the true types removed, at least one, with "uncertain" offered
    L4    L1 + L2-5 + L3

Per (dataset, seed), three orders are drawn once -- the true types to remove, the relatives to add,
the absent types to add -- and each condition takes a prefix of each. L2-2 therefore lies inside
L2-5, and L4 is exactly the union of what L1, L2-5 and L3 do. Relatives and absent types are drawn
one per true type, the types taken in random order, so they spread over the lineages instead of
piling up under the parent with the most children ("fibroblast" has dozens). Added types come from
the CELLxGENE subset of the Cell Ontology, the vocabulary annotators use, without species- or
taxon-qualified terms; parents are never broad (``BROAD_TERMS_BELOW``). True types keep their raw
labels, so E is the benchmark's list.

Each list is a text file, one name per line, sorted the way the prompt builders sort:
``results_lists/lists/<dataset>__<condition>__seed<seed>.txt``. It is the one source for the
prompts, the scLLM wrappers and the reference methods. Which entries are removed, relatives or
absent is read off the ontology when scoring, not stored a second time.

Scoring reuses the benchmark's scoring (``score_run`` in ``bin/evaluate.py``) on every run
that ``score.py --study lists`` scored, and adds what the lists need: balanced accuracy over the
types still listed (removing a type should not be charged for cells no listed type could name) and
over the types a reference method can reach; abstention, an answer of exactly "uncertain", on the
removed and on the listed types; and the answer types (``CellTaxonomy.lineage``) over all types and
over the removed ones, which tell abstaining from naming a listed ancestor (HIHA's "T cell", or an
L4 parent). Every summary is a mean over true types, as balanced accuracy is.

Usage, in the evaluation environment (pronto):
    python src/lists.py --write      # draw every list
    python src/lists.py --score      # after score.py --study lists
"""
from __future__ import annotations

import argparse
import math
import sys
import zlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT / "bin"), str(REPO_ROOT / "paper" / "scripts")]


def condition_modes(cfg):
    """The list conditions as context modes: the base mode, and whether "uncertain" is offered."""
    base = cfg["modes"][cfg["lists"]["mode"]]
    return {name: {**base, "uncertain": bool(spec.get("uncertain"))}
            for name, spec in cfg["lists"]["conditions"].items()}


def list_path(tree, dataset, condition, seed):
    return Path(tree) / "lists" / f"{dataset}__{condition}__seed{seed}.txt"


def read_list(tree, dataset, condition, seed):
    return list_path(tree, dataset, condition, seed).read_text().splitlines()


def prompt_args(cfg, tree, dataset):
    """What the prompt builders take for this study: its arms, its conditions as modes, and the
    list of each (condition, seed)."""
    return {"arms": {name: cfg["arms"][name] for name in cfg["lists"]["arms"]},
            "modes": condition_modes(cfg),
            "candidates": lambda condition, seed: read_list(tree, dataset, condition, seed)}


def _prefix(items, share, n_types):
    """The first round-half-up(share x n_types) items, at least one; none without a share."""
    return items[:max(1, math.floor(share * n_types + 0.5))] if share else []


def draw_lists(true_labels, true_ids, taxonomy, addable, conditions, rng):
    """``{condition: sorted names}`` for one dataset and seed.

    ``true_labels`` are the raw labels, sorted, and ``true_ids`` their Cell Ontology ids in the same
    order; ``addable`` holds the ids that may be added.
    """
    graph, true = taxonomy.graph, set(true_ids)

    def parents(t):                                 # never a broad class
        return [p for p in sorted(graph.predecessors(t))
                if taxonomy.n_below[p] <= taxonomy.broad_terms_below]

    def absent_of(t):          # a parent's children that are no true type, nor above or below one
        return sorted({c for p in parents(t) for c in graph.successors(p) if c in addable
                       and all(taxonomy.lineage(other, c) == "other" for other in true)})

    def walk(options_of):
        """One new option per true type, the types taken in random order; each option list of a
        type is a side, and a side is chosen at random before an option."""
        picked = []
        for i in rng.permutation(len(true_ids)):
            sides = [s for s in ([o for o in side if o not in picked]
                                 for side in options_of(true_ids[i])) if s]
            if sides:
                side = sides[rng.integers(len(sides))]
                picked.append(side[rng.integers(len(side))])
        return picked

    removal = [true_labels[i] for i in rng.permutation(len(true_labels))]
    relatives = walk(lambda t: ([p for p in parents(t) if p in addable and p not in true],
                                [c for c in sorted(graph.successors(t))
                                 if c in addable and c not in true]))
    absent = walk(lambda t: (absent_of(t),))

    lists = {}
    for name, spec in conditions.items():
        removed = set(_prefix(removal, spec.get("remove"), len(true_labels)))
        added = _prefix(relatives, spec.get("relatives"), len(true_ids)) + absent[:spec.get("extra", 0)]
        lists[name] = sorted([label for label in true_labels if label not in removed]
                             + [taxonomy.names[c] for c in added])
    return lists


def ontology(cache_dir):
    """The pinned Cell Ontology of the scoring cache: label index, taxonomy and the ids that may be
    added to a list (CELLxGENE subset, no species- or taxon-qualified name)."""
    import pronto
    from util.cell_ontology_similarity import (CL_FILE, SPECIES_TAG, CellTaxonomy,
                                               build_label_index)
    ont = pronto.Ontology(str(Path(cache_dir) / CL_FILE))
    taxonomy = CellTaxonomy.from_ontology(ont)
    addable = {t.id for t in ont.terms()
               if t.id in taxonomy and "cellxgene_subset" in t.subsets
               and not SPECIES_TAG.search(t.name or "") and "(sensu " not in (t.name or "")}
    return build_label_index(ont), taxonomy, addable


def eval_subset(dataset, cfg):
    ev = cfg["eval"]
    return ROOT / ev["subset_dir"] / f"{dataset}_sub{ev['size']}_{ev['strategy']}.h5ad"


def true_types(dataset, cfg, label_index, review_map):
    """The eval subset's raw labels, sorted as the prompt builders sort them, and their CL ids."""
    import anndata as ad
    from sentences import _annotation_key
    from util.label_constraint import cl_ids
    obs = ad.read_h5ad(eval_subset(dataset, cfg), backed="r").obs
    labels = sorted(obs[_annotation_key(dataset)].astype(str).unique())
    ids = cl_ids(labels, label_index, review_map)
    if None in ids:
        raise SystemExit(f"{dataset}: true labels without a Cell Ontology term: "
                         f"{[label for label, i in zip(labels, ids) if i is None]}")
    return labels, ids


def write_lists(cfg, tree):
    import numpy as np
    from util.ontology_mapping import load_review_status
    cache_dir = ROOT / cfg["scoring"]["cache_dir"]
    label_index, taxonomy, addable = ontology(cache_dir)
    _reviewed, review_map = load_review_status(cache_dir / MAPPINGS)
    (Path(tree) / "lists").mkdir(parents=True, exist_ok=True)
    for dataset in cfg["datasets"]:
        labels, ids = true_types(dataset, cfg, label_index, review_map)
        for seed in cfg["repeats"]["seeds"]:
            rng = np.random.default_rng([zlib.crc32(dataset.encode()), seed])
            for condition, names in draw_lists(labels, ids, taxonomy, addable,
                                               cfg["lists"]["conditions"], rng).items():
                list_path(tree, dataset, condition, seed).write_text("\n".join(names) + "\n")
                print(f"{dataset:15s} seed{seed} {condition:5s} {len(names):3d} types "
                      f"({len(set(names) - set(labels)):2d} added, "
                      f"{len(set(labels) - set(names)):2d} removed)")


RUN = ["arm", "dataset", "condition", "seed"]
MAPPINGS = "cell_type_ontology_mappings.csv"
PAIR_COLUMNS = ["true_label", "predicted_label", "n_cells", "prevalence", "n_direct", "lineage"]


def _answer_types(pairs, prefix):
    """Class-balanced answer-type shares per run (``manuscript_figures._answer_shares``)."""
    from manuscript_figures import _answer_shares
    if pairs.empty:
        return pd.DataFrame()
    run = pairs["arm"] + "|" + pairs["condition"] + "|" + pairs["seed"].astype(str)
    shares = _answer_shares(pairs.assign(model=run)).add_prefix(prefix)
    keys = shares.index.to_frame(index=False)
    keys[["arm", "condition", "seed"]] = keys["model"].str.split("|", expand=True)
    return shares.set_axis(pd.MultiIndex.from_frame(keys[RUN].astype({"seed": int})))


def run_metrics(cells, pairs):
    """One row per run: balanced accuracy over all, listed and reachable listed types; abstention
    on the listed and the removed types; answer types over all and over the removed types."""
    per_type = cells.groupby(RUN + ["true_label"]).agg(
        hit=("hit", "mean"), abstained=("abstained", "mean"), listed=("listed", "first"),
        reachable=("reachable", "first"))

    def mean_over(column, mask=None):
        values = per_type[column] if mask is None else per_type[column].where(mask)
        return values.groupby(level=RUN).mean()

    return pd.concat([pd.DataFrame({
        "balanced_accuracy": mean_over("hit"),
        "balanced_accuracy_listed": mean_over("hit", per_type["listed"]),
        "balanced_accuracy_reachable": mean_over("hit", per_type["listed"] & per_type["reachable"]),
        "abstained_listed": mean_over("abstained", per_type["listed"]),
        "abstained_removed": mean_over("abstained", ~per_type["listed"]),
    }), _answer_types(pairs, "all_"), _answer_types(pairs[~pairs["listed"]], "removed_")], axis=1)


def score_study(cfg, tree):
    from evaluate import score_run
    from util.ontology_mapping import canonical_names, load_review_status
    cache_dir = ROOT / cfg["scoring"]["cache_dir"]
    label_index, taxonomy, _addable = ontology(cache_dir)
    reviewed, review_map = load_review_status(cache_dir / MAPPINGS)
    cells, pairs, overall, gated = [], [], [], []
    for mapped in sorted(Path(tree, "scored").glob("*/predictions_mapped.csv")):
        arm, dataset, condition, variant, seed = mapped.parent.name.split("__")
        seed = int(seed.removeprefix("seed"))
        frame = pd.read_csv(mapped, dtype={"cell_id": str})
        _coverage, metrics, tables = score_run(frame, label_index, taxonomy, reviewed, review_map,
                                               cfg["scoring"]["min_scoring_coverage"])
        if metrics is None:
            gated.append(mapped.parent.name)
            continue
        scored = tables["frame"]
        run_cells = pd.DataFrame({
            "dataset": dataset, "cell_id": scored["cell_id"].astype(str),
            "true_label": scored["true_labels"],
            "hit": (scored["true_labels"] == scored["predicted_labels"]).astype(int)})
        run_pairs = tables["pairs"][PAIR_COLUMNS].assign(dataset=dataset)
        overall_scores = tables["per_class"].iloc[-1]
        listed = set(canonical_names(pd.Series(read_list(tree, dataset, condition, seed)),
                                     label_index, review_map)[0])
        raw = pd.read_csv(Path(tree, "predictions", arm, dataset, condition, variant,
                               f"seed{seed}", "predictions.csv"), dtype={"cell_id": str})
        reachable = raw.set_index("cell_id")["reachable"] if "reachable" in raw else None
        answer = frame.set_index("cell_id")["original_predicted_labels"].astype(str)
        run = {"arm": arm, "condition": condition, "seed": seed}
        cells.append(run_cells.assign(
            **run, listed=lambda d: d["true_label"].isin(listed),
            abstained=lambda d: d["cell_id"].map(answer).eq("uncertain"),
            reachable=lambda d: True if reachable is None else d["cell_id"].map(reachable)))
        pairs.append(run_pairs.assign(
            **run, listed=lambda d: d["true_label"].isin(listed)))
        overall.append({"dataset": dataset, **run, "agreement": overall_scores["agreement"],
                        "lin": overall_scores["lin"]})
    cells, pairs = pd.concat(cells, ignore_index=True), pd.concat(pairs, ignore_index=True)
    runs = run_metrics(cells, pairs).join(pd.DataFrame(overall).set_index(RUN))
    pairs[RUN + [c for c in pairs.columns if c not in RUN]].to_csv(
        Path(tree, "confusion_pairs.csv"), index=False)
    runs.to_csv(Path(tree, "runs.csv"))
    print(f"{len(runs)} runs scored, {len(gated)} below the coverage gate{': ' if gated else ''}"
          f"{', '.join(gated)}\n-> {tree}/{{runs,confusion_pairs}}.csv")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "config.yml"))
    p.add_argument("--write", action="store_true", help="draw every list")
    p.add_argument("--score", action="store_true",
                   help="run metrics and confusion pairs of every scored run")
    args = p.parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    tree = ROOT / cfg["studies"]["lists"]
    if args.write:
        write_lists(cfg, tree)
    if args.score:
        score_study(cfg, tree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
