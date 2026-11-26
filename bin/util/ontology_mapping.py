"""The curated table that maps free-text cell type labels to Cell Ontology terms.

The table is a semicolon-separated file with one row per label that is not itself a Cell
Ontology term: ``reviewed``, ``origin`` (the run or dataset that first produced the label),
``predicted_label``, ``chosen_match`` (the term a reviewer chose) and ``best_match`` (the
automatic suggestion). Labels of new runs are appended as unreviewed rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pronto

from .cell_ontology_similarity import (
    CL_FILE, CL_URL, build_label_index, build_synonym_index, download_cl_ontology,
    map_string_to_term,
)
from .label_normalization import canonicalize_empty_labels

COLUMNS = ["reviewed", "origin", "predicted_label", "chosen_match", "best_match"]
SUGGESTION_ENCODER = "all-MiniLM-L6-v2"


def read_mappings(path) -> pd.DataFrame:
    """The mapping table, empty if the file does not exist. Blank cells stay empty strings."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    table = pd.read_csv(path, sep=";", keep_default_na=False)
    table["predicted_label"] = canonicalize_empty_labels(table["predicted_label"])
    return table


def load_review_status(path):
    """Reviewed labels and their targets, as ``(reviewed_labels, review_map)``.

    An answer is trusted if it is a Cell Ontology term or one of ``reviewed_labels``.
    ``review_map`` gives the chosen term of a reviewed label, keyed by the lowercased label.
    """
    table = read_mappings(path)
    reviewed = table["reviewed"].map(
        lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes"))
    rows = table[reviewed.astype(bool)]
    review_map = {str(label).strip().lower(): str(target)
                  for label, target in zip(rows["predicted_label"], rows["chosen_match"])
                  if pd.notna(target) and str(target).strip()}
    return set(rows["predicted_label"].astype(str)), review_map


def suggest_matches(labels, cache_dir) -> dict[str, str]:
    """A suggested Cell Ontology term name per label, "" if there is none.

    Uses the semantic matcher of omicverse if the package is installed, otherwise an exact match
    on term names, ids and exact synonyms. Suggestions are shown to the reviewer and never scored.
    """
    try:
        import omicverse as ov
        ontology_json = download_cl_ontology(
            cache_dir, cl_url=CL_URL.removesuffix(".obo") + ".json",
            filename=CL_FILE.removesuffix(".obo") + ".json")
        mapper = ov.single.CellOntologyMapper(
            cl_obo_file=str(ontology_json), model_name=SUGGESTION_ENCODER,
            local_model_dir=str(cache_dir))
        results = mapper.map_cells(labels)
        return {label: results.get(label, {}).get("best_match", "") for label in labels}
    except Exception as error:
        print(f"omicverse matcher not available ({error}); suggesting exact name and synonym "
              "matches instead")
    ontology = pronto.Ontology(str(download_cl_ontology(cache_dir)))
    names, synonyms = build_label_index(ontology), build_synonym_index(ontology)
    suggestions = {}
    for label in labels:
        term = map_string_to_term(label, names)
        if term is None:
            term = map_string_to_term(label, synonyms)
        suggestions[label] = term.name if term is not None else ""
    return suggestions


def add_unreviewed(labels, path, cache_dir, origin) -> int:
    """Append the labels that the table does not hold yet as unreviewed rows, each with a
    suggested match. Returns the number of rows added."""
    table = read_mappings(path)
    known = set(table["predicted_label"])
    new = [label for label in labels if label not in known]
    if not new:
        return 0
    suggestions = suggest_matches(new, cache_dir)
    matches = [suggestions.get(label, "") for label in new]
    rows = pd.DataFrame({"reviewed": False, "origin": origin, "predicted_label": new,
                         "chosen_match": matches, "best_match": matches})
    table = (pd.concat([table, rows], ignore_index=True)
             .drop_duplicates(subset=["predicted_label"], keep="first")
             .sort_values("predicted_label").reset_index(drop=True))
    table = table[COLUMNS + [c for c in table.columns if c not in COLUMNS]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep=";", index=False)
    print(f"{len(new)} new label(s) added to {path} for review")
    return len(new)


def resolve_labels(labels, label_index, path, cache_dir, origin, append=True, rewrite=True):
    """Labels with their matches from the table substituted, and the substitutions made.

    A label that is no Cell Ontology term is looked up in the table. With ``append``, labels
    the table does not hold yet are added as unreviewed rows. With ``rewrite``, a label is
    replaced by its ``chosen_match`` if that names a Cell Ontology term. Whether a match counts
    in scoring depends on its review status, which ``evaluate.py`` reads.
    """
    labels = list(labels)
    unknown = {label for label in set(labels) if map_string_to_term(label, label_index) is None}
    if not unknown:
        return labels, {}
    if append:
        add_unreviewed(sorted(unknown), path, cache_dir, origin)
    if not rewrite:
        return labels, {}
    table = read_mappings(path)
    chosen = dict(zip(table["predicted_label"], table["chosen_match"]))
    replacements = {}
    for label in unknown:
        match = str(chosen.get(label, "")).strip()
        if match and match != label and map_string_to_term(match, label_index) is not None:
            replacements[label] = match
    return [replacements.get(label, label) for label in labels], replacements


def map_adata_labels(adata, annotation_key, path, cache_dir, origin, append=True):
    """Replace the labels in ``adata.obs[annotation_key]`` by their Cell Ontology matches.

    The labels as given are kept in ``original_<annotation_key>``, and a repeated call starts
    from them again. With ``append``, labels the table does not hold are added to it.
    """
    backup = f"original_{annotation_key}"
    if backup in adata.obs.columns:
        adata.obs[annotation_key] = adata.obs[backup].copy()
    labels = adata.obs[annotation_key].tolist()
    label_index = build_label_index(pronto.Ontology(str(download_cl_ontology(cache_dir))))
    mapped, replacements = resolve_labels(labels, label_index, path, cache_dir, origin,
                                          append=append)
    if replacements:
        if backup not in adata.obs.columns:
            adata.obs[backup] = labels
        adata.obs[annotation_key] = mapped
        print(f"{len(replacements)} label(s) in '{annotation_key}' mapped to Cell Ontology terms")
    return adata


def canonical_names(labels: pd.Series, label_index, review_map):
    """Each label as the name of its Cell Ontology term, so that true and predicted labels are
    compared in one spelling.

    A label resolves through the ontology (names and ids, ignoring case) or else through a
    reviewed mapping. A label that resolves through neither is kept in lowercase, which never
    merges two different labels. Returns the names and the list of labels that did not resolve.
    Without a label index the labels are returned unchanged.
    """
    labels = labels.astype(str)
    if label_index is None:
        return labels, []
    names, unresolved = {}, []
    for raw in labels.unique():
        term = map_string_to_term(raw, label_index)
        if term is None:
            target = review_map.get(raw.strip().lower())
            if target:
                term = map_string_to_term(target, label_index)
        if term is not None:
            names[raw] = term.name
        else:
            names[raw] = raw.strip().lower()
            unresolved.append(raw)
    return labels.map(names), unresolved
