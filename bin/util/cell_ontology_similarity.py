"""
Cell Ontology similarity between true and predicted cell types.

Labels are resolved to Cell Ontology (CL) terms, and each (true, predicted) pair is scored on the
is_a hierarchy below 'cell' (CL:0000000):

- Lin (1998): 2 * IC(MICA) / (IC(a) + IC(b)), where MICA is the most informative common ancestor
  and IC is the intrinsic information content of Seco et al. (2004),
  IC(t) = 1 - log(n_below(t)) / log(N), with n_below(t) the number of terms at or below t (each
  counted once) and N the number of terms in the hierarchy. IC('cell') is 0 and a leaf has IC 1.
- Rada et al. (1989): 1 / (1 + d), with d the shortest undirected is_a path between the terms.
- Agreement score of Hou & Ji (2024, GPTCelltype): full match 1, partial match 0.5, mismatch 0.
  A partial match is the true type's direct parent (one step broader) or direct child (one step
  finer); allowing any ancestor would give half credit for answering "cell".

A prediction that does not resolve to a term of the hierarchy scores 0 on all three, because it is
a wrong answer, not a missing one.
"""
from __future__ import annotations
import math
import os
import re
import urllib.request
import urllib.error
from collections import Counter
from collections.abc import Iterable
from typing import Optional
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pronto

# Pinned Cell Ontology release (stated in Methods). The local file name carries the release
# date, so a cache holding any other version is never mistaken for this one.
CL_RELEASE = "2026-06-08"
CL_URL = f"https://purl.obolibrary.org/obo/cl/releases/{CL_RELEASE}/cl-basic.obo"
CL_FILE = f"cl-basic_{CL_RELEASE}.obo"
CL_ROOT = "CL:0000000"  # 'cell'
# CL tags species-specific versions of a general term, e.g. "microglial cell (Mmus)".
SPECIES_TAG = re.compile(r" \((?:Mmus|Homo sapiens)\)$|, (?:human|mouse)$")
AGREEMENT = {"exact": 1.0, "broader": 0.5, "finer": 0.5, "other": 0.0}
SIMILARITY_KEYS = ("lin_avg_similarity", "rada_avg_similarity", "agreement_score",
                   "share_broader", "share_finer")


def download_cl_ontology(cache_dir: Path, cl_url: str = CL_URL, filename: str = CL_FILE,
                         force_download: bool = False) -> Path:
    """Path to the pinned Cell Ontology release in ``cache_dir``, downloaded on first use."""
    path = Path(cache_dir) / filename
    if path.exists() and path.stat().st_size > 0 and not force_download:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Cell Ontology from {cl_url} to {path}")
    tmp = path.with_suffix(f".{os.getpid()}.tmp")  # a failed download never leaves a partial file
    try:
        urllib.request.urlretrieve(cl_url, tmp)
        tmp.replace(path)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download Cell Ontology from {cl_url}: {e}")
    return path


def _normalize(s: str) -> str:
    """Lowercase + strip for robust dictionary keys."""
    return " ".join(s.lower().split()) if s is not None else ""


def _cl_terms(ont: pronto.Ontology):
    """Current (non-obsolete) Cell Ontology terms; the file also carries a few BFO/RO/CP stubs."""
    return (t for t in ont.terms() if not t.obsolete and t.id.startswith("CL:"))


def species_neutral(term: pronto.Term) -> pronto.Term:
    """The general term for a species-tagged one ("X (Mmus)" -> "X"), else the term itself.

    Each dataset holds one species, so the tag carries no information there, and without this an
    answer naming the general type would count as wrong. Only a species tag whose general term is
    a direct parent of the same name is dropped; tissue or marker qualifiers are never touched.
    """
    general = SPECIES_TAG.sub("", term.name or "")
    if general != term.name:
        for parent in term.superclasses(distance=1, with_self=False):
            if parent.name == general:
                return parent
    return term


def build_label_index(ont: pronto.Ontology) -> dict[str, pronto.Term]:
    """Map normalized CL term IDs and names to (species-neutral) terms, case-insensitively.

    Synonyms are left out on purpose. They ignore tissue and species: "alpha cell" is an exact
    synonym of a mouse retinal ganglion cell, so matching on synonyms would turn pancreatic alpha
    cells into neurons. Labels that need a synonym go through the reviewed mappings, and
    ``build_synonym_index`` offers synonyms to reviewers as suggestions. Retired IDs are not
    indexed either, so they too go to review.
    """
    idx: dict[str, pronto.Term] = {}
    for term in _cl_terms(ont):
        for key in (term.id, term.name):
            if key:
                idx.setdefault(_normalize(key), species_neutral(term))
    return idx


def build_synonym_index(ont: pronto.Ontology) -> dict[str, pronto.Term]:
    """Map normalized EXACT synonyms to their terms: review suggestions only, never for scoring."""
    idx: dict[str, pronto.Term] = {}
    for term in _cl_terms(ont):
        for syn in term.synonyms:
            if syn.scope == "EXACT":
                idx.setdefault(_normalize(syn.description), species_neutral(term))
    return idx


def map_string_to_term(s: str, index: dict[str, pronto.Term]) -> Optional[pronto.Term]:
    """Map a free-text label/synonym to a CL term (case-insensitive)."""
    if not s or not s.strip():
        return None
    return index.get(_normalize(s))


class CellTaxonomy:
    """The is_a hierarchy below ``root``, with the Seco intrinsic IC of every term."""

    def __init__(self, edges: Iterable[tuple[str, str]], root: str = CL_ROOT,
                 names: dict[str, str] | None = None):
        """``edges`` are (parent, child) is_a pairs; terms outside ``root``'s subtree are dropped.
        ``names`` maps term ids to names, kept for the terms in the hierarchy."""
        graph = nx.DiGraph(edges)
        self.graph = graph.subgraph(nx.descendants(graph, root) | {root}).copy()
        self.names = {t: names[t] for t in self.graph if t in (names or {})}
        self._undirected = self.graph.to_undirected(as_view=True)
        self._ancestors = {t: nx.ancestors(self.graph, t) | {t} for t in self.graph}
        n_below = Counter(a for ancestors in self._ancestors.values() for a in ancestors)
        log_n = math.log(self.graph.number_of_nodes())
        self.ic = {t: 1.0 - math.log(n_below[t]) / log_n for t in self.graph}

    @classmethod
    def from_ontology(cls, ont: pronto.Ontology, root: str = CL_ROOT) -> CellTaxonomy:
        terms = list(_cl_terms(ont))
        return cls(((parent.id, term.id) for term in terms
                    for parent in term.superclasses(distance=1, with_self=False)
                    if not parent.obsolete), root, {t.id: t.name for t in terms if t.name})

    def __contains__(self, term_id: str) -> bool:
        return term_id in self.graph

    def lin(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        mica_ic = max(self.ic[t] for t in self._ancestors[a] & self._ancestors[b])
        return 2.0 * mica_ic / (self.ic[a] + self.ic[b])

    def path_length(self, a: str, b: str) -> int:
        return nx.shortest_path_length(self._undirected, a, b)

    def relation(self, true: str, pred: str) -> str:
        """``pred`` relative to ``true``: exact, one is_a step broader or finer, or other."""
        if pred == true:
            return "exact"
        if self.graph.has_edge(pred, true):
            return "broader"
        if self.graph.has_edge(true, pred):
            return "finer"
        return "other"


def compute_ontology_similarities(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    label_index: dict[str, pronto.Term],
    taxonomy: CellTaxonomy,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score every distinct (true, predicted) label pair once, then summarize over the cells.

    Per pair: Lin and Rada similarity, the is_a path length, and the prediction's relation to the
    true type with its agreement score. An unresolved prediction is a mismatch. An unresolved true
    label cannot be scored, so its cells stay out of the summaries (``evaluate.py`` reports it).

    Summaries (``SIMILARITY_KEYS``): Lin and Rada are means over cells. ``agreement_score`` is a
    mean over true cell types, as GPTCelltype averages over cell types. ``share_broader`` and
    ``share_finer`` are the fractions of cells answered one step too coarse or too fine.
    """
    pairs = df.groupby([true_col, pred_col]).size().reset_index(name="cell_count")
    term_ids = {}
    for label in pd.unique(pairs[[true_col, pred_col]].to_numpy().ravel()):
        term = map_string_to_term(label, label_index)
        term_ids[label] = term.id if term is not None and term.id in taxonomy else None
    pairs["true_term_id"] = pairs[true_col].map(term_ids)
    pairs["pred_term_id"] = pairs[pred_col].map(term_ids)

    def score(t, p):
        if pd.isna(t):
            return np.nan, np.nan, np.nan, None
        if pd.isna(p):
            return 0.0, 0.0, np.nan, "other"
        d = taxonomy.path_length(t, p)
        return taxonomy.lin(t, p), 1.0 / (1.0 + d), float(d), taxonomy.relation(t, p)

    pairs = pairs.join(pd.DataFrame(
        [score(t, p) for t, p in zip(pairs["true_term_id"], pairs["pred_term_id"])],
        columns=["lin_similarity", "rada_similarity", "distance_edges", "relation"],
        index=pairs.index))
    pairs["agreement"] = pairs["relation"].map(AGREEMENT)

    scored = pairs[pairs["true_term_id"].notna()]
    if scored.empty:
        return pairs, dict.fromkeys(SIMILARITY_KEYS, np.nan)
    cells = scored["cell_count"]
    by_type = ((scored["agreement"] * cells).groupby(scored[true_col]).sum()
               / cells.groupby(scored[true_col]).sum())
    share = cells.groupby(scored["relation"]).sum() / cells.sum()
    return pairs, {
        "lin_avg_similarity": float(np.average(scored["lin_similarity"], weights=cells)),
        "rada_avg_similarity": float(np.average(scored["rada_similarity"], weights=cells)),
        "agreement_score": float(by_type.mean()),
        "share_broader": float(share.get("broader", 0.0)),
        "share_finer": float(share.get("finer", 0.0)),
    }
