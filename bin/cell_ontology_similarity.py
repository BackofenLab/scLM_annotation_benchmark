#!/usr/bin/env python3
"""
Compute ontology-based similarity between true/predicted cell types using Cell Ontology (CL).

This script:
1) Loads CL (OBO/OWL) via `pronto`.
2) Maps free-text labels/synonyms to CL terms.
3) Computes pairwise similarity for each (true, predicted) pair.

Supported metrics:
- "rada": path-based similarity S = 1 / (1 + distance_edges), where distance is via the
         lowest common ancestor (LCA) over is_a edges (multiple inheritance allowed).
- "lin":  intrinsic-IC Lin similarity using Seco (2004) IC: S = (2 * IC(LCA)) / (IC(a) + IC(b)),
          with IC(t) = 1 - log((descendants(t)+1)/(max_desc+1)) / log(max_desc+1).

Usage
-----
python cell_ontology_similarity.py \
    --input input.csv \
    --output scored.csv \
    --cl-path /path/to/cl.owl \
    --metric rada        # or lin

CSV requirements
----------------
Columns: id, true_labels, predicted_labels (strings)
Unmapped terms get NaN similarities.

Notes
-----
- Mapping uses case-insensitive match against preferred labels and ALL synonyms.
- We restrict edges to 'is_a' relations to reflect type hierarchy (standard for CL closeness).

References
----------
- Cell Ontology (CL) PURL: http://purl.obolibrary.org/obo/cl.owl
- pronto: Python library for OBO/OWL ontologies.
- Seco et al., ECAI 2004: Intrinsic Information Content.
"""
from __future__ import annotations
import argparse
import math
import sys
from typing import Optional

import numpy as np
import pandas as pd
import pronto
from collections import deque

def _normalize(s: str) -> str:
    """Lowercase + strip for robust dictionary keys."""
    return " ".join(s.lower().split()) if s is not None else ""


def build_label_index(ont: pronto.Ontology) -> dict[str, pronto.Term]:
    """
    Map normalized labels/synonyms -> pronto.Term (case-insensitive).
    """
    idx: dict[str, pronto.Term] = {}
    for term in ont.terms():
        if term.obsolete:
            continue
        # preferred label
        if term.name:
            idx.setdefault(_normalize(term.name), term)
        # synonyms (ALL types)
        for syn in term.synonyms:
            # pronto uses `description`; older examples sometimes show `desc`
            syn_text = getattr(syn, "description", None) or getattr(syn, "desc", None)
            if not syn_text:
                # very defensive fallback
                try:
                    syn_text = str(syn)
                except Exception:
                    syn_text = None
            if syn_text:
                idx.setdefault(_normalize(syn_text), term)
    return idx

def get_is_a_parents(term: pronto.Term) -> set:
    """Direct is_a parents using pronto's taxonomy API."""
    return set(term.superclasses(distance=1, with_self=False))


def compute_depths(ont: pronto.Ontology) -> dict:
    """
    Compute minimal depth (shortest path) from any root via is_a edges.
    Uses BFS from all roots (terms with no direct superclasses).
    """
    from collections import deque

    def has_direct_superclass(term: pronto.Term) -> bool:
        # True if there is at least one direct superclass (is_a parent)
        return any(True for _ in term.superclasses(distance=1, with_self=False))

    # Roots: no direct superclasses (is_a)
    roots = [t for t in ont.terms() if (not t.obsolete) and not has_direct_superclass(t)]
    depth = {t: 0 for t in roots}

    # BFS using subclasses(distance=1)
    q = deque(roots)
    while q:
        parent = q.popleft()
        for child in parent.subclasses(distance=1, with_self=False):
            if child.obsolete:
                continue
            nd = depth[parent] + 1
            if child not in depth or nd < depth[child]:
                depth[child] = nd
                q.append(child)

    # Ensure every non-obsolete term has a depth entry
    for t in ont.terms():
        if not t.obsolete:
            depth.setdefault(t, 0)
    return depth


def ancestors(term: pronto.Term) -> set:
    """All ancestors via is_a (transitive), including the term itself."""
    return set(term.superclasses(distance=None, with_self=True))

def lca_max_depth(a: pronto.Term, b: pronto.Term, depth: dict[pronto.Term, int]) -> Optional[pronto.Term]:
    """
    Lowest Common Ancestor defined as common ancestor with maximum depth.

    Handles DAG by selecting the deepest common ancestor.
    """
    ca = ancestors(a)
    cb = ancestors(b)
    common = ca.intersection(cb)
    if not common:
        return None
    return max(common, key=lambda t: depth.get(t, 0))

def neighbors_is_a(term):
    """Undirected neighbors via is_a: direct parents and direct children."""
    # parents (superclasses) one step up
    for p in term.superclasses(distance=1, with_self=False):
        yield p
    # children (subclasses) one step down
    for c in term.subclasses(distance=1, with_self=False):
        yield c

def edge_distance_undirected(a, b):
    """Shortest number of is_a edges between two terms, via undirected BFS."""
    if a == b:
        return 0
    seen = {a}
    q = deque([(a, 0)])
    while q:
        cur, d = q.popleft()
        for nb in neighbors_is_a(cur):
            if nb == b:
                return d + 1
            if nb not in seen and not nb.obsolete:
                seen.add(nb)
                q.append((nb, d + 1))
    return None  # disconnected (shouldn't happen in CL for valid terms)

def edge_distance_via_lca(a: pronto.Term, b: pronto.Term, depth: dict[pronto.Term, int]) -> Optional[int]:
    """
    Compute undirected edge distance using depths and deepest LCA:
    d(a,b) = (depth(a) - depth(LCA)) + (depth(b) - depth(LCA))
    """
    lca = lca_max_depth(a, b, depth)
    if lca is None:
        return None
    return (depth.get(a, 0) - depth.get(lca, 0)) + (depth.get(b, 0) - depth.get(lca, 0))


def seco_ic(desc_count: dict[pronto.Term, int], max_desc: int, term: pronto.Term) -> float:
    """
    Seco intrinsic IC:
      IC(t) = 1 - log((desc(t)+1)/(max_desc+1)) / log(max_desc+1)
    """
    return 1.0 - (math.log((desc_count[term] + 1) / (max_desc + 1)) / math.log(max_desc + 1))


def descendant_counts(ont: pronto.Ontology) -> dict:
    """
    Count transitive descendants via is_a for each term using the taxonomy API.
    """
    # memoized DFS over subclasses
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def count_desc(t: pronto.Term) -> int:
        total = 0
        for c in t.subclasses(distance=1, with_self=False):
            if c.obsolete:
                continue
            total += 1 + count_desc(c)
        return total

    out = {}
    for t in ont.terms():
        if not t.obsolete:
            out[t] = count_desc(t)
    return out


def compute_similarity(
    a: pronto.Term,
    b: pronto.Term,
    metric: str,
    depth: dict[pronto.Term, int],
    desc_count: Optional[dict[pronto.Term, int]] = None,
    max_desc: Optional[int] = None,
) -> Optional[float]:
    """
    Compute pairwise similarity for two CL terms.

    Parameters
    ----------
    a, b : pronto.Term
        Terms to compare.
    metric : {"rada", "lin"}
        "rada": 1 / (1 + edge_distance_via_lca)
        "lin" : Lin similarity with Seco intrinsic IC.
    depth : dict
        Precomputed depths.
    desc_count : dict, optional
        Required for "lin".
    max_desc : int, optional
        Required for "lin".

    Returns
    -------
    float or None
        Similarity score; None if LCA missing or division-by-zero occurs.
    """
    if metric == "rada":
        d = edge_distance_undirected(a, b)
        return None if d is None else 1.0 / (1.0 + float(d))
    elif metric == "lin":
        if desc_count is None or max_desc is None:
            return None
        lca = lca_max_depth(a, b, depth)
        if lca is None:
            return None
        ic_a = seco_ic(desc_count, max_desc, a)
        ic_b = seco_ic(desc_count, max_desc, b)
        ic_l = seco_ic(desc_count, max_desc, lca)
        denom = ic_a + ic_b
        return None if denom <= 0 else (2.0 * ic_l) / denom
    else:
        raise ValueError(f"Unknown metric: {metric}")


def map_string_to_term(s: str, index: dict[str, pronto.Term]) -> Optional[pronto.Term]:
    """Map a free-text label/synonym to a CL term (case-insensitive)."""
    if not s or not s.strip():
        return None
    return index.get(_normalize(s))


def main():
    parser = argparse.ArgumentParser(description="Cell Ontology similarity scoring")
    parser.add_argument("--input", required=True, help="Input CSV (id,true_labels,predicted_labels)")
    parser.add_argument("--output", required=True, help="Output CSV with similarity column")
    parser.add_argument("--cl-path", required=True, help="Path or URL to cl.obo or cl.owl")
    parser.add_argument("--metric", default="rada", choices=["rada", "lin"], help="Similarity metric")
    args = parser.parse_args()

    # Load CL ontology
    try:
        ont = pronto.Ontology(args.cl_path)  # supports local path or URL
    except Exception as e:
        sys.exit(f"Failed to load ontology from {args.cl_path}: {e}")

    # Build mapping index and graph helpers
    label_index = build_label_index(ont)
    depth = compute_depths(ont)
    desc_count = max_desc = None
    if args.metric == "lin":
        desc_count = descendant_counts(ont)
        max_desc = max(desc_count.values()) if desc_count else 0

    # Load CSV
    df = pd.read_csv(args.input)

    for col in ("cell_id", "true_labels", "predicted_labels"):
        if col not in df.columns:
            sys.exit(f"Missing required column: {col}")

    # Map to terms
    df["true_term_id"] = None
    df["pred_term_id"] = None
    df["similarity"] = np.nan
    df["distance_edges"] = np.nan  # helpful audit when using 'rada'

    for i, row in df.iterrows():
        t = map_string_to_term(row["true_labels"], label_index)
        p = map_string_to_term(row["predicted_labels"], label_index)
        df.at[i, "true_term_id"] = t.id if t else None
        df.at[i, "pred_term_id"] = p.id if p else None
        if t and p:
            # Always compute distance; similarity depends on metric
            d = edge_distance_undirected(t, p)
            if d is not None:
                df.at[i, "distance_edges"] = float(d)
            sim = compute_similarity(t, p, args.metric, depth, desc_count, max_desc)
            if sim is not None:
                df.at[i, "similarity"] = float(sim)

    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(df)} rows).")


if __name__ == "__main__":
    main()