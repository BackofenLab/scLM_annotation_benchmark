"""
Cell Ontology similarity computation utilities.

This module provides functions to compute ontology-based similarity between 
true/predicted cell types using Cell Ontology (CL).

Supported metrics:
- "rada": path-based similarity S = 1 / (1 + distance_edges), where distance is via the
         lowest common ancestor (LCA) over is_a edges (multiple inheritance allowed).
- "lin":  intrinsic-IC Lin similarity using Seco (2004) IC: S = (2 * IC(LCA)) / (IC(a) + IC(b)),
          with IC(t) = 1 - log((descendants(t)+1) / log(max_desc).
- "balanced_id_accuracy": balanced exact match accuracy based on ontology term IDs. 
         Computed as the average of per-class accuracies (1.0 if true_term_id == pred_term_id, 0.0 otherwise),
         giving equal weight to each class regardless of class size.

Functions can be used independently or via the main compute_ontology_similarities
function that handles the full pipeline.
"""
from __future__ import annotations
import math
import sys
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import pronto


def download_cl_ontology(
    cache_dir: Path, 
    cl_url: str = "https://purl.obolibrary.org/obo/cl/cl-base.owl",
    force_download: bool = False
) -> Path:
    """
    Download Cell Ontology if not already present in cache.
    
    Parameters
    ----------
    cache_dir : Path
        Directory to store the downloaded ontology file
    cl_url : str, default "https://purl.obolibrary.org/obo/cl/cl-base.owl"
        URL to download the Cell Ontology from
    force_download : bool, default False
        If True, download even if file already exists
        
    Returns
    -------
    Path
        Path to the cl-base.owl file (either existing or newly downloaded)
        
    Raises
    ------
    RuntimeError
        If download fails or file cannot be accessed
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    #get file path from url
    cl_file_path = Path(cl_url).name
    
    cl_file_path = cache_dir / cl_file_path
    
    # Check if file already exists and is not empty
    if cl_file_path.exists() and cl_file_path.stat().st_size > 0 and not force_download:
        print(f"Cell Ontology already exists at: {cl_file_path}")
        return cl_file_path
    
    print(f"Downloading Cell Ontology from: {cl_url}")
    print(f"Saving to: {cl_file_path}")
    
    try:
        # Download with progress indication
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 100 == 0 or percent >= 100:  # Update every 100 blocks or at completion
                    print(f"Download progress: {percent}%")
        
        urllib.request.urlretrieve(cl_url, cl_file_path, reporthook=progress_hook)
        
        # Verify the downloaded file
        if not cl_file_path.exists() or cl_file_path.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty or missing")
            
        print(f"Successfully downloaded Cell Ontology ({cl_file_path.stat().st_size} bytes)")
        return cl_file_path
        
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download Cell Ontology from {cl_url}: {e}")
    except Exception as e:
        # Clean up partial download
        if cl_file_path.exists():
            cl_file_path.unlink()
        raise RuntimeError(f"Error downloading Cell Ontology: {e}")


def _normalize(s: str) -> str:
    """Lowercase + strip for robust dictionary keys."""
    return " ".join(s.lower().split()) if s is not None else ""


def build_label_index(ont: pronto.Ontology) -> dict[str, pronto.Term]:
    """
    Map normalized labels/synonyms/IDs -> pronto.Term (case-insensitive).
    """
    idx: dict[str, pronto.Term] = {}
    for term in ont.terms():
        if term.obsolete:
            continue
        # The term's own CL ID, so a prediction that already *is* an ontology ID
        # ("CL:0000900") resolves exactly instead of falling through to fuzzy
        # name matching, which scores IDs against unrelated obsolete terms.
        # Obsolete terms are skipped above, so retired IDs still go to review.
        if term.id:
            idx.setdefault(_normalize(term.id), term)
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
    # Special case: identical terms should return the term itself
    if a == b:
        return a
    
    ca = ancestors(a)
    cb = ancestors(b)
    common = ca.intersection(cb)
    if not common:
        return None
    # Break depth ties by term id so the LCA (and thus the Lin similarity) is deterministic.
    # Without the id tiebreaker, max() over the `common` set picks an arbitrary element whose
    # order depends on per-process string-hash randomization, making `lin` irreproducible.
    return max(common, key=lambda t: (depth.get(t, 0), t.id))


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
    return 1.0 - (math.log((desc_count[term] + 1)) / math.log(max_desc))


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


def compute_ontology_similarities(
    df: pd.DataFrame,
    cl_path: str,
    true_col: str = "true_labels",
    pred_col: str = "predicted_labels",
    max_unmapped_ratio: float = 0.0,
    ont: Optional[pronto.Ontology] = None,
    label_index: Optional[dict] = None,
    depth: Optional[dict] = None,
    desc_count: Optional[dict] = None,
    max_desc: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute ontology-based similarities for a dataframe with true and predicted labels.
    Returns unique combinations of true/predicted labels with aggregated scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with true and predicted cell type labels
    cl_path : str
        Path to Cell Ontology file (OBO/OWL)
    true_col : str, default "true_labels"
        Column name for true labels
    pred_col : str, default "predicted_labels" 
        Column name for predicted labels
    max_unmapped_ratio : float, default 0.0
        Maximum acceptable ratio of unmapped cell types (0.0 to 1.0).
        If 0.0, all cell types must be mapped or an error will be raised.
        If 1.0, any ratio of unmapped cell types is acceptable.
    ont, label_index, depth, desc_count, max_desc : optional
        Preloaded ontology + graph helpers. When supplied (e.g. by evaluate.py, which parses the
        OWL once per run), they are reused instead of re-parsing the 63MB OWL on every call.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        - DataFrame with unique label combinations and aggregated similarity scores
        - Dictionary with average accuracies for all metrics (rada, lin, balanced_id_accuracy) (NaN if unmapped ratio exceeded)
        
    Raises
    ------
    RuntimeError
        If cell types remain unmapped and max_unmapped_ratio is 0.0. 
        Note: Ontology mapping should be applied before calling this function.
    """
    # Load CL ontology and build graph helpers. Callers (e.g. evaluate.py) may pass these in
    # preloaded so the 63MB OWL is parsed only once per process instead of per call.
    if ont is None:
        try:
            ont = pronto.Ontology(cl_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ontology from {cl_path}: {e}")
    if label_index is None:
        label_index = build_label_index(ont)
    if depth is None:
        depth = compute_depths(ont)
    if desc_count is None:
        desc_count = descendant_counts(ont)
    if max_desc is None:
        max_desc = max(desc_count.values()) if desc_count else 0

    # Check mapping coverage for all unique cell types using the already-built label_index
    # (no second ontology parse — this used to call check_ontology_mapping_coverage, which
    # re-parsed the OWL on every call).
    all_cell_types = list(set(df[true_col].unique()) | set(df[pred_col].unique()))
    unmapped_types = {c for c in all_cell_types if map_string_to_term(c, label_index) is None}
    mapped_types = set(all_cell_types) - unmapped_types

    # Calculate unmapped ratio
    total_types = len(all_cell_types)
    unmapped_count = len(unmapped_types)
    unmapped_ratio = unmapped_count / total_types if total_types > 0 else 0.0
    
    # Log mapping results
    print(f"Cell type mapping summary:")
    print(f"  Total unique cell types: {total_types}")
    print(f"  Successfully mapped: {len(mapped_types)}")
    print(f"  Unmapped: {unmapped_count}")
    print(f"  Unmapped ratio: {unmapped_ratio:.3f}")
    
    if unmapped_types:
        print(f"  Unmapped cell types: {sorted(unmapped_types)}")
    
    # Warn if there are unmapped cell types and max_unmapped_ratio is 0.0
    if unmapped_ratio > max_unmapped_ratio:
        print(f"WARNING: "
            f"Cell types remain unmapped: {sorted(unmapped_types)}. "
            f"Please apply ontology mapping before evaluation or "
            f"increase max_unmapped_ratio to allow unmapped cell types."
        )
    
    # Check if unmapped ratio exceeds threshold
    if unmapped_ratio > max_unmapped_ratio:
        print(f"WARNING: Unmapped ratio ({unmapped_ratio:.3f}) exceeds threshold ({max_unmapped_ratio:.3f})")
        print("Returning NaN similarities for all combinations.")
        
        # Return empty results with NaN averages
        empty_combinations = df.groupby([true_col, pred_col]).size().reset_index(name='cell_count')
        empty_combinations['rada_similarity_mean'] = np.nan
        empty_combinations['lin_similarity_mean'] = np.nan
        empty_combinations['distance_edges_mean'] = np.nan
        empty_combinations['id_accuracy_mean'] = np.nan
        empty_combinations['true_term_id'] = None
        empty_combinations['pred_term_id'] = None
        
        avg_accuracies = {
            "rada_avg_similarity": np.nan,
            "lin_avg_similarity": np.nan,
            "id_accuracy": np.nan,
            "balanced_id_accuracy": np.nan
        }
        
        return empty_combinations, avg_accuracies

    # Compute scores ONCE per unique (true, pred) pair, then derive cell-weighted aggregates.
    # Similarity depends only on the (true_label, pred_label) pair, so the old per-cell loop
    # repeated identical work n_cells times. Scoring each distinct pair once and weighting by
    # its cell_count gives results identical to the per-cell version (and computes the rada
    # edge distance only once instead of twice).
    pair_counts = df.groupby([true_col, pred_col]).size().reset_index(name="cell_count")

    records = []
    for _, prow in pair_counts.iterrows():
        true_label = prow[true_col]
        pred_label = prow[pred_col]

        t = map_string_to_term(true_label, label_index)
        p = map_string_to_term(pred_label, label_index)
        true_term_id = t.id if t else None
        pred_term_id = p.id if p else None

        # id_accuracy: exact match iff both term IDs exist and are equal (0.0 otherwise).
        idacc = 1.0 if (true_term_id is not None and pred_term_id is not None
                        and true_term_id == pred_term_id) else 0.0

        distance = np.nan
        rada = np.nan
        lin = np.nan
        if t and p:
            d = edge_distance_undirected(t, p)
            if d is not None:
                distance = float(d)
                rada = 1.0 / (1.0 + float(d))  # rada = 1/(1+d); reuse the single BFS distance
            lin_sim = compute_similarity(t, p, "lin", depth, desc_count, max_desc)
            if lin_sim is not None:
                lin = float(lin_sim)

        records.append({
            true_col: true_label,
            pred_col: pred_label,
            "mapped_labels": pred_label,
            "cell_count": int(prow["cell_count"]),
            "rada_similarity": rada,
            "lin_similarity": lin,
            "distance_edges": distance,
            "id_accuracy": idacc,
            "true_term_id": true_term_id,
            "pred_term_id": pred_term_id,
        })

    # Final output column contract (renamed similarity columns). Used for the empty-frame guard.
    out_cols = [
        true_col, pred_col, "mapped_labels", "cell_count",
        "rada_similarity_mean", "lin_similarity_mean", "distance_edges_mean", "id_accuracy_mean",
        "true_term_id", "pred_term_id",
    ]
    if not records:
        empty = pd.DataFrame(columns=out_cols)
        return empty, {
            "rada_avg_similarity": np.nan, "lin_avg_similarity": np.nan,
            "id_accuracy": np.nan, "balanced_id_accuracy": np.nan,
        }

    unique_combinations = pd.DataFrame.from_records(records)

    # --- Cell-weighted average accuracies (identical to the old per-cell means) ---
    counts = unique_combinations["cell_count"]

    def _weighted_mean(col, mask=None):
        vals = unique_combinations[col]
        w = counts
        if mask is not None:
            vals, w = vals[mask], w[mask]
        total = w.sum()
        return float((vals * w).sum() / total) if total > 0 else np.nan

    avg_accuracies = {}
    rada_mask = unique_combinations["rada_similarity"].notna()
    avg_accuracies["rada_avg_similarity"] = _weighted_mean("rada_similarity", rada_mask) if rada_mask.any() else np.nan
    lin_mask = unique_combinations["lin_similarity"].notna()
    avg_accuracies["lin_avg_similarity"] = _weighted_mean("lin_similarity", lin_mask) if lin_mask.any() else np.nan
    avg_accuracies["id_accuracy"] = _weighted_mean("id_accuracy")

    # balanced_id_accuracy: per-true-class cell-weighted id_accuracy, then unweighted mean over classes.
    per_class = unique_combinations.groupby(true_col).apply(
        lambda g: (g["id_accuracy"] * g["cell_count"]).sum() / g["cell_count"].sum()
    )
    avg_accuracies["balanced_id_accuracy"] = float(per_class.mean()) if len(per_class) > 0 else np.nan

    # Round per-combination similarity values (matches old output; the averages above use the
    # unrounded values).
    similarity_cols = ["rada_similarity", "lin_similarity", "distance_edges", "id_accuracy"]
    for col in similarity_cols:
        unique_combinations[col] = unique_combinations[col].round(4)

    # Reorder + rename to the documented output contract (unchanged columns/order).
    column_order = [
        true_col, pred_col, "mapped_labels", "cell_count",
        "rada_similarity", "lin_similarity", "distance_edges", "id_accuracy",
        "true_term_id", "pred_term_id",
    ]
    unique_combinations = unique_combinations[column_order]
    unique_combinations = unique_combinations.rename(columns={
        "rada_similarity": "rada_similarity_mean",
        "lin_similarity": "lin_similarity_mean",
        "distance_edges": "distance_edges_mean",
        "id_accuracy": "id_accuracy_mean",
    })

    return unique_combinations, avg_accuracies
