"""Checks for the scoring metrics in bin/: information content, Lin, Rada, answer types and the
agreement score, the text-similarity baseline, ID answers, macro averages and untrusted answers.
The checks on the Cell Ontology run only when the pinned release is in cache/.

Run with pytest, or directly in the evaluation environment: python tests/test_scoring.py
"""
import functools
import itertools
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pronto

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from evaluate import UNMAPPED, ids_as_names, label_metrics, prepare_for_scoring  # noqa: E402
from util.cell_ontology_similarity import (  # noqa: E402
    BROAD_TERMS_BELOW, CL_FILE, CellTaxonomy, build_label_index, build_synonym_index,
    compute_ontology_similarities, map_string_to_term)
from util.text_embedding_similarity import (  # noqa: E402
    compute_text_embedding_similarities, reference_baseline)


def seco_ic(n_below, n_terms):
    return {t: 1 - math.log(n) / math.log(n_terms) for t, n in n_below.items()}


# C has two parents, so a path count would put 9 terms below root instead of 7.
TOY_EDGES = [("root", "A"), ("root", "B"), ("A", "C"), ("B", "C"), ("A", "F"), ("C", "D"),
             ("C", "E")]
TOY = CellTaxonomy(TOY_EDGES, root="root")
TOY_IC = seco_ic({"root": 7, "A": 5, "B": 4, "C": 3, "D": 1, "E": 1, "F": 1}, 7)

# X and Y share R (depth 2, 4 terms below) and S (depth 1, 3 terms below). The deepest common
# ancestor is R, but the most informative one is S.
MICA = CellTaxonomy([("root", "P"), ("P", "R"), ("R", "X"), ("R", "Y"), ("R", "Z"),
                     ("root", "S"), ("S", "X"), ("S", "Y")], root="root")
MICA_IC = seco_ic({"root": 7, "P": 5, "R": 4, "S": 3, "X": 1, "Y": 1, "Z": 1}, 7)


def test_information_content_counts_each_term_once():
    for term, expected in TOY_IC.items():
        assert math.isclose(TOY.ic[term], expected, abs_tol=1e-12), term
    assert TOY.ic["root"] == 0.0


def test_lin_uses_most_informative_common_ancestor():
    assert math.isclose(TOY.lin("D", "E"), TOY_IC["C"])
    assert math.isclose(TOY.lin("C", "F"), 2 * TOY_IC["A"] / (TOY_IC["C"] + 1))
    assert math.isclose(MICA.lin("X", "Y"), MICA_IC["S"])  # deepest ancestor R would give IC(R)
    assert TOY.lin("D", "D") == 1.0 and TOY.lin("root", "root") == 1.0
    assert TOY.lin("root", "D") == 0.0


def test_rada_path_length_is_undirected():
    assert TOY.path_length("D", "F") == 3  # D - C - A - F
    assert TOY.path_length("D", "E") == 2


def test_unresolved_prediction_scores_zero():
    index = {name.lower(): SimpleNamespace(id=name) for name in "DEF"}
    df = pd.DataFrame({"true": ["D", "D", "E", "E"], "pred": ["D", "nonsense", "E", "F"]})
    pairs, means = compute_ontology_similarities(df, "true", "pred", index, TOY)
    scores = pairs.set_index("pred")
    assert scores.loc["nonsense", "lin_similarity"] == 0.0
    assert scores.loc["nonsense", "rada_similarity"] == 0.0
    assert math.isclose(means["lin_avg_similarity"], (1 + 0 + 1 + TOY.lin("E", "F")) / 4)


def test_answer_types_and_agreement_score():
    assert TOY.lineage("D", "A") == "coarser"  # 5 terms at or below A, far from the default cut
    toy = CellTaxonomy(TOY_EDGES, root="root", broad_terms_below=4)  # now A is broad, B is not
    assert [toy.lineage("D", p) for p in ("D", "C", "B", "A", "E", "F", "root", None, "Z")] == [
        "exact", "coarser", "coarser", "broad", "other", "other", "none", "none", "none"]
    assert toy.lineage("C", "D") == "finer"
    index = {name.lower(): SimpleNamespace(id=name) for name in ("A", "B", "D", "E", "root")}
    df = pd.DataFrame({"true": ["D"] * 4 + ["E"] * 2,
                       "pred": ["D", "B", "A", "root", "E", "nonsense"]})
    _, means = compute_ontology_similarities(df, "true", "pred", index, toy)
    # D: exact 1, B two steps up 0.5, broad A 0, root 0; E: exact 1, unresolved 0
    assert math.isclose(means["agreement_score"], ((1 + 0.5) / 4 + 1 / 2) / 2)  # mean over types
    assert means["n_unique_pred_mapped"] == 4  # D, B, A and E, not the root or the unresolved one


def test_text_baseline_is_mean_cosine_of_distinct_names():
    vectors = {"a": [1.0, 0.0], "b": [0.6, 0.8], "c": [0.0, 1.0]}
    encoder = SimpleNamespace(model_name="toy", encode=lambda names: np.array(
        [vectors[n] for n in names]))
    pairs = [np.dot(vectors[x], vectors[y]) for x, y in itertools.permutations(vectors, 2)]
    with tempfile.TemporaryDirectory() as cache:
        assert math.isclose(reference_baseline(encoder, vectors, cache), np.mean(pairs))
        encoder.encode = None  # a second call must come from the cache
        assert math.isclose(reference_baseline(encoder, vectors, cache), np.mean(pairs))


def test_no_answer_scores_like_a_random_name():
    vectors = {"a": [1.0, 0.0], "b": [0.6, 0.8], "trunk": [0.0, 1.0]}
    encoder = SimpleNamespace(model_name="toy", encode=lambda names: np.array(
        [vectors[n] for n in names]).reshape(len(names), 2))
    # no answer ("no prediction", mapped to "cell"), nonsense mapped to the root, and an answer
    # without a reviewed mapping
    df = pd.DataFrame({"true": ["a"] * 4, "raw": ["a", "no prediction", "trunk", "b"],
                       "mapped": ["a", "cell", "cell", UNMAPPED]})
    combos, _ = compute_text_embedding_similarities(
        df, "true", "raw", "mapped", encoder, baseline=0.5,
        no_answer=(UNMAPPED, "no prediction", "cell"))
    sims = combos.set_index("raw")[["text_emb_similarity_raw", "text_emb_similarity_mapped"]]
    assert sims.loc["no prediction"].tolist() == [0.5, 0.5]
    assert sims.loc["trunk"].tolist() == [0.0, 0.5]  # its raw text still counts, "cell" does not
    assert sims.loc["b"].tolist() == [0.6, 0.5]


@functools.lru_cache(maxsize=None)
def pinned_ontology():
    """The pinned CL release, or None while the pipeline has not downloaded it yet."""
    path = Path(__file__).resolve().parents[1] / "cache" / CL_FILE
    return pronto.Ontology(str(path)) if path.exists() else None


def test_label_index():
    ont = pinned_ontology()
    if ont is None:
        return
    names, synonyms = build_label_index(ont), build_synonym_index(ont)
    resolve = lambda label: map_string_to_term(label, names).name  # noqa: E731
    assert resolve("pancreatic A cell") == resolve("CL:0000171") == "pancreatic A cell"
    # "alpha cell" is an exact synonym of a mouse retinal ganglion cell: a suggestion, not a match
    assert map_string_to_term("alpha cell", names) is None
    assert map_string_to_term("alpha cell", synonyms).name.startswith("alpha retinal ganglion")
    # a species tag is dropped when the general term is a same-named parent, nothing else is
    assert resolve("vascular leptomeningeal cell (Mmus)") == resolve("CL:4023056") == \
        "vascular leptomeningeal cell"
    assert resolve("T cell of appendix") == "T cell of appendix"
    assert resolve("CD16-negative, CD56-bright natural killer cell, human").endswith(", human")


def test_broad_cut_on_the_cell_ontology():
    ont = pinned_ontology()
    if ont is None:
        return
    index, taxonomy = build_label_index(ont), CellTaxonomy.from_ontology(ont)
    term = lambda name: map_string_to_term(name, index).id  # noqa: E731
    # the counts quoted where BROAD_TERMS_BELOW is set
    counts = {"neuron": 829, "epithelial cell": 713, "leukocyte": 573, "lymphocyte": 348,
              "myeloid cell": 260, "T cell": 172}
    assert {name: taxonomy.n_below[term(name)] for name in counts} == counts
    assert BROAD_TERMS_BELOW == 250
    assert taxonomy.lineage(term("glutamatergic neuron"), term("neuron")) == "broad"
    naive_cd4 = term("naive thymus-derived CD4-positive, alpha-beta T cell")
    assert taxonomy.lineage(naive_cd4, term("T cell")) == "coarser"
    assert nx.shortest_path_length(taxonomy.graph, term("T cell"), naive_cd4) == 3  # is_a steps


def test_ontology_id_is_read_as_its_name():
    ont = pinned_ontology()
    if ont is None:
        return
    labels = pd.Series(["CL:0000623", " cl:0000623", "natural killer cell", "CL:9999999", "no prediction"])
    assert ids_as_names(labels, build_label_index(ont)).tolist() == [
        "natural killer cell"] * 3 + ["CL:9999999", "no prediction"]


def test_macro_average_runs_over_true_classes():
    true = pd.Series(["a", "a", "b", "b"])
    pred = pd.Series(["a", "x", "b", "y"])  # x and y are labels that no cell carries
    metrics = label_metrics(true, pred)
    macro = metrics["summary_dict"]["macro avg"]
    assert math.isclose(macro["f1-score"], 2 / 3)  # averaging over a, b, x, y would give 1/3
    assert math.isclose(macro["recall"], metrics["balanced_accuracy"])
    assert metrics["accuracy"] == 0.5


def test_untrusted_answers_count_as_wrong():
    df = pd.DataFrame({"true_labels": ["T cell", "T cell"],
                       "predicted_labels": ["T cell", "T cell"],
                       "original_predicted_labels": ["t cell", "unreviewed answer"]})
    out, _, _ = prepare_for_scoring(df, pd.Series([True, False]), ["true_labels"],
                                    label_index=None, review_map={})
    assert list(out["predicted_labels"]) == ["T cell", UNMAPPED]
    assert label_metrics(out["true_labels"], out["predicted_labels"])["accuracy"] == 0.5


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
