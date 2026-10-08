"""Checks for the imperfect-list study (context_ablation/src/lists.py): the lists follow their
rules on the real datasets, a runner stops, caching nothing, when the account cannot pay, and
balanced accuracy over the listed types agrees between the run metrics and the bootstrap (the
label constraint of the reference methods is tested in test_label_constraint.py).

Run with pytest, or directly in the evaluation environment: python tests/test_candidate_lists.py
"""
import functools
import math
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bin"), str(ROOT / "context_ablation" / "src"), str(ROOT / "tests")]
import lists  # noqa: E402
import run_openrouter  # noqa: E402
from bootstrap_contrasts import replicates  # noqa: E402
from runner_cache import ResponseCache  # noqa: E402

CFG = yaml.safe_load((ROOT / "context_ablation" / "config.yml").read_text())
TREE = ROOT / "context_ablation" / CFG["studies"]["lists"]


@functools.lru_cache(maxsize=None)
def study():
    """Ontology, reviewed mappings and the true types of every dataset, loaded once."""
    from evaluate import load_review_status
    cache_dir = ROOT / "context_ablation" / CFG["scoring"]["cache_dir"]
    label_index, taxonomy, addable = lists.ontology(cache_dir)
    _reviewed, review_map = load_review_status(cache_dir)
    true = {ds: lists.true_types(ds, CFG, label_index, review_map) for ds in CFG["datasets"]}
    return label_index, review_map, taxonomy, addable, true


def _round_half_up(x):
    return math.floor(x + 0.5)


def test_lists_follow_their_rules():
    from evaluate import _canonicalize_to_cl
    from util.cell_ontology_similarity import map_string_to_term
    label_index, review_map, taxonomy, addable, true = study()
    conditions = CFG["lists"]["conditions"]
    for dataset, (labels, ids) in true.items():
        k, seen = len(labels), set()
        for seed in CFG["repeats"]["seeds"]:
            drawn = {c: lists.read_list(TREE, dataset, c, seed) for c in conditions}
            rng = np.random.default_rng([zlib.crc32(dataset.encode()), seed])
            assert drawn == lists.draw_lists(labels, ids, taxonomy, addable, conditions, rng)
            seen.add(tuple(map(tuple, drawn.values())))

            def added(c):
                return [n for n in drawn[c] if n not in labels]

            names = pd.Series([n for c in conditions for n in drawn[c]]).drop_duplicates()
            canon, unresolved = _canonicalize_to_cl(names, label_index, review_map)
            assert not unresolved, unresolved
            term = dict(zip(names, (map_string_to_term(n, label_index).id for n in canon)))
            for c, listed in drawn.items():
                assert listed == sorted(listed) and len({term[n] for n in listed}) == len(listed), c
            assert drawn["E"] == drawn["EU"] == labels
            relatives, extras = added("L1"), added("L2-5")
            assert len(relatives) == _round_half_up(k / 2)
            assert len(extras) == 5 and len(added("L2-2")) == 2 and set(added("L2-2")) <= set(extras)
            for n in relatives:
                kinds = {taxonomy.lineage(t, term[n]) for t in ids}
                assert term[n] in addable and kinds & {"coarser", "finer"}, n
            for n in extras:
                assert term[n] in addable and all(taxonomy.lineage(t, term[n]) == "other"
                                                  for t in ids), n
            removed = set(labels) - set(drawn["L3"])
            assert len(removed) == max(1, _round_half_up(k / 4)) and not added("L3")
            assert set(drawn["L4"]) == (set(labels) - removed) | set(relatives) | set(extras)
        assert len(seen) == len(CFG["repeats"]["seeds"]), f"{dataset}: seeds drew the same lists"


class _Response:
    def __init__(self, status):
        self.status_code, self.text, self.headers = status, '{"error": "test"}', {}


class _Session:
    def __init__(self, status):
        self.status = status

    def post(self, *args, **kwargs):
        return _Response(self.status)


def test_account_errors_stop_the_run_and_are_not_cached():
    for status in (401, 402, 403):
        _text, meta = run_openrouter.call_once(_Session(status), "m", "p", CFG, 1, "key")
        assert meta["fatal"] and meta["transient"], status
    _text, meta = run_openrouter.call_once(_Session(400), "m", "p", CFG, 1, "key")
    assert not meta["transient"] and not meta.get("fatal")

    arm = "A1_llm_percell"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prompts = tmp / f"human_pancreas__{arm}__E__expression_ranked__seed1234.csv"
        pd.DataFrame({"cell_id": ["c1", "c2"], "true_labels": ["t", "t"],
                      "prompt": ["p1", "p2"]}).to_csv(prompts, index=False)
        cache_file = tmp / "cache" / f"{arm}__human_pancreas.jsonl"
        original = run_openrouter.call_once
        try:
            run_openrouter.call_once = lambda *a, **k: (None, {"error": "HTTP 402", "transient": True,
                                                               "fatal": True})
            try:
                run_openrouter.run_file(prompts, CFG, "key", tmp / "out", tmp / "cache")
            except SystemExit:
                pass
            else:
                raise AssertionError("a 402 must stop the run")
            assert not cache_file.exists() and not (tmp / "out").exists()
            run_openrouter.call_once = lambda *a, **k: (None, {"error": "HTTP 400",
                                                               "transient": False})
            run_openrouter.run_file(prompts, CFG, "key", tmp / "out", tmp / "cache")
            assert len(ResponseCache(cache_file, CFG["arms"][arm]["model"], seed=1234)) == 2
        finally:
            run_openrouter.call_once = original


def test_listed_balanced_accuracy_agrees_with_the_bootstrap():
    rng = np.random.default_rng(0)
    types = np.repeat(["a", "b", "c"], [4, 3, 5])
    cells = pd.concat([pd.DataFrame({
        "arm": "A1", "dataset": "d", "condition": condition, "seed": 1,
        "cell_id": [f"x{i}" for i in range(len(types))], "true_label": types,
        "hit": rng.integers(0, 2, len(types)), "abstained": False, "reachable": True,
        "listed": [t != "c" or condition == "E" for t in types]}) for condition in ("E", "L3")])
    pairs = (cells.assign(n_cells=1, lineage="exact")
             .groupby(lists.RUN + ["true_label", "listed", "lineage"], as_index=False)["n_cells"].sum())
    metrics = lists.run_metrics(cells, pairs)
    by_run = cells.assign(model=cells["condition"])
    mask = by_run.groupby(["true_label", "model"])["listed"].first().unstack()
    all_draws, all_observed = replicates(by_run, 50, np.random.default_rng(1))
    draws, observed = replicates(by_run, 50, np.random.default_rng(1), mask.where(mask, False))
    for condition in ("E", "L3"):
        expected = metrics.loc[("A1", "d", condition, 1), "balanced_accuracy_listed"]
        assert math.isclose(observed[condition], expected), condition
    # With every type listed, the mask changes nothing, to the bit.
    assert observed["E"] == all_observed["E"] and (draws["E"] == all_draws["E"]).all()
    full, _ = replicates(by_run, 50, np.random.default_rng(1),
                         pd.DataFrame(True, index=mask.index, columns=mask.columns))
    assert (full.to_numpy() == all_draws.to_numpy()).all()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
