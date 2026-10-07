"""Checks for the datasets table (bin/dataset_table.py) and the sourced facts it is built from.

Run with pytest, or directly in the evaluation environment: python tests/test_dataset_table.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from dataset_table import facts_by_dataset, write_table  # noqa: E402
from manuscript_figures import DATASET_ORDER  # noqa: E402

FACTS = ROOT / "cache" / "result" / "dataset_facts.csv"
BIB = ROOT / "scLLMs_benchmark_paper" / "reference.bib"


def facts():
    return pd.read_csv(FACTS, dtype=str)


def test_every_value_has_a_source():
    f = facts()
    assert f[["value", "source", "evidence", "checked"]].notna().all().all()
    assert set(f["dataset"]) == set(DATASET_ORDER)


def test_counts_are_consistent():
    for ds, v in facts_by_dataset(facts()).items():
        n = {k: int(x) for k, x in v.items() if k.startswith(("cells_", "genes_", "cell_types_",
                                                              "test_cells_"))}
        assert n["cells_after_qc"] <= n["cells_downloaded"], ds
        assert n["genes_after_qc"] <= n["genes_downloaded"], ds
        assert n["cell_types_scored"] <= n["cell_types_after_mapping"] <= n["cell_types_labelled"], ds
        for size in (1000, 10000):
            # Per-type quotas are rounded, so a test set can fall a few cells short, never over.
            assert size - 3 <= n[f"test_cells_{size}"] <= size, (ds, size)


def test_facts_match_the_cached_subsets():
    for ds, v in facts_by_dataset(facts()).items():
        path = ROOT / f"cache/subset_qc/{ds}/sub1000_stratified/cell_type_distribution_stratified.csv"
        if not path.exists():
            continue
        qc = pd.read_csv(path)
        assert int(v["cells_after_qc"]) == qc["original_count"].sum(), ds
        assert int(v["cell_types_scored"]) == (~qc["dropped"]).sum(), ds
        assert int(v["test_cells_1000"]) == qc["subset_count"].sum(), ds


def test_scored_cell_types_match_the_scoring():
    pairs = ROOT / "cache" / "confusion_pairs_1000_stratified.csv"
    if not pairs.exists():
        return
    classes = pd.read_csv(pairs).groupby("dataset")["true_label"].nunique()
    for ds, v in facts_by_dataset(facts()).items():
        assert int(v["cell_types_scored"]) == classes[ds], ds


def test_every_citation_is_in_the_bibliography():
    keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    for ds, v in facts_by_dataset(facts()).items():
        for key in [v["citation"], *v["studies"].split("; ")]:
            assert key in keys, (ds, key)


def test_table_has_one_column_per_dataset():
    table = write_table(facts())
    header = next(line for line in table.splitlines() if "\\textbf{" in line and "&" in line)
    assert header.count("&") == len(DATASET_ORDER)
    assert "\\label{supptab:datasets}" in table


def test_supplementary_table_does_not_cite():
    # The bioRxiv build gives each supplementary part its own bibliography: a \cite there is undefined.
    assert "\\cite" not in write_table(facts())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
