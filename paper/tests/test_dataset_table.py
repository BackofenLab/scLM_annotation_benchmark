"""Checks for the datasets table (paper/scripts/dataset_table.py) and the sourced facts it is built
from.

Run with pytest, or directly in the evaluation environment: python paper/tests/test_dataset_table.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from dataset_table import facts_by_dataset, write_baselines_table, write_table  # noqa: E402
from methods import DATASET_ORDER  # noqa: E402

FACTS = ROOT / "paper" / "data" / "dataset_facts.csv"
BIB = ROOT / "paper" / "data" / "references.bib"


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
        assert (n["cell_types_scored"] <= n["cell_types_after_mapping"]
                <= n["cell_types_labelled"]), ds
        for size in (1000, 10000):
            # Per-type quotas are rounded, so a test set can fall a few cells short, never over.
            assert size - 3 <= n[f"test_cells_{size}"] <= size, (ds, size)


def test_facts_match_the_cached_subsets():
    for ds, v in facts_by_dataset(facts()).items():
        path = ROOT / f"paper/data/cell_type_distributions/{ds}.csv"
        if not path.exists():
            continue
        qc = pd.read_csv(path)
        assert int(v["cells_after_qc"]) == qc["original_count"].sum(), ds
        assert int(v["cell_types_scored"]) == (~qc["dropped"]).sum(), ds
        assert int(v["test_cells_1000"]) == qc["subset_count"].sum(), ds


def test_scored_cell_types_match_the_scoring():
    pairs = ROOT / "paper" / "results" / "confusion_pairs.csv"
    if not pairs.exists():
        return
    pairs = pd.read_csv(pairs)
    classes = pairs[pairs["subset_size"] == 1000].groupby("dataset")["true_label"].nunique()
    for ds, v in facts_by_dataset(facts()).items():
        assert int(v["cell_types_scored"]) == classes[ds], ds


def test_every_citation_is_in_the_bibliography():
    keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    for ds, v in facts_by_dataset(facts()).items():
        for key in [v["citation"], *v["studies"].split("; "), v["singler_reference_citation"]]:
            assert key in keys, (ds, key)


def test_baseline_facts_match_the_dataset_configs():
    """The baselines table states what each wrapper was configured with."""
    import yaml
    for ds, v in facts_by_dataset(facts()).items():
        cfg = yaml.safe_load((ROOT / "datasets" / f"{ds}.yml").read_text())
        assert cfg["celltypist_model"] == f"{v['celltypist_model']}.pkl", ds
        assert cfg["sctype_tissue"] == v["sctype_tissue"], ds
        assert cfg["scsa_tissue"] == v["scsa_tissue"], ds
        source = cfg["singler_reference_source"]
        expected = ("CELLxGENE" if source == "cellxgene"
                    else f"celldex {cfg['singler_reference']}")
        assert v["singler_reference"] == expected, ds


def test_baselines_table_has_one_column_per_dataset_and_does_not_cite():
    table = write_baselines_table(facts())
    header = next(line for line in table.splitlines() if "\\textbf{" in line and "&" in line)
    assert header.count("&") == len(DATASET_ORDER)
    assert "\\label{supptab:baselines}" in table
    assert "\\cite" not in table


def test_table_has_one_column_per_dataset():
    table = write_table(facts())
    header = next(line for line in table.splitlines() if "\\textbf{" in line and "&" in line)
    assert header.count("&") == len(DATASET_ORDER)
    assert "\\label{supptab:datasets}" in table


def test_supplementary_table_does_not_cite():
    # Each part of the supplement has its own bibliography, so a \cite there is undefined.
    assert "\\cite" not in write_table(facts())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
