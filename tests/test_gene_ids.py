"""Checks for the mapping of gene symbols to Ensembl ids in bin/util/pp_utils.py: symbols are
looked up for the dataset's species, and the offline table, which holds human genes, serves human
data only. A stand-in replaces pybiomart, so the checks need no network.

Run with pytest, or directly in the environment of the data steps (envs/download.yml, which has
scanpy): python tests/test_gene_ids.py
"""
import sys
import types
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from util import pp_utils  # noqa: E402

MOUSE_GENES = pd.DataFrame({"Gene stable ID": ["ENSMUSG00000029580", "ENSMUSG00000057666"],
                            "Gene name": ["Actb", "Gapdh"]})


def answering(tables):
    """A stand-in for pybiomart that answers from tables by BioMart dataset name and fails like
    an unreachable host for every other name."""
    class Dataset:
        def __init__(self, name, host):
            if name not in tables:
                raise ConnectionError(f"{host} unreachable")
            self.table = tables[name]

        def query(self, attributes):
            return self.table

    return types.SimpleNamespace(Dataset=Dataset)


def with_symbols(*genes):
    adata = anndata.AnnData(X=np.ones((2, len(genes)), dtype=np.float32))
    adata.var_names = list(genes)
    return adata


def test_mouse_symbols_get_mouse_ids():
    sys.modules["pybiomart"] = answering({"mmusculus_gene_ensembl": MOUSE_GENES})
    adata = pp_utils.ensure_ensembl_ids(with_symbols("Actb", "Gapdh"), species="mmusculus",
                                        raise_on_failure=True)
    assert list(adata.var_names) == ["ENSMUSG00000029580", "ENSMUSG00000057666"]


def test_offline_table_serves_human_data_only():
    sys.modules["pybiomart"] = answering({})
    adata = pp_utils.ensure_ensembl_ids(with_symbols("ACTB", "GAPDH"), species="hsapiens",
                                        raise_on_failure=True)
    assert all(gene.startswith("ENSG") for gene in adata.var_names)
    # Mouse data never get human ids, not even for symbols that the human table holds.
    try:
        pp_utils.ensure_ensembl_ids(with_symbols("ACTB", "GAPDH"), species="mmusculus",
                                    raise_on_failure=True)
    except ValueError as error:
        assert "mmusculus" in str(error)
    else:
        raise AssertionError("mouse data were given ids from the table of human genes")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
