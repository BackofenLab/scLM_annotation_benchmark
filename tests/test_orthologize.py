"""Checks for bin/orthologize.py, which gives mouse data human gene identifiers.

Run with pytest, or directly in the evaluation environment: python tests/test_orthologize.py
"""
import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from orthologize import load_orthologs, to_human_orthologs  # noqa: E402

TABLE = pd.DataFrame({
    "mouse_gene_id": ["ENSMUSG01", "ENSMUSG02"],
    "mouse_gene_name": ["Gnai3", "Abc1"],
    "human_gene_id": ["ENSG01", "ENSG02"],
    "human_gene_name": ["GNAI3", ""],
}).set_index("mouse_gene_id")


def toy() -> anndata.AnnData:
    var = pd.DataFrame({"ensembl_id": ["ENSMUSG01", "ENSMUSG02.3", "ENSMUSG03"],
                        "gene_names": ["Gnai3", "Abc1", "Mouseonly"]},
                       index=["ENSMUSG01", "ENSMUSG02.3", "ENSMUSG03"])
    x = sparse.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]]))
    return anndata.AnnData(X=x, var=var, layers={"counts": x.copy()})


def test_orthologs_replace_ids_and_symbols():
    out = to_human_orthologs(toy(), TABLE)
    assert list(out.var_names) == ["ENSG01", "ENSG02", "ENSMUSG03"]
    assert list(out.var.ensembl_id) == ["ENSG01", "ENSG02", "ENSMUSG03"]
    # a human ortholog without a symbol keeps the mouse symbol; unmapped genes keep both
    assert list(out.var.gene_names) == ["GNAI3", "Abc1", "Mouseonly"]
    assert list(out.var.mouse_ensembl_id) == ["ENSMUSG01", "ENSMUSG02", "ENSMUSG03"]
    assert json.loads(out.uns["orthologs"])["mapped_to_human"] == 2


def test_an_unmapped_gene_never_shadows_an_ortholog_symbol():
    adata = toy()
    adata.var.loc["ENSMUSG03", "gene_names"] = "GNAI3"
    out = to_human_orthologs(adata, TABLE)
    assert list(out.var.gene_names) == ["GNAI3", "Abc1", "ENSMUSG03"]


def test_expression_is_untouched():
    before = toy()
    out = to_human_orthologs(before, TABLE)
    assert (out.X != before.X).nnz == 0
    assert (out.layers["counts"] != before.layers["counts"]).nnz == 0
    assert list(before.var_names) == ["ENSMUSG01", "ENSMUSG02.3", "ENSMUSG03"]


def test_the_pinned_table_is_one_to_one():
    table = load_orthologs()
    assert len(table) == 17178
    assert table.index.str.startswith("ENSMUSG").all()
    assert table.human_gene_id.str.startswith("ENSG").all()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
