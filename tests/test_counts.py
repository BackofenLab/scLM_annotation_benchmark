"""Checks for the raw counts and infinite values in bin/util/pp_utils.py: counts from adata.raw
are taken by gene name, the counts are checked before they become integers, and genes with
infinite values are found in sparse matrices that store zeros.

Run with pytest, or directly in the environment of the data steps (envs/download.yml, which has
scanpy): python tests/test_counts.py
"""
import sys
from pathlib import Path

import anndata
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from util import pp_utils  # noqa: E402


def test_counts_from_raw_follow_the_gene_names():
    counts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    adata = anndata.AnnData(X=np.log1p(counts))
    adata.var_names = ["a", "b", "c"]
    raw = anndata.AnnData(X=counts[:, [2, 0, 1]])
    raw.var_names = ["c", "a", "b"]
    adata.raw = raw
    pp_utils.ensure_raw_counts_layer(adata)
    assert np.array_equal(adata.layers["counts"], counts.astype(np.int32))


def test_counts_with_decimals_are_rounded_down():
    adata = anndata.AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    adata.layers["counts"] = np.array([[150.4, 2.0], [3.6, 400.0]], dtype=np.float32)
    pp_utils.ensure_raw_counts_layer(adata)
    assert np.array_equal(adata.layers["counts"], np.array([[150, 2], [3, 400]]))


def test_log_transformed_or_negative_counts_stop():
    for layer in ([[0.69, 1.1], [1.39, 0.0]], [[-1.0, 2.0], [3.0, 4.0]]):
        adata = anndata.AnnData(X=np.ones((2, 2), dtype=np.float32))
        adata.layers["counts"] = np.array(layer, dtype=np.float32)
        try:
            pp_utils.ensure_raw_counts_layer(adata)
        except ValueError:
            continue
        raise AssertionError(f"{layer} were taken as raw counts")


def test_infinite_values_are_found_next_to_stored_zeros():
    # Row 0 stores an explicit zero in gene 0 and an infinite value in gene 2.
    X = sparse.csr_matrix((np.array([0.0, 5.0, np.inf, 1.0]), np.array([0, 1, 2, 0]),
                           np.array([0, 3, 4])), shape=(2, 3))
    adata = anndata.AnnData(X=X)
    adata.var_names = ["a", "b", "c"]
    assert list(pp_utils.remove_infinite_values(adata).var_names) == ["a", "b"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
