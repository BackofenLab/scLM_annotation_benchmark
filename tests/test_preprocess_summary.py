from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-test-cache")
sys.modules.setdefault("pybiomart", types.ModuleType("pybiomart"))
sys.path.insert(0, str(REPO_ROOT / "bin"))

from util.pp_utils import summarize_adata


def test_summarize_adata_creates_parent_directory(tmp_path):
    adata = ad.AnnData(
        X=np.array([[1.0, 0.0], [0.0, 2.0]]),
        obs=pd.DataFrame(index=["cell1", "cell2"]),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    out_path = tmp_path / "nested" / "summary.txt"

    summarize_adata(adata, out_path=out_path)

    assert out_path.exists()
    assert "# AnnData summary" in out_path.read_text()
