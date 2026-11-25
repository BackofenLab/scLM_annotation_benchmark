"""Checks for the runs a method does not support (paper/scripts/plot_model_comparison.py).

Run with pytest, or directly: python paper/tests/test_unsupported_runs.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper" / "scripts"))
import plot_model_comparison  # noqa: E402
from plot_model_comparison import unsupported_pairs, with_unsupported_as_zero  # noqa: E402


def test_unsupported_runs_count_as_zero():
    # The pipeline's own skip list is empty at present, so the zero rows are checked on a fixed one.
    runs = pd.DataFrame({"dataset": ["HFRA", "Zeisel", "HFRA"],
                         "model": ["mmcontext_unblind", "langcell_unblind", "langcell_unblind"],
                         "subset_size": 1000, "balanced_accuracy": [0.4, 0.2, 0.5],
                         "n_unique_pred_raw": [9, 29, 9]})
    plot_model_comparison.unsupported_pairs = lambda: frozenset({("mmcontext", "Zeisel")})
    try:
        filled = with_unsupported_as_zero(runs)
        added = filled[filled["unsupported"]]
        assert added[["dataset", "model", "balanced_accuracy"]].values.tolist() == [
            ["Zeisel", "mmcontext_unblind", 0.0]]
        assert added["subset_size"].item() == 1000 and added["n_unique_pred_raw"].isna().all()
        assert with_unsupported_as_zero(filled).equals(filled)  # idempotent
    finally:
        plot_model_comparison.unsupported_pairs = unsupported_pairs


def test_unsupported_pairs_reads_the_organism_skip_list():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "datasets").mkdir()
        (tmp / "datasets" / "Zeisel.yml").write_text('organism: "Mus musculus"\n')
        (tmp / "datasets" / "HFRA.yml").write_text("annotation_key: cell_type\n")
        listed, empty = tmp / "listed.config", tmp / "empty.config"
        listed.write_text("params.model_blacklist_by_organism = [\n"
                          "    'mus musculus': ['mmcontext_blind', 'mmcontext_unblind'],\n]\n")
        empty.write_text("params.model_blacklist_by_organism = [:]\n")
        assert unsupported_pairs(listed, tmp / "datasets") == {("mmcontext", "Zeisel")}
        assert unsupported_pairs(empty, tmp / "datasets") == frozenset()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
