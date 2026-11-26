"""Checks for the share of answers outside the list (paper/scripts/methods.py): each run is held to
its own list, an answer naming no cell type is not outside it, and shares are averaged over cell
types.

Run with pytest, or directly in the evaluation environment:
python paper/tests/test_unlisted_answers.py
"""
import sys
from math import isclose
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper" / "scripts"))
from methods import _unlisted  # noqa: E402

# Two constrained runs of one method, whose lists are their true types: {a, b} in D and {b, c}
# in E. Rows are (dataset, true label, predicted label, cells, lineage).
PAIRS = pd.DataFrame(
    [("D", "a", "a", 2, "exact"), ("D", "a", "c", 2, "other"), ("D", "b", "b", 1, "exact"),
     ("D", "b", "cell", 1, "none"), ("D", "b", "a", 2, "other"),
     ("E", "c", "c", 3, "exact"), ("E", "c", "a", 1, "other"), ("E", "b", "b", 2, "exact")],
    columns=["dataset", "true_label", "predicted_label", "n_cells", "lineage"],
).assign(model="m_constrained")


def test_only_types_off_the_run_s_own_list_count():
    shares = _unlisted(PAIRS)
    # D: c is off the list for 2 of a's 4 cells; 'cell' names no type, and a is listed.
    assert isclose(shares[("m_constrained", "D")], (2 / 4 + 0) / 2)
    # E: a is off the list for 1 of c's 4 cells, although D lists it.
    assert isclose(shares[("m_constrained", "E")], (1 / 4 + 0) / 2)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
