"""Checks for the rare-type rates (paper/scripts/figures_rare_types.py): recall is compared within
each dataset, and the outcomes partition the rare cell types.

Run with pytest, or directly in the evaluation environment: python paper/tests/test_rare_types.py
"""
import sys
from math import isclose
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper" / "scripts"))
from figures_rare_types import _rarity_rates  # noqa: E402

# One run on two datasets: per type its recall, whether the method named it, its share of the
# full dataset, and whether the method has a label for it.
TYPES = [("D1", "a", 0.8, True, 0.30, True), ("D1", "b", 0.6, True, 0.20, True),
         ("D1", "r1", 0.0, False, 0.005, False), ("D1", "r2", 0.6, True, 0.001, True),
         ("D2", "c", 0.4, True, 0.50, True),
         ("D2", "r3", 0.2, True, 0.002, True), ("D2", "r4", 0.0, True, 0.009, True)]
RARE = pd.DataFrame([dict(method="singler", mode="unblind", dataset=d, k=k, recall=recall,
                          emitted=emitted, orig_prevalence=prevalence, reachable=reachable)
                     for d, k, recall, emitted, prevalence, reachable in TYPES])


def test_recall_is_averaged_within_each_dataset_first():
    rates = _rarity_rates(RARE).loc["singler_unblind"]
    assert isclose(rates["common"], (0.7 + 0.4) / 2)
    assert isclose(rates["rare"], (0.3 + 0.1) / 2)


def test_the_outcomes_partition_the_rare_types():
    rates = _rarity_rates(RARE).loc["singler_unblind"]
    assert rates[["most", "some", "wrong", "unreachable"]].tolist() == [0.25] * 4
    assert rates["never"] == 0


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
