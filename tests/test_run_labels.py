"""Checks for the run labels and groups the figures and tables share (bin/plot_model_comparison.py):
every method, baselines included, carries its setting, and the chance models carry none.

Run with pytest, or directly in the evaluation environment: python tests/test_run_labels.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from manuscript_figures import (  # noqa: E402
    BLOCK_FAMILY, BLOCK_ORDER, BLOCK_SHORT, MAPPING_ROWS, RARE_BLOCKS, select_flow_models)
from plot_model_comparison import group_name, run_label  # noqa: E402

RUNS = {"sctype_blind": "unconstrained baselines", "singler_unblind": "constrained baselines",
        "mmcontext_blind": "unconstrained scLLM", "c2s_scale_2B_unblind": "constrained scLLM",
        "gpt-5.5_blind": "unconstrained LLM", "gemini-3.1-pro-preview_unblind": "constrained LLM"}


def test_every_run_but_chance_carries_its_setting():
    assert run_label("sctype_blind") == "sctype (U)"
    assert run_label("singler_unblind", "SingleR") == "SingleR (C)"
    assert run_label("gpt-5.5_blind", "GPT-5.5") == "GPT-5.5 (U)"
    assert run_label("random_stratified") == "random_stratified"


def test_groups_split_every_family_by_setting():
    assert {model: group_name(model) for model in RUNS} == RUNS


def test_the_figure_groups_cover_both_settings_of_every_family():
    assert set(RUNS.values()) <= set(BLOCK_ORDER) == set(BLOCK_SHORT) == set(BLOCK_FAMILY)
    assert {block for block, _family, _style in RARE_BLOCKS} == set(RUNS.values())
    assert {(family, mode) for family, mode, _label in MAPPING_ROWS} == {
        (family, mode) for family in ("classical", "scLLM", "LLM") for mode in ("blind", "unblind")}


def test_flow_models_are_the_best_constrained_runs_with_their_unconstrained_twins():
    scores = {"sctype": (0.31, 0.38), "scsa": (0.30, 0.41), "langcell": (0.30, 0.45),
              "mmcontext": (0.25, 0.47), "gpt-5.5": (0.40, 0.60), "kimi-k2.6": (0.45, 0.55)}
    df = pd.DataFrame([{"model": f"{base}_{mode}", "dataset": dataset, "balanced_accuracy": score}
                       for base, pair in scores.items()
                       for mode, score in zip(("blind", "unblind"), pair)
                       for dataset in ("HFRA", "HIHA")])
    assert select_flow_models(df) == [["scsa_unblind", "mmcontext_unblind", "gpt-5.5_unblind"],
                                      ["scsa_blind", "mmcontext_blind", "gpt-5.5_blind"]]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
