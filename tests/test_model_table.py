"""Checks for Table 1 (bin/model_table.py) and the sourced facts it is built from.

Run with pytest, or directly in the evaluation environment: python tests/test_model_table.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from model_table import parameters, reasoning, rows, si  # noqa: E402

FACTS = ROOT / "cache" / "result" / "model_facts.csv"


def facts():
    return pd.read_csv(FACTS, dtype=str)


def test_counts_are_shown_with_three_significant_digits():
    assert si(2_614_341_888) == "2.61B"
    assert si(495_821_769) == "496M"
    assert si(23_256) == "23.3K"
    assert si(1_000_000_000_000) == "1T"


def test_parameters_cell_names_the_parts():
    instructcell = {"parameters_total": "495821769", "parameters.text_encoder": "109616256",
                    "parameters.text_decoder": "113275008",
                    "parameters.expression_encoder": "167668224",
                    "parameters.expression_decoder": "105262281"}
    assert parameters(instructcell) == "496M (TE 110M, TD 113M, EE 168M, ED 105M)"
    assert parameters({"parameters_total": "117000000000", "parameters_active": "5100000000"}) \
        == "117B (5.1B active)"
    assert parameters({"parameters_total": "2614341888",
                       "parameters.text_decoder": "2614341888"}) == "2.61B (TD)"
    assert parameters({"parameters_total": "not disclosed"}) == "not disclosed"


def test_every_value_has_a_source():
    f = facts()
    assert f.model.nunique() == 21
    assert not f.model.str.contains("agent").any()
    assert f[["value", "source", "evidence", "checked"]].notna().all().all()


def test_component_counts_add_up_to_the_total():
    f = facts()
    for model, g in f.groupby("model"):
        v = dict(zip(g.field, g.value))
        parts = [int(x) for k, x in v.items()
                 if k.startswith("parameters.") and not k.startswith("parameters.model:")
                 and k != "parameters.vision_encoder"]
        if parts:
            assert sum(parts) == int(v["parameters_total"]), model


def test_reasoning_is_given_for_every_llm():
    assert reasoning({"reasoning": "effort xhigh"}) == "xhigh"
    assert reasoning({"reasoning": "budget 10000"}) == "10,000 tokens"
    assert reasoning({"reasoning": "none"}) == "none"
    assert reasoning({}) == "--"
    f = facts()
    llms = set(f[(f.field == "group") & (f.value == "LLM")].model)
    setting = f[f.field == "reasoning"].set_index("model")["value"]
    assert set(setting.index) == llms
    tier = f[f.field == "subcategory"].set_index("model")["value"]
    assert all((setting[m] == "none") == (tier[m] == "open_nonreasoning") for m in llms)


def test_table_follows_the_figure_classification():
    f = facts()
    assert len(rows(f)) == 21
    wrong = f.copy()
    wrong.loc[(wrong.model == "langcell") & (wrong.field == "subcategory"), "value"] = "generative"
    try:
        rows(wrong)
    except SystemExit:
        return
    raise AssertionError("a subcategory that disagrees with the figures must stop the script")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
