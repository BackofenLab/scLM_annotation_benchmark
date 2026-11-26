"""Names of runs and how they are written into file names.

A method is a folder in ``models/``. Its name ends in ``_unconstrained`` (no list of expected cell
types) or ``_constrained`` (with the list). Most methods write one ``predictions.csv`` per dataset.
A wrapper for several served models writes one ``predictions_<provider>--<model>.csv`` per model,
and each of these files is a run of its own.
"""
from __future__ import annotations

from pathlib import Path

SETTINGS = ("_constrained", "_unconstrained")
MAPPED_SUFFIX = "_mapped"
# The row of all cells in the tables of scores per cell type.
OVERALL = "__OVERALL__"


def prediction_stem(path) -> str:
    """File name of a prediction file without extension and without the mapped suffix."""
    return Path(path).stem.removesuffix(MAPPED_SUFFIX)


def run_name(method: str, stem: str) -> str:
    """Name of a run in the result tables: the method, or the served model with the method's
    setting if the prediction file belongs to one model of a wrapper."""
    if not stem.startswith("predictions_"):
        return method
    served = stem.removeprefix("predictions_").split("--")[-1]
    setting = next((s for s in SETTINGS if method.endswith(s)), "")
    return f"{served}{setting}"


def setting_of(model: str) -> str:
    """``unconstrained``, ``constrained`` or, for a method without settings, ``baseline``."""
    name = str(model).lower()
    if name.endswith("_constrained"):
        return "constrained"
    if name.endswith("_unconstrained"):
        return "unconstrained"
    return "baseline"


def mapping_origin(method: str, stem: str, dataset: str) -> str:
    """The origin recorded in the mapping table for labels first seen in this run."""
    served = stem.removeprefix("predictions_") if stem.startswith("predictions_") else ""
    return "_".join(part for part in (method, served, dataset) if part)
