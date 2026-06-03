"""
Shared cell-type label normalization used by the evaluation workflow.

These functions hold the *single source of truth* for how raw predicted/true
labels are cleaned up before ontology lookup and mapping. They are used both by
the evaluation pipeline (``bin/evaluate.py``) and by the mapping-coverage check
(``bin/util/check_prediction_mapping_coverage.py``) so the coverage check
reflects the exact behavior of evaluation.

This module intentionally depends only on pandas + numpy so it can be imported
in lightweight contexts without pulling in scanpy/matplotlib/etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_predicted_labels(labels: pd.Series) -> pd.Series:
    """
    Normalize predicted cell-type labels.

    Applies, in order:
    1. ``astype(str)`` so non-string/NaN values become strings.
    2. ``strip()`` to remove surrounding whitespace (incl. trailing ``\\n``).
       This must happen BEFORE removing a trailing period, because some models
       (e.g. c2s_scale_27B_blind) emit ``"label.\\n"`` and a naive
       ``rstrip('.')`` would see the newline as the trailing char and leave the
       period in place.
    3. ``rstrip('.')`` to drop a trailing period.
    4. Strip leading/trailing single/double quotes, e.g. ``'"B cell"'`` or
       ``'\"\"\"gabaergic neuron\"\"\"'`` -> the bare label.
    5. ``strip()`` again to clean up any space the period/quote removal exposed.
    6. ``lower()`` for case-insensitive ontology matching.
    7. Replace every remaining internal newline (``\\r\\n``, ``\\n``, ``\\r``)
       with a single space. Many models (especially LLM wrappers) emit
       multi-line answers; the ontology mapping CSV stores their labels with
       newlines flattened to spaces this way (each newline -> one space, runs
       are NOT collapsed, so ``"...is:\\n\\nfoo"`` -> ``"...is:  foo"``), so this
       flattening is required for those rows to match their mapping entries.
    8. Convert resulting blanks to ``NaN``.

    Parameters
    ----------
    labels : pd.Series
        Raw predicted labels.

    Returns
    -------
    pd.Series
        Normalized labels (blanks replaced with ``np.nan``).
    """
    return (labels.astype(str)
                  .str.strip()
                  .str.rstrip('.')
                  .str.replace(r"^[\'\"]+|[\'\"]+$", "", regex=True)
                  .str.strip()
                  .str.lower()
                  .str.replace('\r\n', ' ', regex=False)
                  .str.replace('\n', ' ', regex=False)
                  .str.replace('\r', ' ', regex=False)
                  .replace('', np.nan))


def normalize_true_labels(labels: pd.Series) -> pd.Series:
    """
    Normalize true (ground-truth) cell-type labels: strip whitespace + lowercase.

    Mirrors the handling of true-label columns in the evaluation workflow.

    Parameters
    ----------
    labels : pd.Series
        Raw true labels.

    Returns
    -------
    pd.Series
        Normalized labels.
    """
    return labels.str.strip().str.lower()
