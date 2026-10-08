"""
Shared cell-type label normalization used by the evaluation workflow.

These functions hold the *single source of truth* for how raw predicted/true
labels are cleaned up before ontology lookup and mapping. They are used both by
the evaluation pipeline (``bin/evaluate.py``) and by the mapping-coverage check
(``bin/util/check_prediction_mapping_coverage.py``) so the coverage check
reflects the exact behavior of evaluation.

This module intentionally depends only on pandas so it can be imported in
lightweight contexts without pulling in scanpy/matplotlib/etc.
"""
from __future__ import annotations

import pandas as pd


# Explicit, human-readable sentinel for predictions that are missing/blank.
# An empty prediction is a real outcome (the model returned nothing) and must be
# reviewable in ``cell_type_ontology_mappings.csv`` like any other label. Using a
# plain string -- rather than ``np.nan`` or the accidental ``"nan"`` produced by
# ``astype(str)`` on a NaN -- keeps the label stable across pandas CSV
# round-trips, whose default ``na_values`` (which include ``"nan"`` and blanks)
# would otherwise turn it back into ``NaN`` and break key matching. The matching
# read side lives in ``util.ontology_mapper._read_mapping_csv``.
EMPTY_PREDICTION_LABEL = "no prediction"


def canonicalize_empty_labels(labels: pd.Series) -> pd.Series:
    """Map blank / literal-``"nan"`` placeholders to :data:`EMPTY_PREDICTION_LABEL`.

    Used when reading already-written labels (e.g. the ``predicted_label`` column
    of the ontology mapping CSV) so legacy rows whose key was stored as an empty
    cell or the text ``"nan"`` line up with the sentinel that
    :func:`normalize_predicted_labels` emits for empty predictions.

    Parameters
    ----------
    labels : pd.Series
        Labels possibly containing blanks / ``NaN`` / the text ``"nan"``.

    Returns
    -------
    pd.Series
        Labels with every empty placeholder replaced by the sentinel.
    """
    s = labels.astype(str).str.strip()
    return s.mask(s.str.lower().isin(["", "nan"]), EMPTY_PREDICTION_LABEL)


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
    8. Collapse every "empty" form -- a genuinely blank cell (read as ``NaN``,
       turned into the string ``"nan"`` by step 1), a whitespace-only cell
       (now ``""``), or an originally-missing value -- to the single explicit
       sentinel :data:`EMPTY_PREDICTION_LABEL`. This replaces the old behavior
       where blanks became either the fragile string ``"nan"`` or ``np.nan``;
       both were silently turned back into ``NaN`` when the mapping CSV was
       re-read, so empty predictions could never be matched to a manual mapping.
       The sentinel is an ordinary, round-trip-safe label, so empty predictions
       can be reviewed/mapped in ``cell_type_ontology_mappings.csv`` like any
       other prediction.

    Parameters
    ----------
    labels : pd.Series
        Raw predicted labels.

    Returns
    -------
    pd.Series
        Normalized labels (empty predictions replaced with
        :data:`EMPTY_PREDICTION_LABEL`).
    """
    normalized = (labels.astype(str)
                  .str.strip()
                  .str.rstrip('.')
                  .str.replace(r"^[\'\"]+|[\'\"]+$", "", regex=True)
                  .str.strip()
                  .str.lower()
                  .str.replace('\r\n', ' ', regex=False)
                  .str.replace('\n', ' ', regex=False)
                  .str.replace('\r', ' ', regex=False))
    missing = labels.isna() | normalized.isin(['', 'nan'])
    return normalized.mask(missing, EMPTY_PREDICTION_LABEL)


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


def label_key(s) -> str:
    """The key under which a label is matched against Cell Ontology names and reviewed mappings:
    lowercase, whitespace runs collapsed; ``None`` gives ``""``. Mirrored in ``label_constraint.R``."""
    return " ".join(s.lower().split()) if s is not None else ""
