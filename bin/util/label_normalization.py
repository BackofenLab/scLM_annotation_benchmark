"""Normalization of cell type labels before they are matched to the Cell Ontology."""
from __future__ import annotations

import pandas as pd

# Label of a missing or blank prediction. A plain string survives a CSV round trip, unlike NaN.
EMPTY_PREDICTION_LABEL = "no prediction"


def canonicalize_empty_labels(labels: pd.Series) -> pd.Series:
    """Replace blank and "nan" entries by EMPTY_PREDICTION_LABEL."""
    s = labels.astype(str).str.strip()
    return s.mask(s.str.lower().isin(["", "nan"]), EMPTY_PREDICTION_LABEL)


def normalize_predicted_labels(labels: pd.Series) -> pd.Series:
    """Lowercase predicted labels and remove surrounding whitespace, quotes and a trailing period.

    A line break inside an answer becomes one space, so several line breaks leave several
    spaces. Missing and blank answers become EMPTY_PREDICTION_LABEL.
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
    """Strip and lowercase true labels."""
    return labels.str.strip().str.lower()


def label_key(s) -> str:
    """Key for matching a label to Cell Ontology names and reviewed mappings: lowercase with
    whitespace runs collapsed, "" for None. Mirrored in ``label_constraint.R``."""
    return " ".join(s.lower().split()) if s is not None else ""
