"""Helpers of the scripts that write LaTeX tables: escaping, numbers, lists and the best values
in bold."""
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd


def tex(text):
    """text with the LaTeX specials that occur in names escaped."""
    return (str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")
            .replace("%", r"\%"))


def fmt(value, digits=3):
    """value rounded half up as the decimal number it is written as, or -- if it is missing. The
    result tables keep four decimals, and rounding the binary number would print a stored 0.3495
    as 0.349."""
    if value is None or pd.isna(value):
        return "--"
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def join(names):
    """The names as running text: a, b and c."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def best_as_printed(values, lowest=False, digits=3):
    """The highest of values, or the lowest, as printed, so that every value that prints the same
    counts as best. None if there is no value."""
    values = pd.to_numeric(values, errors="coerce")
    if not values.notna().any():
        return None
    return fmt(values.min() if lowest else values.max(), digits)


def bold(text, best):
    """text in bold if it prints the best value."""
    return f"\\textbf{{{text}}}" if text != "--" and text == best else text
