"""Small, generic helpers shared across the ``src`` package.

This module intentionally contains only utility functions. Business rules,
validation policy, feature engineering and pricing logic live in their
dedicated modules.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "normalize_column_name",
    "get_available_columns",
    "ensure_numeric",
    "safe_divide",
    "calculate_percentage_change",
]


def normalize_column_name(name: object) -> str:
    """Return a comparison key that ignores case, spaces and underscores.

    ``"Units Sold"``, ``"units_sold"`` and ``"UnitsSold"`` all normalize to
    ``"unitssold"`` so column aliases can be matched robustly.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_available_columns(df: pd.DataFrame, columns) -> list[str]:
    """Return the actual column names of ``df`` matching the requested names.

    Matching is case-, space- and underscore-insensitive. Requested order is
    preserved; names without a match are omitted.
    """
    lookup = {normalize_column_name(column): column for column in df.columns}
    available: list[str] = []
    for name in columns:
        actual = lookup.get(normalize_column_name(name))
        if actual is not None and actual not in available:
            available.append(actual)
    return available


def ensure_numeric(value, default: float = float("nan")):
    """Convert a scalar or a pandas Series to numeric values.

    Unparseable scalars become ``default``; unparseable Series entries become
    ``default`` (``NaN`` by default) instead of raising.
    """
    if isinstance(value, pd.Series):
        numeric = pd.to_numeric(value, errors="coerce").astype("float64")
        if default is not None:
            numeric = numeric.fillna(default)
        return numeric
    try:
        return float(value)
    except (TypeError, ValueError):
        return default if default is None else float(default)


def safe_divide(numerator: float, denominator: float, default: float = float("nan")) -> float:
    """Divide ``numerator`` by ``denominator``, returning ``default`` when undefined.

    ``default`` is returned when either input is missing/non-finite or the
    denominator is zero.
    """
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float(default)
    return num / den


def calculate_percentage_change(new_value: float, old_value: float) -> float:
    """Percentage change from ``old_value`` to ``new_value``.

    Returns ``NaN`` when the reference value is missing, non-finite or zero.
    """
    try:
        new = float(new_value)
        old = float(old_value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(new) or not np.isfinite(old) or old == 0:
        return float("nan")
    return (new - old) / abs(old) * 100.0
