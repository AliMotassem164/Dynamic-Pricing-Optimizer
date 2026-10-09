"""Generic dataset validation for the pricing pipeline.

Validation **reports** problems; it never deletes rows, never invents values
and never repairs data behind the caller's back. The main entry point,
:func:`validate_dataset`, returns a structured result::

    {
        "is_valid": True,
        "errors": [],     # problems that block the pipeline
        "warnings": [],   # problems that must be visible but do not block
        "info": {...},    # dataset statistics for reporting
    }

Canonical schema
----------------
Required: ``Date``, ``Product_ID``, ``Price``, ``Demand``.
Strongly recommended for profit optimization: ``Unit_Cost`` - its absence is
a **warning**, never an error: revenue optimization still works without it.
Optional: ``Store_ID``, ``Competitor_Price``, ``Inventory``, ``Discount``,
``Promotion``, ``Category``, ``Region``.

Column matching is case-, space- and underscore-insensitive, so ``"Product
ID"`` resolves ``Product_ID`` before the column-mapping step has run.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .pricing import assess_price_variation
from .utils import normalize_column_name

__all__ = [
    "CANONICAL_REQUIRED_COLUMNS",
    "RECOMMENDED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "UNIT_COST_MISSING_WARNING",
    "validate_required_columns",
    "validate_data_types",
    "validate_price_data",
    "validate_demand_data",
    "validate_unit_cost",
    "validate_dataset",
]

CANONICAL_REQUIRED_COLUMNS: list[str] = ["Date", "Product_ID", "Price", "Demand"]
RECOMMENDED_COLUMNS: list[str] = ["Unit_Cost"]
OPTIONAL_COLUMNS: list[str] = [
    "Store_ID", "Competitor_Price", "Inventory", "Discount",
    "Promotion", "Category", "Region",
]

#: Canonical warning emitted whenever the data carries no usable unit cost.
UNIT_COST_MISSING_WARNING: str = (
    "Unit Cost is unavailable. Revenue optimization is possible, but true profit "
    "optimization requires Unit Cost."
)

#: A column missing more than half of its values is reported as excessive.
MAX_MISSING_FRACTION: float = 0.5


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {normalize_column_name(column): column for column in df.columns}


def _resolve(df: pd.DataFrame, canonical: str, lookup: dict[str, str] | None = None) -> str | None:
    """Resolve a canonical field name to the actual column in ``df`` (None when absent)."""
    table = lookup if lookup is not None else _column_lookup(df)
    return table.get(normalize_column_name(canonical))


def _as_result(
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"errors": errors or [], "warnings": warnings or [], "info": info or {}}


def _fraction(count: int, total: int) -> str:
    return f"{count:,} row(s) ({count / max(total, 1):.1%})"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str] = CANONICAL_REQUIRED_COLUMNS,
) -> dict[str, Any]:
    """Check that every required column is present (case/spacing-insensitive).

    Returns ``{"errors", "warnings", "info"}`` where ``info`` contains the
    ``resolved_columns`` mapping (canonical -> actual column name) and the
    ``missing_columns`` list.
    """
    lookup = _column_lookup(df)
    resolved = {name: _resolve(df, name, lookup) for name in required_columns}
    missing = [name for name, actual in resolved.items() if actual is None]
    errors = []
    if missing:
        errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    return _as_result(
        errors=errors,
        info={
            "required_columns": list(required_columns),
            "resolved_columns": resolved,
            "missing_columns": missing,
        },
    )


def validate_data_types(df: pd.DataFrame) -> dict[str, Any]:
    """Check that date, numeric and flag columns can be interpreted correctly.

    Reports per-column dtypes, values that fail conversion (missing dates,
    non-numeric business numbers, non 0/1 indicators) and the parsed date
    range. Nothing is modified.
    """
    errors: list[str] = []
    warnings: list[str] = []
    lookup = _column_lookup(df)
    invalid_values: dict[str, int] = {}

    date_col = _resolve(df, "Date", lookup)
    date_range = None
    if date_col is None:
        errors.append("Date column not found - dates cannot be validated.")
    else:
        original = df[date_col]
        parsed = pd.to_datetime(original, errors="coerce")
        n_invalid = int((original.notna() & parsed.isna()).sum())
        if n_invalid:
            invalid_values["Date"] = n_invalid
            warnings.append(f"{_fraction(n_invalid, len(df))} are invalid/unparseable dates")
        valid = parsed.dropna()
        if valid.empty:
            errors.append(f"Date column '{date_col}' contains no valid dates.")
        else:
            date_range = (valid.min(), valid.max())

    numeric_columns = [
        actual for actual in (
            _resolve(df, name, lookup) for name in
            ("Price", "Demand", "Unit_Cost", "Inventory", "Competitor_Price", "Discount")
        ) if actual is not None
    ]
    for column in numeric_columns:
        original = df[column]
        if pd.api.types.is_numeric_dtype(original) and not pd.api.types.is_bool_dtype(original):
            continue
        converted = pd.to_numeric(original, errors="coerce")
        n_invalid = int((original.notna() & converted.isna()).sum())
        if n_invalid:
            invalid_values[column] = n_invalid
            warnings.append(
                f"{_fraction(n_invalid, len(df))} in '{column}' are non-numeric"
            )

    for column in ("Promotion", "Epidemic"):
        actual = _resolve(df, column, lookup)
        if actual is None:
            continue
        converted = pd.to_numeric(df[actual], errors="coerce").dropna()
        off_values = int((~converted.isin([0, 1])).sum())
        if off_values:
            warnings.append(
                f"{_fraction(off_values, len(df))} in '{actual}' are not 0/1 indicator values"
            )

    return _as_result(
        errors=errors,
        warnings=warnings,
        info={
            "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
            "invalid_values": invalid_values,
            "date_range": date_range,
        },
    )


def validate_price_data(df: pd.DataFrame, price_col: str = "Price") -> dict[str, Any]:
    """Validate the price column: usability, value sanity and price variation.

    * missing column or no valid numeric values -> **error**
    * negative prices -> **error** (physically invalid; reported, never removed)
    * missing values, zero prices, insufficient/limited variation -> warnings
    """
    errors: list[str] = []
    warnings: list[str] = []
    lookup = _column_lookup(df)
    actual = _resolve(df, price_col, lookup)
    if actual is None:
        return _as_result(
            errors=[f"Price column '{price_col}' not found. Available columns: {list(df.columns)}"]
        )

    price = pd.to_numeric(df[actual], errors="coerce")
    n_valid = int(price.notna().sum())
    if n_valid == 0:
        return _as_result(
            errors=[f"Price column '{actual}' contains no valid numeric values."],
            info={"column": actual},
        )

    n_missing = int(price.isna().sum())
    n_negative = int((price < 0).sum())
    n_zero = int((price == 0).sum())

    info: dict[str, Any] = {
        "column": actual,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "min": float(price.min()),
        "max": float(price.max()),
        "mean": float(price.mean()),
        "std": float(price.std()) if n_valid > 1 else None,
    }
    if n_missing:
        warnings.append(f"Price: {_fraction(n_missing, len(df))} missing or non-numeric")
    if n_negative:
        errors.append(
            f"Price: {_fraction(n_negative, len(df))} are negative - negative prices are "
            "invalid and must be fixed before pricing (no rows were removed)"
        )
    if n_zero:
        warnings.append(
            f"Price: {_fraction(n_zero, len(df))} are zero - zero prices cannot be used for "
            "elasticity estimates or price scenarios"
        )

    positive = price.dropna()
    positive = positive[positive > 0]
    if len(positive):
        variation = assess_price_variation(positive, price_col=actual)
        info["variation"] = {
            "verdict": variation["verdict"],
            "n_unique_prices": variation["n_unique_prices"],
            "price_cv": variation["price_cv"],
            "scope": "pooled dataset rows (per-series variation is checked when a product "
                     "series is selected for pricing)",
        }
        if variation["verdict"] == "INSUFFICIENT":
            warnings.append(
                "Historical price variation is insufficient for reliable price-response "
                "estimation. Optimization should be treated as scenario analysis only."
            )
        elif variation["verdict"] == "LIMITED":
            warnings.append(
                "Historical price variation is limited. Price-response estimates are "
                "unstable; treat results as indicative scenario analysis."
            )
    return _as_result(errors=errors, warnings=warnings, info=info)


def validate_demand_data(df: pd.DataFrame, demand_col: str = "Demand") -> dict[str, Any]:
    """Validate the demand column: usability, negative values and dispersion.

    * missing column or no valid numeric values -> **error**
    * negative demand -> **error** (physically invalid; reported, never removed)
    * missing values, constant demand -> warnings
    """
    errors: list[str] = []
    warnings: list[str] = []
    lookup = _column_lookup(df)
    actual = _resolve(df, demand_col, lookup)
    if actual is None:
        return _as_result(
            errors=[f"Demand column '{demand_col}' not found. Available columns: {list(df.columns)}"]
        )

    demand = pd.to_numeric(df[actual], errors="coerce")
    n_valid = int(demand.notna().sum())
    if n_valid == 0:
        return _as_result(
            errors=[f"Demand column '{actual}' contains no valid numeric values."],
            info={"column": actual},
        )

    n_missing = int(demand.isna().sum())
    n_negative = int((demand < 0).sum())
    n_zero = int((demand == 0).sum())

    info: dict[str, Any] = {
        "column": actual,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "min": float(demand.min()),
        "max": float(demand.max()),
        "mean": float(demand.mean()),
        "std": float(demand.std()) if n_valid > 1 else None,
        "n_unique": int(demand.nunique()),
    }
    if n_missing:
        warnings.append(f"Demand: {_fraction(n_missing, len(df))} missing or non-numeric")
    if n_negative:
        errors.append(
            f"Demand: {_fraction(n_negative, len(df))} are negative - negative demand is "
            "invalid and must be fixed before modeling (no rows were removed)"
        )
    if info["n_unique"] <= 1:
        warnings.append(
            f"Demand is constant across the dataset (unique values: {info['n_unique']}) - "
            "no price response can be learned from a flat series"
        )
    return _as_result(errors=errors, warnings=warnings, info=info)


def validate_unit_cost(df: pd.DataFrame, unit_cost_col: str = "Unit_Cost") -> dict[str, Any]:
    """Check the optional ``Unit_Cost`` column.

    A missing (or unusable) unit cost is **never** an error: it produces the
    canonical warning :data:`UNIT_COST_MISSING_WARNING` and revenue
    optimization stays available. Non-positive/invalid existing values are
    reported as warnings.
    """
    warnings: list[str] = []
    lookup = _column_lookup(df)
    actual = _resolve(df, unit_cost_col, lookup)

    if actual is None:
        return _as_result(
            warnings=[UNIT_COST_MISSING_WARNING],
            info={"column": None, "available": False},
        )

    cost = pd.to_numeric(df[actual], errors="coerce")
    original = df[actual]
    n_non_numeric = int((original.notna() & cost.isna()).sum())
    valid = cost.dropna()
    positive = valid[valid > 0]
    n_negative = int((valid < 0).sum())

    available = bool(len(positive))
    info: dict[str, Any] = {
        "column": actual,
        "available": available,
        "n_valid": int(len(valid)),
        "n_positive": int(len(positive)),
    }
    if not available:
        warnings.append(
            f"Unit Cost column '{actual}' exists but holds no positive value - revenue "
            "optimization can be performed, but true profit optimization requires "
            "business-provided unit cost."
        )
    if n_non_numeric:
        warnings.append(
            f"Unit Cost: {_fraction(n_non_numeric, len(df))} are non-numeric and unusable"
        )
    if n_negative:
        warnings.append(
            f"Unit Cost: {_fraction(n_negative, len(df))} are negative - costs must be >= 0"
        )
    if len(positive):
        info.update({
            "min": float(positive.min()),
            "max": float(positive.max()),
            "mean": float(positive.mean()),
            "median": float(positive.median()),
        })
    return _as_result(warnings=warnings, info=info)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full validation suite and return a structured result.

    Returns ``{"is_valid", "errors", "warnings", "info"}``.

    * ``errors`` - problems that block the pipeline (missing required
      columns, unusable date/price/demand columns, negative prices or
      demand).
    * ``warnings`` - problems that must be visible but do not block, e.g. a
      missing ``Unit_Cost`` (revenue optimization still possible), row-level
      missing values, duplicate rows or insufficient price variation.
    * ``info`` - dataset statistics for reporting.

    Nothing is dropped, imputed or invented.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate_dataset() expects a pandas DataFrame, got {type(df).__name__}")

    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
    }

    def absorb(result: dict[str, Any], section: str) -> None:
        errors.extend(result["errors"])
        warnings.extend(result["warnings"])
        if result["info"]:
            info[section] = result["info"]

    required = validate_required_columns(df)
    absorb(required, "required_columns")
    resolved: dict[str, str | None] = required["info"]["resolved_columns"]

    if resolved.get("Price"):
        absorb(validate_price_data(df, price_col=resolved["Price"]), "price")
    if resolved.get("Demand"):
        absorb(validate_demand_data(df, demand_col=resolved["Demand"]), "demand")
    if not required["info"]["missing_columns"]:
        absorb(validate_data_types(df), "data_types")
    absorb(validate_unit_cost(df), "unit_cost")

    if "Date" in info.get("data_types", {}):
        info["date_range"] = info["data_types"]["date_range"]
    if resolved.get("Product_ID"):
        info["n_products"] = int(df[resolved["Product_ID"]].nunique())
    if resolved.get("Store_ID"):
        info["n_stores"] = int(df[resolved["Store_ID"]].nunique())

    missing_counts = df.isna().sum()
    info["missing_values"] = {str(column): int(count) for column, count in missing_counts.items()
                              if count}
    for column, count in missing_counts.items():
        if count / max(len(df), 1) > MAX_MISSING_FRACTION:
            warnings.append(
                f"Column '{column}' is {_fraction(int(count), len(df))} missing "
                f"(exceeds the {MAX_MISSING_FRACTION:.0%} threshold)"
            )

    n_duplicates = int(df.duplicated().sum())
    info["duplicate_rows"] = n_duplicates
    if n_duplicates:
        warnings.append(f"{_fraction(n_duplicates, len(df))} are exact duplicate rows")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }
