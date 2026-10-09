"""Generic, leakage-safe feature engineering.

Extracted from ``Notebooks/Feature_Engineering.ipynb`` into reusable
functions. The module turns raw company data into the engineered features the
saved demand model expects, using only information available at prediction
time:

* time features from ``Date``,
* price features (previous price, price change, effective price, competitor
  comparison) computed *per series* after chronological sorting,
* lag/rolling demand features built with ``shift(1)`` **before** any rolling
  window, so current or future demand never leaks into a feature,
* inventory and promotion interaction features when those optional columns
  exist.

Optional columns are handled gracefully: missing ``Competitor_Price``,
``Inventory``, ``Discount`` or ``Promotion`` simply means the related
features are skipped and reported in ``df.attrs["feature_report"]``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL_SCHEMA",
    "REQUIRED_COLUMNS",
    "standardize_columns",
    "validate_and_prepare_data",
    "create_time_features",
    "create_price_features",
    "create_lag_features",
    "create_inventory_features",
    "create_interaction_features",
    "engineer_features",
]

CANONICAL_SCHEMA: dict[str, str] = {
    "Date": "required",
    "Product_ID": "required",
    "Price": "required",
    "Demand": "required",
    "Store_ID": "optional",
    "Unit_Cost": "optional",
    "Inventory": "optional",
    "Competitor_Price": "optional",
    "Discount": "optional",
    "Promotion": "optional",
    "Category": "optional",
    "Region": "optional",
    "Weather": "optional",
    "Seasonality": "optional",
    "Epidemic": "optional",
}

REQUIRED_COLUMNS: list[str] = [name for name, kind in CANONICAL_SCHEMA.items() if kind == "required"]
NUMERIC_COLUMNS: list[str] = ["Price", "Demand", "Unit_Cost", "Inventory", "Competitor_Price", "Discount"]
FLAG_COLUMNS: list[str] = ["Promotion", "Epidemic"]
TEXT_COLUMNS: list[str] = ["Product_ID", "Store_ID", "Category", "Region", "Weather", "Seasonality"]

_FLAG_TEXT: dict[str, float] = {
    "1": 1.0, "true": 1.0, "yes": 1.0, "y": 1.0, "t": 1.0,
    "0": 0.0, "false": 0.0, "no": 0.0, "n": 0.0, "f": 0.0,
}


# ---------------------------------------------------------------------------
# Column mapping and data type preparation
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame, column_mapping: dict[str, str | None]) -> pd.DataFrame:
    """Rename company-specific columns to the canonical schema.

    ``column_mapping`` maps ``{canonical_name: company_column_name}``, for
    example ``{"Date": "order_date", "Product_ID": "sku", "Price": "unit_price"}``.
    Entries whose value is ``None``/empty mean "the company does not have this
    column" and are skipped. Unmapped columns are left untouched.

    Raises ``ValueError`` for unknown canonical names, unmapped required
    fields, source columns mapped to several canonical names, or clashes with
    columns that already carry a canonical name. Missing *optional* mapped
    columns are skipped; they never cause an error.
    """
    if not isinstance(column_mapping, dict):
        raise TypeError("column_mapping must be a dict of {canonical_name: company_column_name}")

    unknown = [name for name in column_mapping if name not in CANONICAL_SCHEMA]
    if unknown:
        raise ValueError(
            f"Unknown canonical name(s) in column mapping: {unknown}. "
            f"Valid canonical names: {list(CANONICAL_SCHEMA)}"
        )

    mapping = {canon: source for canon, source in column_mapping.items() if source}
    missing_required = [name for name in REQUIRED_COLUMNS if name not in mapping]
    if missing_required:
        raise ValueError(f"No source column mapped for required field(s): {missing_required}")

    sources = list(mapping.values())
    duplicated = sorted({source for source in sources if sources.count(source) > 1})
    if duplicated:
        raise ValueError(
            f"The same source column is mapped to several canonical names: {duplicated}"
        )

    absent = {canon: source for canon, source in mapping.items() if source not in df.columns}
    absent_required = {canon: source for canon, source in absent.items()
                       if CANONICAL_SCHEMA[canon] == "required"}
    if absent_required:
        raise KeyError(f"Mapped required column(s) not found in the data: {absent_required}")
    for canon in absent:
        mapping.pop(canon)

    clashes = [column for column in df.columns if column in mapping and mapping[column] != column]
    if clashes:
        raise ValueError(
            f"Columns already named like canonical fields but mapped elsewhere: {clashes}"
        )

    out = df.rename(columns={source: canon for canon, source in mapping.items()}).copy()
    out.attrs["column_mapping_report"] = {
        "applied": {canon: source for canon, source in mapping.items()},
        "skipped": absent,
        "unmapped_columns": [column for column in out.columns if column not in CANONICAL_SCHEMA],
    }
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    if not pd.api.types.is_numeric_dtype(series):
        text = series.astype("string").str.strip()
        series = text.mask(text == "", pd.NA)
    out = pd.to_numeric(series, errors="coerce").astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _to_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return _to_numeric(series)
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(lambda value: _FLAG_TEXT.get(value, np.nan) if pd.notna(value) else np.nan)
    return pd.to_numeric(mapped, errors="coerce").astype("float64")


def _id_to_string(series: pd.Series) -> pd.Series:
    """IDs/categories -> string. Integer-like floats (1.0) become '1', blanks become missing."""
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if len(non_null) and bool((non_null == non_null.round()).all()):
            series = series.astype("Int64")
    out = series.astype("string").str.strip()
    return out.mask(out == "", pd.NA)


def validate_and_prepare_data(
    df: pd.DataFrame,
    date_format: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize dtypes and check that the canonical schema is usable.

    Converts ``Date`` to datetime, business numbers to numeric, 0/1
    indicators to numeric (yes/no text is understood) and identifiers /
    categories to string. Values that cannot be converted become
    ``NaN``/``NaT`` - nothing is dropped or invented.

    Returns ``(prepared_df, report)`` where ``report`` contains
    ``invalid_values`` (per column), ``missing_values`` (per column),
    ``date_range`` and ``issues``. Raises ``KeyError`` when a required
    canonical column is absent.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate_and_prepare_data() expects a DataFrame, got {type(df).__name__}")
    missing_required = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing_required:
        raise KeyError(
            f"Required column(s) missing from the data: {missing_required}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    invalid_values: dict[str, int] = {}
    issues: list[str] = []

    def record(column: str, original: pd.Series, converted: pd.Series) -> None:
        invalid = int((original.notna() & converted.isna()).sum())
        if invalid:
            invalid_values[column] = invalid
            issues.append(f"{invalid:,} value(s) in '{column}' could not be converted and are missing")
        if original.isna().any():
            issues.append(f"{original.isna().sum():,} pre-existing missing value(s) in '{column}'")

    original_date = out["Date"]
    converted_date = pd.to_datetime(original_date, errors="coerce", format=date_format)
    record("Date", original_date, converted_date)
    out["Date"] = converted_date

    for column in NUMERIC_COLUMNS:
        if column in out.columns:
            original = out[column]
            converted = _to_numeric(original)
            record(column, original, converted)
            out[column] = converted

    for column in FLAG_COLUMNS:
        if column in out.columns:
            original = out[column]
            converted = _to_flag(original)
            record(column, original, converted)
            if converted.notna().all() and bool(converted.isin([0, 1]).all()):
                converted = converted.astype("int64")
            out[column] = converted

    for column in TEXT_COLUMNS:
        if column in out.columns:
            original = out[column]
            converted = _id_to_string(original)
            record(column, original, converted)
            out[column] = converted

    valid_dates = out["Date"].dropna()
    report: dict[str, Any] = {
        "invalid_values": invalid_values,
        "missing_values": {column: int(count) for column, count in out.isna().sum().items()
                           if count},
        "date_range": (valid_dates.min(), valid_dates.max()) if len(valid_dates) else None,
        "issues": issues,
    }
    return out, report


def _infer_group_keys(df: pd.DataFrame) -> list[str]:
    """Series grouping used for every lag/rolling feature: store + product when available."""
    keys = [column for column in ("Store_ID", "Product_ID") if column in df.columns]
    if "Product_ID" not in keys:
        raise KeyError("Product_ID is required to build per-series lag features.")
    return keys


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features from ``Date``: Year, Month, Day, DayOfWeek, WeekOfYear, Quarter, IsWeekend.

    Raises ``KeyError`` if ``Date`` is absent and ``ValueError`` if it holds
    missing/invalid values (rows must be cleaned before feature engineering).
    """
    if "Date" not in df.columns:
        raise KeyError("create_time_features() requires a 'Date' column.")
    if df["Date"].isna().any():
        n_bad = int(df["Date"].isna().sum())
        raise ValueError(
            f"Date contains {n_bad:,} missing/invalid value(s) - remove or fix these rows "
            "before feature engineering (see src.validator.validate_dataset())."
        )
    dt = df["Date"].dt
    df["Year"] = dt.year.astype("int64")
    df["Month"] = dt.month.astype("int64")
    df["Day"] = dt.day.astype("int64")
    df["DayOfWeek"] = dt.dayofweek.astype("int64")
    df["WeekOfYear"] = dt.isocalendar().week.astype("int64")
    df["Quarter"] = dt.quarter.astype("int64")
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype("int64")
    return df


def create_price_features(df: pd.DataFrame, group_keys: Sequence[str] | None = None) -> pd.DataFrame:
    """Price features; ``df`` must already be sorted chronologically per series.

    * ``Previous_Price`` - previous observation's price within the series (backward-looking only)
    * ``Price_Change`` / ``Price_Change_Pct`` - change versus that previous price
    * ``Effective_Price`` - ``Price * (1 - Discount/100)`` (no ``Discount`` column -> 0%)
    * ``Price_Difference`` / ``Price_Ratio`` - versus ``Competitor_Price`` (skipped when absent)
    """
    if "Price" not in df.columns:
        raise KeyError("create_price_features() requires a 'Price' column.")
    keys = list(group_keys) if group_keys else _infer_group_keys(df)

    grouped_price = df.groupby(keys, sort=False, dropna=False)["Price"]
    df["Previous_Price"] = grouped_price.shift(1)
    df["Price_Change"] = df["Price"] - df["Previous_Price"]
    df["Price_Change_Pct"] = (
        df["Price_Change"] / df["Previous_Price"].where(df["Previous_Price"] > 0) * 100.0
    )

    discount = df["Discount"].fillna(0) if "Discount" in df.columns else 0.0
    df["Effective_Price"] = df["Price"] * (1.0 - discount / 100.0)

    if "Competitor_Price" in df.columns:
        competitor = df["Competitor_Price"]
        df["Price_Difference"] = df["Price"] - competitor
        df["Price_Ratio"] = df["Price"] / competitor.where(competitor > 0)
    return df


def create_lag_features(
    df: pd.DataFrame,
    window: int = 7,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Lag and rolling demand features built strictly from earlier observations.

    * ``Previous_Demand`` = ``Demand.shift(1)`` per series
    * ``Previous_{window}Day_Avg_Demand`` = rolling mean of the **already shifted**
      series, so the current row's demand is never included
    * ``Demand_Trend`` = ``Previous_Demand - Previous_{window}Day_Avg_Demand``

    ``df`` must already be sorted chronologically per series.
    """
    if "Demand" not in df.columns:
        raise KeyError("create_lag_features() requires a 'Demand' column.")
    if window < 1:
        raise ValueError(f"window must be >= 1 (got {window})")
    keys = list(group_keys) if group_keys else _infer_group_keys(df)

    average_column = f"Previous_{window}Day_Avg_Demand"
    grouped = df.groupby(keys, sort=False, dropna=False)
    df["Previous_Demand"] = grouped["Demand"].shift(1)  # shift BEFORE any rolling window
    df[average_column] = (
        df.groupby(keys, sort=False, dropna=False)["Previous_Demand"]
        .transform(lambda series: series.rolling(window, min_periods=window).mean())
    )
    df["Demand_Trend"] = df["Previous_Demand"] - df[average_column]
    return df


def create_inventory_features(
    df: pd.DataFrame,
    window: int = 7,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Inventory features, created only when ``Inventory`` exists.

    * ``Inventory_Coverage_Days`` - stock divided by the *previous* window's
      average demand (uses past demand only)
    * ``Low_Inventory`` - 1 when stock is below the series' own earlier median
      inventory (expanding median of past observations, shifted by one)

    Returns the frame unchanged when ``Inventory`` is unavailable.
    """
    if "Inventory" not in df.columns:
        return df
    average_column = f"Previous_{window}Day_Avg_Demand"
    if average_column not in df.columns:
        raise KeyError(
            f"'{average_column}' is missing - run create_lag_features() before "
            "create_inventory_features()."
        )
    keys = list(group_keys) if group_keys else _infer_group_keys(df)

    recent_demand = df[average_column]
    df["Inventory_Coverage_Days"] = df["Inventory"] / recent_demand.where(recent_demand > 0)

    prior_median = (
        df.groupby(keys, sort=False, dropna=False)["Inventory"]
        .transform(lambda series: series.shift(1).expanding(min_periods=1).median())
    )
    low = (df["Inventory"] < prior_median).astype("float64")
    undefined = prior_median.isna() | df["Inventory"].isna()
    low[undefined] = np.nan
    df["Low_Inventory"] = low.astype("int64") if not bool(undefined.any()) else low.astype("Int64")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Promotion interactions, created only when their inputs exist.

    * ``Weekend_Promotion`` = ``IsWeekend * Promotion``
    * ``Promotion_Discount`` = ``Promotion * Discount``

    Missing optional inputs mean the related feature is skipped - no default
    business data is invented.
    """
    if "Promotion" not in df.columns:
        return df
    if "IsWeekend" in df.columns:
        df["Weekend_Promotion"] = df["IsWeekend"] * df["Promotion"]
    if "Discount" in df.columns:
        df["Promotion_Discount"] = df["Promotion"] * df["Discount"]
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    column_mapping: dict[str, str | None] | None = None,
    history_window: int = 7,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Run the full, leakage-safe feature engineering pipeline on a copy of ``df``.

    Steps: optional column mapping -> dtype preparation -> chronological
    sorting per series -> time / price / lag / inventory / interaction
    features. No rows are dropped and no values are invented.

    ``column_mapping`` maps ``{canonical_name: company_column_name}`` (see
    :func:`standardize_columns`). ``history_window`` controls the rolling
    demand average (feature name follows it, e.g. ``Previous_7Day_Avg_Demand``).

    The returned frame carries a report in ``df.attrs["feature_report"]`` with
    ``created`` features, ``skipped`` feature groups, ``group_keys`` and
    ``history_window``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"engineer_features() expects a DataFrame, got {type(df).__name__}")
    if history_window < 1:
        raise ValueError(f"history_window must be >= 1 (got {history_window})")

    work = df.copy()
    if column_mapping:
        work = standardize_columns(work, column_mapping)
    work, preparation_report = validate_and_prepare_data(work, date_format=date_format)

    if work["Date"].isna().any() or work["Product_ID"].isna().any():
        n_bad = int(work["Date"].isna().sum() + work["Product_ID"].isna().sum())
        raise ValueError(
            f"{n_bad:,} row(s) have a missing/invalid Date or Product_ID - remove or fix them "
            "before feature engineering (see src.validator.validate_dataset())."
        )

    group_keys = _infer_group_keys(work)
    work = work.sort_values(group_keys + ["Date"], kind="mergesort").reset_index(drop=True)
    input_columns = set(work.columns)

    work = create_time_features(work)
    work = create_price_features(work, group_keys=group_keys)
    work = create_lag_features(work, window=history_window, group_keys=group_keys)
    work = create_inventory_features(work, window=history_window, group_keys=group_keys)
    work = create_interaction_features(work)

    created = [column for column in work.columns if column not in input_columns]
    skipped = {
        f"Optional column {name}": f"{name} not provided - nothing derived from it."
        for name, kind in CANONICAL_SCHEMA.items()
        if kind == "optional" and name not in input_columns
    }
    if "Promotion" in input_columns and "Discount" not in input_columns:
        skipped["Promotion_Discount"] = "Promotion_Discount skipped because Discount is unavailable."

    work.attrs["feature_report"] = {
        "created": created,
        "skipped": skipped,
        "group_keys": group_keys,
        "history_window": history_window,
        "preparation": preparation_report,
    }
    return work
