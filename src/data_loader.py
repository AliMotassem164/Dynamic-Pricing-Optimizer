"""Reusable data loading helpers.

The loaders are dataset-agnostic: they detect the file type from the
extension, return a ``pandas.DataFrame`` and raise clear errors for
missing or unsupported files. Nothing is printed; callers decide how to
report progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "EXCEL_EXTENSIONS",
    "load_csv",
    "load_excel",
    "load_data",
    "get_basic_dataset_info",
]

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xlsb": "excel",
    ".xls": "excel",
    ".ods": "excel",
}

EXCEL_EXTENSIONS: tuple[str, ...] = tuple(
    extension for extension, kind in SUPPORTED_EXTENSIONS.items() if kind == "excel"
)


def _existing_file(path: str | Path) -> Path:
    """Return ``path`` as a Path, raising a clear error when it does not exist."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"Expected a file but got a directory: {file_path}")
    return file_path


def load_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    file_path = _existing_file(path)
    return pd.read_csv(file_path, **read_csv_kwargs)


def load_excel(path: str | Path, sheet_name: str | int = 0, **read_excel_kwargs: Any) -> pd.DataFrame:
    """Load one sheet of an Excel/ODS file into a DataFrame.

    Reading Excel files requires an optional pandas engine (for example
    ``openpyxl`` for ``.xlsx``). A missing engine raises an ``ImportError``
    with installation instructions.
    """
    file_path = _existing_file(path)
    suffix = file_path.suffix.lower()
    if suffix not in EXCEL_EXTENSIONS:
        raise ValueError(
            f"load_excel() expects an Excel file {sorted(EXCEL_EXTENSIONS)}, got '{suffix}'"
        )
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, **read_excel_kwargs)
    except ImportError as exc:
        raise ImportError(
            f"Reading '{suffix}' files requires an optional Excel engine "
            f"(for example: pip install openpyxl). Original error: {exc}"
        ) from exc


def load_data(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load a supported data file, detecting the format from its extension.

    Supported: ``.csv``, ``.xlsx``, ``.xlsm``, ``.xlsb``, ``.xls``, ``.ods``.
    Unsupported extensions raise ``ValueError``; missing files raise
    ``FileNotFoundError``.
    """
    file_path = _existing_file(path)
    suffix = file_path.suffix.lower()
    kind = SUPPORTED_EXTENSIONS.get(suffix)
    if kind is None:
        raise ValueError(
            f"Unsupported file type '{suffix}' for '{file_path.name}'. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if kind == "csv":
        return load_csv(file_path, **read_kwargs)
    return load_excel(file_path, **read_kwargs)


def get_basic_dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic structural information about a DataFrame.

    Keys: ``rows``, ``columns``, ``column_names``, ``dtypes``,
    ``missing_values`` (per column), ``missing_percent`` (per column),
    ``total_missing`` and ``duplicate_rows``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"get_basic_dataset_info() expects a DataFrame, got {type(df).__name__}")
    row_count = len(df)
    missing = df.isna().sum()
    return {
        "rows": int(row_count),
        "columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {str(column): int(count) for column, count in missing.items()},
        "missing_percent": {
            str(column): round(float(count) / max(row_count, 1) * 100.0, 4)
            for column, count in missing.items()
        },
        "total_missing": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
