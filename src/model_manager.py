"""Model artifact management: loading, input preparation, prediction and saving.

The saved scikit-learn pipeline already contains its own preprocessing, so
this module never re-implements encoding, scaling or imputation. The input
contract comes exclusively from ``model_metadata.json`` (``feature_columns``),
which keeps the app generic: a different model for a different company
requires no code changes here.

Typical usage::

    model = load_model("Models/demand_forecasting_model.pkl")
    metadata = load_metadata("Models/model_metadata.json")
    compatibility = check_model_compatibility(model, metadata, df)
    predictions = predict_demand(model, df, metadata)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

__all__ = [
    "load_model",
    "load_metadata",
    "save_model",
    "save_metadata",
    "get_model_features",
    "apply_column_renames",
    "prepare_model_input",
    "predict_demand",
    "check_model_compatibility",
]


def _existing_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {file_path}")
    return file_path


def load_model(path: str | Path) -> Any:
    """Load a saved model/pipeline artifact with joblib.

    Raises ``FileNotFoundError`` when the file is absent and ``ValueError``
    when the loaded object cannot predict.
    """
    file_path = _existing_file(path)
    model = joblib.load(file_path)
    if not hasattr(model, "predict"):
        raise ValueError(
            f"Object loaded from '{file_path}' ({type(model).__name__}) has no predict() "
            "method - it is not a usable model artifact."
        )
    return model


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load the model metadata JSON (the model's input contract)."""
    file_path = _existing_file(path)
    try:
        with open(file_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model metadata file '{file_path}': {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Model metadata '{file_path}' must be a JSON object, got {type(metadata).__name__}"
        )
    return metadata


def save_model(model: Any, path: str | Path) -> Path:
    """Persist a model/pipeline artifact with joblib; returns the written path."""
    if not hasattr(model, "predict"):
        raise ValueError(
            f"Refusing to save {type(model).__name__}: it has no predict() method."
        )
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    return file_path


def save_metadata(metadata: dict[str, Any], path: str | Path) -> Path:
    """Persist model metadata as pretty-printed JSON; returns the written path."""
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_default)
    return file_path


def get_model_features(metadata: dict[str, Any]) -> list[str]:
    """Return the model's expected input feature columns from its metadata."""
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")
    features = metadata.get("feature_columns")
    if not features or not isinstance(features, list):
        raise ValueError(
            "Model metadata does not declare 'feature_columns' - the input contract is "
            "unknown, so the model cannot be fed safely."
        )
    features = [str(column) for column in features]
    target = metadata.get("target_column")
    if target and str(target) in features:
        raise ValueError(
            f"Invalid input contract: the target column '{target}' is listed among "
            "feature_columns - the model would be trained on its own target."
        )
    return features


def apply_column_renames(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Re-apply the canonical renames recorded at training time, if needed.

    ``metadata["column_renames_applied"]`` maps original column names to the
    canonical names the model was trained with (e.g. ``"Store ID" ->
    "Store_ID"``). Renames are applied only when the old name is present and
    the new name is not already in the frame.
    """
    renames = {
        old: new
        for old, new in (metadata.get("column_renames_applied") or {}).items()
        if old in df.columns and new not in df.columns
    }
    return df.rename(columns=renames) if renames else df


def prepare_model_input(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Select exactly the columns of the saved pipeline's input contract.

    Canonical training renames are applied first (see
    :func:`apply_column_renames`). Raises ``KeyError`` listing every missing
    feature when the frame does not contain the full contract.
    """
    features = get_model_features(metadata)
    prepared = apply_column_renames(df, metadata)
    missing = [column for column in features if column not in prepared.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} model input feature column(s) missing from the data: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return prepared[features].copy()


def predict_demand(
    model: Any,
    df: pd.DataFrame,
    metadata: dict[str, Any],
    clip_negative: bool = True,
) -> np.ndarray:
    """Predict demand for the rows of ``df`` using the saved pipeline contract.

    Predictions are returned as a flat ``float64`` array. By default negative
    predictions are clipped at 0 (demand cannot be negative), as recorded in
    the metadata notes; pass ``clip_negative=False`` to inspect raw output.
    """
    if not hasattr(model, "predict"):
        raise ValueError(f"{type(model).__name__} has no predict() method.")
    model_input = prepare_model_input(df, metadata)
    predictions = np.asarray(model.predict(model_input), dtype=float).ravel()
    if len(predictions) != len(model_input):
        raise ValueError(
            f"Model returned {len(predictions)} prediction(s) for {len(model_input)} row(s)."
        )
    return np.clip(predictions, 0.0, None) if clip_negative else predictions


def _dtype_kind_matches(expected: str, actual: str) -> bool:
    """True when a training-time dtype and a current dtype are compatible."""
    expected, actual = str(expected).lower(), str(actual).lower()
    text_kinds = {"object", "str", "string", "category"}
    if expected in text_kinds and actual in text_kinds:
        return True
    numeric_prefixes = ("int", "uint", "float")
    if expected.startswith(numeric_prefixes) and actual.startswith(numeric_prefixes + ("boolean",)):
        return True
    if expected == "bool" and actual.startswith(("bool", "int", "uint")):
        return True
    return expected == actual


def check_model_compatibility(
    model: Any,
    metadata: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Check that a model, its metadata and a dataset agree with each other.

    Returns ``{"compatible", "errors", "warnings", "notes", "missing_features",
    "feature_columns"}``. ``compatible`` is ``True`` only when no errors were
    found; warnings/notes flag softer issues (dtype drift, library version
    differences, target column absent from ``df``).
    """
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    missing_features: list[str] = []

    if model is None or not hasattr(model, "predict"):
        errors.append("The model object has no predict() method.")

    features: list[str] = []
    try:
        features = get_model_features(metadata)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    prepared = df
    if features:
        prepared = apply_column_renames(df, metadata)
        missing_features = [column for column in features if column not in prepared.columns]
        if missing_features:
            errors.append(
                f"{len(missing_features)} declared feature column(s) missing from the data: "
                + ", ".join(missing_features)
            )

        pipeline_features_attr = getattr(model, "feature_names_in_", None)
        pipeline_features = (
            [str(column) for column in pipeline_features_attr]
            if pipeline_features_attr is not None
            else []
        )
        if pipeline_features:
            if set(pipeline_features) != set(features):
                errors.append(
                    "Metadata feature_columns do not match the pipeline's feature_names_in_ "
                    f"(metadata: {len(features)}, pipeline: {len(pipeline_features)})."
                )
            elif pipeline_features != features:
                warnings.append(
                    "Metadata feature order differs from the pipeline's training order "
                    "(prediction still works; order is normalized when predicting)."
                )

        expected_dtypes = metadata.get("feature_input_dtypes") or {}
        mismatched = [
            f"{column} (training {expected_dtypes[column]}, now {prepared[column].dtype})"
            for column in features
            if column in expected_dtypes
            and column in prepared.columns
            and not _dtype_kind_matches(expected_dtypes[column], prepared[column].dtype)
        ]
        if mismatched:
            warnings.append(
                f"{len(mismatched)} feature dtype(s) differ from training: " + "; ".join(mismatched)
            )

    target = metadata.get("target_column")
    if target and target not in prepared.columns:
        notes.append(
            f"Target column '{target}' is not in the data - fine for prediction, required "
            "only for evaluation."
        )

    trained_with = (metadata.get("library_versions") or {}).get("scikit_learn")
    if trained_with:
        import sklearn

        if trained_with != sklearn.__version__:
            warnings.append(
                f"Model was trained with scikit-learn {trained_with}; current version is "
                f"{sklearn.__version__}."
            )

    pipeline_file = metadata.get("pipeline_file")
    if pipeline_file:
        notes.append(f"Metadata pipeline_file: {pipeline_file}")
    if metadata.get("model_name"):
        notes.append(
            f"Model: {metadata.get('model_name')} (version {metadata.get('model_version')})"
        )

    return {
        "compatible": not errors,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "missing_features": missing_features,
        "feature_columns": features,
    }
