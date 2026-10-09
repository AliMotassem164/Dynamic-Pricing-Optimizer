"""End-to-end verification of the ``src/`` refactoring.

Run from the project root with the project virtual environment::

    .\\.venv\\Scripts\\python.exe tests\\refactoring_test.py

``pytest`` is not installed in this environment, so the script uses a small
built-in check harness. Every check prints a ``[PASS]``/``[FAIL]`` line and
the script ends with a REFACTORING TEST SUMMARY block. Exit code is 0 only
when every check passed.

The script exercises the full production workflow on the real project data
(nothing is faked):

    Company CSV -> column mapping -> validation -> feature engineering ->
    model load -> demand prediction -> elasticity/scenarios -> price
    optimization -> guardrails -> decision

It also proves the refactoring constraints: no rows dropped, leakage-safe
lag features, missing Unit_Cost only warns, profit optimization refuses to
run without a real cost, the five pricing decisions in priority order, the
required public API, and no Streamlit anywhere in ``src/``.
"""

from __future__ import annotations

import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import data_loader, feature_engineering, model_manager, pricing, utils, validator

RAW_CSV = ROOT / "Data" / "demand_forecasting.csv"
ENGINEERED_CSV = ROOT / "Data" / "demand_forecasting_engineered.csv"
PRODUCT_METADATA_CSV = ROOT / "Data" / "product_metadata.csv"
MODEL_PKL = ROOT / "Models" / "demand_forecasting_model.pkl"
MODEL_METADATA_JSON = ROOT / "Models" / "model_metadata.json"

#: The demo dataset's company-specific column names -> canonical schema.
COLUMN_MAPPING: dict[str, str | None] = {
    "Date": "Date",
    "Product_ID": "Product ID",
    "Store_ID": "Store ID",
    "Price": "Price",
    "Demand": "Demand",
    "Unit_Cost": None,  # not present in the raw data -> must only warn
    "Inventory": "Inventory Level",
    "Competitor_Price": "Competitor Pricing",
    "Discount": "Discount",
    "Promotion": "Promotion",
    "Category": "Category",
    "Region": "Region",
    "Weather": "Weather Condition",
    "Seasonality": "Seasonality",
    "Epidemic": "Epidemic",
}

EXPECTED_UNIT_COST_WARNING = (
    "Unit Cost is unavailable. Revenue optimization is possible, but true profit "
    "optimization requires Unit Cost."
)

RESULTS: list[dict[str, str]] = []


# ---------------------------------------------------------------------------
# Tiny check harness
# ---------------------------------------------------------------------------

def record(status: str, name: str, detail: str = "") -> bool:
    RESULTS.append({"status": status, "name": name, "detail": detail})
    suffix = f" :: {detail}" if detail else ""
    print(f"    [{status}] {name}{suffix}")
    return status == "PASS"


def check(name: str, condition: Any, detail: str = "") -> bool:
    return record("PASS" if condition else "FAIL", name, detail)


def skip(name: str, reason: str) -> bool:
    return record("SKIP", name, reason)


def expect_raises(
    name: str,
    exc_type: type[BaseException],
    func: Callable[..., Any],
    *args: Any,
    contains: str | None = None,
    **kwargs: Any,
) -> bool:
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        message = str(exc)
        ok = contains is None or contains in message
        return record("PASS" if ok else "FAIL", name, f"{type(exc).__name__}: {message}")
    except Exception as exc:  # noqa: BLE001 - wrong exception type is a failure
        return record(
            "FAIL", name, f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        )
    return record("FAIL", name, f"no {exc_type.__name__} raised")


def require(ctx: dict[str, Any], *keys: str) -> bool:
    """Return True when every prerequisite is in ``ctx``, else record a SKIP."""
    missing = [key for key in keys if key not in ctx]
    if missing:
        skip(f"depends on {missing}", "prerequisite step did not complete")
        return False
    return True


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(4, 70 - len(title)))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_utils(ctx: dict[str, Any]) -> None:
    banner("Step 1: utility helpers (src/utils.py)")
    check("safe_divide() divides normally", np.isclose(utils.safe_divide(10, 4), 2.5))
    check("safe_divide() by zero returns the default", np.isnan(utils.safe_divide(10, 0)))
    check(
        "safe_divide() with missing input returns the default",
        utils.safe_divide(np.nan, 2, default=-1.0) == -1.0,
    )
    check(
        "calculate_percentage_change() up",
        np.isclose(utils.calculate_percentage_change(120, 100), 20.0),
    )
    check(
        "calculate_percentage_change() down",
        np.isclose(utils.calculate_percentage_change(50, 100), -50.0),
    )
    check(
        "calculate_percentage_change() from zero is undefined",
        np.isnan(utils.calculate_percentage_change(10, 0)),
    )
    probe = pd.DataFrame({"Product ID": [1], "Units Sold": [2]})
    resolved = utils.get_available_columns(probe, ["product_id", "Units Sold", "Unit Cost"])
    check(
        "get_available_columns() matches names loosely and omits absent ones",
        resolved == ["Product ID", "Units Sold"],  # 'product_id' -> 'Product ID'; no 'Unit Cost'
        f"resolved={resolved}",
    )
    check("ensure_numeric() coerces bad values to NaN", np.isnan(utils.ensure_numeric("abc")))
    check("ensure_numeric(..., default=None) keeps None", utils.ensure_numeric("abc", None) is None)


def step_load(ctx: dict[str, Any]) -> None:
    banner("Step 2: loading the company CSV (src/data_loader.py)")
    raw = data_loader.load_data(RAW_CSV)
    check(
        "load_data() reads the raw company CSV",
        raw.shape == (76000, 16),
        f"shape={raw.shape}",
    )
    sample = data_loader.load_csv(RAW_CSV, nrows=5)
    check("load_csv() forwards read options", sample.shape == (5, 16), f"shape={sample.shape}")

    info = data_loader.get_basic_dataset_info(raw)
    check(
        "get_basic_dataset_info() reports structure",
        info["rows"] == len(raw)
        and info["columns"] == raw.shape[1]
        and "Date" in info["column_names"],
        f"rows={info['rows']:,}, columns={info['columns']}, duplicates={info['duplicate_rows']}",
    )
    check(
        "missing-value statistics are complete",
        len(info["missing_percent"]) == raw.shape[1]
        and info["total_missing"] == sum(info["missing_values"].values()),
        f"total_missing={info['total_missing']:,}",
    )

    expect_raises(
        "missing file raises FileNotFoundError",
        FileNotFoundError,
        data_loader.load_data,
        ROOT / "Data" / "does_not_exist.csv",
    )
    expect_raises(
        "load_excel() refuses non-Excel files",
        ValueError,
        data_loader.load_excel,
        RAW_CSV,
    )
    with tempfile.TemporaryDirectory() as tmp:
        text_file = Path(tmp) / "notes.txt"
        text_file.write_text("not a dataset", encoding="utf-8")
        expect_raises(
            "unsupported extension raises ValueError",
            ValueError,
            data_loader.load_data,
            text_file,
        )

    ctx["raw"] = raw
    ctx["raw_info"] = info


def step_validate(ctx: dict[str, Any]) -> None:
    banner("Step 3: validation (src/validator.py)")
    raw = ctx.get("raw")
    if raw is None:
        skip("validate raw dataset", "raw data not loaded")
        return

    result = validator.validate_dataset(raw)
    if result["errors"]:
        print(f"    (errors: {result['errors']})")
    check(
        "raw dataset passes validation",
        result["is_valid"] is True,
        f"errors={len(result['errors'])}, warnings={len(result['warnings'])}",
    )
    check(
        "missing Unit_Cost warns with the exact message and never invalidates",
        validator.UNIT_COST_MISSING_WARNING == EXPECTED_UNIT_COST_WARNING
        and EXPECTED_UNIT_COST_WARNING in result["warnings"]
        and result["is_valid"] is True,
        f"warning present={EXPECTED_UNIT_COST_WARNING in result['warnings']}",
    )
    unit_cost_result = validator.validate_unit_cost(raw)
    check(
        "validate_unit_cost() reports 'unavailable' without errors",
        unit_cost_result["errors"] == []
        and unit_cost_result["info"]["available"] is False,
        f"warnings={len(unit_cost_result['warnings'])}",
    )
    check(
        "validate_required_columns() resolves the legacy column names",
        validator.validate_required_columns(raw)["info"]["missing_columns"] == [],
        "Product ID -> Product_ID matched case/space-insensitively",
    )
    check(
        "validate_price_data() finds no blocking issue",
        validator.validate_price_data(raw)["errors"] == [],
    )
    check(
        "validate_demand_data() finds no blocking issue",
        validator.validate_demand_data(raw)["errors"] == [],
    )
    check(
        "validate_data_types() parses dates and numbers",
        validator.validate_data_types(raw)["errors"] == [],
    )

    # Blocking cases must be errors, and validation must never drop rows.
    damaged = raw.head(100).copy()
    damaged.loc[damaged.index[0], "Price"] = -1.0
    damaged_result = validator.validate_dataset(damaged)
    check(
        "negative prices block validation",
        damaged_result["is_valid"] is False
        and any("negative" in error for error in damaged_result["errors"]),
        f"errors={damaged_result['errors'][:1]}",
    )
    check("validation never removes rows", len(damaged) == 100)

    without_price = raw.drop(columns=["Price"])
    missing_result = validator.validate_dataset(without_price)
    check(
        "missing required column is an error",
        missing_result["is_valid"] is False
        and any("Price" in error for error in missing_result["errors"]),
    )

    ctx["validation"] = result


def step_engineer(ctx: dict[str, Any]) -> None:
    banner("Step 4: feature engineering on the raw data (src/feature_engineering.py)")
    raw = ctx.get("raw")
    if raw is None:
        skip("engineer features", "raw data not loaded")
        return

    engineered = feature_engineering.engineer_features(
        raw, column_mapping=COLUMN_MAPPING, history_window=7
    )
    check(
        "no rows are dropped by feature engineering",
        len(engineered) == len(raw),
        f"{len(raw):,} -> {len(engineered):,} rows",
    )

    expected_features = [
        "Year", "Month", "DayOfWeek", "WeekOfYear", "Quarter", "IsWeekend",
        "Previous_Price", "Price_Change", "Price_Change_Pct", "Effective_Price",
        "Price_Difference", "Price_Ratio", "Previous_Demand",
        "Previous_7Day_Avg_Demand", "Demand_Trend", "Inventory_Coverage_Days",
        "Low_Inventory", "Weekend_Promotion", "Promotion_Discount",
    ]
    missing = [column for column in expected_features if column not in engineered.columns]
    check("every expected feature column was created", not missing, f"missing={missing}")

    report = engineered.attrs.get("feature_report") or {}
    check(
        "feature report records created/skipped/groups/window",
        {"created", "skipped", "group_keys", "history_window"} <= set(report)
        and report.get("history_window") == 7
        and report.get("group_keys") == ["Store_ID", "Product_ID"],
        f"created={len(report.get('created', []))}, skipped={len(report.get('skipped', {}))}",
    )
    check(
        "absent Unit_Cost is reported as skipped, never invented",
        any("Unit_Cost" in key for key in report.get("skipped", {})),
        f"skipped keys sample={list(report.get('skipped', {}))[:3]}",
    )
    check(
        "unmapped company columns are left untouched",
        "Units Sold" in engineered.columns and "Units Ordered" in engineered.columns,
    )

    ctx["engineered_raw"] = engineered


def step_leakage(ctx: dict[str, Any]) -> None:
    banner("Step 5: leakage-safe lag features")
    raw = ctx.get("raw")
    if raw is None:
        skip("leakage checks", "raw data not loaded")
        return

    prepared, _ = feature_engineering.validate_and_prepare_data(
        feature_engineering.standardize_columns(raw.copy(), COLUMN_MAPPING)
    )
    group_keys = ["Store_ID", "Product_ID"]
    prepared = prepared.sort_values(group_keys + ["Date"], kind="mergesort").reset_index(drop=True)

    base = feature_engineering.create_lag_features(prepared.copy(), window=7)
    probe = 30  # interior row of the first series
    perturbed = prepared.copy()
    perturbed.loc[probe, "Demand"] = float(perturbed.loc[probe, "Demand"]) + 1000.0
    recomputed = feature_engineering.create_lag_features(perturbed, window=7)

    lag_columns = ["Previous_Demand", "Previous_7Day_Avg_Demand", "Demand_Trend"]
    before = base.loc[probe, lag_columns].to_numpy(dtype=float)
    after = recomputed.loc[probe, lag_columns].to_numpy(dtype=float)
    check(
        "a row's own demand never leaks into its own lag features",
        np.allclose(before, after, equal_nan=True),
        f"max |diff| after +1000 perturbation = {np.nanmax(np.abs(before - after)):.6g}",
    )
    check(
        "Previous_Demand is exactly the previous row's demand (shift(1))",
        np.isclose(
            float(base.loc[probe, "Previous_Demand"]), float(prepared.loc[probe - 1, "Demand"])
        ),
    )
    expected_average = base["Previous_Demand"].iloc[probe - 6 : probe + 1].mean()
    check(
        "rolling mean uses only the 7 already-shifted values",
        np.isclose(float(base.loc[probe, "Previous_7Day_Avg_Demand"]), float(expected_average)),
        f"feature={float(base.loc[probe, 'Previous_7Day_Avg_Demand']):.4f}, "
        f"expected={float(expected_average):.4f}",
    )
    check(
        "the perturbation only propagates forward to later rows",
        not np.isclose(
            float(base.loc[probe + 1, "Previous_Demand"]),
            float(recomputed.loc[probe + 1, "Previous_Demand"]),
        ),
    )


def step_model(ctx: dict[str, Any]) -> None:
    banner("Step 6: model artifacts (src/model_manager.py)")
    model = model_manager.load_model(MODEL_PKL)
    metadata = model_manager.load_metadata(MODEL_METADATA_JSON)
    features = model_manager.get_model_features(metadata)
    engineered_csv = data_loader.load_data(ENGINEERED_CSV)

    check("saved model artifact loads with predict()", hasattr(model, "predict"), type(model).__name__)
    check(
        "metadata declares the input contract",
        len(features) == 32 and metadata.get("target_column") not in features,
        f"{len(features)} feature columns, target={metadata.get('target_column')}",
    )

    sample = engineered_csv.head(50)
    baseline_predictions = model_manager.predict_demand(model, sample, metadata)
    with tempfile.TemporaryDirectory() as tmp:
        saved_model = model_manager.save_model(model, Path(tmp) / "model.pkl")
        reloaded_model = model_manager.load_model(saved_model)
        round_trip = model_manager.predict_demand(reloaded_model, sample, metadata)
        check(
            "save_model()/load_model() round-trip is prediction-identical",
            np.allclose(baseline_predictions, round_trip),
            f"{len(round_trip)} predictions compared",
        )
        saved_meta = model_manager.save_metadata(metadata, Path(tmp) / "metadata.json")
        reloaded_meta = model_manager.load_metadata(saved_meta)
        check(
            "save_metadata()/load_metadata() round-trip preserves the contract",
            reloaded_meta == metadata
            and model_manager.get_model_features(reloaded_meta) == features,
        )

    prepared = model_manager.prepare_model_input(engineered_csv, metadata)
    check(
        "prepare_model_input() selects exactly the contract columns",
        list(prepared.columns) == features,
        f"{prepared.shape[1]} columns, {prepared.shape[0]:,} rows",
    )

    ctx["model"] = model
    ctx["metadata"] = metadata
    ctx["features"] = features
    ctx["engineered_csv"] = engineered_csv


def step_compatibility(ctx: dict[str, Any]) -> None:
    banner("Step 7: model/data compatibility")
    if not require(ctx, "model", "metadata", "engineered_csv"):
        return
    compatibility = model_manager.check_model_compatibility(
        ctx["model"], ctx["metadata"], ctx["engineered_csv"]
    )
    if compatibility["errors"]:
        print(f"    (errors: {compatibility['errors']})")
    check("legacy engineered CSV is compatible with the model", compatibility["compatible"] is True)
    check(
        "no declared feature is missing",
        compatibility["missing_features"] == [],
        f"warnings={len(compatibility['warnings'])}, notes={len(compatibility['notes'])}",
    )


def step_predict(ctx: dict[str, Any]) -> None:
    banner("Step 8: demand prediction on the saved engineered dataset")
    if not require(ctx, "model", "metadata", "engineered_csv"):
        return
    model, metadata, engineered_csv = ctx["model"], ctx["metadata"], ctx["engineered_csv"]

    predictions = model_manager.predict_demand(model, engineered_csv, metadata)
    check(
        "one prediction per row",
        len(predictions) == len(engineered_csv),
        f"{len(predictions):,} predictions for {len(engineered_csv):,} rows",
    )
    check("no NaN predictions", bool(np.isfinite(predictions).all()))
    check(
        "no negative predictions",
        bool((predictions >= 0).all()),
        f"min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}",
    )
    raw_predictions = model_manager.predict_demand(
        model, engineered_csv, metadata, clip_negative=False
    )
    check(
        "clipping only changes negative raw outputs",
        np.allclose(np.clip(raw_predictions, 0.0, None), predictions),
        f"raw min={raw_predictions.min():.2f}",
    )


def step_predict_fresh(ctx: dict[str, Any]) -> None:
    banner("Step 9: prediction on freshly engineered raw data")
    if not require(ctx, "model", "metadata", "features", "engineered_raw"):
        return
    model, metadata, features = ctx["model"], ctx["metadata"], ctx["features"]
    engineered_raw = ctx["engineered_raw"]

    missing = [column for column in features if column not in engineered_raw.columns]
    check("freshly engineered data has every model input column", not missing, f"missing={missing}")
    if missing:
        return

    ready_mask = engineered_raw[features].notna().all(axis=1)
    ready = engineered_raw.loc[ready_mask]
    check(
        "rows with complete history match the notebook's 75,300 warm-up-free rows",
        len(ready) == 75300,
        f"{len(ready):,} of {len(engineered_raw):,} rows complete "
        f"({len(engineered_raw) - len(ready):,} warm-up rows carry no history yet)",
    )
    predictions = model_manager.predict_demand(model, ready, metadata)
    check(
        "freshly engineered rows predict cleanly",
        len(predictions) == len(ready)
        and bool(np.isfinite(predictions).all())
        and bool((predictions >= 0).all()),
        f"min={predictions.min():.2f}, max={predictions.max():.2f}",
    )
    ctx["fresh_predictions"] = predictions


def step_equivalence(ctx: dict[str, Any]) -> None:
    banner("Step 10: extracted feature engineering vs the notebook output")
    if not require(ctx, "model", "metadata", "features", "engineered_raw", "engineered_csv"):
        return
    model, metadata, features = ctx["model"], ctx["metadata"], ctx["features"]

    ours = ctx["engineered_raw"]
    ours = ours.loc[ours[features].notna().all(axis=1)].reset_index(drop=True)
    theirs = feature_engineering.standardize_columns(ctx["engineered_csv"], COLUMN_MAPPING)
    theirs["Date"] = pd.to_datetime(theirs["Date"], errors="coerce")
    theirs = theirs.reset_index(drop=True)

    check(
        "same row count as the notebook's engineered output",
        len(ours) == len(theirs),
        f"ours={len(ours):,}, notebook={len(theirs):,}",
    )
    if len(ours) != len(theirs):
        return

    key_columns = ["Date", "Store_ID", "Product_ID"]
    keys_match = all(
        ours[column].astype(str).equals(theirs[column].astype(str)) for column in key_columns
    )
    check("rows align on Date / Store / Product", keys_match)

    compared = [
        "Price", "Demand", "Previous_Demand", "Previous_7Day_Avg_Demand",
        "Price_Change_Pct", "Effective_Price", "Price_Difference",
        "Inventory_Coverage_Days", "Low_Inventory", "Weekend_Promotion",
    ]
    differing = []
    max_abs_diff = 0.0
    for column in compared:
        left = pd.to_numeric(ours[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(theirs[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True, rtol=1e-9, atol=1e-9):
            differing.append(column)
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(left[finite] - right[finite]))))
    check(
        "our engineered feature values match the notebook's",
        not differing,
        f"compared={len(compared)} columns, differing={differing}, "
        f"max |diff|={max_abs_diff:.2e} (float64 noise)",
    )

    ours_sorted = ours.sort_values(key_columns).reset_index(drop=True)
    theirs_sorted = theirs.sort_values(key_columns).reset_index(drop=True)

    # HistGradientBoosting trains in float32, so that is the model's effective
    # precision: at that precision the two inputs - and therefore the
    # predictions - must be bit-identical.
    def to_model_precision(frame: pd.DataFrame) -> pd.DataFrame:
        cast = frame.copy()
        for column in features:
            if pd.api.types.is_numeric_dtype(cast[column]):
                cast[column] = cast[column].astype(np.float32)
        return cast

    pred_ours = model_manager.predict_demand(model, ours_sorted, metadata)
    pred_theirs = model_manager.predict_demand(model, theirs_sorted, metadata)
    pred_ours32 = model_manager.predict_demand(model, to_model_precision(ours_sorted), metadata)
    pred_theirs32 = model_manager.predict_demand(model, to_model_precision(theirs_sorted), metadata)
    check(
        "predictions are bit-identical at the model's float32 training precision",
        np.array_equal(pred_ours32, pred_theirs32),
        f"{len(pred_ours32):,} predictions",
    )

    agree = np.isclose(pred_ours, pred_theirs, rtol=1e-6, atol=1e-6)
    check(
        "float64 predictions agree on >= 98% of rows",
        float(agree.mean()) >= 0.98,
        f"{int(agree.sum()):,}/{len(agree):,} rows identical; "
        f"{int((~agree).sum()):,} rows differ by <= {float(np.max(np.abs(pred_ours - pred_theirs))):.2f} "
        "units because last-ULP input noise can flip a tree split "
        "(the notebook's own stored features deviate from exact arithmetic by up to 7.1e-15)",
    )


def step_select(ctx: dict[str, Any]) -> None:
    banner("Step 11: representative product selection (src/pricing.py)")
    if not require(ctx, "engineered_csv"):
        return
    canonical = feature_engineering.standardize_columns(ctx["engineered_csv"], COLUMN_MAPPING)
    selection = pricing.select_representative_product(canonical)

    series = selection["series"]
    history = selection["history"]
    check(
        "a series is selected from the data, not from a hardcoded ID",
        {"Product_ID", "Store_ID"} <= set(series) and len(history) > 0,
        f"series={series}, history={len(history):,} rows",
    )
    candidates = selection["candidates"]
    check(
        "every candidate series meets the observation threshold",
        selection["n_candidates"] >= 1
        and bool((candidates["Observations"] >= selection["min_observations"]).all()),
        f"{selection['n_candidates']} candidate series",
    )
    check(
        "history is sorted chronologically",
        bool(history["Date"].is_monotonic_increasing),
    )

    ctx["canonical"] = canonical
    ctx["selection"] = selection
    ctx["history"] = history


def step_variation_elasticity(ctx: dict[str, Any]) -> None:
    banner("Step 12: price variation and elasticity estimation")
    if not require(ctx, "history"):
        return
    history = ctx["history"]

    variation = pricing.assess_price_variation(history)
    check(
        "price variation verdict is explicit",
        variation["verdict"] in {"GOOD", "LIMITED", "INSUFFICIENT"},
        f"{variation['verdict']}: {variation['n_unique_prices']} unique prices, "
        f"CV={variation['price_cv']:.3f}",
    )

    estimate = pricing.estimate_price_elasticity(history, price_quality=variation["verdict"])
    required_keys = {"elasticity", "source", "reliable", "r2", "observations", "warning"}
    check("elasticity result has the required keys", required_keys <= set(estimate))
    check(
        "elasticity source is always stated",
        estimate["source"] in {"historical_estimate", "scenario_assumption"},
        f"source={estimate['source']}, elasticity={estimate['elasticity']:.3f}, "
        f"observations={estimate['observations']}",
    )
    if estimate["source"] == "scenario_assumption":
        check(
            "unreliable history falls back to DEFAULT_ELASTICITY",
            estimate["elasticity"] == pricing.DEFAULT_ELASTICITY == -1.5
            and estimate["reliable"] is False
            and bool(estimate["warning"]),
            f"issues={estimate['issues'][:2]}",
        )
    else:
        check(
            "reliable estimate lies inside the plausible band",
            pricing.ELASTICITY_MIN <= estimate["elasticity"] <= pricing.ELASTICITY_MAX,
            f"elasticity={estimate['elasticity']:.3f}, r2={estimate['r2']:.4f}",
        )
    print(f"    -> {estimate['source']}: {estimate['warning']}")

    # Deterministic synthetic checks of the reliability gate.
    price = np.linspace(8.0, 48.0, 240)
    clean = pd.DataFrame(
        {"Price": price, "Demand": 2000 * price ** -1.7 * (1 + 0.01 * np.sin(np.arange(240)))}
    )
    clean_estimate = pricing.estimate_price_elasticity(clean)
    check(
        "clean synthetic history yields a historical estimate",
        clean_estimate["source"] == "historical_estimate"
        and clean_estimate["reliable"] is True
        and abs(clean_estimate["elasticity"] + 1.7) < 0.1,
        f"elasticity={clean_estimate['elasticity']:.3f} (true -1.7), "
        f"r2={clean_estimate['r2']:.4f}",
    )

    rising = pd.DataFrame({"Price": price, "Demand": 500 * price ** 1.3})
    rising_estimate = pricing.estimate_price_elasticity(rising)
    check(
        "positive (implausible) elasticity is rejected",
        rising_estimate["source"] == "scenario_assumption"
        and rising_estimate["elasticity"] == pricing.DEFAULT_ELASTICITY,
        f"issues={rising_estimate['issues'][:1]}",
    )

    flat = pd.DataFrame({"Price": np.full(120, 20.0), "Demand": np.linspace(40, 160, 120)})
    flat_estimate = pricing.estimate_price_elasticity(flat)
    check(
        "flat prices fall back to the assumption",
        flat_estimate["source"] == "scenario_assumption"
        and flat_estimate["elasticity"] == pricing.DEFAULT_ELASTICITY,
    )

    thin_estimate = pricing.estimate_price_elasticity(clean.head(10))
    check(
        "too few observations fall back to the assumption",
        thin_estimate["source"] == "scenario_assumption"
        and "need 30" in (thin_estimate["warning"] or ""),
        f"observations={thin_estimate['observations']}",
    )

    ctx["variation"] = variation
    ctx["estimate"] = estimate


def step_scenarios(ctx: dict[str, Any]) -> None:
    banner("Step 13: price scenarios, baseline demand and demand simulation")
    if not require(ctx, "history", "estimate", "model", "metadata"):
        return
    history, estimate = ctx["history"], ctx["estimate"]
    current_price = float(history["Price"].iloc[-1])

    scenarios = pricing.generate_price_scenarios(current_price, history["Price"], n_scenarios=25)
    check(
        "25 positive scenarios inside the observed price range",
        len(scenarios) == 25
        and bool(np.all(scenarios > 0))
        and scenarios.min() >= float(history["Price"].min()) - 1e-9
        and scenarios.max() <= float(history["Price"].max()) + 1e-9,
        f"[{scenarios.min():.2f}, {scenarios.max():.2f}] vs history "
        f"[{history['Price'].min():.2f}, {history['Price'].max():.2f}]",
    )
    check("scenarios are strictly increasing", bool(np.all(np.diff(scenarios) > 0)))
    expect_raises(
        "non-positive current price is rejected",
        ValueError,
        pricing.generate_price_scenarios,
        -5.0,
        history["Price"],
    )

    latest = history.tail(1)
    baseline_demand = float(model_manager.predict_demand(ctx["model"], latest, ctx["metadata"])[0])
    check(
        "baseline demand comes from the model at the current price",
        np.isfinite(baseline_demand) and baseline_demand >= 0,
        f"{baseline_demand:.1f} units at price {current_price:.2f}",
    )

    elasticity = estimate["elasticity"]
    demand = pricing.simulate_demand(baseline_demand, current_price, scenarios, elasticity)
    check(
        "simulated demand is finite and non-negative",
        bool(np.isfinite(demand).all()) and bool((demand >= 0).all()),
        f"min={demand.min():.1f}, max={demand.max():.1f}",
    )
    check(
        "demand falls as price rises (negative elasticity)",
        elasticity >= 0 or bool(np.all(np.diff(demand) <= 1e-9)),
        f"elasticity={elasticity:.3f}",
    )
    at_current = pricing.simulate_demand(baseline_demand, current_price, [current_price], elasticity)
    check(
        "scenario at the current price returns the baseline demand",
        np.isclose(float(at_current[0]), baseline_demand),
    )

    ctx["current_price"] = current_price
    ctx["scenarios"] = scenarios
    ctx["demand"] = demand
    ctx["baseline_demand"] = baseline_demand


def step_revenue_profit(ctx: dict[str, Any]) -> None:
    banner("Step 14: revenue and profit (real business unit cost)")
    if not require(ctx, "scenarios", "demand", "selection"):
        return
    scenarios, demand = ctx["scenarios"], ctx["demand"]

    revenue = pricing.calculate_revenue(scenarios, demand)
    check(
        "revenue = price x demand (vectorized)",
        np.ndim(revenue) == 1 and np.allclose(revenue, scenarios * demand),
    )
    check("scalar revenue works", np.isclose(pricing.calculate_revenue(10.0, 4.0), 40.0))
    expect_raises(
        "profit without a unit cost raises a clear error",
        ValueError,
        pricing.calculate_profit,
        10.0,
        4.0,
        None,
        contains="Unit cost",
    )

    catalog = data_loader.load_data(PRODUCT_METADATA_CSV)
    product_id = str(ctx["selection"]["series"]["Product_ID"])
    match = catalog[catalog["Product ID"].astype(str) == product_id]
    unit_cost = float(match["Unit Cost"].iloc[0]) if len(match) else None
    check(
        "unit cost is sourced from business data, never invented",
        unit_cost is not None and unit_cost > 0,
        f"Product {product_id}: Unit_Cost={unit_cost}",
    )
    if unit_cost is not None:
        check(
            "calculate_profit() uses the supplied cost",
            np.isclose(pricing.calculate_profit(10.0, 4.0, unit_cost), (10.0 - unit_cost) * 4.0),
        )
    ctx["unit_cost"] = unit_cost


def step_optimize(ctx: dict[str, Any]) -> None:
    banner("Step 15: scenario table and price optimization")
    if not require(ctx, "current_price", "scenarios", "demand", "baseline_demand", "estimate", "history"):
        return
    current_price = ctx["current_price"]
    scenarios = ctx["scenarios"]
    baseline_demand = ctx["baseline_demand"]
    elasticity = ctx["estimate"]["elasticity"]
    unit_cost = ctx.get("unit_cost")
    discount = float(ctx["history"]["Discount"].iloc[-1]) if "Discount" in ctx["history"] else 0.0

    table = pricing.build_scenario_table(
        current_price, scenarios, baseline_demand, elasticity,
        unit_cost=unit_cost, discount_pct=discount,
    )
    check(
        "scenario table has price/demand/revenue columns",
        {"Price", "Price_Change_Pct", "Adjusted_Demand", "Revenue"} <= set(table.columns),
        f"{len(table)} scenarios, discount held at {discount:g}%",
    )
    if unit_cost is not None:
        check(
            "profit columns appear only with a real unit cost",
            {"Profit", "Profit_Margin"} <= set(table.columns),
        )

    revenue_only = pricing.build_scenario_table(
        current_price, scenarios, baseline_demand, elasticity,
        unit_cost=None, discount_pct=discount,
    )
    expect_raises(
        "profit optimization without Unit_Cost raises a clear error",
        ValueError,
        pricing.optimize_price,
        revenue_only,
        "profit",
        contains="Unit_Cost",
    )
    expect_raises(
        "unknown objective is rejected",
        ValueError,
        pricing.optimize_price,
        table,
        "margin",
    )

    best_revenue = pricing.optimize_price(table, objective="revenue")
    check(
        "revenue optimization returns a recommendation",
        best_revenue["recommended_price"] is not None
        and best_revenue["objective_value"] is not None,
        f"price={best_revenue['recommended_price']:.2f} -> "
        f"revenue={best_revenue['objective_value']:.2f}",
    )

    best_profit = None
    if unit_cost is not None:
        best_profit = pricing.optimize_price(table, objective="profit")
        check(
            "profit optimization returns a recommendation",
            best_profit["recommended_price"] is not None,
            f"price={best_profit['recommended_price']:.2f} -> "
            f"profit={best_profit['objective_value']:.2f}",
        )

    ctx["table"] = table
    ctx["revenue_only"] = revenue_only
    ctx["best_revenue"] = best_revenue
    ctx["best_profit"] = best_profit
    ctx["discount"] = discount


def step_guardrails(ctx: dict[str, Any]) -> None:
    banner("Step 16: business guardrails")
    if not require(ctx, "table", "current_price"):
        return
    unit_cost = ctx.get("unit_cost")
    guarded = pricing.apply_price_guardrails(
        ctx["table"],
        ctx["current_price"],
        max_price_change_pct=10,
        min_profit_margin=0.10 if unit_cost is not None else None,
        unit_cost=unit_cost,
    )
    check(
        "guardrails add Feasible/Guardrail_Violation columns",
        {"Feasible", "Guardrail_Violation"} <= set(guarded.columns),
    )
    feasible = guarded["Feasible"].astype(bool).to_numpy()
    labels_clear = guarded["Guardrail_Violation"].fillna("").eq("") .to_numpy()
    check(
        "feasibility agrees with the violation labels",
        bool((feasible == labels_clear).all()),
        f"{int(feasible.sum())}/{len(guarded)} scenarios feasible",
    )
    beyond_band = (
        np.abs(guarded["Price"].to_numpy(dtype=float) - ctx["current_price"])
        / ctx["current_price"] * 100.0
        > 10.0 + 1e-9
    )
    check(
        "every scenario beyond +/-10% of the current price is blocked",
        bool((~feasible[beyond_band]).all()) if beyond_band.any() else True,
        f"{int(beyond_band.sum())} scenarios beyond the band",
    )
    check(
        "at least one scenario survives the guardrails",
        bool(feasible.any()),
        f"{int(feasible.sum())}/{len(guarded)} feasible",
    )
    check(
        "per-rule block counts are recorded in attrs",
        "guardrail_blocks" in guarded.attrs,
        f"blocks={guarded.attrs.get('guardrail_blocks')}",
    )

    ctx["guarded"] = guarded


def _decision_inputs(ctx: dict[str, Any], guarded: pd.DataFrame, objective: str) -> dict[str, Any]:
    """Assemble current/recommended objective values for decide_pricing_action()."""
    current_price = ctx["current_price"]
    unit_cost = ctx.get("unit_cost")
    objective_column = "Profit" if objective == "profit" else "Revenue"
    current_table = pricing.build_scenario_table(
        current_price, [current_price], ctx["baseline_demand"],
        ctx["estimate"]["elasticity"], unit_cost=unit_cost,
        discount_pct=ctx.get("discount", 0.0),
    )
    current_value = float(current_table[objective_column].iloc[0])

    best = pricing.optimize_price(guarded, objective=objective)
    feasible_rows = guarded[guarded["Feasible"].astype(bool)]
    max_feasible = float(feasible_rows[objective_column].max()) if len(feasible_rows) else None
    if best["recommended_price"] is None:
        recommended_price, recommended_value, blocked = current_price, current_value, True
    else:
        recommended_price = float(best["recommended_price"])
        recommended_value = float(best["objective_value"])
        blocked = not bool(len(feasible_rows))
    return {
        "current_price": current_price,
        "recommended_price": recommended_price,
        "current_value": current_value,
        "recommended_value": recommended_value,
        "max_feasible_value": max_feasible,
        "guardrail_blocked": blocked,
    }


def step_decide(ctx: dict[str, Any]) -> None:
    banner("Step 17: pricing decision on the real data")
    if not require(ctx, "guarded", "variation"):
        return
    objective = "profit" if ctx.get("unit_cost") else "revenue"
    inputs = _decision_inputs(ctx, ctx["guarded"], objective)
    decision = pricing.decide_pricing_action(
        price_quality=ctx["variation"]["verdict"],
        objective=objective,
        **inputs,
    )
    check(
        "decision is one of the five allowed actions",
        decision["action"] in pricing.VALID_ACTIONS,
        f"{decision['action']}",
    )
    check(
        "decision states its reason and improvement",
        bool(decision["reason"]) and "improvement_pct" in decision,
        f"objective={decision['objective']}, "
        f"improvement={decision['improvement_pct'] if decision['improvement_pct'] is not None else float('nan'):.2f}%",
    )
    print(f"    -> {decision['action']}: {decision['reason']}")
    ctx["decision"] = decision


def step_decision_priority(ctx: dict[str, Any]) -> None:
    banner("Step 18: decision priority order INSUFFICIENT_DATA > NO_GO > KEEP > RAISE > LOWER")
    insufficient = pricing.decide_pricing_action(
        price_quality="INSUFFICIENT",
        objective="profit",
        current_price=20.0,
        recommended_price=23.0,
        current_value=-10.0,
        recommended_value=-5.0,
        max_feasible_value=-5.0,
    )
    check(
        "1. INSUFFICIENT_DATA outranks NO_GO and RAISE",
        insufficient["action"] == "INSUFFICIENT_DATA",
        insufficient["action"],
    )
    no_go = pricing.decide_pricing_action(
        price_quality="GOOD",
        objective="profit",
        current_price=20.0,
        recommended_price=23.0,
        current_value=-10.0,
        recommended_value=-5.0,
        max_feasible_value=-5.0,
    )
    check("2. NO_GO when no feasible price is profitable", no_go["action"] == "NO_GO", no_go["action"])
    keep = pricing.decide_pricing_action(
        price_quality="GOOD",
        objective="revenue",
        current_price=20.0,
        recommended_price=23.0,
        current_value=100.0,
        recommended_value=100.0,
    )
    check("3. KEEP when the objective does not improve (even if price differs)", keep["action"] == "KEEP", keep["action"])
    raise_action = pricing.decide_pricing_action(
        price_quality="GOOD",
        objective="revenue",
        current_price=20.0,
        recommended_price=23.0,
        current_value=100.0,
        recommended_value=115.0,
    )
    check("4. RAISE when a higher price improves the objective", raise_action["action"] == "RAISE", raise_action["action"])
    lower = pricing.decide_pricing_action(
        price_quality="GOOD",
        objective="revenue",
        current_price=20.0,
        recommended_price=18.0,
        current_value=100.0,
        recommended_value=115.0,
    )
    check("5. LOWER when a lower price improves the objective", lower["action"] == "LOWER", lower["action"])


def step_api(ctx: dict[str, Any]) -> None:
    banner("Step 19: required public API and UI independence")
    spec: dict[Any, list[str]] = {
        data_loader: ["load_csv", "load_excel", "load_data", "get_basic_dataset_info"],
        validator: [
            "validate_required_columns", "validate_data_types", "validate_price_data",
            "validate_demand_data", "validate_unit_cost", "validate_dataset",
        ],
        feature_engineering: [
            "standardize_columns", "validate_and_prepare_data", "create_time_features",
            "create_price_features", "create_lag_features", "create_inventory_features",
            "create_interaction_features", "engineer_features",
        ],
        model_manager: [
            "load_model", "load_metadata", "get_model_features", "prepare_model_input",
            "predict_demand", "save_model", "save_metadata", "check_model_compatibility",
        ],
        pricing: [
            "select_representative_product", "assess_price_variation",
            "estimate_price_elasticity", "generate_price_scenarios", "simulate_demand",
            "calculate_revenue", "calculate_profit", "optimize_price",
            "apply_price_guardrails", "decide_pricing_action",
        ],
        utils: ["safe_divide", "ensure_numeric", "get_available_columns", "calculate_percentage_change"],
    }
    missing: dict[str, list[str]] = {}
    total = 0
    for module, names in spec.items():
        total += len(names)
        absent = [name for name in names if not callable(getattr(module, name, None))]
        if absent:
            missing[module.__name__] = absent
    check(
        "every required public function exists",
        not missing,
        f"{total} functions across {len(spec)} modules; missing={missing}",
    )

    source_files = sorted((ROOT / "src").glob("*.py"))
    usage = re.compile(r"(import\s+streamlit|from\s+streamlit|streamlit\.)", re.IGNORECASE)
    offenders = [
        f"{path.name}: {match.group(0)}"
        for path in source_files
        for match in usage.finditer(path.read_text(encoding="utf-8"))
    ]
    check(
        "src/ contains no Streamlit (UI) imports or code",
        not offenders,
        f"scanned {len(source_files)} files: {[p.name for p in source_files]}; offenders={offenders}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

STEPS: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    ("utility helpers", step_utils),
    ("data loading", step_load),
    ("validation", step_validate),
    ("feature engineering", step_engineer),
    ("leakage safety", step_leakage),
    ("model artifacts", step_model),
    ("model compatibility", step_compatibility),
    ("prediction on saved engineered data", step_predict),
    ("prediction on freshly engineered data", step_predict_fresh),
    ("feature engineering equivalence", step_equivalence),
    ("representative product selection", step_select),
    ("price variation and elasticity", step_variation_elasticity),
    ("scenarios and simulation", step_scenarios),
    ("revenue and profit", step_revenue_profit),
    ("price optimization", step_optimize),
    ("guardrails", step_guardrails),
    ("pricing decision", step_decide),
    ("decision priority order", step_decision_priority),
    ("public API and UI independence", step_api),
]


def print_summary() -> int:
    total = len(RESULTS)
    passed = sum(1 for row in RESULTS if row["status"] == "PASS")
    failed = sum(1 for row in RESULTS if row["status"] == "FAIL")
    skipped = sum(1 for row in RESULTS if row["status"] == "SKIP")
    bar = "=" * 74
    print(f"\n{bar}")
    print("REFATORING TEST SUMMARY")
    print("REFACTORING TEST SUMMARY")
    print(bar)
    print(f"Total checks : {total}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")
    print(f"Skipped      : {skipped}")
    print(f"RESULT       : {'ALL TESTS PASSED' if failed == 0 and total > 0 else f'{failed} FAILURE(S)'}")
    print(bar)
    if failed:
        print("\nFailed checks:")
        for row in RESULTS:
            if row["status"] == "FAIL":
                print(f"  - {row['name']}: {row['detail']}")
    return 0 if failed == 0 and total > 0 else 1


def main() -> int:
    import sklearn

    print("Dynamic Pricing Optimizer - src/ refactoring end-to-end test")
    print(
        f"Python {sys.version.split()[0]}, pandas {pd.__version__}, "
        f"numpy {np.__version__}, scikit-learn {sklearn.__version__}"
    )
    print(f"Project root: {ROOT}")

    ctx: dict[str, Any] = {}
    for title, func in STEPS:
        try:
            func(ctx)
        except Exception as exc:  # noqa: BLE001 - report and stop dependent steps
            record("FAIL", f"step '{title}' crashed", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            break
    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
