"""Reusable pricing engine: product selection, elasticity, scenarios,
optimization, business guardrails and decision logic.

Extracted from ``Notebooks/Price_Optimization.ipynb``. Every function is
dataset-agnostic: no product ID, store ID or company value is hardcoded.

Honesty rules encoded in this module
------------------------------------
* The demand model is **predictive, not causal**. Price scenarios are built
  with an explicit, transparent elasticity formula - never by re-running the
  forecasting model on counterfactual prices.
* Historical (observational) elasticity is never trusted blindly. It must
  pass reliability checks (sign, magnitude, stability, fit, price variation);
  otherwise the configurable :data:`DEFAULT_ELASTICITY` assumption is used and
  the result is labelled ``source = "scenario_assumption"``.
* Profit optimization requires a business-provided ``Unit_Cost``. No cost is
  ever invented; without a cost only revenue optimization is available.
* A price is never recommended merely because it is the mathematical maximum -
  business guardrails and the decision rules in :func:`decide_pricing_action`
  decide whether a change is justified.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import calculate_percentage_change

__all__ = [
    "DEFAULT_ELASTICITY",
    "ELASTICITY_MIN",
    "ELASTICITY_MAX",
    "VALID_ACTIONS",
    "select_representative_product",
    "assess_price_variation",
    "estimate_price_elasticity",
    "generate_price_scenarios",
    "simulate_demand",
    "calculate_revenue",
    "calculate_profit",
    "build_scenario_table",
    "optimize_price",
    "apply_price_guardrails",
    "decide_pricing_action",
]

# ---------------------------------------------------------------------------
# Configuration defaults (assumptions are always explicit, never silent)
# ---------------------------------------------------------------------------

#: Used whenever the historical elasticity estimate is unreliable. This is a
#: configurable *assumption*, not an estimate and never a causal claim.
DEFAULT_ELASTICITY: float = -1.5

#: Plausible band for an elasticity used in scenario simulation.
ELASTICITY_MIN: float = -3.0
ELASTICITY_MAX: float = -0.5

#: Reliability thresholds for the historical log-log estimate.
ELASTICITY_R2_MIN: float = 0.05
MIN_ELASTICITY_OBS: int = 30
CONFIDENCE_Z: float = 1.96

#: Price-variation quality thresholds.
QUALITY_GOOD_MIN_UNIQUE_PRICES: int = 20
QUALITY_GOOD_MIN_CV: float = 0.05
QUALITY_LIMITED_MIN_UNIQUE_PRICES: int = 5
QUALITY_LIMITED_MIN_CV: float = 0.01

#: Default series-selection thresholds.
MIN_OBSERVATIONS: int = 50

VALID_ACTIONS: frozenset[str] = frozenset(
    {"KEEP", "RAISE", "LOWER", "NO_GO", "INSUFFICIENT_DATA"}
)

OBJECTIVE_LABELS = {
    "profit": "Profit maximization using the business-provided unit cost",
    "revenue": "Revenue-maximizing price - not profit-maximizing price",
}


def _to_python(value: Any) -> Any:
    """Convert numpy scalars to native Python types for clean result dictionaries."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_dataframe(df: Any, function_name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{function_name}() expects a pandas DataFrame, got {type(df).__name__}")


def _require_positive(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return number


# ---------------------------------------------------------------------------
# Product selection and data quality
# ---------------------------------------------------------------------------

def select_representative_product(
    df: pd.DataFrame,
    price_col: str = "Price",
    product_col: str = "Product_ID",
    store_col: str | None = "Store_ID",
    date_col: str | None = "Date",
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, Any]:
    """Select the product series whose history best supports price analysis.

    No product or store ID is hardcoded. A *series* is a ``Product_ID``
    (plus ``Store_ID`` when that column exists). Candidate series must have at
    least ``min_observations`` rows with a positive price; they are ranked by
    historical price variation (coefficient of variation), ties broken by
    observation count, because estimating any price response requires prices
    that actually moved.

    Returns a dict with ``series_keys``, ``series`` (the selected key
    values), ``history`` (that series' rows sorted by date), ``candidates``
    (ranked statistics for every qualifying series) and ``n_candidates``.
    """
    _require_dataframe(df, "select_representative_product")
    for column in (price_col, product_col):
        if column not in df.columns:
            raise KeyError(
                f"Column '{column}' not found - available columns: {list(df.columns)}"
            )

    keys = [product_col] + ([store_col] if store_col and store_col in df.columns else [])
    rows: list[dict[str, Any]] = []
    for key, sub in df.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        if any(pd.isna(value) for value in key):
            continue
        if len(sub) < min_observations:
            continue
        price = pd.to_numeric(sub[price_col], errors="coerce").dropna()
        price = price[price > 0]
        if price.empty:
            continue
        mean_price = float(price.mean())
        std_price = float(price.std()) if len(price) > 1 else float("nan")
        rows.append({
            **{name: _to_python(value) for name, value in zip(keys, key)},
            "Observations": int(len(sub)),
            "Unique_Prices": int(price.nunique()),
            "Price_Mean": mean_price,
            "Price_Std": std_price,
            "Price_CV": std_price / mean_price if mean_price else float("nan"),
            "Price_Min": float(price.min()),
            "Price_Max": float(price.max()),
        })

    if not rows:
        raise ValueError(
            f"No product series has at least {min_observations} observations with a "
            "positive price. Provide a longer history or lower min_observations."
        )

    candidates = (
        pd.DataFrame(rows)
        .sort_values(["Price_CV", "Observations"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
    best = candidates.iloc[0]
    series = {name: _to_python(best[name]) for name in keys}

    mask = np.ones(len(df), dtype=bool)
    for name in keys:
        mask &= (df[name] == series[name]).to_numpy()
    history = df.loc[mask]
    if history.empty:
        raise ValueError(f"Selected series {series} produced no rows - check the identifier columns.")
    if date_col and date_col in history.columns:
        history = history.sort_values(date_col, kind="mergesort")
    history = history.reset_index(drop=True)

    return {
        "series_keys": keys,
        "series": series,
        "history": history,
        "candidates": candidates,
        "n_candidates": int(len(candidates)),
        "min_observations": int(min_observations),
    }


def assess_price_variation(
    product_df: pd.DataFrame | pd.Series,
    price_col: str = "Price",
    good_min_unique: int = QUALITY_GOOD_MIN_UNIQUE_PRICES,
    good_min_cv: float = QUALITY_GOOD_MIN_CV,
    limited_min_unique: int = QUALITY_LIMITED_MIN_UNIQUE_PRICES,
    limited_min_cv: float = QUALITY_LIMITED_MIN_CV,
) -> dict[str, Any]:
    """Measure historical price variation and issue a quality verdict.

    Verdicts:

    * ``GOOD`` - enough unique prices and dispersion for a descriptive
      price-response estimate (still observational, never causal),
    * ``LIMITED`` - indicative scenario analysis only,
    * ``INSUFFICIENT`` - elasticity must not be estimated from this history.
    """
    if isinstance(product_df, pd.DataFrame):
        if price_col not in product_df.columns:
            raise KeyError(
                f"Column '{price_col}' not found - available columns: {list(product_df.columns)}"
            )
        series = product_df[price_col]
    elif isinstance(product_df, pd.Series):
        series = product_df
    else:
        raise TypeError(
            "assess_price_variation() expects a DataFrame or a price Series, got "
            f"{type(product_df).__name__}"
        )

    price = pd.to_numeric(series, errors="coerce").dropna()
    price = price[price > 0]
    n_unique = int(price.nunique())
    n_observations = int(len(price))
    mean = float(price.mean()) if n_observations else float("nan")
    std = float(price.std()) if n_observations > 1 else float("nan")
    cv = std / mean if n_observations > 1 and np.isfinite(mean) and mean != 0 else float("nan")

    if n_unique >= good_min_unique and np.isfinite(cv) and cv >= good_min_cv:
        verdict = "GOOD"
    elif n_unique >= limited_min_unique and np.isfinite(cv) and cv >= limited_min_cv:
        verdict = "LIMITED"
    else:
        verdict = "INSUFFICIENT"

    if n_observations:
        quantiles = {
            "q05": float(price.quantile(0.05)),
            "q25": float(price.quantile(0.25)),
            "q50": float(price.quantile(0.50)),
            "q75": float(price.quantile(0.75)),
            "q95": float(price.quantile(0.95)),
        }
    else:
        quantiles = {name: float("nan") for name in ("q05", "q25", "q50", "q75", "q95")}

    return {
        "verdict": verdict,
        "n_observations": n_observations,
        "n_unique_prices": n_unique,
        "price_mean": mean,
        "price_std": std,
        "price_cv": cv,
        "price_min": float(price.min()) if n_observations else float("nan"),
        "price_max": float(price.max()) if n_observations else float("nan"),
        "quantiles": quantiles,
        "thresholds": {
            "good_min_unique_prices": int(good_min_unique),
            "good_min_cv": float(good_min_cv),
            "limited_min_unique_prices": int(limited_min_unique),
            "limited_min_cv": float(limited_min_cv),
        },
    }


# ---------------------------------------------------------------------------
# Elasticity
# ---------------------------------------------------------------------------

def _assess_elasticity_reliability(
    estimate: dict[str, Any],
    price_quality: str,
    elasticity_min: float,
    elasticity_max: float,
    r2_min: float,
) -> list[str]:
    """Return the list of failed reliability checks (empty list = usable estimate)."""
    issues: list[str] = []
    if not estimate.get("available"):
        issues.append("estimate unavailable: " + str(estimate.get("reason", "not enough data")))
        return issues

    beta = estimate["beta"]
    if not np.isfinite(beta):
        issues.append("estimated elasticity is undefined (degenerate price/demand history)")
        return issues
    if price_quality == "INSUFFICIENT":
        issues.append("historical price variation is INSUFFICIENT")
    elif price_quality == "LIMITED":
        issues.append("historical price variation is LIMITED")
    if beta > 0:
        issues.append(
            "estimated elasticity is POSITIVE (implausible for a normal good - historical "
            "price and demand are confounded, so the sign cannot be used for pricing)"
        )
    elif beta > elasticity_max:
        issues.append(
            f"estimated elasticity {beta:+.3f} is weaker than the plausible band "
            f"(max {elasticity_max})"
        )
    if beta < elasticity_min:
        issues.append(
            f"estimated elasticity {beta:+.3f} is more extreme than the plausible band "
            f"(min {elasticity_min})"
        )
    std_error = estimate.get("std_error")
    if std_error is not None and np.isfinite(std_error):
        if estimate["ci_low"] <= 0 <= estimate["ci_high"]:
            issues.append(
                f"95% confidence interval [{estimate['ci_low']:+.3f}, "
                f"{estimate['ci_high']:+.3f}] includes 0 - the sign is unstable"
            )
    r2 = estimate.get("r2")
    if r2 is None or not np.isfinite(r2):
        issues.append("R-squared of the log-log fit is undefined")
    elif r2 < r2_min:
        issues.append(
            f"log-log fit explains only {r2:.1%} of demand variance (minimum {r2_min:.0%})"
        )
    return issues


def estimate_price_elasticity(
    product_df: pd.DataFrame,
    price_col: str = "Price",
    demand_col: str = "Demand",
    price_quality: str | None = None,
    default_elasticity: float = DEFAULT_ELASTICITY,
    min_observations: int = MIN_ELASTICITY_OBS,
    confidence_z: float = CONFIDENCE_Z,
    elasticity_min: float = ELASTICITY_MIN,
    elasticity_max: float = ELASTICITY_MAX,
    r2_min: float = ELASTICITY_R2_MIN,
) -> dict[str, Any]:
    """Estimate price elasticity with an auditable log-log OLS fit, with fallback.

    Fits ``log(Demand) = alpha + beta * log(Price)`` on positive observations
    and puts ``beta`` through explicit reliability checks (price variation,
    sign, magnitude, confidence interval, explanatory power).

    **The result is an observational association, never a causally identified
    elasticity.** When any check fails, the configurable
    ``default_elasticity`` assumption (:data:`DEFAULT_ELASTICITY` by default)
    is returned with ``source = "scenario_assumption"``.

    Returned keys: ``elasticity`` (the value to use downstream), ``source``
    (``"historical_estimate"`` or ``"scenario_assumption"``), ``reliable``,
    ``r2``, ``observations``, ``warning``, plus diagnostics
    (``raw_estimate``, ``std_error``, ``ci_low``, ``ci_high``, ``alpha``,
    ``price_quality``, ``issues``, ``default_elasticity``).
    """
    _require_dataframe(product_df, "estimate_price_elasticity")
    if price_col not in product_df.columns or demand_col not in product_df.columns:
        missing = [c for c in (price_col, demand_col) if c not in product_df.columns]
        raise KeyError(
            f"Column(s) {missing} not found - available columns: {list(product_df.columns)}"
        )
    if not (elasticity_min <= default_elasticity <= elasticity_max):
        raise ValueError(
            f"default_elasticity must lie in the plausible band "
            f"[{elasticity_min}, {elasticity_max}] (got {default_elasticity})"
        )

    if price_quality is None:
        price_quality = assess_price_variation(product_df, price_col=price_col)["verdict"]

    data = product_df[[price_col, demand_col]].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[(data[price_col] > 0) & (data[demand_col] > 0)]
    n_observations = int(len(data))

    result: dict[str, Any] = {
        "elasticity": float(default_elasticity),
        "source": "scenario_assumption",
        "reliable": False,
        "r2": None,
        "observations": n_observations,
        "warning": None,
        "raw_estimate": None,
        "alpha": None,
        "std_error": None,
        "ci_low": None,
        "ci_high": None,
        "price_quality": price_quality,
        "issues": [],
        "default_elasticity": float(default_elasticity),
    }

    if n_observations < min_observations:
        reason = (
            f"only {n_observations} positive {price_col}/{demand_col} observations "
            f"(need {min_observations})"
        )
        result["issues"] = _assess_elasticity_reliability(
            {"available": False, "reason": reason}, price_quality,
            elasticity_min, elasticity_max, r2_min,
        )
        result["warning"] = (
            f"{reason}; scenario assumption {default_elasticity} used instead "
            "(not causally estimated)."
        )
        return result

    x = np.log(data[price_col].to_numpy(dtype=float))
    y = np.log(data[demand_col].to_numpy(dtype=float))
    design = np.column_stack([np.ones(n_observations), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    sigma2 = ss_res / max(n_observations - 2, 1)
    sxx = float(np.sum((x - x.mean()) ** 2))
    std_error = float(np.sqrt(sigma2 / sxx)) if sxx > 0 else float("nan")
    beta = float(coefficients[1])

    estimate = {
        "available": True,
        "beta": beta,
        "std_error": std_error,
        "ci_low": beta - confidence_z * std_error,
        "ci_high": beta + confidence_z * std_error,
        "r2": float(r2),
    }
    issues = _assess_elasticity_reliability(
        estimate, price_quality, elasticity_min, elasticity_max, r2_min
    )

    result.update({
        "alpha": float(coefficients[0]),
        "raw_estimate": beta,
        "std_error": std_error,
        "ci_low": estimate["ci_low"],
        "ci_high": estimate["ci_high"],
        "r2": float(r2) if np.isfinite(r2) else None,
        "issues": issues,
        "reliable": not issues,
    })
    if not issues:
        result["elasticity"] = float(np.clip(beta, elasticity_min, elasticity_max))
        result["source"] = "historical_estimate"
        result["warning"] = (
            "Historical estimate is observational: promotions, seasonality, inventory and "
            "competition may confound the price-demand association. Validate any price change "
            "with a controlled experiment."
        )
    else:
        result["warning"] = (
            "Historical estimate rejected (" + "; ".join(issues)
            + f"); scenario assumption {default_elasticity} used instead "
            "(not causally estimated)."
        )
    return result


# ---------------------------------------------------------------------------
# Scenario generation and simulation
# ---------------------------------------------------------------------------

def generate_price_scenarios(
    current_price: float,
    historical_prices: Sequence[float] | pd.Series | np.ndarray,
    n_scenarios: int = 30,
    low_quantile: float = 0.05,
    high_quantile: float = 0.95,
) -> np.ndarray:
    """Generate evenly spaced candidate prices inside the historical price range.

    The search space spans the ``low_quantile``-to-``high_quantile`` window of
    the observed prices (falling back to the observed min/max when that window
    is degenerate), so scenarios never extrapolate far beyond prices the
    business has actually seen. Returns a float array of ``n_scenarios``
    positive prices.
    """
    _require_positive(current_price, "current_price")
    if not isinstance(n_scenarios, (int, np.integer)) or n_scenarios < 1:
        raise ValueError(f"n_scenarios must be a positive integer (got {n_scenarios!r})")
    if not (0.0 <= low_quantile < high_quantile <= 1.0):
        raise ValueError(
            f"Require 0 <= low_quantile < high_quantile <= 1 "
            f"(got low={low_quantile}, high={high_quantile})"
        )

    prices = pd.to_numeric(pd.Series(historical_prices), errors="coerce").dropna().to_numpy(dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size == 0:
        raise ValueError("historical_prices contains no finite numeric values.")

    low = float(np.quantile(prices, low_quantile))
    high = float(np.quantile(prices, high_quantile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(prices.min()), float(prices.max())
    if not np.isfinite(low) or not np.isfinite(high) or high < low or low <= 0:
        raise ValueError(
            "Cannot build a usable price scenario range from the historical prices "
            f"(range [{low}, {high}]). Check the price data."
        )
    return np.linspace(low, high, int(n_scenarios))


def simulate_demand(
    baseline_demand: float,
    current_price: float,
    scenario_prices: Sequence[float] | np.ndarray,
    elasticity: float,
) -> np.ndarray:
    """Constant-elasticity scenario demand.

    ``adjusted = baseline_demand * (scenario_price / current_price) ** elasticity``

    This is a transparent, auditable scenario rule - **not** a causal effect
    estimated by the forecasting model. The result is non-negative by
    construction (positive baseline, positive price ratio).
    """
    baseline = float(baseline_demand)
    if not np.isfinite(baseline) or baseline < 0:
        raise ValueError(f"baseline_demand must be finite and >= 0 (got {baseline_demand!r})")
    current = _require_positive(current_price, "current_price")
    elasticity = float(elasticity)
    if not np.isfinite(elasticity):
        raise ValueError(f"elasticity must be finite (got {elasticity!r})")

    prices = np.asarray(scenario_prices, dtype=float)
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("scenario_prices must all be finite and strictly positive.")
    ratio = prices / current
    demand = baseline * np.power(ratio, elasticity)
    return np.clip(demand, 0.0, None)


def calculate_revenue(price, demand):
    """Revenue = price x demand (scalar in -> scalar out, array in -> array out)."""
    price_array = np.asarray(price, dtype=float)
    demand_array = np.asarray(demand, dtype=float)
    revenue = price_array * demand_array
    if revenue.ndim == 0:
        return float(revenue)
    return revenue


def calculate_profit(price, demand, unit_cost: float | None):
    """Profit = (price - unit_cost) x demand.

    ``unit_cost`` must be supplied by the business. ``None`` (or a
    non-finite/negative cost) raises ``ValueError`` - a cost is never
    fabricated, because a profit number built on an invented cost would be
    worse than no profit number at all.
    """
    if unit_cost is None:
        raise ValueError(
            "Unit cost is required to calculate profit. True profit optimization needs a "
            "business-provided Unit_Cost; use revenue optimization until one is supplied."
        )
    try:
        cost = float(unit_cost)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unit_cost must be a number, got {unit_cost!r}") from exc
    if not np.isfinite(cost) or cost < 0:
        raise ValueError(f"unit_cost must be finite and >= 0 (got {unit_cost!r})")

    price_array = np.asarray(price, dtype=float)
    demand_array = np.asarray(demand, dtype=float)
    profit = (price_array - cost) * demand_array
    if profit.ndim == 0:
        return float(profit)
    return profit


def build_scenario_table(
    current_price: float,
    scenario_prices: Sequence[float] | np.ndarray,
    baseline_demand: float,
    elasticity: float,
    unit_cost: float | None = None,
    discount_pct: float = 0.0,
) -> pd.DataFrame:
    """Build the full scenario table: demand, revenue and (optionally) profit.

    Columns: ``Price``, ``Price_Change_Pct``, ``Price_Ratio``,
    ``Effective_Price``, ``Adjusted_Demand``, ``Revenue`` and, only when a
    real ``unit_cost`` is given, ``Profit`` / ``Profit_Margin``. The current
    discount is held constant across scenarios (changing it would be a
    separate lever and a separate study).
    """
    current = _require_positive(current_price, "current_price")
    discount = float(discount_pct)
    if not np.isfinite(discount) or not 0.0 <= discount <= 100.0:
        raise ValueError(f"discount_pct must be between 0 and 100 (got {discount_pct!r})")

    prices = np.asarray(scenario_prices, dtype=float)
    adjusted = simulate_demand(baseline_demand, current, prices, elasticity)

    table = pd.DataFrame({"Price": prices})
    table["Price_Change_Pct"] = (table["Price"] - current) / current * 100.0
    table["Price_Ratio"] = table["Price"] / current
    table["Effective_Price"] = table["Price"] * (1.0 - discount / 100.0)
    table["Adjusted_Demand"] = adjusted
    table["Revenue"] = calculate_revenue(table["Effective_Price"].to_numpy(), adjusted)

    if unit_cost is not None:
        table["Profit"] = calculate_profit(table["Effective_Price"].to_numpy(), adjusted, unit_cost)
        revenue = table["Revenue"].to_numpy(dtype=float)
        profit = table["Profit"].to_numpy(dtype=float)
        margin = np.full(len(table), np.nan)
        positive = np.isfinite(revenue) & (revenue > 0)
        margin[positive] = profit[positive] / revenue[positive]
        table["Profit_Margin"] = margin
    return table


# ---------------------------------------------------------------------------
# Optimization and guardrails
# ---------------------------------------------------------------------------

def optimize_price(scenarios: pd.DataFrame, objective: str = "profit") -> dict[str, Any]:
    """Pick the best scenario for ``objective`` (``"profit"`` or ``"revenue"``).

    ``scenarios`` must contain ``Price`` and the matching objective column
    (``Profit``/``Revenue``) - typically from :func:`build_scenario_table`.
    When a ``Feasible`` column exists (added by
    :func:`apply_price_guardrails`), only feasible scenarios compete.

    Optimizing profit without a ``Profit`` column raises a clear error: a
    profit optimization requires a business-provided ``Unit_Cost``; it is
    never invented here.

    Returns a dict with ``objective``, ``feasible``, ``n_scenarios``,
    ``n_feasible``, ``recommended_index``, ``recommended_price``,
    ``objective_value`` and ``recommended_row``.
    """
    _require_dataframe(scenarios, "optimize_price")
    objective_key = str(objective).lower()
    if objective_key not in {"profit", "revenue"}:
        raise ValueError(f"objective must be 'profit' or 'revenue', got {objective!r}")

    objective_column = "Profit" if objective_key == "profit" else "Revenue"
    if objective_column not in scenarios.columns:
        if objective_key == "profit":
            raise ValueError(
                "Profit optimization requires Unit_Cost: the scenarios contain no 'Profit' "
                "column. Rebuild the scenario table with a business-provided unit_cost, or "
                "optimize objective='revenue'."
            )
        raise ValueError(
            f"The scenarios contain no '{objective_column}' column - build them with "
            "build_scenario_table() first."
        )
    if "Price" not in scenarios.columns:
        raise KeyError("The scenarios must contain a 'Price' column.")

    feasible_mask = (
        scenarios["Feasible"].fillna(False).astype(bool).to_numpy()
        if "Feasible" in scenarios.columns
        else np.ones(len(scenarios), dtype=bool)
    )
    n_feasible = int(feasible_mask.sum())
    result = {
        "objective": objective_key,
        "feasible": n_feasible > 0,
        "n_scenarios": int(len(scenarios)),
        "n_feasible": n_feasible,
        "recommended_index": None,
        "recommended_price": None,
        "objective_value": None,
        "recommended_row": None,
    }
    if n_feasible == 0:
        return result

    feasible_rows = scenarios.loc[feasible_mask]
    values = pd.to_numeric(feasible_rows[objective_column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).any():
        raise ValueError(
            f"The '{objective_column}' column contains no finite values for feasible scenarios."
        )
    best_index = int(values.idxmax())
    best_row = scenarios.loc[best_index]
    result.update({
        "recommended_index": best_index,
        "recommended_price": float(best_row["Price"]),
        "objective_value": float(best_row[objective_column]),
        "recommended_row": {key: _to_python(value) for key, value in best_row.to_dict().items()},
    })
    return result


def apply_price_guardrails(
    scenarios: pd.DataFrame,
    current_price: float,
    max_price_change_pct: float = 10,
    min_profit_margin: float | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_expected_demand: float | None = None,
    unit_cost: float | None = None,
) -> pd.DataFrame:
    """Mark each scenario as feasible or blocked by a business guardrail.

    Guardrails:

    * ``max_price_change_pct`` - a scenario may not move the price further
      than this percentage from the current price (default 10%),
    * ``min_price`` / ``max_price`` - absolute price bounds (default: the
      scenario range itself),
    * ``min_profit_margin`` - minimum margin (``profit / revenue``); requires
      ``unit_cost`` or ``Profit``/``Revenue`` columns and raises otherwise
      rather than silently skipping the rule,
    * ``min_expected_demand`` - optional absolute demand floor.

    Returns a copy of ``scenarios`` with ``Feasible`` and
    ``Guardrail_Violation`` columns; per-rule blocked counts are stored in
    ``result.attrs["guardrail_blocks"]``.
    """
    _require_dataframe(scenarios, "apply_price_guardrails")
    current = _require_positive(current_price, "current_price")
    if "Price" not in scenarios.columns:
        raise KeyError("The scenarios must contain a 'Price' column.")
    change_limit = float(max_price_change_pct)
    if not np.isfinite(change_limit) or change_limit < 0:
        raise ValueError(f"max_price_change_pct must be >= 0 (got {max_price_change_pct!r})")

    out = scenarios.copy()
    price = out["Price"].to_numpy(dtype=float)
    lower = float(min_price) if min_price is not None else float(price.min())
    upper = float(max_price) if max_price is not None else float(price.max())
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower <= 0 or upper < lower:
        raise ValueError(f"Invalid price bounds [{lower}, {upper}].")

    within_change = np.abs(price - current) / current * 100.0 <= change_limit + 1e-9
    within_bounds = (price >= lower - 1e-9) & (price <= upper + 1e-9)

    if min_expected_demand is not None:
        if "Adjusted_Demand" not in out.columns:
            raise KeyError(
                "A minimum expected demand guardrail requires an 'Adjusted_Demand' column."
            )
        demand_floor = float(min_expected_demand)
        demand_ok = out["Adjusted_Demand"].to_numpy(dtype=float) >= demand_floor
    else:
        demand_ok = np.ones(len(out), dtype=bool)

    if min_profit_margin is not None:
        margin_floor = float(min_profit_margin)
        if unit_cost is not None and "Profit" in out.columns:
            effective = (
                out["Effective_Price"].to_numpy(dtype=float)
                if "Effective_Price" in out.columns
                else price
            )
            margin = np.where(effective > 0, (effective - float(unit_cost)) / effective, -np.inf)
        elif "Profit" in out.columns and "Revenue" in out.columns:
            revenue = out["Revenue"].to_numpy(dtype=float)
            profit = out["Profit"].to_numpy(dtype=float)
            margin = np.where(revenue > 0, profit / revenue, -np.inf)
        else:
            raise ValueError(
                "A minimum profit margin guardrail requires a unit_cost (or 'Profit'/'Revenue' "
                "columns) - a margin cannot be checked without real cost information."
            )
        margin_ok = margin >= margin_floor
    else:
        margin_ok = np.ones(len(out), dtype=bool)

    margin_label = (
        f"margin below {float(min_profit_margin):.0%}"
        if min_profit_margin is not None
        else "margin below the configured minimum"
    )
    out["Guardrail_Violation"] = np.select(
        [
            ~within_change,
            ~within_bounds,
            ~margin_ok,
            ~demand_ok,
        ],
        [
            f"price change beyond +/-{change_limit:.1f}%",
            "outside the configured min/max price",
            margin_label,
            "below the minimum expected demand",
        ],
        default="",
    )
    out["Feasible"] = within_change & within_bounds & margin_ok & demand_ok

    out.attrs["guardrail_blocks"] = {
        "max_price_change_pct": int((~within_change).sum()),
        "price_bounds": int((~within_bounds).sum()),
        "min_profit_margin": int((~margin_ok).sum()),
        "min_expected_demand": int((~demand_ok).sum()),
    }
    return out


# ---------------------------------------------------------------------------
# Business decision logic
# ---------------------------------------------------------------------------

def decide_pricing_action(
    price_quality: str,
    objective: str,
    current_price: float,
    recommended_price: float,
    current_value: float,
    recommended_value: float,
    max_feasible_value: float | None = None,
    min_improvement_pct: float = 0.0,
    guardrail_blocked: bool = False,
) -> dict[str, Any]:
    """Map an optimization result to one of five explicit business actions.

    Evaluated in order:

    1. insufficient price variation -> ``INSUFFICIENT_DATA`` (scenario
       analysis only, hold the price),
    2. profit mode and the best feasible scenario is not profitable
       (``max_feasible_value <= 0``) -> ``NO_GO``,
    3. the recommendation does not improve the objective over the current
       price (by more than ``min_improvement_pct``) -> ``KEEP``,
    4. recommended price above the current price -> ``RAISE``,
    5. recommended price below the current price -> ``LOWER``.

    ``current_value`` / ``recommended_value`` are the objective amounts
    (expected profit or expected revenue) at those prices. The returned dict
    contains ``action``, ``reason``, ``objective``, ``basis_label``,
    ``improvement_abs`` and ``improvement_pct``. No causal claim is made: a
    recommendation is always conditional on the stated elasticity assumption
    and guardrails.
    """
    objective_key = str(objective).lower()
    if objective_key not in {"profit", "revenue"}:
        raise ValueError(f"objective must be 'profit' or 'revenue', got {objective!r}")
    current = _require_positive(current_price, "current_price")
    recommended = _require_positive(recommended_price, "recommended_price")

    improvement_abs = float(recommended_value) - float(current_value)
    improvement_pct = calculate_percentage_change(recommended_value, current_value)
    quality = str(price_quality).upper()

    result: dict[str, Any] = {
        "action": None,
        "reason": "",
        "objective": objective_key,
        "basis_label": OBJECTIVE_LABELS[objective_key],
        "improvement_abs": float(improvement_abs),
        "improvement_pct": float(improvement_pct) if np.isfinite(improvement_pct) else None,
    }

    # 1) The data cannot support a price-response estimate.
    if quality == "INSUFFICIENT":
        result["action"] = "INSUFFICIENT_DATA"
        result["reason"] = (
            "Historical price variation is insufficient for reliable price-response "
            "estimation, so the output is scenario analysis only and the current price "
            "is held."
        )
        return result

    # 2) No tested price generates a positive profit.
    if objective_key == "profit" and max_feasible_value is not None and max_feasible_value <= 0:
        result["action"] = "NO_GO"
        result["reason"] = (
            "No price within the tested historical range is expected to generate positive "
            "profit under the current assumptions."
        )
        return result

    # 3) Changing the price buys nothing.
    improves = improvement_abs > 0
    if np.isfinite(improvement_pct) and improvement_pct <= float(min_improvement_pct):
        improves = False
    if not improves:
        result["action"] = "KEEP"
        result["reason"] = (
            f"The best feasible scenario does not improve expected {objective_key} over "
            "the current price."
        )
        if guardrail_blocked:
            result["reason"] += " No candidate price satisfied the configured guardrails."
        return result

    # 4)/5) Direction of the change.
    if recommended > current:
        result["action"] = "RAISE"
        result["reason"] = (
            f"The best feasible scenario sets a higher price than the current one "
            f"({(recommended / current - 1) * 100:+.2f}%), improving expected "
            f"{objective_key} under the stated assumptions."
        )
    elif recommended < current:
        result["action"] = "LOWER"
        result["reason"] = (
            f"The best feasible scenario sets a lower price than the current one "
            f"({(recommended / current - 1) * 100:+.2f}%), improving expected "
            f"{objective_key} under the stated assumptions."
        )
    else:
        result["action"] = "KEEP"
        result["reason"] = (
            f"The recommended price equals the current price, so expected "
            f"{objective_key} is unchanged."
        )
    return result
