"""
Dynamic Pricing Optimizer
=========================
A generic, decision-support Streamlit application for turning historical
pricing data into transparent pricing scenarios.

The application is intentionally thin: it owns the user interface, the
session state and the workflow. Every business or machine-learning rule lives
in the reusable ``src`` package:

    src/data_loader.py        - loading CSV/Excel datasets + basic structure
    src/validator.py          - schema / type / price / demand validation
    src/feature_engineering.py- canonical schema + leakage-safe feature build
    src/model_manager.py      - saved model contract, compatibility, prediction
    src/pricing.py            - elasticity, scenarios, revenue/profit, guardrails
    src/utils.py              - small shared helpers

Run from the project root:
    python -m streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import get_basic_dataset_info, load_data
from src.feature_engineering import (
    CANONICAL_SCHEMA,
    engineer_features,
    standardize_columns,
    validate_and_prepare_data,
)
from src.model_manager import (
    check_model_compatibility,
    load_metadata,
    load_model,
    predict_demand,
)
from src.pricing import (
    DEFAULT_ELASTICITY,
    ELASTICITY_MAX,
    ELASTICITY_MIN,
    apply_price_guardrails,
    assess_price_variation,
    build_scenario_table,
    calculate_profit,
    calculate_revenue,
    decide_pricing_action,
    estimate_price_elasticity,
    generate_price_scenarios,
    optimize_price,
)
from src.utils import calculate_percentage_change
from src.validator import (
    CANONICAL_REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    RECOMMENDED_COLUMNS,
    validate_dataset,
)

# ===========================================================================
# Configuration
# ===========================================================================

st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DEMO_RAW_PATH = BASE_DIR / "Data" / "demand_forecasting.csv"
MODEL_PATH = BASE_DIR / "Models" / "demand_forecasting_model.pkl"
METADATA_PATH = BASE_DIR / "Models" / "model_metadata.json"

N_SCENARIOS = 30
HISTORY_WINDOW = 7
BASELINE_RECENT_OBS = 14
SENSITIVITY_ELASTICITIES = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]

SUPPORTED_UPLOAD_TYPES = ["csv", "xlsx", "xls", "xlsm", "xlsb", "ods"]

# Restrained palette: blue = action, green = positive, amber = warning,
# red = error / NO_GO.
INK = "#0B1220"
SURFACE = "#111B2E"
SURFACE_2 = "#0F1728"
BORDER = "#22324F"
TEXT = "#E6ECF5"
MUTED = "#8C9BB5"
BLUE = "#5B9BF5"
GREEN = "#3FC48A"
AMBER = "#E3B25A"
RED = "#E5484D"

STAGES: list[tuple[str, str]] = [
    ("upload", "Upload Data"),
    ("overview", "Dataset Overview"),
    ("mapping", "Map Columns"),
    ("validation", "Validate Data"),
    ("prepare", "Prepare & Model"),
    ("pricing", "Pricing Analysis"),
    ("recommendation", "Recommendation"),
]
STAGE_LABELS = dict(STAGES)

CSS = """
<style>
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.stApp { background: #0B1220; color: #E6ECF5; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1120px; }

.app-title { font-size: 2.1rem; font-weight: 650; letter-spacing: -0.02em; color: #E6ECF5; margin: 0; }
.app-tagline { color: #8C9BB5; font-size: 1.02rem; margin-top: 0.3rem; }

.section { font-size: 1.15rem; font-weight: 600; color: #E6ECF5; margin: 2.2rem 0 0.2rem 0; }
.section-note { color: #8C9BB5; font-size: 0.85rem; margin-bottom: 0.6rem; }
.note { color: #8C9BB5; font-size: 0.82rem; line-height: 1.5; margin-top: 0.35rem; }

.stage-pill { display: inline-block; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: #5B9BF5; border: 1px solid #22324F; border-radius: 999px;
    padding: 0.15rem 0.7rem; background: #0F1728; }

.stat { border-top: 1px solid #22324F; padding-top: 0.65rem; }
.stat-label { color: #8C9BB5; font-size: 0.82rem; }
.stat-value { color: #E6ECF5; font-size: 1.55rem; font-weight: 650; font-variant-numeric: tabular-nums; margin-top: 0.12rem; }
.stat-sub { color: #8C9BB5; font-size: 0.78rem; margin-top: 0.1rem; }

.card { background: #111B2E; border: 1px solid #22324F; border-radius: 14px; padding: 1.4rem 1.6rem; }

.hero-card { background: #111B2E; border: 1px solid #2E4A7A; border-radius: 16px; padding: 1.8rem 2rem; }
.hero-eyebrow { color: #8C9BB5; font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; }
.hero-action { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.01em; margin-top: 0.2rem; }
.hero-price { font-size: 2.9rem; font-weight: 650; color: #E6ECF5; font-variant-numeric: tabular-nums; margin-top: 0.4rem; }
.hero-price-note { color: #8C9BB5; font-size: 0.95rem; margin-top: 0.1rem; }
.hero-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1.1rem; margin-top: 1.5rem; }
.hero-cell { border-top: 1px solid #22324F; padding-top: 0.55rem; }
.hero-cell-label { color: #8C9BB5; font-size: 0.78rem; }
.hero-cell-value { color: #E6ECF5; font-size: 1.05rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 0.12rem; }
.hero-reason { color: #C6D0E0; font-size: 0.92rem; line-height: 1.55; margin-top: 1.4rem; border-top: 1px solid #22324F; padding-top: 1rem; }

.step-list { counter-reset: steps; list-style: none; padding-left: 0; margin: 0.4rem 0 0 0; }
.step-list li { position: relative; padding: 0.55rem 0 0.55rem 2.3rem; border-bottom: 1px solid #1B2740; color: #C6D0E0; font-size: 0.92rem; line-height: 1.5; }
.step-list li:last-child { border-bottom: none; }
.step-list li::before { counter-increment: steps; content: counter(steps); position: absolute; left: 0; top: 0.5rem;
    width: 1.55rem; height: 1.55rem; border-radius: 50%; background: #0F1728; border: 1px solid #22324F; color: #5B9BF5;
    font-size: 0.8rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }

.badge { display: inline-block; padding: 0.14rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
</style>
"""

# Column-mapping suggestions (canonical field -> common company aliases).
FIELD_ALIASES: dict[str, list[str]] = {
    "Date": ["date", "sale_date", "transaction_date", "order_date", "datetime", "timestamp", "day"],
    "Product_ID": ["product", "product_id", "sku", "sku_id", "item_id", "item", "product_code"],
    "Price": ["price", "selling_price", "sale_price", "unit_price", "retail_price", "mrp"],
    "Demand": ["demand", "units_sold", "quantity_sold", "sales_quantity", "sales", "qty_sold", "quantity"],
    "Unit_Cost": ["unit_cost", "cost", "product_cost", "cost_per_unit", "cost_price", "cogs"],
    "Store_ID": ["store", "store_id", "branch_id", "location_id", "shop", "branch", "outlet"],
    "Competitor_Price": ["competitor_price", "competitor_pricing", "rival_price", "competition_price"],
    "Inventory": ["inventory", "inventory_level", "stock", "stock_level", "stock_on_hand"],
    "Discount": ["discount", "discount_percentage", "discount_pct"],
    "Promotion": ["promotion", "promotional_flag", "is_promotion", "promo"],
    "Category": ["category", "product_category", "product_group"],
    "Region": ["region", "market", "zone", "territory"],
    "Weather": ["weather", "weather_condition"],
    "Seasonality": ["seasonality", "season"],
    "Epidemic": ["epidemic", "outbreak"],
}

# Tokens that are too generic to identify a field on their own.
GENERIC_TOKENS = {"id", "code", "level", "condition", "name", "type", "value", "flag",
                  "indicator", "number", "no", "col", "column"}

NO_MAPPING = "— not provided —"


# ===========================================================================
# Small display helpers (formatting only - no business logic)
# ===========================================================================

def _stretch() -> dict[str, Any]:
    try:
        parts = tuple(int(p) for p in st.__version__.split(".")[:2])
    except ValueError:
        parts = (0, 0)
    return {"width": "stretch"} if parts >= (1, 50) else {"use_container_width": True}


def _fmt(value: Any, digits: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(number):
        return dash
    return f"{number:,.{digits}f}"


def _fmt_int(value: Any, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return dash


def _fmt_signed_pct(value: Any, digits: int = 1, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(number):
        return dash
    return f"{number:+.{digits}f}%"


def _fmt_date(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(stamp):
        return "—"
    return stamp.strftime("%Y-%m-%d")


def _badge(text: str, color: str) -> str:
    return (
        f'<span class="badge" style="border:1px solid {color};color:{color};">'
        f"{html.escape(str(text))}</span>"
    )


def _stat_html(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="stat"><div class="stat-label">{html.escape(str(label))}</div>'
        f'<div class="stat-value">{value}</div>{sub_html}</div>'
    )


def render_stats(items: list[tuple[str, str, str]], columns: int = 4) -> None:
    for start in range(0, len(items), columns):
        chunk = items[start:start + columns]
        cols = st.columns(len(chunk))
        for col, (label, value, sub) in zip(cols, chunk):
            col.markdown(_stat_html(label, value, sub), unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="section">{html.escape(title)}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{html.escape(note)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{html.escape(text)}</div>', unsafe_allow_html=True)


# ===========================================================================
# Column-mapping suggestions (UI helper)
# ===========================================================================

def _norm_key(name: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def _tokens(name: Any) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", str(name).lower()) if t]


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Suggest a canonical column for each field using exact then token matching.

    Never forces optional fields and never assigns one source column twice.
    """
    columns = list(columns)
    col_norm = {c: _norm_key(c) for c in columns}
    col_tokens = {c: _tokens(c) for c in columns}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}

    def exact_from(names: set[str]) -> str | None:
        for column in columns:
            if column not in used and col_norm[column] in names:
                return column
        return None

    # Phase 1a: exact canonical-name match.
    for field in CANONICAL_SCHEMA:
        pick = exact_from({_norm_key(field)})
        mapping[field] = pick
        if pick:
            used.add(pick)

    # Phase 1b: exact alias match for fields still unassigned.
    for field in CANONICAL_SCHEMA:
        if mapping.get(field):
            continue
        alias_norm = {_norm_key(a) for a in FIELD_ALIASES.get(field, [])}
        pick = exact_from(alias_norm)
        if pick:
            mapping[field] = pick
            used.add(pick)

    # Phase 2: distinctive token match.
    for field in CANONICAL_SCHEMA:
        if mapping.get(field):
            continue
        alias_tokens = {
            token
            for alias in [field, *FIELD_ALIASES.get(field, [])]
            for token in _tokens(alias)
            if len(token) >= 3 and token not in GENERIC_TOKENS
        }
        for column in columns:
            if column in used:
                continue
            if alias_tokens and alias_tokens.intersection(col_tokens[column]):
                mapping[field] = column
                used.add(column)
                break
    return mapping


# ===========================================================================
# Cached loaders and deterministic computations
# ===========================================================================

@st.cache_data(show_spinner=False)
def _load_uploaded(file_name: str, data: bytes) -> pd.DataFrame:
    """Load an uploaded file through src.data_loader from a temporary file."""
    suffix = Path(file_name).suffix.lower() or ".csv"
    handle_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(data)
        return load_data(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@st.cache_data(show_spinner=False)
def _prepare_pipeline(standardized_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministic dtype preparation + leakage-safe feature engineering."""
    prepared, report = validate_and_prepare_data(standardized_df)
    engineered = engineer_features(prepared, history_window=HISTORY_WINDOW)
    return engineered, report


@st.cache_data(show_spinner=False)
def _estimate_elasticity(series: pd.DataFrame) -> dict[str, Any]:
    """Elasticity estimate for one product series (cached per series content)."""
    return estimate_price_elasticity(series)


@st.cache_resource(show_spinner=False)
def _model_bundle() -> tuple[Any, dict[str, Any], str | None]:
    """Load the saved model and its metadata once per server process."""
    try:
        model = load_model(MODEL_PATH)
        metadata = load_metadata(METADATA_PATH)
        return model, metadata, None
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, never a traceback
        return None, {}, str(exc)


# ===========================================================================
# Session state
# ===========================================================================

DEFAULT_STATE: dict[str, Any] = {
    "current_stage": "upload",
    "uploaded_file_name": None,
    "is_demo": False,
    "raw_df": None,
    "column_mapping": {},
    "standardized_df": None,
    "validation_result": None,
    "engineered_df": None,
    "prepare_report": None,
    "model": None,
    "metadata": None,
    "model_compatibility": None,
    "selected_product": None,
    "selected_store": None,
    "objective": None,
    "elasticity": None,
    "optimization_result": None,
    "scenario_table": None,
    "sensitivity_table": None,
}


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        st.session_state.setdefault(key, value)


def clear_analysis_state(keep: set[str] | None = None) -> None:
    keep = keep or set()
    for key in list(st.session_state.keys()):
        if key not in keep:
            del st.session_state[key]
    init_state()


def full_reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def go_to(stage: str) -> None:
    st.session_state.current_stage = stage
    st.rerun()


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Workflow")
        current = st.session_state.current_stage
        order = [sid for sid, _ in STAGES]
        current_index = order.index(current) if current in order else 0
        for index, (sid, label) in enumerate(STAGES):
            if index < current_index:
                prefix = "✓"
            elif index == current_index:
                prefix = "▸"
            else:
                prefix = "·"
            weight = "600" if index == current_index else "400"
            color = BLUE if index == current_index else ("#8C9BB5" if index > current_index else "#7E8CA5")
            st.markdown(
                f'<div style="font-weight:{weight};color:{color};padding:0.12rem 0;">{prefix} {index + 1}. '
                f"{html.escape(label)}</div>",
                unsafe_allow_html=True,
            )
        st.divider()
        if st.session_state.uploaded_file_name:
            tag = " (demo)" if st.session_state.is_demo else ""
            st.caption("Loaded: " + str(st.session_state.uploaded_file_name) + tag)
        else:
            st.caption("No dataset loaded")
        if st.button("Start New Analysis", key="sidebar_reset"):
            full_reset()
            st.rerun()


# ===========================================================================
# Stage 1 - upload
# ===========================================================================

def _activate_dataset(name: str, df: pd.DataFrame, demo: bool) -> bool:
    if df is None or len(df) == 0:
        st.error(
            "The dataset is empty — no rows were found. Upload a file that contains "
            "historical sales rows."
        )
        return False
    clear_analysis_state(keep={"_upload_sig"})
    st.session_state.uploaded_file_name = name
    st.session_state.is_demo = demo
    st.session_state.raw_df = df
    st.session_state.current_stage = "overview"
    return True


def render_upload() -> None:
    st.markdown('<div class="app-title">Dynamic Pricing Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-tagline">Turn historical pricing data into actionable pricing scenarios.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Upload your data")
        note(
            "Upload a CSV or Excel file with historical sales and pricing rows. "
            "Supported formats: CSV, XLSX, XLS, XLSM, XLSB, ODS."
        )
        uploaded = st.file_uploader(
            "Historical sales and pricing file",
            type=SUPPORTED_UPLOAD_TYPES,
            key="file_uploader",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Status")
        st.markdown(
            f'<div style="margin-top:0.6rem;">{_badge("No dataset loaded", MUTED)}</div>',
            unsafe_allow_html=True,
        )
        note(
            "The analysis starts from an upload. Nothing is loaded automatically, so no "
            "company data is ever shown by default."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    demo_clicked = st.button("Load Demo Dataset (DEMO MODE)", key="demo_button")

    if uploaded is not None:
        signature = f"{uploaded.name}|{uploaded.size}"
        if st.session_state.get("_upload_sig") != signature:
            try:
                with st.spinner("Reading dataset..."):
                    frame = _load_uploaded(uploaded.name, uploaded.getvalue())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read '{uploaded.name}'. {exc}")
            else:
                st.session_state["_upload_sig"] = signature
                if _activate_dataset(uploaded.name, frame, demo=False):
                    st.rerun()

    if demo_clicked:
        try:
            with st.spinner("Loading demo dataset..."):
                frame = _load_uploaded(DEMO_RAW_PATH.name, DEMO_RAW_PATH.read_bytes())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load the demo dataset. {exc}")
        else:
            if _activate_dataset(f"{DEMO_RAW_PATH.name} (demo)", frame, demo=True):
                st.rerun()


# ===========================================================================
# Stage 2 - dataset overview
# ===========================================================================

def _guess_columns(columns: list[str], keywords: tuple[str, ...]) -> list[str]:
    found = []
    for column in columns:
        tokens = _tokens(column)
        if any(keyword in tokens for keyword in keywords):
            found.append(column)
    return found


def _best_effort_date_range(df: pd.DataFrame) -> str:
    candidates = _guess_columns(list(df.columns), ("date", "day", "time", "timestamp"))
    for column in candidates:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return f"{_fmt_date(parsed.min())} → {_fmt_date(parsed.max())}"
    return "Available after column mapping"


def render_overview() -> None:
    df: pd.DataFrame = st.session_state.raw_df
    section("Dataset Overview", "Structure of the uploaded dataset before any preparation.")
    info = get_basic_dataset_info(df)

    columns = list(df.columns)
    product_cols = _guess_columns(columns, ("product", "sku", "item"))
    store_cols = _guess_columns(columns, ("store", "shop", "branch", "outlet"))

    render_stats(
        [
            ("Rows", _fmt_int(info["rows"]), ""),
            ("Columns", _fmt_int(info["columns"]), ""),
            ("Date Range", _best_effort_date_range(df), ""),
            ("Products", _fmt_int(df[product_cols[0]].nunique()) if product_cols else "After mapping", ""),
            ("Stores", _fmt_int(df[store_cols[0]].nunique()) if store_cols else "After mapping", ""),
        ]
    )

    st.write("")
    st.markdown("**Data preview (first 10 rows)**")
    st.dataframe(df.head(10), **_stretch())

    if st.session_state.is_demo:
        st.info("DEMO MODE — this dataset is a bundled example, not company data.")

    st.write("")
    nav = st.columns([1, 1, 2])
    if nav[0].button("Back to upload", key="overview_back"):
        go_to("upload")
    if nav[1].button("Continue to column mapping", type="primary", key="overview_next"):
        go_to("mapping")


# ===========================================================================
# Stage 3 - column mapping
# ===========================================================================

def _field_kind(field: str) -> str:
    if field in CANONICAL_REQUIRED_COLUMNS:
        return "required"
    if field in RECOMMENDED_COLUMNS:
        return "recommended"
    return "optional"


def render_mapping() -> None:
    raw_df: pd.DataFrame = st.session_state.raw_df
    columns = list(raw_df.columns)
    suggestions = suggest_mapping(columns)
    options = [NO_MAPPING, *columns]

    section(
        "Column Mapping",
        "Match your columns to the pricing schema. Required fields must be mapped; "
        "optional fields can be left unmapped.",
    )

    with st.form("mapping_form"):
        st.markdown("**Required fields**")
        required_fields = [f for f in CANONICAL_SCHEMA if _field_kind(f) == "required"]
        rec_fields = [f for f in CANONICAL_SCHEMA if _field_kind(f) == "recommended"]
        opt_fields = [f for f in CANONICAL_SCHEMA if _field_kind(f) == "optional"]

        selected: dict[str, str | None] = {}
        for index, field in enumerate(required_fields):
            col = st.columns([1, 2])[1]
            default = suggestions.get(field)
            choice = col.selectbox(
                f"{field} *",
                options,
                index=(options.index(default) if default in options else 0),
                key=f"map__{field}",
            )
            selected[field] = None if choice == NO_MAPPING else choice

        st.markdown("**Recommended**")
        for field in rec_fields:
            col = st.columns([1, 2])[1]
            default = suggestions.get(field)
            choice = col.selectbox(
                f"{field} (recommended)",
                options,
                index=(options.index(default) if default in options else 0),
                key=f"map__{field}",
            )
            selected[field] = None if choice == NO_MAPPING else choice

        with st.expander("Optional fields", expanded=False):
            for field in opt_fields:
                col = st.columns([1, 2])[1]
                default = suggestions.get(field)
                choice = col.selectbox(
                    field,
                    options,
                    index=(options.index(default) if default in options else 0),
                    key=f"map__{field}",
                )
                selected[field] = None if choice == NO_MAPPING else choice

        submitted = st.form_submit_button("Validate mapping", type="primary")

    back = st.columns([1, 4])[0]
    if back.button("Back to overview", key="mapping_back"):
        go_to("overview")

    if submitted:
        try:
            standardized = standardize_columns(raw_df, selected)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Column mapping problem: {exc}")
        else:
            st.session_state.column_mapping = selected
            st.session_state.standardized_df = standardized
            with st.spinner("Validating dataset..."):
                st.session_state.validation_result = validate_dataset(standardized)
            go_to("validation")


# ===========================================================================
# Stage 4 - validation and data quality
# ===========================================================================

_VARIATION_TEXT = {
    "GOOD": (
        "There is substantial historical price variation, so a descriptive price-response "
        "estimate can be attempted. It remains observational, not causal."
    ),
    "LIMITED": (
        "Historical price variation is limited. Price-response estimates are unstable, so "
        "any output should be treated as indicative scenario analysis only."
    ),
    "INSUFFICIENT": (
        "Historical price variation is insufficient to estimate a price response. No reliable "
        "pricing recommendation can be derived from this data; the analysis is scenario-only."
    ),
}


def _render_field_status(std_df: pd.DataFrame) -> None:
    st.markdown("**Schema status**")
    groups = [("Required", "required"), ("Recommended", "recommended"), ("Optional", "optional")]
    for title, kind in groups:
        fields = [f for f in CANONICAL_SCHEMA if _field_kind(f) == kind]
        if not fields:
            continue
        badges = []
        for field in fields:
            present = field in std_df.columns
            if present:
                color = GREEN if kind != "optional" else BLUE
                badges.append(_badge(f"{field} ✓", color))
            else:
                color = RED if kind == "required" else (AMBER if kind == "recommended" else MUTED)
                badges.append(_badge(f"{field} ⚠", color))
        st.markdown(
            f'<div style="margin:0.35rem 0;"><span style="color:{MUTED};font-size:0.82rem;">'
            f"{html.escape(title)}:</span> " + " ".join(badges) + "</div>",
            unsafe_allow_html=True,
        )


def _render_data_quality(validation: dict[str, Any]) -> None:
    info = validation.get("info", {})
    std_df: pd.DataFrame = st.session_state.standardized_df

    render_stats(
        [
            ("Rows", _fmt_int(info.get("rows")), ""),
            ("Columns", _fmt_int(info.get("columns")), ""),
            ("Products", _fmt_int(info.get("n_products")), ""),
            ("Stores", _fmt_int(info.get("n_stores")) if info.get("n_stores") is not None else "Single-store", ""),
            ("Duplicate rows", _fmt_int(info.get("duplicate_rows")), ""),
        ]
    )

    st.write("")
    date_range = info.get("date_range") or info.get("data_types", {}).get("date_range")
    price_info = info.get("price", {})
    demand_info = info.get("demand", {})

    stat_items = [
        (
            "Date range",
            f"{_fmt_date(date_range[0])} → {_fmt_date(date_range[1])}" if date_range else "—",
            "",
        ),
        ("Price min / max", f"{_fmt(price_info.get('min'))} / {_fmt(price_info.get('max'))}", "per unit"),
        ("Price mean", _fmt(price_info.get("mean")), ""),
        ("Demand min / max", f"{_fmt(demand_info.get('min'), 0)} / {_fmt(demand_info.get('max'), 0)}", "units"),
        ("Demand mean", _fmt(demand_info.get("mean"), 0), "units"),
    ]
    render_stats(stat_items, columns=3)

    st.write("")
    missing = info.get("missing_values", {})
    if missing:
        missing_df = pd.DataFrame(
            {"Column": list(missing.keys()), "Missing": list(missing.values())}
        ).sort_values("Missing", ascending=False)
        st.markdown("**Missing values**")
        st.dataframe(missing_df, **_stretch())
    else:
        st.success("No missing values detected.")

    variation = price_info.get("variation", {})
    if variation:
        verdict = variation.get("verdict")
        section("Historical Price Variation")
        st.markdown(
            _badge(verdict, {"GOOD": GREEN, "LIMITED": AMBER}.get(verdict, RED)),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="note">Unique prices: {_fmt_int(variation.get("n_unique_prices"))} · '
            f'Coefficient of variation: {_fmt(variation.get("price_cv"), 3)}</div>',
            unsafe_allow_html=True,
        )
        note(_VARIATION_TEXT.get(verdict, ""))


def render_validation() -> None:
    if st.session_state.standardized_df is None or st.session_state.validation_result is None:
        st.warning("No standardized dataset is available. Please map your columns first.")
        if st.button("Back to column mapping", key="validation_no_data"):
            go_to("mapping")
        return

    validation: dict[str, Any] = st.session_state.validation_result
    std_df: pd.DataFrame = st.session_state.standardized_df

    section("Data Validation", "Required columns, data types and value sanity.")
    _render_field_status(std_df)

    st.write("")
    if validation["errors"]:
        for error in validation["errors"]:
            st.error(error)
    if validation["warnings"]:
        for warning in validation["warnings"]:
            st.warning(warning)
    if not validation["errors"] and not validation["warnings"]:
        st.success("All validation checks passed.")

    section("Data Quality Analysis", "Descriptive statistics of the standardized dataset.")
    _render_data_quality(validation)

    st.write("")
    nav = st.columns([1, 1, 2])
    if nav[0].button("Back to mapping", key="validation_back"):
        go_to("mapping")
    if validation["is_valid"]:
        if nav[1].button("Continue to preparation", type="primary", key="validation_next"):
            go_to("prepare")
    else:
        st.caption("Resolve the errors above before continuing. Warnings do not block the workflow.")


# ===========================================================================
# Stage 5 - preparation and model
# ===========================================================================

def _computed_compatibility(engineered: pd.DataFrame) -> dict[str, Any]:
    model, metadata, load_error = _model_bundle()
    st.session_state.model = model
    st.session_state.metadata = metadata
    if load_error:
        return {
            "compatible": False,
            "errors": [f"The trained model could not be loaded: {load_error}"],
            "warnings": [],
            "notes": [],
            "missing_features": [],
            "feature_columns": [],
        }
    try:
        return check_model_compatibility(model, metadata, engineered)
    except Exception as exc:  # noqa: BLE001
        return {
            "compatible": False,
            "errors": [f"Compatibility check failed: {exc}"],
            "warnings": [],
            "notes": [],
            "missing_features": [],
            "feature_columns": [],
        }


def _run_preparation() -> None:
    std_df: pd.DataFrame = st.session_state.standardized_df
    progress = st.progress(0, text="Preparing data...")
    try:
        progress.progress(30, text="Preparing data...")
        engineered, report = _prepare_pipeline(std_df)
        progress.progress(70, text="Engineering historical features...")
        progress.progress(85, text="Checking data quality...")
        assess_price_variation(std_df, price_col="Price")
        compatibility = _computed_compatibility(engineered)
        progress.progress(100, text="Ready")
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        st.error(f"Preparation failed: {exc}")
        return
    progress.empty()
    st.session_state.engineered_df = engineered
    st.session_state.prepare_report = report
    st.session_state.model_compatibility = compatibility
    st.rerun()


def render_prepare() -> None:
    section("Data Preparation & Model", "Dtype preparation, leakage-safe features and model check.")

    if st.session_state.standardized_df is None:
        st.warning("No standardized dataset is available.")
        if st.button("Back to column mapping", key="prepare_no_data"):
            go_to("mapping")
        return

    if st.session_state.engineered_df is None:
        note(
            "This step converts data types, builds the model's historical features and checks "
            "whether the saved demand model can be applied to this dataset. The original rows "
            "are preserved; only derived feature columns are added."
        )
        if st.button("Prepare data and check model", type="primary", key="prepare_run"):
            _run_preparation()
        return

    engineered: pd.DataFrame = st.session_state.engineered_df
    report = st.session_state.prepare_report or {}
    created = report.get("created", [])
    first_rows_missing = int(engineered["Previous_Demand"].isna().sum()) if "Previous_Demand" in engineered.columns else 0

    render_stats(
        [
            ("Rows prepared", _fmt_int(len(engineered)), ""),
            ("Features created", _fmt_int(len(created)), "derived columns"),
            ("Rows without lag history", _fmt_int(first_rows_missing), "handled by the model"),
            ("Series window", f"{HISTORY_WINDOW} rows", "rolling history"),
        ]
    )
    skipped = report.get("skipped", {})
    if skipped:
        with st.expander("Optional feature groups skipped"):
            for key, value in skipped.items():
                st.markdown(f"- **{key}** — {value}")

    # ----- Step 7: model handling -----
    section("Demand Model")
    compatibility: dict[str, Any] = st.session_state.model_compatibility or {}
    metadata = st.session_state.metadata or {}

    if compatibility.get("compatible"):
        st.success("A compatible demand model is available for this dataset.")
        model_rows = [
            ("Model", str(metadata.get("model_name", "—")), ""),
            ("MAE", _fmt(metadata.get("MAE"), 3), "test set"),
            ("RMSE", _fmt(metadata.get("RMSE"), 3), "test set"),
            ("R²", _fmt(metadata.get("R2"), 4), "test set"),
        ]
        render_stats(model_rows)
        st.info(
            "Model performance measures demand prediction accuracy. It does not establish that "
            "changing price will cause the predicted demand change."
        )
    else:
        st.markdown("#### New dataset detected")
        st.warning(
            "The saved model was trained on a different dataset and cannot be applied to this "
            "data. Demand prediction requires a model trained on a compatible dataset. Training "
            "is not performed inside this application."
        )
        for error in compatibility.get("errors", [])[:5]:
            st.markdown(f"- {error}")
        missing = compatibility.get("missing_features", [])
        if missing:
            with st.expander(f"Missing model inputs ({len(missing)})"):
                st.markdown(", ".join(str(item) for item in missing[:40]))
        st.info(
            "Pricing scenario analysis can still run using recent observed demand as the "
            "baseline. Model-based demand prediction is disabled until a compatible model exists."
        )

    for warning in compatibility.get("warnings", []):
        st.caption(f"Note: {warning}")

    st.write("")
    nav = st.columns([1, 1, 2])
    if nav[0].button("Back to validation", key="prepare_back"):
        go_to("validation")
    if nav[1].button("Continue to pricing analysis", type="primary", key="prepare_next"):
        go_to("pricing")


# ===========================================================================
# Stage 6 - pricing analysis
# ===========================================================================

def _series_for(engineered: pd.DataFrame, product: str, store: str | None) -> pd.DataFrame:
    mask = engineered["Product_ID"].astype(str) == str(product)
    if store is not None and "Store_ID" in engineered.columns:
        mask &= engineered["Store_ID"].astype(str) == str(store)
    series = engineered.loc[mask].copy()
    if "Date" in series.columns:
        series = series.sort_values("Date", kind="mergesort")
    return series.reset_index(drop=True)


def _latest_value(series: pd.DataFrame, column: str) -> float | None:
    if column not in series.columns:
        return None
    values = pd.to_numeric(series[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _positive_unit_cost(series: pd.DataFrame) -> float | None:
    if "Unit_Cost" not in series.columns:
        return None
    cost = pd.to_numeric(series["Unit_Cost"], errors="coerce").dropna()
    cost = cost[cost > 0]
    return float(cost.iloc[-1]) if len(cost) else None


def _current_price(series: pd.DataFrame) -> float:
    price = pd.to_numeric(series["Price"], errors="coerce")
    valid = series.loc[price.notna() & (price > 0)]
    if valid.empty:
        raise ValueError("The selected series has no positive historical price.")
    return float(pd.to_numeric(valid["Price"], errors="coerce").iloc[-1])


def _baseline_demand(series: pd.DataFrame, compatibility: dict[str, Any]) -> tuple[float, str]:
    model = st.session_state.model
    metadata = st.session_state.metadata or {}
    if compatibility.get("compatible") and model is not None:
        try:
            prediction = predict_demand(model, series.tail(1), metadata)
            baseline = float(prediction[0])
            if np.isfinite(baseline) and baseline >= 0:
                return baseline, "model"
        except Exception:  # noqa: BLE001 - fall back to observed demand
            pass
    recent = pd.to_numeric(series["Demand"], errors="coerce").dropna().tail(BASELINE_RECENT_OBS)
    if recent.empty:
        raise ValueError("No observed demand is available for the selected series.")
    return float(recent.mean()), "recent_observed"


def _analyze(
    *,
    series: pd.DataFrame,
    current_price: float,
    unit_cost: float | None,
    discount_pct: float,
    price_quality: str,
    elasticity_value: float,
    elasticity_source: str,
    baseline: float,
    baseline_source: str,
    objective: str,
    low_q: float,
    high_q: float,
    max_change: float,
    min_margin: float | None,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    prices = generate_price_scenarios(
        current_price, series["Price"], n_scenarios=N_SCENARIOS,
        low_quantile=low_q, high_quantile=high_q,
    )
    table = build_scenario_table(
        current_price, prices, baseline, elasticity_value,
        unit_cost=unit_cost, discount_pct=discount_pct,
    )
    guarded = apply_price_guardrails(
        table, current_price, max_price_change_pct=max_change,
        min_profit_margin=min_margin, unit_cost=unit_cost,
    )
    optimization = optimize_price(guarded, objective=objective)

    effective_current = current_price * (1.0 - discount_pct / 100.0)
    current_revenue = calculate_revenue(effective_current, baseline)
    current_profit = (
        calculate_profit(effective_current, baseline, unit_cost) if unit_cost is not None else None
    )
    current_value = current_profit if objective == "profit" else current_revenue

    feasible = bool(optimization["feasible"])
    recommended_price = optimization["recommended_price"] if feasible else current_price
    recommended_value = optimization["objective_value"] if feasible else current_value
    recommended_row = optimization["recommended_row"] or {}

    decision = decide_pricing_action(
        price_quality,
        objective,
        current_price,
        recommended_price,
        current_value,
        recommended_value,
        max_feasible_value=optimization["objective_value"] if feasible else None,
        guardrail_blocked=bool((~guarded["Feasible"]).any()),
    )

    if feasible:
        recommended_demand = recommended_row.get("Adjusted_Demand", baseline)
        recommended_revenue = recommended_row.get("Revenue", current_revenue)
        recommended_profit = recommended_row.get("Profit") if unit_cost is not None else None
        recommended_margin = recommended_row.get("Profit_Margin") if unit_cost is not None else None
    else:
        recommended_demand = baseline
        recommended_revenue = current_revenue
        recommended_profit = current_profit
        recommended_margin = None

    sensitivity: list[dict[str, Any]] = []
    for elasticity in SENSITIVITY_ELASTICITIES:
        sens_table = build_scenario_table(
            current_price, prices, baseline, elasticity,
            unit_cost=unit_cost, discount_pct=discount_pct,
        )
        sens_guarded = apply_price_guardrails(
            sens_table, current_price, max_price_change_pct=max_change,
            min_profit_margin=min_margin, unit_cost=unit_cost,
        )
        sens_opt = optimize_price(sens_guarded, objective=objective)
        sens_row = sens_opt["recommended_row"] or {}
        sensitivity.append(
            {
                "Elasticity": elasticity,
                "Optimal Price": sens_opt["recommended_price"],
                "Expected Demand": sens_row.get("Adjusted_Demand"),
                "Expected Revenue": sens_row.get("Revenue"),
                "Expected Profit": sens_row.get("Profit") if unit_cost is not None else None,
            }
        )

    result: dict[str, Any] = {
        "objective": objective,
        "action": decision["action"],
        "action_reason": decision["reason"],
        "basis_label": decision["basis_label"],
        "improvement_abs": decision["improvement_abs"],
        "improvement_pct": decision["improvement_pct"],
        "price_quality": price_quality,
        "series_label": st.session_state.get("_series_label", "selected series"),
        "current_price": current_price,
        "recommended_price": recommended_price if feasible else None,
        "current_demand": baseline,
        "recommended_demand": recommended_demand,
        "current_revenue": current_revenue,
        "recommended_revenue": recommended_revenue,
        "current_profit": current_profit,
        "recommended_profit": recommended_profit,
        "recommended_margin": recommended_margin,
        "unit_cost": unit_cost,
        "discount_pct": discount_pct,
        "elasticity_value": elasticity_value,
        "elasticity_source": elasticity_source,
        "baseline_demand": baseline,
        "baseline_source": baseline_source,
        "n_scenarios": int(optimization["n_scenarios"]),
        "n_feasible": int(optimization["n_feasible"]),
        "guardrail_blocks": guarded.attrs.get("guardrail_blocks", {}),
        "max_change": max_change,
        "min_margin": min_margin,
        "low_q": low_q,
        "high_q": high_q,
        "observation_count": int(len(series)),
    }
    return result, guarded, sensitivity


def _scenario_chart(
    table: pd.DataFrame,
    y_column: str,
    y_title: str,
    title: str,
    color: str,
    current_price: float,
    recommended_price: float | None,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=table["Price"], y=table[y_column], mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=5), name=y_title,
        )
    )
    if "Feasible" in table.columns:
        blocked = ~table["Feasible"].fillna(False).astype(bool)
        if blocked.any():
            figure.add_trace(
                go.Scatter(
                    x=table.loc[blocked, "Price"], y=table.loc[blocked, y_column],
                    mode="markers", marker=dict(size=8, color="#5A6B87", symbol="x"),
                    name="Blocked by guardrails",
                )
            )
    figure.add_vline(
        x=current_price, line_dash="dash", line_color=MUTED,
        annotation_text="Current", annotation_position="top left",
    )
    if recommended_price is not None and abs(recommended_price - current_price) > 1e-9:
        figure.add_vline(
            x=recommended_price, line_dash="dot", line_color=GREEN,
            annotation_text="Recommended", annotation_position="top right",
        )
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=10, r=10, t=48, b=10), title=title,
        xaxis_title="Price", yaxis_title=y_title, showlegend=False,
    )
    return figure


def _render_scenarios(result: dict[str, Any]) -> None:
    table: pd.DataFrame = st.session_state.scenario_table
    has_cost = result["unit_cost"] is not None

    section("Price Scenarios", f"{result['n_scenarios']} simulated prices · "
            f"{result['n_feasible']} pass the guardrails.")

    display_columns = ["Price", "Adjusted_Demand", "Revenue"]
    display_names = ["Price", "Expected Demand", "Expected Revenue"]
    if has_cost and "Profit" in table.columns:
        display_columns += ["Profit", "Profit_Margin"]
        display_names += ["Expected Profit", "Profit Margin"]
    display = table[display_columns].copy()
    display.columns = display_names
    if "Profit Margin" in display.columns:
        display["Profit Margin"] = display["Profit Margin"].map(
            lambda value: f"{value * 100:.1f}%" if pd.notna(value) else "—"
        )
    st.dataframe(display, **_stretch())

    recommended = result["recommended_price"]
    st.plotly_chart(
        _scenario_chart(table, "Adjusted_Demand", "Scenario Demand",
                        "Elasticity-Adjusted Demand Scenarios", BLUE,
                        result["current_price"], recommended),
        **_stretch(),
    )
    st.plotly_chart(
        _scenario_chart(table, "Revenue", "Expected Revenue", "Expected Revenue", GREEN,
                        result["current_price"], recommended),
        **_stretch(),
    )
    if has_cost and "Profit" in table.columns:
        st.plotly_chart(
            _scenario_chart(table, "Profit", "Expected Profit", "Expected Profit", AMBER,
                            result["current_price"], recommended),
            **_stretch(),
        )


def render_pricing() -> None:
    if st.session_state.engineered_df is None:
        st.warning("Data must be prepared before pricing analysis.")
        if st.button("Back to preparation", key="pricing_no_data"):
            go_to("prepare")
        return

    engineered: pd.DataFrame = st.session_state.engineered_df
    section("Pricing Analysis", "Select a series, set the objective and configuration, then run the analysis.")

    # ----- Step 8-9: series selection + snapshot -----
    if "Product_ID" not in engineered.columns:
        st.error("This dataset has no Product_ID column — pricing analysis cannot proceed.")
        return

    products = sorted(engineered["Product_ID"].dropna().astype(str).unique().tolist())
    if not products:
        st.error("No product identifiers are available in the prepared dataset.")
        return

    selection = st.columns([1, 1])
    product = selection[0].selectbox("Product", products, key="product_select")
    has_store = (
        "Store_ID" in engineered.columns
        and engineered["Store_ID"].notna().any()
        and engineered["Store_ID"].astype(str).nunique() > 1
    )
    store: str | None = None
    if has_store:
        stores = sorted(
            engineered.loc[engineered["Product_ID"].astype(str) == product, "Store_ID"]
            .dropna().astype(str).unique().tolist()
        )
        if stores:
            store = selection[1].selectbox("Store", stores, key="store_select")
        else:
            selection[1].info("No store rows for the selected product.")
    else:
        selection[1].markdown("**Store**")
        selection[1].caption("Single-store dataset")

    st.session_state.selected_product = product
    st.session_state.selected_store = store

    series = _series_for(engineered, product, store)
    st.session_state["_series_label"] = f"Product {product}" + (f" · Store {store}" if store else "")

    if series.empty:
        st.error("The selected series has no observations. Choose a different product or store.")
        return

    try:
        current_price = _current_price(series)
    except ValueError as exc:
        st.error(str(exc))
        return

    unit_cost = _positive_unit_cost(series)
    discount = _latest_value(series, "Discount")
    discount_pct = float(discount) if (discount is not None and 0 <= discount <= 100) else 0.0

    price_stats = pd.to_numeric(series["Price"], errors="coerce")
    demand_stats = pd.to_numeric(series["Demand"], errors="coerce")
    variation = assess_price_variation(series)

    snapshot = [
        ("Current price", _fmt(current_price), "latest observed"),
        ("Average price", _fmt(price_stats.mean()), "historical"),
        ("Latest demand", _fmt(demand_stats.iloc[-1] if len(demand_stats) else None, 0), "units"),
        ("Average demand", _fmt(demand_stats.mean(), 0), "units"),
        ("Observations", _fmt_int(len(series)), ""),
        ("Price variation", str(variation["verdict"]), f"{variation['n_unique_prices']} unique prices"),
    ]
    if unit_cost is not None:
        snapshot.append(("Unit cost", _fmt(unit_cost), "business-provided"))
    competitor = _latest_value(series, "Competitor_Price")
    if competitor is not None:
        snapshot.append(("Competitor price", _fmt(competitor), "latest observed"))
    inventory = _latest_value(series, "Inventory")
    if inventory is not None:
        snapshot.append(("Inventory", _fmt(inventory, 0), "latest observed"))
    render_stats(snapshot, columns=4)
    note("Current price is the most recent positive observed price for the selected series.")

    # ----- Step 10: objective -----
    section("Optimization Objective")
    if "objective_radio" not in st.session_state:
        st.session_state.objective_radio = "Profit" if unit_cost is not None else "Revenue"

    def _switch_to_revenue() -> None:
        st.session_state.objective_radio = "Revenue"
        st.session_state.objective = "revenue"

    objective_label = st.radio(
        "Objective", ["Revenue", "Profit"], key="objective_radio", horizontal=True,
        help="Profit requires a business-provided Unit Cost.",
    )
    objective = objective_label.lower()
    st.session_state.objective = objective
    if objective == "profit" and unit_cost is None:
        st.error("Profit optimization requires a business-provided Unit Cost.")
        st.button("Switch to Revenue Optimization", key="switch_revenue", on_click=_switch_to_revenue)

    # ----- Step 11: elasticity -----
    section("Price Elasticity", "Observational estimate; falls back to a scenario assumption when unreliable.")
    estimate = _estimate_elasticity(series[["Price", "Demand"]].copy())
    st.session_state.elasticity = estimate
    source_label = (
        "Historical Estimate" if estimate["source"] == "historical_estimate" else "Scenario Assumption"
    )
    reliability = "Reliable" if estimate["reliable"] else "Not reliable — using default assumption"
    render_stats(
        [
            ("Elasticity", _fmt(estimate["elasticity"], 3), ""),
            ("Source", source_label, ""),
            ("Reliability", reliability, ""),
            ("Observations", _fmt_int(estimate["observations"]), ""),
        ]
    )
    if estimate["source"] == "scenario_assumption":
        st.markdown(_badge("Scenario assumption — not a causal estimate.", AMBER), unsafe_allow_html=True)
    if estimate.get("warning"):
        st.warning(estimate["warning"])

    default_elasticity = round(
        float(np.clip(estimate["elasticity"], ELASTICITY_MIN, ELASTICITY_MAX)), 1
    )
    with st.expander("Advanced settings"):
        elasticity_value = st.slider(
            "Scenario elasticity assumption",
            float(ELASTICITY_MIN), float(ELASTICITY_MAX), default_elasticity, 0.1,
            key="elasticity_slider",
        )
        st.caption(
            f"Default from src.pricing.DEFAULT_ELASTICITY is {DEFAULT_ELASTICITY}. "
            "Changing this value does not retrain the model."
        )
    elasticity_source = (
        estimate["source"]
        if abs(elasticity_value - default_elasticity) < 1e-9
        else "manual_override"
    )

    # ----- Step 12: guardrails -----
    section("Business Guardrails", "Constraints a recommended price must satisfy.")
    guard = st.columns([1, 1, 1])
    max_change = guard[0].slider("Maximum price change (%)", 0, 50, 10, 1, key="max_change")
    low_pct = guard[1].slider("Lower price percentile", 5, 50, 5, 5, key="low_pct")
    high_pct = guard[2].slider("Upper price percentile", 50, 95, 95, 5, key="high_pct")
    use_margin = st.checkbox("Enforce minimum profit margin", key="use_margin")
    min_margin = None
    if use_margin:
        min_margin = st.slider("Minimum profit margin (%)", 0, 90, 20, 5, key="min_margin") / 100.0
        if unit_cost is None:
            st.error("A minimum profit margin requires a business-provided Unit Cost.")

    blocking_error = (objective == "profit" and unit_cost is None) or (use_margin and unit_cost is None)

    # ----- Step 13: run -----
    st.write("")
    run_clicked = st.button(
        "Run Pricing Analysis", type="primary", key="run_pricing", disabled=blocking_error
    )

    settings_signature = (
        product, store, objective, round(float(elasticity_value), 3),
        int(max_change), min_margin, int(low_pct), int(high_pct),
    )

    if run_clicked:
        with st.spinner("Simulating price scenarios..."):
            try:
                baseline, baseline_source = _baseline_demand(
                    series, st.session_state.model_compatibility or {}
                )
                result, guarded, sensitivity = _analyze(
                    series=series,
                    current_price=current_price,
                    unit_cost=unit_cost,
                    discount_pct=discount_pct,
                    price_quality=variation["verdict"],
                    elasticity_value=float(elasticity_value),
                    elasticity_source=elasticity_source,
                    baseline=baseline,
                    baseline_source=baseline_source,
                    objective=objective,
                    low_q=low_pct / 100.0,
                    high_q=high_pct / 100.0,
                    max_change=float(max_change),
                    min_margin=min_margin,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pricing analysis failed: {exc}")
            else:
                st.session_state.optimization_result = result
                st.session_state.scenario_table = guarded
                st.session_state.sensitivity_table = sensitivity
                st.session_state["_analysis_sig"] = settings_signature

    result = st.session_state.optimization_result
    if result is not None:
        if st.session_state.get("_analysis_sig") != settings_signature:
            st.info("Settings changed since the last run — run the analysis again to refresh the results.")
        _render_scenarios(result)
        st.write("")
        if st.button("View Recommendation", type="primary", key="view_recommendation"):
            go_to("recommendation")


# ===========================================================================
# Stage 7 - recommendation
# ===========================================================================

_ACTION_COLOR = {"KEEP": BLUE, "RAISE": GREEN, "LOWER": GREEN, "NO_GO": RED, "INSUFFICIENT_DATA": AMBER}
_ACTION_TEXT = {
    "KEEP": "Hold the current price.",
    "RAISE": "Raise the price to the recommended level.",
    "LOWER": "Lower the price to the recommended level.",
    "NO_GO": "Do not change the price.",
    "INSUFFICIENT_DATA": "Hold the current price.",
}


def _hero_cell(label: str, before: str, after: str | None = None) -> str:
    value = f"{before} → {after}" if after is not None else before
    return (
        f'<div class="hero-cell"><div class="hero-cell-label">{html.escape(label)}</div>'
        f'<div class="hero-cell-value">{value}</div></div>'
    )


def _render_hero(result: dict[str, Any]) -> None:
    action = result["action"]
    color = _ACTION_COLOR.get(action, BLUE)
    has_cost = result["unit_cost"] is not None

    recommended = result["recommended_price"]
    if recommended is not None and action in {"RAISE", "LOWER", "KEEP"}:
        price_html = f"{_fmt(recommended)}"
        price_note = "recommended price"
    else:
        price_html = _fmt(result["current_price"])
        price_note = "current price retained"

    cells = [
        _hero_cell("Current price", _fmt(result["current_price"])),
        _hero_cell("Expected demand", _fmt(result["current_demand"], 0),
                   _fmt(result["recommended_demand"], 0)),
        _hero_cell("Expected revenue", _fmt(result["current_revenue"]),
                   _fmt(result["recommended_revenue"])),
    ]
    if has_cost:
        cells.append(
            _hero_cell("Expected profit", _fmt(result["current_profit"]),
                       _fmt(result["recommended_profit"]))
        )

    improvement_html = ""
    if action in {"RAISE", "LOWER", "KEEP"} and result["objective"] == "profit":
        current_profit = result["current_profit"]
        if current_profit is not None and current_profit > 0 and result["improvement_pct"] is not None:
            improvement_html = _hero_cell("Profit change", _fmt_signed_pct(result["improvement_pct"]))
    elif action in {"RAISE", "LOWER", "KEEP"} and result["objective"] == "revenue":
        if result["current_revenue"] > 0 and result["improvement_pct"] is not None:
            improvement_html = _hero_cell("Revenue change", _fmt_signed_pct(result["improvement_pct"]))

    price_change = ""
    if recommended is not None and action in {"RAISE", "LOWER"}:
        change = calculate_percentage_change(recommended, result["current_price"])
        if np.isfinite(change):
            price_change = f'<div class="hero-price-note">Price change: {_fmt_signed_pct(change)}</div>'

    st.markdown(
        '<div class="hero-card">'
        '<div class="hero-eyebrow">Pricing Recommendation</div>'
        f'<div class="hero-action" style="color:{color};">{html.escape(action)}</div>'
        f'<div class="hero-price">{price_html}</div>'
        f'<div class="hero-price-note">{html.escape(price_note)}</div>'
        f"{price_change}"
        f'<div class="hero-grid">{"".join(cells)}{improvement_html}</div>'
        f'<div class="hero-reason">{html.escape(result["action_reason"])}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_decision_context(result: dict[str, Any]) -> None:
    action = result["action"]
    if action == "NO_GO":
        st.error(
            "No tested feasible price is expected to generate positive profit under the current "
            "assumptions. Hold the current price and review costs, the margin structure or the "
            "elasticity assumption before acting."
        )
    elif action == "INSUFFICIENT_DATA":
        st.warning(
            "The available data does not support a reliable pricing recommendation. The results "
            "below are scenario analysis only; collect more varied price history before changing "
            "prices."
        )


def _render_explanation(result: dict[str, Any]) -> None:
    baseline_label = (
        "the saved demand model at the current price"
        if result["baseline_source"] == "model"
        else f"recent observed average demand ({BASELINE_RECENT_OBS} observations)"
    )
    elasticity_source = {
        "historical_estimate": "historical observational estimate",
        "scenario_assumption": "configurable scenario assumption (not causal)",
        "manual_override": "manual scenario override (not causal)",
    }.get(result["elasticity_source"], result["elasticity_source"])
    objective_label = "expected profit" if result["objective"] == "profit" else "expected revenue"

    with st.expander("How was this recommendation calculated?", expanded=False):
        st.markdown(
            "<ol class='step-list'>"
            f"<li>The uploaded data was validated and mapped to the pricing schema ({result['observation_count']:,} "
            f"observations for {html.escape(str(result['series_label']))}).</li>"
            f"<li>The historical price and demand of the selected series were analyzed; current price is "
            f"{_fmt(result['current_price'])}.</li>"
            f"<li>Baseline demand ({_fmt(result['baseline_demand'], 0)} units) was established from {baseline_label}.</li>"
            f"<li>A price response was applied using a constant-elasticity assumption of "
            f"{_fmt(result['elasticity_value'], 2)} ({elasticity_source}).</li>"
            f"<li>{result['n_scenarios']} candidate prices were simulated between the "
            f"{_fmt(result['low_q'] * 100, 0)}th and {_fmt(result['high_q'] * 100, 0)}th percentile of observed prices.</li>"
            f"<li>Business guardrails (maximum price change ±{_fmt(result['max_change'], 0)}%"
            + (f", minimum profit margin {_fmt((result['min_margin'] or 0) * 100, 0)}%" if result['min_margin'] is not None else "")
            + f") were applied; {result['n_feasible']} of {result['n_scenarios']} scenarios were feasible.</li>"
            f"<li>The best feasible scenario for {objective_label} was selected and the decision rules "
            f"produced <strong>{html.escape(result['action'])}</strong>.</li>"
            "</ol>",
            unsafe_allow_html=True,
        )
        st.info("This is a decision-support scenario, not a guaranteed causal outcome.")


def _render_model_performance() -> None:
    metadata = st.session_state.metadata or {}
    compatibility = st.session_state.model_compatibility or {}
    with st.expander("Model Performance", expanded=False):
        if not compatibility.get("compatible"):
            st.caption("No compatible demand model was available for this dataset.")
            return
        render_stats(
            [
                ("Model", str(metadata.get("model_name", "—")), ""),
                ("MAE", _fmt(metadata.get("MAE"), 3), "test set"),
                ("RMSE", _fmt(metadata.get("RMSE"), 3), "test set"),
                ("R²", _fmt(metadata.get("R2"), 4), "test set"),
            ]
        )
        render_stats(
            [
                ("Training period", f"{_fmt_date(metadata.get('training_start_date'))} → "
                                    f"{_fmt_date(metadata.get('training_end_date'))}", ""),
                ("Test period", f"{_fmt_date(metadata.get('test_start_date'))} → "
                                f"{_fmt_date(metadata.get('test_end_date'))}", ""),
            ],
            columns=2,
        )
        st.caption(
            "Metrics come from the saved model metadata. Model performance measures demand "
            "prediction accuracy, not the causal effect of price."
        )


def _render_sensitivity() -> None:
    sensitivity = st.session_state.sensitivity_table or []
    with st.expander("Elasticity Sensitivity", expanded=False):
        if not sensitivity:
            st.caption("Run the pricing analysis to compute sensitivity.")
            return
        frame = pd.DataFrame(sensitivity)
        st.dataframe(
            frame.style.format(
                {
                    "Elasticity": "{:.1f}",
                    "Optimal Price": "{:,.2f}",
                    "Expected Demand": "{:,.1f}",
                    "Expected Revenue": "{:,.2f}",
                    "Expected Profit": "{:,.2f}",
                },
                na_rep="—",
            ),
            **_stretch(),
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=frame["Elasticity"], y=frame["Optimal Price"], mode="lines+markers",
                line=dict(color=BLUE, width=2), marker=dict(size=6),
            )
        )
        figure.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Elasticity", yaxis_title="Optimal price", showlegend=False,
        )
        st.plotly_chart(figure, **_stretch())
        st.caption("Scenario sensitivity, not causal estimation.")


def render_recommendation() -> None:
    result = st.session_state.optimization_result
    if result is None:
        st.warning("No analysis result is available. Run the pricing analysis first.")
        if st.button("Back to pricing analysis", key="reco_no_data"):
            go_to("pricing")
        return

    section("Recommendation")
    _render_hero(result)
    _render_decision_context(result)

    st.write("")
    _render_explanation(result)
    _render_model_performance()
    _render_sensitivity()

    st.write("")
    nav = st.columns([1, 1, 2])
    if nav[0].button("Back to pricing analysis", key="reco_back"):
        go_to("pricing")
    if nav[1].button("Start New Analysis", type="primary", key="reco_reset"):
        full_reset()
        st.rerun()


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()

    stage = st.session_state.current_stage
    if stage == "overview" and st.session_state.raw_df is None:
        stage = st.session_state.current_stage = "upload"

    renderers = {
        "upload": render_upload,
        "overview": render_overview,
        "mapping": render_mapping,
        "validation": render_validation,
        "prepare": render_prepare,
        "pricing": render_pricing,
        "recommendation": render_recommendation,
    }
    renderer = renderers.get(stage, render_upload)
    try:
        renderer()
    except Exception as exc:  # noqa: BLE001 - never expose a raw traceback
        st.error(f"Something went wrong while rendering this step: {exc}")
        st.caption("Use 'Start New Analysis' in the sidebar to reset the workflow.")


main()
