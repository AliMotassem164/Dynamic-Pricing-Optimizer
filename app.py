"""
Dynamic Pricing Optimizer
=========================
AI-powered pricing recommendations for smarter retail decisions.

Run from the project root:
    python -m streamlit run app.py

Expected project layout:
    Data/demand_forecasting_engineered.csv
    Data/store_metadata.csv      (optional demo labels; created automatically if missing)
    Data/product_metadata.csv    (optional demo labels + demo unit cost; created automatically if missing)
    Models/demand_forecasting_model.pkl
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# 1. Configuration
# ==========================================================

st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "demand_forecasting_engineered.csv"
MODEL_PATH = BASE_DIR / "Models" / "demand_forecasting_model.pkl"
STORE_META_PATH = BASE_DIR / "Data" / "store_metadata.csv"
PRODUCT_META_PATH = BASE_DIR / "Data" / "product_metadata.csv"

# Exact feature names (and order) expected by the trained pipeline.
MODEL_FEATURES = [
    "Price", "Discount", "Promotion", "Competitor Pricing",
    "Price_Difference", "Price_Ratio", "Effective_Price", "Promotion_Discount",
    "Price_Change", "Price_Change_Pct", "Previous_Demand",
    "Previous_7Day_Avg_Demand", "Demand_Trend", "Inventory Level",
    "Inventory_Coverage_Days", "Category", "Region", "Weather Condition",
    "Seasonality", "Epidemic", "Year", "Month", "Day", "DayOfWeek",
    "IsWeekend", "Weekend_Promotion", "Low_Inventory", "Store ID", "Product ID",
]
REQUIRED_COLUMNS = MODEL_FEATURES + ["Date", "Demand"]

N_SCENARIOS = 30
PRICE_LOW_Q, PRICE_HIGH_Q = 0.05, 0.95
FALLBACK_UNIT_COST = 40.0
SENSITIVITY_COSTS = [30, 35, 40, 45, 50, 55, 60]
KEEP_THRESHOLD_PCT = 1.0      # price moves smaller than this are treated as "keep"
BOUNDARY_TOLERANCE = 0.02     # share of the price range that counts as "at the boundary"

# BUSINESS ASSUMPTION (not estimated from data). Must be negative.
DEFAULT_ELASTICITY = -1.5
ELASTICITY_MIN, ELASTICITY_MAX = -3.0, -0.5

MODEL_METRICS = {"MAE": 13.75, "RMSE": 17.94, "R²": 0.847}

# ---- Demo labels (display names only; no numbers are invented) ----
DEMO_STORES = [
    ("Cairo Downtown Store", "Cairo"), ("Giza Retail Store", "Giza"),
    ("Nasr City Store", "Cairo"), ("Alexandria Store", "Alexandria"),
    ("6th October Store", "Giza"),
]
# Realistic retail products per category. Assigned in sorted Product ID order, so the mapping is stable.
PRODUCT_POOLS = {
    "electronics": ["Smartphone", "Laptop", "Wireless Headphones", "Bluetooth Speaker", "Smartwatch",
                    "Power Bank", "USB-C Cable", "Tablet", "Webcam", "Wireless Mouse"],
    "clothing": ["T-Shirt", "Jeans", "Running Shoes", "Backpack", "Hoodie", "Denim Jacket",
                 "Polo Shirt", "Sneakers", "Winter Scarf", "Baseball Cap"],
    "groceries": ["Breakfast Cereal", "Cooking Oil", "Coffee", "Bottled Water", "Basmati Rice",
                  "Pasta", "Black Tea", "Olive Oil", "Orange Juice", "Mixed Nuts"],
    "furniture": ["Office Chair", "Desk Lamp", "Bookshelf", "Coffee Table", "Standing Desk",
                  "TV Stand", "Storage Cabinet", "Shoe Rack", "Bed Frame", "Floor Mirror"],
    "toys": ["Building Blocks Set", "Jigsaw Puzzle", "Plush Teddy Bear", "Remote Control Car",
             "Board Game", "Art & Craft Kit", "Toy Kitchen Set", "Action Figure Set", "Play Dough Set", "Toy Train Set"],
    "home & kitchen": ["Coffee Maker", "Electric Kettle", "Water Bottle", "Air Fryer", "Blender",
                       "Toaster", "Non-Stick Frying Pan", "Food Storage Set", "Dinner Plate Set", "Cutting Board"],
    "personal care": ["Shampoo", "Toothpaste", "Body Lotion", "Face Wash", "Deodorant",
                      "Hand Soap", "Conditioner", "Sunscreen", "Electric Toothbrush", "Hair Dryer"],
}
POOL_ALIASES = {
    "home and kitchen": "home & kitchen", "home": "home & kitchen", "kitchen": "home & kitchen",
    "personal-care": "personal care", "beauty": "personal care", "grocery": "groceries",
    "toy": "toys", "apparel": "clothing", "clothes": "clothing",
}
PLACEHOLDER_NAME = re.compile(r"^\s*(example\s+|random\s+)?(product|item)\s*\d*\s*$", re.I)
DEMO_COST_SHARE = 0.60  # demo unit cost = 60% of the product's median historical price

# Palette (kept small on purpose: blue = action/line, green = recommendation)
INK = "#0B1220"
SURFACE = "#111B2E"
BORDER = "#22324F"
TEXT = "#E6ECF5"
MUTED = "#8C9BB5"
BLUE = "#5B9BF5"
STEEL = "#3A64B8"
GREEN = "#3FC48A"
AMBER = "#E3B25A"

# Streamlit renamed `use_container_width` to `width="stretch"` in newer releases.
_ST_VERSION = tuple(int(v) for v in st.__version__.split(".")[:2] if v.isdigit())
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}

# ==========================================================
# 2. Styling
# ==========================================================

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background: {INK}; color: {TEXT}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1040px; }}

.hero-title {{ font-size: 2.4rem; font-weight: 600; letter-spacing: -0.02em; color: {TEXT}; margin: 0; }}
.hero-subtitle {{ color: {MUTED}; font-size: 1.05rem; margin-top: 0.35rem; }}

.section-title {{ font-size: 1.25rem; font-weight: 600; color: {TEXT}; margin: 3rem 0 0.9rem 0; }}
.note {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.5; margin-top: 0.4rem; }}

.stat {{ border-top: 1px solid {BORDER}; padding-top: 0.7rem; }}
.stat-label {{ color: {MUTED}; font-size: 0.85rem; }}
.stat-value {{ color: {TEXT}; font-size: 1.7rem; font-weight: 600; font-variant-numeric: tabular-nums;
    margin-top: 0.15rem; }}
.stat-sub {{ color: {MUTED}; font-size: 0.8rem; margin-top: 0.15rem; }}

.reco {{ background: linear-gradient(160deg, #162743 0%, {SURFACE} 70%); border: 1px solid #2E4A7A;
    border-radius: 16px; padding: 2rem 2.2rem; }}
.reco-pill {{ display: inline-block; font-size: 0.82rem; font-weight: 500; padding: 0.2rem 0.65rem;
    border-radius: 999px; border: 1px solid {BORDER}; background: {INK}; }}
.reco-label {{ color: {MUTED}; font-size: 0.95rem; margin-top: 1.1rem; }}
.reco-price {{ font-size: 4rem; font-weight: 600; line-height: 1.05; color: {TEXT};
    font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }}
.reco-profit-row {{ display: flex; align-items: baseline; gap: 0.8rem; margin-top: 1rem; flex-wrap: wrap; }}
.reco-profit {{ font-size: 1.8rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.badge {{ font-size: 0.9rem; font-weight: 600; }}
.cmp-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1.8rem;
    padding-top: 1.4rem; border-top: 1px solid {BORDER}; }}
.cmp-label {{ color: {MUTED}; font-size: 0.85rem; }}
.cmp-row {{ font-size: 1.35rem; font-variant-numeric: tabular-nums; margin-top: 0.2rem; }}
.cmp-row .arrow {{ color: {MUTED}; margin: 0 0.55rem; }}
.cmp-cap {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.15rem; }}
.explain {{ font-size: 1.05rem; color: {TEXT}; margin: 1.3rem 0 0.2rem 0; }}
.impact-line {{ color: {MUTED}; font-size: 0.88rem; }}
.pos {{ color: {GREEN}; }}
.neu {{ color: {BLUE}; }}
.muted {{ color: {MUTED}; }}

.warn {{ border: 1px solid #5E5026; background: rgba(227,178,90,0.07); color: #E8CF98;
    border-radius: 8px; padding: 0.65rem 1rem; font-size: 0.88rem; margin-top: 1rem; }}

/* The single primary action */
div.stButton > button, div[data-testid="stButton"] > button {{
    background: {STEEL} !important; color: #FFFFFF !important; border: 1px solid #4F7BD0 !important;
    font-weight: 600; font-size: 1.1rem; padding: 0.8rem 1.5rem; border-radius: 10px; width: 100%; }}
div.stButton > button:hover, div[data-testid="stButton"] > button:hover {{
    background: #4676D6 !important; border-color: #6C93E0 !important; color: #FFFFFF !important; }}
div.stButton > button:focus-visible, div[data-testid="stButton"] > button:focus-visible {{
    outline: 2px solid {TEXT} !important; outline-offset: 2px; }}

div[data-testid="stExpander"] {{ background: transparent; border: 1px solid {BORDER}; border-radius: 10px; }}
label, .stSelectbox label, .stNumberInput label, .stSlider label {{ color: {MUTED} !important; }}
button[data-baseweb="tab"] {{ color: {MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEXT}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ==========================================================
# 3. Formatting & UI helpers
# ==========================================================


def money(x: float) -> str:
    return f"${x:,.2f}"


def money0(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def pct_change(new: float, old: float) -> float:
    """Percentage change; NaN when the baseline is zero or invalid."""
    if not np.isfinite(old) or old == 0:
        return np.nan
    return (new - old) / abs(old) * 100.0


def fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+.1f}%"


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def stat(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return (f'<div class="stat"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div>{sub_html}</div>')


# ==========================================================
# 4. Data, model and metadata loading
# ==========================================================


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return data


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return joblib.load(path)


def _pool_key(category: str) -> str:
    key = str(category).strip().lower()
    return POOL_ALIASES.get(key, key)


def build_demo_metadata(data: pd.DataFrame):
    """Deterministic demo labels from the dataset's real IDs and categories.

    Only display names (and a demo unit cost derived from real historical prices) are created.
    """
    stores = []
    for i, store_id in enumerate(sorted(data["Store ID"].dropna().astype(str).unique())):
        name, region = DEMO_STORES[i] if i < len(DEMO_STORES) else (f"Store {store_id}", "Demo")
        stores.append({"Store ID": store_id, "Store Name": name, "Region": region})

    products, used = [], {}
    grouped = data.groupby(data["Product ID"].astype(str))
    for product_id in sorted(grouped.groups.keys()):
        sub = grouped.get_group(product_id)
        category = str(sub["Category"].mode().iloc[0])
        key = _pool_key(category)
        pool = PRODUCT_POOLS.get(key, [])
        k = used.get(key, 0)
        used[key] = k + 1
        name = pool[k] if k < len(pool) else f"{category} Item {k + 1}"
        products.append({
            "Product ID": product_id, "Product Name": name, "Category": category,
            "Unit Cost": round(float(sub["Price"].median()) * DEMO_COST_SHARE, 2),
        })
    return pd.DataFrame(stores), pd.DataFrame(products)


@st.cache_data(show_spinner=False)
def load_metadata(_data: pd.DataFrame, store_path: str, product_path: str) -> dict:
    """Metadata lookups. Existing files are used (never overwritten); missing ones are created.

    Placeholder names such as "Product 001" found in an existing file are replaced at display
    time by the deterministic demo names, without modifying the file.
    """
    demo_stores, demo_products = build_demo_metadata(_data)
    store_names = dict(zip(demo_stores["Store ID"], demo_stores["Store Name"]))
    prod = demo_products.set_index("Product ID").to_dict("index")

    sp, pp = Path(store_path), Path(product_path)
    if sp.exists():
        try:
            ex = pd.read_csv(sp, dtype={"Store ID": str})
            for sid, name in zip(ex["Store ID"], ex["Store Name"]):
                if isinstance(name, str) and name.strip() and not PLACEHOLDER_NAME.match(name):
                    store_names[sid] = name.strip()
        except Exception:  # noqa: BLE001 - fall back to demo labels
            pass
    else:
        try:
            demo_stores.to_csv(sp, index=False)
        except Exception:  # noqa: BLE001
            pass

    if pp.exists():
        try:
            ex = pd.read_csv(pp, dtype={"Product ID": str})
            for _, row in ex.iterrows():
                pid = row.get("Product ID")
                if pid not in prod:
                    continue
                name = row.get("Product Name")
                if isinstance(name, str) and name.strip() and not PLACEHOLDER_NAME.match(name):
                    prod[pid]["Product Name"] = name.strip()
                cost = pd.to_numeric(row.get("Unit Cost"), errors="coerce")
                if np.isfinite(cost) and cost >= 0:
                    prod[pid]["Unit Cost"] = float(cost)
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            demo_products.to_csv(pp, index=False)
        except Exception:  # noqa: BLE001
            pass

    return {
        "store_names": store_names,
        "product_names": {k: v["Product Name"] for k, v in prod.items()},
        "product_costs": {k: v["Unit Cost"] for k, v in prod.items()},
    }


def load_assets():
    """Load dataset and model; stop with a clear message on failure."""
    errors, data, model = [], None, None

    if not DATA_PATH.exists():
        errors.append(f"Dataset not found: `{DATA_PATH.relative_to(BASE_DIR)}`")
    else:
        try:
            data = load_data(str(DATA_PATH))
            missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
            if missing:
                errors.append(f"Dataset is missing required columns: {missing}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Could not read the dataset: {exc}")

    if not MODEL_PATH.exists():
        errors.append(f"Model not found: `{MODEL_PATH.relative_to(BASE_DIR)}`")
    else:
        try:
            model = load_model(str(MODEL_PATH))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"Could not load the model: {exc}. "
                "The model was trained with scikit-learn 1.6.1 and numpy 2.1.3."
            )

    if errors:
        render_header()
        for message in errors:
            st.error(message)
        st.info("Run the app from the project root: `python -m streamlit run app.py`")
        st.stop()

    return data, model


# ==========================================================
# 5. Pricing engine (unchanged logic)
# ==========================================================


def build_scenarios(base_row: pd.DataFrame, prices: np.ndarray, previous_price: float) -> pd.DataFrame:
    """Copy the selected record once per price and recompute price-derived features."""
    sim = base_row.loc[base_row.index.repeat(len(prices))].reset_index(drop=True)

    sim["Price"] = prices.astype(float)
    competitor = sim["Competitor Pricing"].astype(float).replace(0, np.nan)

    sim["Price_Difference"] = sim["Price"] - sim["Competitor Pricing"]
    sim["Price_Ratio"] = sim["Price"] / competitor
    sim["Price_Change"] = sim["Price"] - previous_price
    if previous_price and np.isfinite(previous_price):
        sim["Price_Change_Pct"] = sim["Price_Change"] / previous_price * 100.0
    else:
        sim["Price_Change_Pct"] = 0.0
    sim["Effective_Price"] = sim["Price"] * (1 - sim["Discount"] / 100.0)
    return sim


def predict_demand(model, sim: pd.DataFrame) -> np.ndarray:
    """Model-predicted demand (demand forecast); cannot be negative."""
    prediction = model.predict(sim[MODEL_FEATURES])
    return np.clip(np.asarray(prediction, dtype=float), 0, None)


@st.cache_data(show_spinner=False)
def run_simulation(_data: pd.DataFrame, _model, store: str, product: str) -> dict:
    """Model-predicted demand for 30 prices in the historical 5th-95th percentile range."""
    combo = _data[(_data["Store ID"] == store) & (_data["Product ID"] == product)].sort_values("Date")
    if combo.empty:
        raise ValueError("No records for the selected Store and Product.")

    base_row = combo.iloc[[-1]]
    current = base_row.iloc[0]

    min_price = float(combo["Price"].quantile(PRICE_LOW_Q))
    max_price = float(combo["Price"].quantile(PRICE_HIGH_Q))
    prices = np.linspace(min_price, max_price, N_SCENARIOS)

    # Previous price is recovered from the record's own Price_Change value.
    previous_price = float(current["Price"] - current["Price_Change"])

    scenarios = build_scenarios(base_row, prices, previous_price)
    scenarios["Model_Predicted_Demand"] = predict_demand(_model, scenarios)

    current_sim = build_scenarios(base_row, np.array([float(current["Price"])]), previous_price)
    current_eff = float(current_sim["Effective_Price"].iloc[0])
    current_model_demand = float(predict_demand(_model, current_sim)[0])

    return {
        "scenarios": scenarios[["Price", "Effective_Price", "Model_Predicted_Demand"]].copy(),
        "current": {
            "price": float(current["Price"]),
            "effective_price": current_eff,
            "model_demand": current_model_demand,
        },
        "snapshot": {
            "price": float(current["Price"]),
            "competitor": float(current["Competitor Pricing"]),
            "discount": float(current["Discount"]),
            "avg_demand": float(combo["Demand"].mean()),
            "latest_demand": float(current["Demand"]),
            "promotion": current["Promotion"],
            "inventory": float(current["Inventory Level"]),
            "date": current["Date"],
        },
        "price_range": (min_price, max_price),
    }


def adjust_demand(price, current_price: float, baseline_demand: float, elasticity: float) -> np.ndarray:
    """Adjusted = Current Predicted Demand x (Price / Current Price) ** Elasticity (scenario assumption)."""
    return baseline_demand * np.power(np.asarray(price, dtype=float) / current_price, elasticity)


def optimize(sim_result: dict, unit_cost: float, elasticity: float) -> dict:
    """Apply the pricing-response assumption and pick the best simulated profit scenario."""
    cur = sim_result["current"]
    if cur["price"] <= 0 or cur["model_demand"] <= 0:
        raise ValueError(
            "The model predicts zero demand (or the price is zero) at the current price, "
            "so a baseline for the pricing-response assumption cannot be built."
        )

    scen = sim_result["scenarios"].copy()
    baseline = cur["model_demand"]
    scen["Adjusted_Demand"] = adjust_demand(scen["Price"], cur["price"], baseline, elasticity)
    scen["Revenue"] = scen["Effective_Price"] * scen["Adjusted_Demand"]
    scen["Profit"] = (scen["Effective_Price"] - unit_cost) * scen["Adjusted_Demand"]

    best_idx = int(scen["Profit"].idxmax())
    best = scen.loc[best_idx]

    cur_revenue = cur["effective_price"] * baseline
    cur_profit = (cur["effective_price"] - unit_cost) * baseline

    change = {
        "price": best["Price"] - cur["price"],
        "price_pct": pct_change(best["Price"], cur["price"]),
        "profit": best["Profit"] - cur_profit,
    }

    rows = []
    for cost in SENSITIVITY_COSTS:
        profit = (scen["Effective_Price"] - cost) * scen["Adjusted_Demand"]
        top = scen.loc[int(profit.idxmax())]
        rows.append({
            "Unit Cost": float(cost),
            "Recommended Price": float(top["Price"]),
            "Predicted Demand (adjusted)": float(top["Adjusted_Demand"]),
            "Expected Profit": float(profit.max()),
        })

    lo, hi = float(scen["Price"].min()), float(scen["Price"].max())
    span = hi - lo
    return {
        "scenarios": scen,
        "best": best,
        "best_idx": best_idx,
        "current": {"price": cur["price"], "demand": baseline, "revenue": cur_revenue, "profit": cur_profit},
        "change": change,
        "sensitivity": pd.DataFrame(rows),
        "at_upper": span > 0 and best["Price"] >= hi - BOUNDARY_TOLERANCE * span,
        "at_lower": span > 0 and best["Price"] <= lo + BOUNDARY_TOLERANCE * span,
    }


def decide(opt: dict) -> dict:
    """Turn the optimization result into what the business user sees.

    If the best simulated price is not meaningfully better than the current one, the
    recommendation is to keep the current price (so the headline never shows a worse price).
    """
    best, cur, ch = opt["best"], opt["current"], opt["change"]
    keep = ch["profit"] <= 0 or (np.isfinite(ch["price_pct"]) and abs(ch["price_pct"]) < KEEP_THRESHOLD_PCT)

    if keep:
        reco = {"price": cur["price"], "demand": cur["demand"], "revenue": cur["revenue"], "profit": cur["profit"]}
        action, cls = "Keep current price", "neu"
        text = "Keeping the current price is expected to provide the strongest simulated profit."
    else:
        reco = {"price": float(best["Price"]), "demand": float(best["Adjusted_Demand"]),
                "revenue": float(best["Revenue"]), "profit": float(best["Profit"])}
        if reco["price"] < cur["price"]:
            action, cls = "Lower price", "pos"
            text = "Lowering the price is expected to increase demand and improve profitability."
        else:
            action, cls = "Raise price", "pos"
            text = "Raising the price is expected to improve profit despite the expected reduction in demand."

    return {
        "keep": keep, "reco": reco, "action": action, "cls": cls, "text": text,
        "profit_pct": pct_change(reco["profit"], cur["profit"]),
        "demand_pct": pct_change(reco["demand"], cur["demand"]),
        "revenue_pct": pct_change(reco["revenue"], cur["revenue"]),
    }


# ==========================================================
# 6. Charts
# ==========================================================


def _base_layout(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXT),
        xaxis=dict(title=x_title, gridcolor=BORDER, zeroline=False),
        yaxis=dict(title=y_title, gridcolor=BORDER, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=70, b=10),
        height=height,
        hovermode="x unified",
    )
    return fig


def line_chart(x, y, title: str, y_title: str, y_prefix: str, current_pt, reco_pt,
               series_name: str, extra_trace=None) -> go.Figure:
    fig = go.Figure()
    if extra_trace is not None:
        fig.add_trace(extra_trace)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=series_name,
        line=dict(color=BLUE, width=3), marker=dict(size=5),
        hovertemplate=f"Price: $%{{x:,.2f}}<br>{y_title}: {y_prefix}%{{y:,.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[current_pt[0]], y=[current_pt[1]], mode="markers", name="Current price",
        marker=dict(size=14, color="#C9D4E8", symbol="diamond", line=dict(color=INK, width=1.5)),
        hovertemplate=f"Current price: $%{{x:,.2f}}<br>{y_title}: {y_prefix}%{{y:,.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[reco_pt[0]], y=[reco_pt[1]], mode="markers", name="Recommended price",
        marker=dict(size=17, color=GREEN, symbol="star", line=dict(color=INK, width=1.5)),
        hovertemplate=f"Recommended price: $%{{x:,.2f}}<br>{y_title}: {y_prefix}%{{y:,.2f}}<extra></extra>",
    ))
    fig = _base_layout(fig, title, "Price", y_title)
    fig.update_xaxes(tickprefix="$")
    if y_prefix:
        fig.update_yaxes(tickprefix=y_prefix)
    return fig


def sensitivity_chart(sens: pd.DataFrame, selected_cost: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens["Unit Cost"], y=sens["Expected Profit"], mode="lines+markers",
        name="Best simulated profit", line=dict(color=BLUE, width=3), marker=dict(size=8),
        customdata=sens["Recommended Price"],
        hovertemplate="Unit cost: $%{x:,.2f}<br>Expected profit: $%{y:,.2f}"
                      "<br>Recommended price: $%{customdata:,.2f}<extra></extra>",
    ))
    match = sens[sens["Unit Cost"] == selected_cost]
    if not match.empty:
        fig.add_trace(go.Scatter(
            x=match["Unit Cost"], y=match["Expected Profit"], mode="markers", name="Selected unit cost",
            marker=dict(size=15, color=GREEN, symbol="star", line=dict(color=INK, width=1.5)),
            hoverinfo="skip",
        ))
    fig = _base_layout(fig, "Profit Sensitivity to Unit Cost", "Unit cost", "Expected profit", height=360)
    fig.update_xaxes(tickprefix="$")
    fig.update_yaxes(tickprefix="$")
    return fig


# ==========================================================
# 7. Page sections
# ==========================================================


def render_header() -> None:
    st.markdown(
        '<div class="hero-title">Dynamic Pricing Optimizer</div>'
        '<div class="hero-subtitle">AI-powered pricing recommendations for smarter retail decisions</div>',
        unsafe_allow_html=True,
    )


def render_selection(data: pd.DataFrame, meta: dict):
    section("Choose Product")
    names = meta["store_names"]
    pnames = meta["product_names"]

    c1, c2 = st.columns(2)
    stores = sorted(data["Store ID"].dropna().astype(str).unique().tolist())
    store = c1.selectbox("Store", stores, key="store", format_func=lambda s: names.get(s, s))

    products = sorted(data.loc[data["Store ID"] == store, "Product ID"].dropna().astype(str).unique().tolist())
    product = c2.selectbox("Product", products, key=f"product_{store}", format_func=lambda p: pnames.get(p, p))

    note(f"Store and product names are realistic demo labels mapped to the dataset's IDs "
         f"({store} · {product}).")

    default_cost = meta["product_costs"].get(product)
    if default_cost is None or not np.isfinite(default_cost) or default_cost < 0:
        default_cost = FALLBACK_UNIT_COST
    return store, product, float(default_cost), names.get(store, store), pnames.get(product, product)


def render_situation(snap: dict) -> None:
    section("Current Situation")
    cols = st.columns(4)
    cols[0].markdown(stat("Current Price", money(snap["price"])), unsafe_allow_html=True)
    cols[1].markdown(stat("Avg. Demand", f"{snap['avg_demand']:,.0f} units"), unsafe_allow_html=True)
    cols[2].markdown(stat("Competitor Price", money(snap["competitor"])), unsafe_allow_html=True)
    cols[3].markdown(stat("Inventory", f"{snap['inventory']:,.0f} units"), unsafe_allow_html=True)

    extras = []
    if float(snap["promotion"]) == 1:
        extras.append("Promotion active")
    if snap["discount"] > 0:
        extras.append(f"{snap['discount']:.0f}% discount applied")
    if extras:
        note(" · ".join(extras))


def render_settings(default_cost: float, product: str):
    section("Pricing Settings")
    c1, _ = st.columns([1, 1])
    unit_cost = c1.number_input(
        "Demo Unit Cost ($)", min_value=0.0, value=default_cost, step=1.0, format="%.2f",
        key=f"unit_cost_{product}",
        help="Business input, not provided in the original dataset. The default is a demo value; "
             "replace it with your actual product cost.",
    )
    note("Unit cost is a business input. The default is a demo value, so replace it with your actual cost.")

    with st.expander("Advanced Pricing Assumptions"):
        elasticity = st.slider(
            "Price Sensitivity (Elasticity)", min_value=ELASTICITY_MIN, max_value=ELASTICITY_MAX,
            value=DEFAULT_ELASTICITY, step=0.1, key="elasticity",
        )
        note("Used as a scenario assumption to estimate how demand may respond to price changes. "
             "It is not learned from the data.")
    return float(unit_cost), float(elasticity)


def render_recommendation(opt: dict) -> dict:
    d = decide(opt)
    reco, cur = d["reco"], opt["current"]
    section("Recommended Pricing")

    if d["keep"] or not np.isfinite(d["profit_pct"]):
        badge = '<span class="badge neu">No profit change expected</span>' if d["keep"] else ""
    else:
        badge = f'<span class="badge pos">{fmt_pct(d["profit_pct"])} Expected Profit</span>'

    st.markdown(
        f'<div class="reco">'
        f'<span class="reco-pill {d["cls"]}">{d["action"]}</span>'
        f'<div class="reco-label">Recommended Price</div>'
        f'<div class="reco-price">{money(reco["price"])}</div>'
        f'<div class="reco-profit-row"><span class="muted">Expected Profit</span>'
        f'<span class="reco-profit">{money0(reco["profit"])}</span>{badge}</div>'
        f'<div class="cmp-grid">'
        f'<div><div class="cmp-label">Price</div>'
        f'<div class="cmp-row">{money(cur["price"])}<span class="arrow">→</span><b>{money(reco["price"])}</b></div>'
        f'<div class="cmp-cap">Current → Recommended</div></div>'
        f'<div><div class="cmp-label">Expected Profit</div>'
        f'<div class="cmp-row">{money0(cur["profit"])}<span class="arrow">→</span><b>{money0(reco["profit"])}</b></div>'
        f'<div class="cmp-cap">Current → Recommended</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="explain">{d["text"]}</div>', unsafe_allow_html=True)
    if not d["keep"]:
        st.markdown(
            f'<div class="impact-line">Expected demand {fmt_pct(d["demand_pct"])} · '
            f'Expected revenue {fmt_pct(d["revenue_pct"])}</div>',
            unsafe_allow_html=True,
        )
        if opt["at_upper"] or opt["at_lower"]:
            end = "upper" if opt["at_upper"] else "lower"
            st.markdown(
                f'<div class="warn">⚠️ The recommended price is at the {end} end of the historical price range. '
                "Further validation is recommended before deployment.</div>",
                unsafe_allow_html=True,
            )
    return d


def render_main_chart(opt: dict, d: dict, price_range: tuple) -> None:
    section("Price vs Expected Profit")
    scen, cur, reco = opt["scenarios"], opt["current"], d["reco"]
    st.plotly_chart(
        line_chart(
            scen["Price"], scen["Profit"], "", "Expected profit", "$",
            (cur["price"], cur["profit"]), (reco["price"], reco["profit"]),
            series_name="Expected profit",
        ),
        **STRETCH,
    )
    note(f"{N_SCENARIOS} prices simulated between {money(price_range[0])} and {money(price_range[1])} "
         "(the product's historical 5th–95th percentile price range).")


def render_advanced(opt, d, result, live_sim, shown_cost, shown_elasticity) -> None:
    section("Advanced Analysis")
    with st.expander("Show detailed analysis", expanded=False):
        tab_demand, tab_sens, tab_scen, tab_info = st.tabs(
            ["Demand Analysis", "Sensitivity Analysis", "All Pricing Scenarios", "Model Information & Limitations"]
        )

        # --- Demand analysis ---
        with tab_demand:
            src = result if result is not None else {"snapshot": live_sim["snapshot"],
                                                      "model_demand": live_sim["current"]["model_demand"]}
            rows = [
                ("Historical Average Demand", f"{src['snapshot']['avg_demand']:,.1f}",
                 "Average of recorded demand for this store and product"),
                ("Latest Recorded Demand", f"{src['snapshot']['latest_demand']:,.0f}",
                 "Demand in the most recent record"),
                ("Current Model-Predicted Demand", f"{src['model_demand']:,.1f}",
                 "Model forecast at the current price (baseline)"),
            ]
            if opt is not None:
                rows.append(("Adjusted Predicted Demand (recommended)", f"{d['reco']['demand']:,.1f}",
                             "Baseline adjusted by the price sensitivity assumption"))
            st.dataframe(pd.DataFrame(rows, columns=["Measure", "Units", "What it is"]),
                         hide_index=True, **STRETCH)
            if opt is None:
                st.info("Click **Optimize Price** to see adjusted demand across prices.")
            else:
                scen, cur = opt["scenarios"], opt["current"]
                raw = go.Scatter(
                    x=scen["Price"], y=scen["Model_Predicted_Demand"], mode="lines",
                    name="Raw model prediction (not used)", line=dict(color=MUTED, width=1.5, dash="dot"),
                    hovertemplate="Price: $%{x:,.2f}<br>Raw model prediction: %{y:,.2f}<extra></extra>",
                )
                st.plotly_chart(
                    line_chart(scen["Price"], scen["Adjusted_Demand"], "Price vs Adjusted Predicted Demand",
                               "Predicted demand", "", (cur["price"], cur["demand"]),
                               (d["reco"]["price"], d["reco"]["demand"]),
                               series_name="Adjusted predicted demand", extra_trace=raw),
                    **STRETCH,
                )
                note(f"Adjusted Demand = Current Predicted Demand × (Price / Current Price) ^ Elasticity, "
                     f"with elasticity {shown_elasticity:.1f} (scenario assumption). The raw model prediction "
                     "is shown only for transparency.")

        # --- Sensitivity ---
        with tab_sens:
            if opt is None:
                st.info("Click **Optimize Price** to see how unit cost affects the recommendation.")
            else:
                note("Unit cost always changes profit; it may or may not change the recommended price.")
                st.dataframe(
                    opt["sensitivity"].style.format({
                        "Unit Cost": "${:,.2f}", "Recommended Price": "${:,.2f}",
                        "Predicted Demand (adjusted)": "{:,.1f}", "Expected Profit": "${:,.2f}",
                    }),
                    hide_index=True, **STRETCH,
                )
                st.plotly_chart(sensitivity_chart(opt["sensitivity"], shown_cost), **STRETCH)

        # --- Scenarios ---
        with tab_scen:
            if opt is None:
                st.info("Click **Optimize Price** to generate the pricing scenarios.")
            else:
                table = opt["scenarios"][
                    ["Price", "Model_Predicted_Demand", "Adjusted_Demand", "Revenue", "Profit"]
                ].rename(columns={
                    "Model_Predicted_Demand": "Model-Predicted Demand", "Adjusted_Demand": "Adjusted Demand",
                    "Revenue": "Expected Revenue", "Profit": "Expected Profit",
                })
                st.dataframe(
                    table.style.format({
                        "Price": "${:,.2f}", "Model-Predicted Demand": "{:,.2f}", "Adjusted Demand": "{:,.2f}",
                        "Expected Revenue": "${:,.2f}", "Expected Profit": "${:,.2f}",
                    }),
                    hide_index=True, **STRETCH,
                )

        # --- Model info ---
        with tab_info:
            st.markdown(
                f"**Model:** HistGradientBoosting demand forecast · MAE {MODEL_METRICS['MAE']:.2f} · "
                f"RMSE {MODEL_METRICS['RMSE']:.2f} · R² {MODEL_METRICS['R²']:.3f}"
            )
            st.markdown(
                """
- The ML model is predictive, not causal. Historical data does not identify a causal price elasticity.
- Elasticity is a configurable business scenario assumption, and results depend directly on it.
- Unit cost may be a synthetic/demo input when no real cost data is available.
- Product and store names are realistic demo labels mapped to existing IDs.
- The recommended price is the best simulated scenario within the selected historical price range, not a guaranteed optimum.
- Validate results with real business data (for example, controlled price tests) before deployment.
                """
            )


# ==========================================================
# 8. Main
# ==========================================================


def main() -> None:
    data, model = load_assets()
    meta = load_metadata(data, str(STORE_META_PATH), str(PRODUCT_META_PATH))

    render_header()
    store, product, default_cost, store_label, product_label = render_selection(data, meta)

    try:
        sim_result = run_simulation(data, model, store, product)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not prepare the selected product: {exc}")
        st.stop()

    render_situation(sim_result["snapshot"])
    unit_cost, elasticity = render_settings(default_cost, product)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    _, button_col, _ = st.columns([1, 1.2, 1])
    clicked = button_col.button("Optimize Price")

    inputs_key = (store, product, unit_cost, elasticity)
    if clicked:
        try:
            with st.spinner("Evaluating pricing scenarios..."):
                st.session_state["result"] = {
                    "key": inputs_key,
                    "labels": (store_label, product_label),
                    "opt": optimize(sim_result, unit_cost, elasticity),
                    "snapshot": sim_result["snapshot"],
                    "model_demand": sim_result["current"]["model_demand"],
                    "price_range": sim_result["price_range"],
                }
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimization failed: {exc}")
            st.stop()

    result = st.session_state.get("result")
    opt = d = None
    shown_cost, shown_elasticity = unit_cost, elasticity
    if result is not None:
        if result["key"] != inputs_key:
            st.warning(
                f"Showing results for {result['labels'][0]} / {result['labels'][1]}. "
                "Inputs have changed. Click **Optimize Price** to refresh."
            )
        opt = result["opt"]
        _, _, shown_cost, shown_elasticity = result["key"]
        d = render_recommendation(opt)
        render_main_chart(opt, d, result["price_range"])

    render_advanced(opt, d, result, sim_result, shown_cost, shown_elasticity)


main()
