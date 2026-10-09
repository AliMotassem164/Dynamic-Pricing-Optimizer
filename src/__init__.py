"""Reusable core logic for the Dynamic Pricing Optimizer.

This package contains the production-style, UI-independent building blocks
extracted from the project's ML notebooks:

- :mod:`src.data_loader` — load CSV/Excel datasets and inspect their structure.
- :mod:`src.validator` — validate required columns, data types, prices,
  demand and unit cost before any modelling or optimisation work.
- :mod:`src.feature_engineering` — standardise column names and build the
  time, price, lag, inventory and interaction features used by the demand
  model (leakage-safe: history features are shifted before rolling).
- :mod:`src.model_manager` — persist, load and query the trained demand
  model, prepare model inputs and generate demand predictions.
- :mod:`src.pricing` — representative product selection, elasticity
  estimation, price scenario generation, demand simulation, revenue/profit
  calculation, price optimisation, guardrails and pricing decisions.
- :mod:`src.utils` — small shared helpers (safe division, numeric coercion,
  column discovery, percentage changes).

The package never imports Streamlit (or any other UI framework) so it can be
reused from scripts, notebooks, tests or any front end.
"""

from . import data_loader, feature_engineering, model_manager, pricing, utils, validator

__version__ = "1.0.0"

__all__ = [
    "data_loader",
    "validator",
    "feature_engineering",
    "model_manager",
    "pricing",
    "utils",
]
