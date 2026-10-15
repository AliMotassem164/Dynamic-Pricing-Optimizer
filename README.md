# Dynamic Pricing Optimizer

**Demand Forecasting · Pricing Scenarios · Revenue & Profit Analysis**

A data-driven decision-support application that helps retailers explore pricing strategies by forecasting product demand, simulating alternative prices, and comparing expected business outcomes. It combines a trained machine-learning pipeline with transparent pricing assumptions, business guardrails, and an interactive Streamlit interface.

> **Note:** This is a decision-support prototype, not an autonomous pricing system. Demand predictions are predictive rather than causal, and scenario results depend on assumptions and input-data quality.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Workflow](#workflow)
- [Machine Learning](#machine-learning)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run the App](#run-the-app)
- [Model Evaluation](#model-evaluation)
- [Limitations](#limitations)
- [Technologies](#technologies)

## Overview

Retail pricing involves balancing demand, revenue, costs, competitor prices, inventory, discounts, and promotions. This project provides a workflow to inspect and validate data, prepare model inputs, simulate candidate prices, and review pricing recommendations.

The application is designed to work with uploaded datasets rather than relying on hardcoded product or store choices.

## Features

- Interactive Streamlit application.
- Dataset inspection, validation, and column mapping.
- Demand forecasting using a saved scikit-learn pipeline.
- Price scenario simulation and estimated-demand comparison.
- Revenue and profit analysis when unit-cost data is available.
- Pricing guardrails and recommendation logic.
- Cost sensitivity analysis and reusable Python modules.
- Jupyter notebooks covering EDA, feature engineering, modeling, and price optimization.

## Workflow

1. Upload a CSV or supported spreadsheet.
2. Inspect the dataset and map columns.
3. Validate and prepare the input data.
4. Forecast demand with the trained model.
5. Simulate candidate prices using explicit pricing assumptions.
6. Compare estimated demand, revenue, and—when unit cost is supplied—profit.
7. Review the recommendation and business guardrails before making a decision.

## Machine Learning

The repository includes a saved **HistGradientBoosting** demand-forecasting pipeline. Its metadata identifies `Demand` as the target and records the feature contract, time-based split, and evaluation metrics.

Features include price, discounts, promotions, competitor pricing, inventory, calendar attributes, and historical demand features. Lagged and rolling features must use only information available at prediction time to avoid data leakage.

**Important distinction:** the forecasting model predicts demand; it does not independently establish the causal effect of changing a price. The pricing engine uses explicit elasticity assumptions for counterfactual scenarios and distinguishes assumption-based results from reliable historical estimates. Observational elasticity should not be treated as causal evidence.

## Dataset

Data and generated artifacts are stored in `Data/`.

| File | Purpose |
|---|---|
| `demand_forecasting.csv` | Raw demand-forecasting dataset |
| `demand_forecasting_engineered.csv` | Dataset with engineered features |
| `demand_predictions.csv` | Demand prediction output |
| `price_optimization_results.csv` | Example pricing scenario results |
| `product_metadata.csv` | Product metadata, including example unit costs |
| `store_metadata.csv` | Store metadata |
| `cost_sensitivity_analysis.csv` | Cost sensitivity output |
| `random_forest_feature_importance.csv` | Feature-importance artifact |

The saved model metadata reports **75,300 rows** in the engineered dataset: **60,200 training rows** and **15,100 test rows**. The recorded chronological split uses training data from 2022-01-08 to 2023-09-01 and test data from 2023-09-02 to 2024-01-30.

## Project Structure

```text
Dynamic-Pricing-Optimizer/
├── app.py
├── Data/
├── Models/
│   ├── demand_forecasting_model.pkl
│   └── model_metadata.json
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Modeling.ipynb
│   └── Price_Optimization.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_manager.py
│   ├── pricing.py
│   ├── utils.py
│   └── validator.py
└── tests/
    └── refactoring_test.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AliMotassem164/Dynamic-Pricing-Optimizer.git
cd Dynamic-Pricing-Optimizer
```

### 2. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install core dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn openpyxl joblib
```

The repository does not currently include a dependency lock or `requirements.txt`. Spreadsheet formats beyond those supported by the installed readers may require additional optional packages. Pin tested dependency versions when reproducing the environment.

## Run the App

From the repository root:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Use Your Own Dataset

Upload your CSV or supported spreadsheet and follow the app's inspection, mapping, and validation steps. Depending on the analysis, useful columns include product/store identifiers, date, price, demand or sales history, discounts, promotions, inventory, and competitor price. **Unit cost is needed for profit calculations**; without it, the application should not invent a cost or present profit estimates as factual.

Check validation messages and ensure lagged/rolling features are calculated from past observations only.

## Model Evaluation

The checked-in metadata for the saved HistGradientBoosting model reports:

| Metric | Recorded value |
|---|---:|
| Mean Absolute Error (MAE) | 10.78 |
| Root Mean Squared Error (RMSE) | 14.28 |
| R² | 0.895 |
| Time-based validation mean MAE | 12.45 ± 1.34 |
| Previous-demand baseline MAE | 40.14 |

These values describe the recorded model run and time split; they do not guarantee performance on new products, stores, periods, or external datasets. MAE and RMSE are expressed in the target's units.

## Limitations

- **Forecasting is not causal inference.** Predictions alone do not prove how demand will change after a price intervention.
- Scenario results depend on elasticity assumptions, guardrails, and data quality.
- Profit analysis requires reliable unit-cost data.
- Unseen products, stores, or data distributions may reduce model reliability.
- Controlled experiments and ongoing monitoring are recommended before operational use.
- Only load pickle-based model artifacts from trusted sources; deserialization can execute code.

## Technologies

Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly · Jupyter Notebook

## Author

**Ali Motassem** · [GitHub @AliMotassem164](https://github.com/AliMotassem164)
