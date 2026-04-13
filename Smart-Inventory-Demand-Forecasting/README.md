# Smart Inventory Demand Forecasting

An end-to-end pipeline that loads retail-style time series data, trains a regression model, and forecasts **demand for the next 7 days** per store–product. The **Dash** dashboard shows KPIs (including total predicted demand over that 7-day window), charts, reorder hints, and a **7-day forecast** table.

## What you need

- **Python 3.10+** (3.9 often works; use what your course recommends)
- Dependencies listed in `requirements.txt`

## Setup

From the project root (`Smart-Inventory-Demand-Forecasting-main`):

```bash
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

Pretrained model files are expected under `models/` (`final_model.joblib`, `scaler.joblib`, `feature_columns.joblib`, `scaled_columns.joblib`). If they are missing, run training (below) first.

## Run the dashboard

```bash
python app/app.py
```

Open the URL shown in the terminal (usually `http://127.0.0.1:8050`). Use **Load from database / default CSV** to refresh, pick a **product** from the dropdown, and explore the charts and tables.

## Train or retrain the model

From the project root:

```bash
python run_training.py
```

This reads data via `src/data_loader.py`, fits models, saves the best one to `models/final_model.joblib`, and writes `models/model_comparison_metrics.json`.

## Data and database

| Location | Role |
|----------|------|
| `data/retail_simple.csv` | Default flat CSV if no DB or as fallback |
| `data/retail_store_inventory.csv` | Alternate CSV if `retail_simple.csv` is absent |
| `data/inventory.db` | SQLite DB; if present and the `sales` table exists, training and load use the joined query in `src/db.py` |

You can **upload a CSV** in the app if it includes at least `store_id` and the columns expected after `ensure_training_columns` in `src/columns.py`.

## Optional HTTP API

A **FastAPI** app lives under `api/main.py`. If your assignment uses it, run it with Uvicorn (after installing requirements). Adjust host/port as needed:

```bash
uvicorn api.main:app --reload
```

## Project layout 

- `app/app.py` — Dash UI and callbacks
- `src/` — data load, DB, features, training, prediction, pipeline, reorder logic
- `models/` — saved model and scaler artifacts
- `data/` — CSV and SQLite database
- `run_training.py` — entry point for training
- `scripts/` — helper scripts (e.g. dataset or forecast exports), if present

## Notes

- Planning and reporting focus on the **next 7 days** of demand (aligned with `FORECAST_HORIZON_DAYS` in `src/forecast_config.py`).
- Reorder messaging combines predicted demand with a simple reorder-point style rule (see `src/reorder.py`).

