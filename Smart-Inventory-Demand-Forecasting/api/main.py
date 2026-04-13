from __future__ import annotations
import io
import sys
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.columns import DATE, INVENTORY_LEVEL, PREDICTED_DEMAND, PRODUCT_ID, REORDER_ALERT, REORDER_POINT, STORE_ID, UNITS_SOLD, ensure_training_columns
from src.data_loader import CSV_SIMPLE, load_data
from src.db import connect, latest_forecast_run, table_exists
from src.forecast_config import FORECAST_HORIZON_DAYS
from src.pipeline import ACTUAL_NEXT_DAY, run_batch_predict
from src.predict import load_artifacts
app = FastAPI(title='Inventory Demand Forecasting API', version='1.0.0')
_model = _scaler = _feat = _scaled = None

def _artifacts():
    global _model, _scaler, _feat, _scaled
    if _model is None:
        _model, _scaler, _feat, _scaled = load_artifacts()
    return (_model, _scaler, _feat, _scaled)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/forecasts/latest')
def forecasts_latest():
    db = ROOT / 'data' / 'inventory.db'
    if not db.exists():
        return {'rows': [], 'message': 'no database'}
    conn = connect(db)
    try:
        if not table_exists(conn, 'forecast_results'):
            return {'rows': [], 'message': 'no forecast_results table'}
        df = latest_forecast_run(conn)
        return {'rows': df.to_dict(orient='records'), 'count': len(df)}
    finally:
        conn.close()

@app.post('/predict/upload')
async def predict_upload(file: UploadFile=File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.StringIO(raw.decode('utf-8')))
    except Exception as e:
        raise HTTPException(400, f'Invalid CSV: {e}') from e
    df = ensure_training_columns(df)
    if STORE_ID not in df.columns:
        raise HTTPException(400, 'CSV must include store_id or Store ID')
    m, sc, fc, sd = _artifacts()
    scored, _full = run_batch_predict(df, m, sc, fc, sd, persist_db=False)
    if scored.empty:
        raise HTTPException(400, 'No rows after feature engineering (need lag history).')
    cols = [STORE_ID, PRODUCT_ID, DATE, INVENTORY_LEVEL, UNITS_SOLD, ACTUAL_NEXT_DAY, PREDICTED_DEMAND, REORDER_POINT, REORDER_ALERT]
    cols = [c for c in cols if c in scored.columns]
    out = scored[cols].copy()
    out[DATE] = pd.to_datetime(out[DATE], errors='coerce').dt.strftime('%Y-%m-%d')
    return {'count': len(out), 'forecast_horizon_days': FORECAST_HORIZON_DAYS, 'prediction': 'next_day_units_sold_per_row', 'rows': out.to_dict(orient='records')}

@app.post('/predict/default-dataset')
def predict_default_dataset():
    if not CSV_SIMPLE.exists():
        raise HTTPException(404, 'retail_simple.csv not found')
    df = load_data()
    m, sc, fc, sd = _artifacts()
    scored, _full = run_batch_predict(df, m, sc, fc, sd, persist_db=False)
    if scored.empty:
        raise HTTPException(500, 'Scoring produced no rows.')
    tail = scored.tail(500)
    tail = tail.assign(**{DATE: pd.to_datetime(tail[DATE], errors='coerce').dt.strftime('%Y-%m-%d')})
    return {'count': len(scored), 'sample': tail.to_dict(orient='records')}
