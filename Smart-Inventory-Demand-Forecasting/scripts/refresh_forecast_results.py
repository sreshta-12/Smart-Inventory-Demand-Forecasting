from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_data
from src.db import clear_forecast_results, connect, latest_forecast_run
from src.pipeline import run_batch_predict
from src.predict import load_artifacts
from src.reorder import DEFAULT_LEAD_TIME_DAYS

def main() -> None:
    df = load_data()
    model, scaler, feature_cols, scaled_columns = load_artifacts()
    conn = connect()
    try:
        clear_forecast_results(conn)
    finally:
        conn.close()
    _, _ = run_batch_predict(df, model, scaler, feature_cols, scaled_columns, lead_time_days=DEFAULT_LEAD_TIME_DAYS, persist_db=True)
    conn = connect()
    try:
        out = latest_forecast_run(conn)
    finally:
        conn.close()
    csv_path = ROOT / 'data' / 'forecast_results_latest.csv'
    out.to_csv(csv_path, index=False)
    runs = out['run_at'].unique().tolist() if not out.empty and 'run_at' in out.columns else []
    print(f'forecast_results: {len(out)} rows, run_at={runs}')
    print(f'Wrote {csv_path}')
if __name__ == '__main__':
    main()
