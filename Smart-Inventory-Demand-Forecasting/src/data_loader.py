import pandas as pd
from pathlib import Path

from .columns import DATE, ensure_training_columns
from .db import SQLITE_PATH, connect, load_flat_training_frame, table_exists

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
DATA_PATH = DATA_DIR / 'retail_store_inventory.csv'
CSV_SIMPLE = DATA_DIR / 'retail_simple.csv'

LEGACY_INVENTORY_SQL = 'SELECT * FROM inventory ORDER BY "Date", "Product ID"'


def load_data():
    if not SQLITE_PATH.exists():
        return _load_from_csv()

    conn = connect()
    try:
        if table_exists(conn, 'sales'):
            frame = load_flat_training_frame(conn)
            frame = frame.dropna(subset=[DATE])
            return frame

        frame = pd.read_sql_query(LEGACY_INVENTORY_SQL, conn)
        frame = ensure_training_columns(frame)
        frame[DATE] = pd.to_datetime(frame[DATE], errors='coerce')
        frame = frame.dropna(subset=[DATE])
        return frame
    finally:
        conn.close()


def _load_from_csv():
    if CSV_SIMPLE.exists():
        path = CSV_SIMPLE
    else:
        path = DATA_PATH
    frame = pd.read_csv(path)
    frame = ensure_training_columns(frame)
    frame[DATE] = pd.to_datetime(frame[DATE], errors='coerce')
    frame = frame.dropna(subset=[DATE])
    return frame
