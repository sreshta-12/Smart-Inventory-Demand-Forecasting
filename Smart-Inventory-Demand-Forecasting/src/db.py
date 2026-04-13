from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .columns import DATE, INVENTORY_LEVEL, PRICE, PRODUCT_ID, PRODUCT_NAME, SEASONALITY, STORE_ID, UNITS_SOLD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
SQLITE_PATH = DATA_DIR / 'inventory.db'

DDL = """
CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    location TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT
);

CREATE TABLE IF NOT EXISTS sales (
    date TEXT NOT NULL,
    store_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    units_sold REAL,
    price REAL,
    seasonality TEXT,
    PRIMARY KEY (date, store_id, product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS inventory_daily (
    date TEXT NOT NULL,
    store_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    inventory_level REAL,
    PRIMARY KEY (date, store_id, product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS forecast_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    target_date TEXT NOT NULL,
    store_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    predicted_demand REAL,
    reorder_point REAL,
    reorder_alert TEXT,
    inventory_at_run REAL,
    UNIQUE (run_at, target_date, store_id, product_id)
);

CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_inv_date ON inventory_daily(date);
CREATE INDEX IF NOT EXISTS idx_fc_target ON forecast_results(target_date);
"""

JOINED_TRAINING_QUERY = """
    SELECT
        s.date AS date,
        s.store_id AS store_id,
        s.product_id AS product_id,
        p.product_name AS product_name,
        s.seasonality AS seasonality,
        i.inventory_level AS inventory_level,
        s.units_sold AS units_sold,
        s.price AS price
    FROM sales s
    JOIN products p ON p.product_id = s.product_id
    JOIN inventory_daily i
      ON i.date = s.date AND i.store_id = s.store_id AND i.product_id = s.product_id
    ORDER BY s.date, s.store_id, s.product_id
    """

LATEST_RUN_QUERY = """
    SELECT fr.* FROM forecast_results fr
    INNER JOIN (
        SELECT MAX(run_at) AS mx FROM forecast_results
    ) t ON fr.run_at = t.mx
    """

DROP_LEGACY = """
        PRAGMA foreign_keys = OFF;
        DROP TABLE IF EXISTS forecast_results;
        DROP TABLE IF EXISTS inventory_daily;
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS stores;
        DROP TABLE IF EXISTS inventory;
        PRAGMA foreign_keys = ON;
        """

FORECAST_WRITE_COLS = [
    'run_at', 'target_date', 'store_id', 'product_id',
    'predicted_demand', 'reorder_point', 'reorder_alert', 'inventory_at_run',
]


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


def clear_operational_tables(conn: sqlite3.Connection) -> None:
    for table in ('forecast_results', 'inventory_daily', 'sales', 'products', 'stores'):
        conn.execute(f'DELETE FROM {table}')
    conn.commit()


def clear_forecast_results(conn: sqlite3.Connection) -> None:
    conn.execute('DELETE FROM forecast_results')
    conn.commit()


def ingest_timeseries_dataframe(df: pd.DataFrame, conn: sqlite3.Connection, replace: bool = True) -> None:
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    d = d.dropna(subset=['date'])
    if replace:
        clear_operational_tables(conn)

    stores = d[[STORE_ID]].drop_duplicates()
    stores = stores.assign(location=lambda x: x[STORE_ID].astype(str))
    stores = stores.rename(columns={STORE_ID: 'store_id'})
    stores.to_sql('stores', conn, if_exists='append', index=False)

    prod_rows = d.sort_values(DATE).groupby(PRODUCT_ID, as_index=False).last()
    products = prod_rows[[PRODUCT_ID, PRODUCT_NAME]].copy()
    products = products.rename(columns={PRODUCT_ID: 'product_id', PRODUCT_NAME: 'product_name'})
    products.to_sql('products', conn, if_exists='append', index=False)

    sales = d[['date', STORE_ID, PRODUCT_ID, UNITS_SOLD, PRICE]].copy()
    if SEASONALITY in d.columns:
        sales['seasonality'] = d[SEASONALITY].astype(str)
    else:
        sales['seasonality'] = None
    sales = sales.rename(columns={
        STORE_ID: 'store_id',
        PRODUCT_ID: 'product_id',
        UNITS_SOLD: 'units_sold',
        PRICE: 'price',
    })
    sales.to_sql('sales', conn, if_exists='append', index=False)

    inv = d[['date', STORE_ID, PRODUCT_ID, INVENTORY_LEVEL]].copy()
    inv = inv.rename(columns={
        STORE_ID: 'store_id',
        PRODUCT_ID: 'product_id',
        INVENTORY_LEVEL: 'inventory_level',
    })
    inv.to_sql('inventory_daily', conn, if_exists='append', index=False)
    conn.commit()


def load_flat_training_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(JOINED_TRAINING_QUERY, conn)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def write_forecast_batch(conn: sqlite3.Connection, rows: pd.DataFrame, run_at: str | None = None) -> None:
    clear_forecast_results(conn)
    stamp = run_at or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    out = rows.copy()
    out.insert(0, 'run_at', stamp)
    out = out.rename(columns={'date': 'target_date'}, errors='ignore')
    if 'target_date' not in out.columns and 'date' in out.columns:
        out['target_date'] = out['date']
    if 'inventory_at_run' not in out.columns:
        out['inventory_at_run'] = None
    out[FORECAST_WRITE_COLS].to_sql('forecast_results', conn, if_exists='append', index=False)
    conn.commit()


def latest_forecast_run(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(LATEST_RUN_QUERY, conn)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def rebuild_database_from_flat(df: pd.DataFrame, db_path: Path | None = None) -> None:
    path = db_path or SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    conn.executescript(DROP_LEGACY)
    conn.commit()
    init_schema(conn)
    ingest_timeseries_dataframe(df, conn, replace=False)
    conn.close()
