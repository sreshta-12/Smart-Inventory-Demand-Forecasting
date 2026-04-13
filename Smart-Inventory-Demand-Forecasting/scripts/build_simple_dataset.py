from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
import sys
sys.path.insert(0, str(ROOT))
from src.columns import DATE, INVENTORY_LEVEL, PRICE, PRODUCT_ID, PRODUCT_NAME, SEASONALITY, STORE_ID, UNITS_SOLD, ensure_training_columns
from src.db import SQLITE_PATH, rebuild_database_from_flat
PRODUCT_NAMES = {'P0001': 'Oil', 'P0002': 'Salt', 'P0003': 'Rice', 'P0004': 'Sugar', 'P0005': 'Wheat Flour', 'P0006': 'Pulses', 'P0007': 'Tea', 'P0008': 'Coffee', 'P0009': 'Biscuits', 'P0010': 'Milk Powder', 'P0011': 'Detergent', 'P0012': 'Soap', 'P0013': 'Toothpaste', 'P0014': 'Matches', 'P0015': 'Candles', 'P0016': 'Spices', 'P0017': 'Noodles', 'P0018': 'Honey', 'P0019': 'Jam', 'P0020': 'Pickle'}

def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return 'Winter'
    if m in (3, 4, 5):
        return 'Summer'
    if m in (6, 7, 8, 9):
        return 'Monsoon'
    return 'Autumn'

def load_source() -> pd.DataFrame:
    one_store = DATA_DIR / 'retail_one_store_20_products.csv'
    full = DATA_DIR / 'retail_store_inventory.csv'
    if one_store.exists():
        return pd.read_csv(one_store)
    if full.exists():
        df = pd.read_csv(full)
        df = df[df['Store ID'] == 'S001'].copy()
        df = df[df['Product ID'].isin(list(PRODUCT_NAMES.keys()))]
        return df
    raise FileNotFoundError(f'Need one of: {one_store}, {full}, or an existing retail_simple.csv (run main() fallback — see build script).')

def simplify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    pname = 'Product ID'
    df['_product_name'] = df['Product ID'].map(PRODUCT_NAMES).fillna(df['Product ID'])
    df[SEASONALITY] = df['Date'].dt.month.map(season_from_month)
    if 'Units Sold' not in df.columns:
        df['Units Sold'] = 0
    legacy = df[['Date', 'Store ID', pname, '_product_name', 'Inventory Level', 'Units Sold', 'Price', SEASONALITY]].copy()
    legacy = legacy.rename(columns={pname: PRODUCT_ID, '_product_name': PRODUCT_NAME})
    legacy = legacy.rename(columns={'Date': DATE, 'Store ID': STORE_ID, 'Inventory Level': INVENTORY_LEVEL, 'Units Sold': UNITS_SOLD, 'Price': PRICE})
    legacy = legacy.sort_values([DATE, PRODUCT_ID]).drop_duplicates(subset=[DATE, STORE_ID, PRODUCT_ID])
    legacy[DATE] = legacy[DATE].dt.strftime('%Y-%m-%d')
    return legacy.reset_index(drop=True)

def regenerate_persistent_units_sold(df: pd.DataFrame, seed: int=42) -> pd.DataFrame:
    out = df.copy()
    out[DATE] = pd.to_datetime(out[DATE], errors='coerce')
    out = out.dropna(subset=[DATE])
    out = out.sort_values([STORE_ID, PRODUCT_ID, DATE])
    rng = np.random.default_rng(seed)
    phi = 0.97
    noise_std = 3.5
    for (_, _), grp in out.groupby([STORE_ID, PRODUCT_ID], sort=False):
        idx = grp.index.to_numpy()
        n = len(idx)
        if n == 0:
            continue
        months = grp[DATE].dt.month.to_numpy()
        key = f'{grp[STORE_ID].iloc[0]}|{grp[PRODUCT_ID].iloc[0]}'.encode()
        h = int(hashlib.md5(key).hexdigest()[:8], 16)
        base = 55.0 + h % 140
        y = np.zeros(n, dtype=float)
        eps = rng.normal(0.0, noise_std, size=n)
        for i in range(n):
            seas = 12.0 * np.sin(2.0 * np.pi * months[i] / 12.0)
            if i == 0:
                y[i] = max(5.0, base + 0.25 * seas + eps[i])
            else:
                y[i] = phi * y[i - 1] + (1.0 - phi) * base + 0.08 * seas + eps[i]
        y = np.clip(np.round(y), 0, None).astype(int)
        out.loc[idx, UNITS_SOLD] = y
    out[DATE] = out[DATE].dt.strftime('%Y-%m-%d')
    return out.reset_index(drop=True)

def main() -> None:
    csv_path = DATA_DIR / 'retail_simple.csv'
    full = DATA_DIR / 'retail_store_inventory.csv'
    one_store = DATA_DIR / 'retail_one_store_20_products.csv'
    if csv_path.exists() and (not full.exists()) and (not one_store.exists()):
        simple = ensure_training_columns(pd.read_csv(csv_path))
        for c in ('demand_forecast',):
            if c in simple.columns:
                simple = simple.drop(columns=[c])
        if 'category' in simple.columns:
            simple = simple.drop(columns=['category'], errors='ignore')
        if PRODUCT_NAME not in simple.columns:
            simple[PRODUCT_NAME] = simple[PRODUCT_ID].astype(str)
        simple = regenerate_persistent_units_sold(simple)
        simple.to_csv(csv_path, index=False)
        rebuild_database_from_flat(simple, SQLITE_PATH)
        print(f'Rebuilt {SQLITE_PATH} from existing {csv_path} ({len(simple)} rows).')
        print('Columns:', list(simple.columns))
        return
    raw = load_source()
    simple = simplify(raw)
    simple = regenerate_persistent_units_sold(simple)
    simple.to_csv(csv_path, index=False)
    rebuild_database_from_flat(simple, SQLITE_PATH)
    print(f'Wrote {csv_path} ({len(simple)} rows, {simple[PRODUCT_ID].nunique()} products)')
    print(f'Wrote {SQLITE_PATH} (normalized schema: stores, products, sales, inventory_daily)')
    print('Columns:', list(simple.columns))
if __name__ == '__main__':
    main()
