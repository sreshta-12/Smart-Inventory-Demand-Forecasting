from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.db import rebuild_database_from_flat
from src.columns import DATE, INVENTORY_LEVEL, PRICE, PRODUCT_ID, STORE_ID, UNITS_SOLD
from src.data_loader import CSV_SIMPLE
from src.db import SQLITE_PATH

def main() -> None:
    if not CSV_SIMPLE.exists():
        raise SystemExit(f'Missing {CSV_SIMPLE}; run scripts/build_simple_dataset.py first.')
    df = pd.read_csv(CSV_SIMPLE)
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    df = df.dropna(subset=[DATE])
    mx = df[DATE].max()
    last = df[df[DATE] == mx].copy()
    if last.empty:
        raise SystemExit('No rows to extend.')
    nxt = mx + pd.Timedelta(days=1)
    last[DATE] = nxt
    rng = np.random.default_rng()
    noise_u = rng.uniform(0.82, 1.18, size=len(last))
    last[UNITS_SOLD] = np.maximum(0, np.round(last[UNITS_SOLD].values * noise_u)).astype(int)
    last[PRICE] = np.round(last[PRICE].values * rng.uniform(0.95, 1.05, size=len(last)), 2)
    sold = last[UNITS_SOLD].values
    prev_inv = last[INVENTORY_LEVEL].values
    restock = rng.integers(0, 45, size=len(last))
    last[INVENTORY_LEVEL] = np.maximum(0, prev_inv - sold + restock)
    out = pd.concat([df, last], ignore_index=True)
    out[DATE] = out[DATE].dt.strftime('%Y-%m-%d')
    out.to_csv(CSV_SIMPLE, index=False)
    rebuild_database_from_flat(out, SQLITE_PATH)
    print(f'Appended synthetic day {nxt.date()}; total rows={len(out)}')
    print('Rebuilt normalized DB; re-run training if you want the model to include the new history.')
if __name__ == '__main__':
    main()
