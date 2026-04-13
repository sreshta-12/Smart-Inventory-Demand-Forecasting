from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import (
    DATE,
    INVENTORY_LEVEL,
    PREDICTED_DEMAND,
    PRODUCT_ID,
    REORDER_ALERT,
    REORDER_POINT,
    STORE_ID,
    UNITS_SOLD,
    ensure_training_columns,
)
from .db import connect, write_forecast_batch
from .forecast_config import FORECAST_HORIZON_DAYS
from .forecast_future import forecast_group_n_days
from .predict import load_artifacts, preprocess_for_prediction
from .reorder import DEFAULT_LEAD_TIME_DAYS, demand_std_by_group, reorder_point

ACTUAL_NEXT_DAY = 'actual_next_day_units'
PERSIST_FORECAST_MAX_GROUPS = 100


def _rows_for_db_batch(
    out: pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
    scaled_columns: list | None,
    *,
    lead_time_days: float,
    n_days: int,
    max_groups: int,
) -> pd.DataFrame:
    if INVENTORY_LEVEL in out.columns:
        inv_col = INVENTORY_LEVEL
    else:
        inv_col = None

    groups = list(out.groupby([STORE_ID, PRODUCT_ID], sort=False))
    groups.sort(key=lambda x: len(x[1]), reverse=True)
    groups = groups[:max_groups]

    records = []
    for _k, g in groups:
        g = g.sort_values(DATE)
        if len(g) < 35:
            continue
        preds = forecast_group_n_days(g, n_days, model, scaler, feature_cols, scaled_columns)
        if preds is None or len(preds) < n_days:
            continue

        std_series = demand_std_by_group(g)
        last = g.iloc[-1]
        if len(std_series):
            last_std = float(std_series.iloc[-1])
        else:
            last_std = float('nan')
        if not np.isfinite(last_std):
            last_std = None

        last_date = pd.Timestamp(g[DATE].max()).normalize()
        if inv_col is not None and pd.notna(last.get(inv_col)):
            inv0 = float(last[inv_col])
        else:
            inv0 = 0.0

        for k in range(1, n_days + 1):
            mu = int(max(0, round(float(preds[k - 1]))))
            used = sum(max(0.0, float(preds[j])) for j in range(k - 1))
            inv_start = max(0.0, inv0 - used)
            rop_val = reorder_point(float(mu), lead_time_days=lead_time_days, demand_std_daily=last_std)
            rop_i = int(max(0, round(float(rop_val))))
            if inv_start < rop_i:
                alert = 'Reorder Needed'
            else:
                alert = 'OK'
            tgt = (last_date + pd.Timedelta(days=k)).strftime('%Y-%m-%d')
            records.append({
                'target_date': tgt,
                'store_id': str(last[STORE_ID]),
                'product_id': str(last[PRODUCT_ID]),
                'predicted_demand': float(mu),
                'reorder_point': rop_i,
                'reorder_alert': alert,
                'inventory_at_run': int(round(inv_start)),
            })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def run_batch_predict(
    df: pd.DataFrame,
    model=None,
    scaler=None,
    feature_cols=None,
    scaled_columns=None,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
    persist_db: bool = False,
):
    if model is None:
        model, scaler, feature_cols, scaled_columns = load_artifacts()

    d = ensure_training_columns(df.copy())
    d[DATE] = pd.to_datetime(d[DATE], errors='coerce')
    d = d.dropna(subset=[DATE])
    d = d.sort_values([STORE_ID, PRODUCT_ID, DATE])
    d[ACTUAL_NEXT_DAY] = d.groupby([STORE_ID, PRODUCT_ID], group_keys=False)[UNITS_SOLD].shift(-1)

    X_ready = preprocess_for_prediction(d, feature_cols, scaler, scaled_columns=scaled_columns)
    if X_ready.empty:
        return (pd.DataFrame(), d)

    raw_preds = model.predict(X_ready)
    out = d.copy()
    out[PREDICTED_DEMAND] = np.nan
    out.loc[X_ready.index, PREDICTED_DEMAND] = np.maximum(0, np.rint(raw_preds)).astype(int)

    std_series = demand_std_by_group(out)
    rop_series = reorder_point(
        out.loc[X_ready.index, PREDICTED_DEMAND],
        lead_time_days=lead_time_days,
        demand_std_daily=std_series.reindex(X_ready.index),
    )
    if isinstance(rop_series, pd.Series):
        out.loc[X_ready.index, REORDER_POINT] = np.maximum(0, np.rint(rop_series.values)).astype(int)
    else:
        out.loc[X_ready.index, REORDER_POINT] = int(max(0, round(float(rop_series))))

    if INVENTORY_LEVEL in out.columns:
        inv_vals = out.loc[X_ready.index, INVENTORY_LEVEL].values
    else:
        inv_vals = np.zeros(len(X_ready))

    rp_vals = out.loc[X_ready.index, REORDER_POINT].values
    out.loc[X_ready.index, REORDER_ALERT] = np.where(inv_vals < rp_vals, 'Reorder Needed', 'OK')

    if persist_db:
        conn = connect()
        try:
            batch = _rows_for_db_batch(
                out,
                model,
                scaler,
                feature_cols,
                scaled_columns,
                lead_time_days=lead_time_days,
                n_days=FORECAST_HORIZON_DAYS,
                max_groups=PERSIST_FORECAST_MAX_GROUPS,
            )
            if not batch.empty:
                write_forecast_batch(conn, batch)
        finally:
            conn.close()

    return (out.loc[X_ready.index], out)
