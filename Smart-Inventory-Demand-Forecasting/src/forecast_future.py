from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import DATE, PRODUCT_ID, STORE_ID, UNITS_SOLD
from .forecast_config import FORECAST_HORIZON_DAYS
from .feature_engineering import engineer_features
from .preprocessing import apply_scaler_inference

TARGET = UNITS_SOLD


def feature_matrix_last_row(working: pd.DataFrame, feature_cols: list, scaler, scaled_columns: list | None):
    enc = engineer_features(working)
    last = enc.iloc[[-1]]
    without_y = [c for c in (TARGET, DATE) if c in last.columns]
    check = last.drop(columns=without_y, errors='ignore')
    if check.isna().any().any():
        return None
    X = last.drop(columns=without_y, errors='ignore')
    try:
        X = pd.get_dummies(X, dtype=bool)
    except TypeError:
        X = pd.get_dummies(X)
    X = X.reindex(columns=feature_cols, fill_value=0)
    if X.isna().any().any():
        return None
    return apply_scaler_inference(X, feature_cols, scaler, scaled_columns=scaled_columns)


def forecast_group_n_days(
    group_df: pd.DataFrame,
    n: int,
    model,
    scaler,
    feature_cols: list,
    scaled_columns: list | None,
    max_history: int = 200,
) -> list[float] | None:
    g = group_df.sort_values(DATE).copy()
    g[DATE] = pd.to_datetime(g[DATE], errors='coerce')
    g = g.dropna(subset=[DATE])
    if len(g) < 35:
        return None
    if TARGET not in g.columns:
        return None

    g = g.tail(max_history).reset_index(drop=True)
    g = g.ffill().bfill()

    preds: list[float] = []
    working = g.copy()
    inv_col = 'inventory_level'

    for _ in range(n):
        Xt = feature_matrix_last_row(working, feature_cols, scaler, scaled_columns)
        if Xt is None or len(Xt) == 0:
            return None
        yhat = float(model.predict(Xt)[0])
        preds.append(yhat)

        new_row = working.iloc[-1].copy()
        new_row[DATE] = working[DATE].max() + pd.Timedelta(days=1)
        new_row[UNITS_SOLD] = max(0.0, yhat)

        if inv_col in new_row.index and inv_col in working.columns:
            try:
                prev_inv = float(working.iloc[-1][inv_col])
                new_row[inv_col] = max(0.0, prev_inv - yhat)
            except (TypeError, ValueError):
                pass

        working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
        working = working.ffill().bfill()

    return preds


def forecast_next_n_days_aggregate(
    df: pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
    scaled_columns: list | None,
    n: int = FORECAST_HORIZON_DAYS,
    max_groups: int = 100,
) -> pd.DataFrame:
    d = df.copy()
    d[DATE] = pd.to_datetime(d[DATE], errors='coerce')
    d = d.dropna(subset=[DATE])

    empty_row = {
        'horizon_day': list(range(1, n + 1)),
        'mean_predicted_demand': [np.nan] * n,
        'std_predicted_demand': [np.nan] * n,
        'n_series': [0] * n,
    }
    if STORE_ID not in d.columns or PRODUCT_ID not in d.columns:
        return pd.DataFrame(empty_row)

    groups = list(d.groupby([STORE_ID, PRODUCT_ID], sort=False))
    groups.sort(key=lambda x: len(x[1]), reverse=True)
    groups = groups[:max_groups]

    buckets: list[list[float]] = [[] for _ in range(n)]
    for _name, chunk in groups:
        seq = forecast_group_n_days(chunk, n, model, scaler, feature_cols, scaled_columns)
        if seq is None or len(seq) < n:
            continue
        for i, val in enumerate(seq):
            if val == val and np.isfinite(val):
                buckets[i].append(float(val))

    rows = []
    for i in range(n):
        vals = buckets[i]
        count = len(vals)
        if count == 0:
            mean_v = float('nan')
            std_v = float('nan')
        else:
            arr = np.asarray(vals, dtype=float)
            mean_v = float(np.mean(arr))
            if count > 1:
                std_v = float(np.std(arr, ddof=1))
            else:
                std_v = 0.0
        rows.append({
            'horizon_day': i + 1,
            'mean_predicted_demand': mean_v,
            'std_predicted_demand': std_v,
            'n_series': count,
        })
    return pd.DataFrame(rows)


def forecast_calendar_series(
    df: pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
    scaled_columns: list | None,
    n: int = FORECAST_HORIZON_DAYS,
    max_groups: int = 100,
) -> pd.DataFrame:
    agg = forecast_next_n_days_aggregate(
        df, model, scaler, feature_cols, scaled_columns, n=n, max_groups=max_groups
    )
    d = df.copy()
    d[DATE] = pd.to_datetime(d[DATE], errors='coerce')
    last_day = d[DATE].max()
    dates = []
    for h in agg['horizon_day']:
        day = last_day + pd.Timedelta(days=int(h))
        dates.append(day.strftime('%Y-%m-%d'))
    agg['forecast_date'] = dates
    return agg
