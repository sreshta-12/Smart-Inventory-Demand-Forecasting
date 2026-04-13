from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import DATE, PRODUCT_ID, STORE_ID, UNITS_SOLD

DEFAULT_LEAD_TIME_DAYS = 2.0
DEFAULT_SERVICE_Z = 1.28
SAFETY_STOCK_MAX_AS_FRACTION_OF_MU = 0.35


def demand_std_by_group(df: pd.DataFrame, window: int = 30) -> pd.Series:
    d = df.copy()
    key = [STORE_ID, PRODUCT_ID]
    d = d.sort_values([*key, DATE])
    if UNITS_SOLD not in d.columns:
        return pd.Series(np.nan, index=d.index)

    def rolling_std(s: pd.Series) -> pd.Series:
        return s.rolling(window=window, min_periods=3).std()

    return d.groupby(key, group_keys=False)[UNITS_SOLD].transform(rolling_std)


def reorder_point(
    predicted_daily_demand: float | pd.Series,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
    demand_std_daily: float | pd.Series | None = None,
    service_z: float = DEFAULT_SERVICE_Z,
) -> float | pd.Series:
    L = float(max(lead_time_days, 0.25))

    if isinstance(predicted_daily_demand, pd.Series):
        mu = pd.to_numeric(predicted_daily_demand, errors='coerce').clip(lower=0)
        base = mu * L
        if demand_std_daily is None:
            safety = 0.15 * mu * np.sqrt(L)
        else:
            sig = pd.to_numeric(demand_std_daily, errors='coerce')
            safety = service_z * sig * np.sqrt(L)
            safety = safety.fillna(0.15 * mu * np.sqrt(L))
        cap = SAFETY_STOCK_MAX_AS_FRACTION_OF_MU * mu
        safety = safety.clip(upper=cap)
        return base + safety

    mu = float(np.nan_to_num(pd.to_numeric(predicted_daily_demand, errors='coerce'), nan=0.0))
    mu = max(0.0, mu)
    base = mu * L
    sig_scalar = pd.to_numeric(demand_std_daily, errors='coerce') if demand_std_daily is not None else np.nan
    if demand_std_daily is None or pd.isna(sig_scalar):
        safety = 0.15 * mu * np.sqrt(L)
    else:
        sig = float(sig_scalar)
        safety = min(service_z * sig * np.sqrt(L), SAFETY_STOCK_MAX_AS_FRACTION_OF_MU * mu)
    return base + safety
