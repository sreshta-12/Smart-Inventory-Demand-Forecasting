import numpy as np
import pandas as pd

from .columns import DATE, PRODUCT_ID, STORE_ID, UNITS_SOLD


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE] = pd.to_datetime(out[DATE], errors='coerce')
    out = out.dropna(subset=[DATE])
    if UNITS_SOLD not in out.columns:
        raise ValueError('Need a units_sold column for feature engineering.')

    out = out.sort_values([STORE_ID, PRODUCT_ID, DATE])
    grp = [STORE_ID, PRODUCT_ID]
    by_sp = out.groupby(grp, group_keys=False)

    sold = by_sp[UNITS_SOLD]
    out['demand_lag1'] = sold.shift(1)
    out['demand_lag2'] = sold.shift(2)
    out['demand_lag7'] = sold.shift(7)
    out['demand_lag30'] = sold.shift(30)

    def roll_mean_shifted(s):
        return s.shift(1).rolling(7, min_periods=1).mean()

    def roll_std_shifted(s):
        return s.shift(1).rolling(7, min_periods=3).std()

    out['rolling_mean_7_units'] = by_sp[UNITS_SOLD].transform(roll_mean_shifted)
    out['rolling_std_7_units'] = by_sp[UNITS_SOLD].transform(roll_std_shifted)

    dt = out[DATE].dt
    out['month'] = dt.month
    out['day_of_year'] = dt.dayofyear
    two_pi = 2 * np.pi
    out['month_sin'] = np.sin(two_pi * out['month'] / 12.0)
    out['month_cos'] = np.cos(two_pi * out['month'] / 12.0)
    out['time_index'] = by_sp.cumcount()
    return out
