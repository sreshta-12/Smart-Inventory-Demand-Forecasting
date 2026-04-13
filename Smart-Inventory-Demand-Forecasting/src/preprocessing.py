import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_numeric(df: pd.DataFrame, numerical_cols: list, scaler: StandardScaler, fit: bool = False):
    out = df.copy()
    out[numerical_cols] = out[numerical_cols].astype(np.float64)
    block = out[numerical_cols].to_numpy(dtype=np.float64, copy=True)
    if fit:
        out[numerical_cols] = scaler.fit_transform(block)
    else:
        out[numerical_cols] = scaler.transform(block)
    return out


def apply_scaler_inference(
    df: pd.DataFrame,
    feature_cols: list,
    scaler: StandardScaler,
    scaled_columns: list | None = None,
) -> pd.DataFrame:
    out = df[feature_cols].copy()
    names = getattr(scaler, 'feature_names_in_', None)
    if names is not None:
        cols_to_scale = [c for c in names if c in out.columns]
    elif scaled_columns is not None:
        cols_to_scale = [c for c in scaled_columns if c in out.columns]
    else:
        cols_to_scale = []
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c]):
                cols_to_scale.append(c)

    if not cols_to_scale:
        return out

    block = out[cols_to_scale].astype(np.float64)
    out[cols_to_scale] = scaler.transform(block)
    return out
