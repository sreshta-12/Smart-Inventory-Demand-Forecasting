import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .columns import DATE, PRODUCT_ID, STORE_ID, TARGET_NEXT_DAY, UNITS_SOLD
from .data_loader import load_data
from .feature_engineering import engineer_features
from .forecast_config import FORECAST_HORIZON_DAYS

MODEL_DIR = Path('models')
TRAIN_END_EXCLUSIVE = '2023-01-01'


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    raw = load_data()
    raw = raw.sort_values([STORE_ID, PRODUCT_ID, DATE])
    raw = engineer_features(raw)
    raw[TARGET_NEXT_DAY] = raw.groupby([STORE_ID, PRODUCT_ID], group_keys=False)[UNITS_SOLD].shift(-1)
    raw = raw.dropna()
    raw = raw.dropna(subset=[TARGET_NEXT_DAY])

    cutoff = pd.Timestamp(TRAIN_END_EXCLUSIVE)
    train_df = raw.loc[raw[DATE] < cutoff].copy()
    test_df = raw.loc[raw[DATE] >= cutoff].copy()

    if train_df.empty or test_df.empty:
        cut_i = int(len(raw) * 0.8)
        train_df = raw.iloc[:cut_i].copy()
        test_df = raw.iloc[cut_i:].copy()
        split_note = 'fallback: 80/20 temporal order (insufficient rows for 2023 cut)'
    else:
        split_note = f'train: {DATE} < {TRAIN_END_EXCLUSIVE}; test: {DATE} >= {TRAIN_END_EXCLUSIVE}'

    drop_y = [TARGET_NEXT_DAY, UNITS_SOLD, DATE]
    X_train = train_df.drop(columns=drop_y)
    y_train = train_df[TARGET_NEXT_DAY]
    X_test = test_df.drop(columns=drop_y)
    y_test = test_df[TARGET_NEXT_DAY]

    combined = pd.concat([X_train, X_test])
    combined = pd.get_dummies(combined)
    X_train = combined.iloc[: len(X_train)].copy()
    X_test = combined.iloc[len(X_train) :].copy()

    feature_cols = X_train.columns.tolist()
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    joblib.dump(feature_cols, MODEL_DIR / 'feature_columns.joblib')
    joblib.dump(scaler, MODEL_DIR / 'scaler.joblib')
    joblib.dump(num_cols, MODEL_DIR / 'scaled_columns.joblib')

    candidates = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }
    results = {}

    print('\nModel comparison:')
    print(f'Split: {split_note}\n')

    for label, est in candidates.items():
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results[label] = {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)}
        print(f'{label:20s} R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')

    best_name = max(results, key=lambda k: results[k]['r2'])
    best = candidates[best_name]
    print('\nBest model:', best_name)
    joblib.dump(best, MODEL_DIR / 'final_model.joblib')
    print('Saved final_model.joblib')

    rationale = (
        f'Target is next calendar day units_sold per store–product (one-step-ahead). '
        f'Features use lagged units_sold and 7-day mean/std using only history through day T '
        f'(shifted rolls, no same-day leakage). Longer horizons use the same model recursively '
        f'({FORECAST_HORIZON_DAYS} days in the app). Trees often help on count data; '
        f'linear models are a strong baseline with clear lag structure.'
    )

    ranking = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)
    payload = {
        'models': results,
        'best_model': best_name,
        'best_metrics': results[best_name],
        'ranking': ranking,
        'deployed_model': best_name,
        'forecast_system': {
            'training_target': 'next_day_units_sold',
            'label_definition': 'groupby(store,product)[units_sold].shift(-1) — exactly 1 day ahead',
            'not_used': ['shift(-7)', 'direct_multi_day_target', '7_day_y_for_training'],
            'horizon_days_recursive': FORECAST_HORIZON_DAYS,
            'inference': 'same model applied recursively for 7-day chart; test metrics are 1-step only',
        },
        'validation': {
            'method': 'time_series_holdout',
            'split_rule': split_note,
            'cutoff_date': TRAIN_END_EXCLUSIVE,
            'metrics_scope': 'one_step_ahead_only_r2_rmse_mae',
            'metrics_not': 'recursive_7_day_path',
        },
        'model_choice_rationale': rationale,
    }

    out_path = MODEL_DIR / 'model_comparison_metrics.json'
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)


if __name__ == '__main__':
    main()
