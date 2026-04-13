from pathlib import Path

import joblib
import pandas as pd

from .columns import DATE, TARGET, TARGET_NEXT_DAY
from .feature_engineering import engineer_features
from .preprocessing import apply_scaler_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / 'models'


def load_artifacts(models_dir: Path | None = None):
    folder = models_dir or MODEL_DIR
    clf = joblib.load(folder / 'final_model.joblib')
    scaler = joblib.load(folder / 'scaler.joblib')
    names = joblib.load(folder / 'feature_columns.joblib')
    scaled_path = folder / 'scaled_columns.joblib'
    if scaled_path.exists():
        scaled_cols = joblib.load(scaled_path)
    else:
        scaled_cols = None
    return clf, scaler, names, scaled_cols


def preprocess_for_prediction(df: pd.DataFrame, feature_cols: list, scaler, scaled_columns: list | None = None):
    wide = engineer_features(df)
    wide = wide.dropna()
    drop_me = [c for c in (TARGET_NEXT_DAY, TARGET, DATE) if c in wide.columns]
    X = wide.drop(columns=drop_me)
    try:
        X = pd.get_dummies(X, dtype=bool)
    except TypeError:
        X = pd.get_dummies(X)
    X = X.reindex(columns=feature_cols, fill_value=0)
    return apply_scaler_inference(X, feature_cols, scaler, scaled_columns=scaled_columns)


def predict_demand(df: pd.DataFrame, model=None, scaler=None, feature_cols=None, scaled_columns=None):
    if model is None or scaler is None or feature_cols is None:
        model, scaler, feature_cols, scaled_columns = load_artifacts()
    X = preprocess_for_prediction(df, feature_cols, scaler, scaled_columns=scaled_columns)
    return model.predict(X)
