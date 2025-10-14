import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from .config import NUMERIC_COLS, CATEGORICAL_COLS, MODELS_DIR

def build_preprocessor():
    numeric = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    categoric = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLS),
            ("cat", categoric, CATEGORICAL_COLS)
        ]
    )
    return pre

def save_preprocessor(preprocessor):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
