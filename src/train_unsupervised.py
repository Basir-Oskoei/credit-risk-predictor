import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

from .config import (
    RAW_DATA, PROCESSED_DIR, MODELS_DIR, ID_COL,
    NUMERIC_COLS, CATEGORICAL_COLS, TEST_SIZE,
    RANDOM_STATE, CONTAMINATION
)

def load_data():
    df = pd.read_csv(RAW_DATA)
    if ID_COL and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df

def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERIC_COLS),
            ("cat", Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_COLS),
        ]
    )
    clf = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe

def main():
    df = load_data()
    X_train, X_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

    pipe = build_pipeline()
    pipe.fit(X_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipe, MODELS_DIR / "credit_model.joblib")
    print("✅ Trained unsupervised model → models/credit_model.joblib")

if __name__ == "__main__":
    main()
