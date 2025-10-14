import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .config import (
    RAW_DATA, TARGET_COL, ID_COL, TEST_SIZE, RANDOM_STATE,
    USE_SMOTE, MODEL_TYPE, MODELS_DIR, PROCESSED_DIR
)
from .preprocess import build_preprocessor, save_preprocessor

def load_data():
    df = pd.read_csv(RAW_DATA)
    if ID_COL and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # If target is text labels, map them
    if df[TARGET_COL].dtype == "object":
        mapping = { "good": 0, "bad": 1, "no": 0, "yes": 1 }
        df[TARGET_COL] = df[TARGET_COL].map(lambda x: mapping.get(str(x).lower(), x))

    # Ensure binary numeric target
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df

def build_model():
    pre = build_preprocessor()

    if MODEL_TYPE == "logreg":
        base_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        # RF tuned like an XGB-lite baseline
        base_clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if USE_SMOTE:
        model = ImbPipeline(steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", base_clf),
        ])
    else:
        from sklearn.pipeline import Pipeline
        model = Pipeline(steps=[
            ("pre", pre),
            ("clf", base_clf),
        ])

    return model, pre

def main():
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    model, pre = build_model()
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODELS_DIR / "credit_model.joblib")
    # Save preprocessor separately for schema introspection if needed
    save_preprocessor(pre)

    print("✅ Trained and saved: models/credit_model.joblib & models/preprocessor.joblib")

if __name__ == "__main__":
    main()
