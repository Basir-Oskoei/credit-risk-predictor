import json
import pandas as pd
from joblib import load
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
)
from .config import PROCESSED_DIR, MODELS_DIR, TARGET_COL

def main():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET_COL]

    model = load(MODELS_DIR / "credit_model.joblib")
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, digits=4)
    }
    print(json.dumps(metrics, indent=2))
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved metrics to models/metrics.json")

if __name__ == "__main__":
    main()
