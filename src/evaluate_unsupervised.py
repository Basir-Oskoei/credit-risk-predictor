import json
import numpy as np
import pandas as pd
from joblib import load
from .config import PROCESSED_DIR, MODELS_DIR

def main():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    model = load(MODELS_DIR / "credit_model.joblib")

    # IsolationForest: decision_function → higher = more normal; lower = more anomalous
    scores = model.decision_function(X_test)
    preds = model.predict(X_test)   # 1 = normal, -1 = anomaly

    # Convert to a "risk probability"-ish value for reporting (min-max normalize inverted score)
    inv = -scores
    risk = (inv - inv.min()) / (inv.max() - inv.min() + 1e-12)

    metrics = {
        "n_test": int(len(X_test)),
        "anomaly_rate": float((preds == -1).mean()),
        "score_mean": float(scores.mean()),
        "score_p10": float(np.percentile(scores, 10)),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
    }
    print(json.dumps(metrics, indent=2))
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved metrics to models/metrics.json")

if __name__ == "__main__":
    main()
