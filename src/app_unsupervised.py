import os
import pandas as pd
import streamlit as st
from joblib import load

# dual import for streamlit
try:
    from .config import NUMERIC_COLS, CATEGORICAL_COLS, MODELS_DIR
except ImportError:
    from config import NUMERIC_COLS, CATEGORICAL_COLS, MODELS_DIR

st.set_page_config(page_title="Credit Risk (Unsupervised)", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "credit_model.joblib"
    if not model_path.exists() or os.path.getsize(model_path) < 1024:
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run:  python -m src.train_unsupervised   then   python -m src.evaluate_unsupervised"
        )
    return load(model_path)

model = load_model()

st.title("💳 Credit Risk Scorer (Unsupervised)")
st.caption("Flags unusual applicants without needing labels (Isolation Forest).")

with st.form("input_form"):
    st.subheader("Applicant Information")

    num_inputs = {}
    for col in NUMERIC_COLS:
        num_inputs[col] = st.number_input(col, value=0.0, step=1.0, format="%.2f")

    cat_inputs = {}
    for col in CATEGORICAL_COLS:
        cat_inputs[col] = st.text_input(col, value="unknown")

    submitted = st.form_submit_button("Score Risk")

if submitted:
    X = pd.DataFrame([{**num_inputs, **cat_inputs}])

    # decision_function: higher is more normal; lower is more anomalous
    score = float(model.decision_function(X)[0])
    # convert to simple risk label using model's predict
    pred = int(model.predict(X)[0])  # 1 normal, -1 anomaly

    st.metric("Anomaly score (higher = safer)", f"{score:.4f}")
    if pred == -1:
        st.error("High Risk (anomalous applicant)")
    else:
        st.success("Low Risk (typical applicant)")

st.info("Unsupervised demo: no labels required. For hiring portfolios, a labeled model is still stronger.")
