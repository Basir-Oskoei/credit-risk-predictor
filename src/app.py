import pandas as pd
import streamlit as st
from joblib import load

# --- handle both relative & direct runs ---
try:
    from .config import NUMERIC_COLS, CATEGORICAL_COLS, MODELS_DIR
except ImportError:
    from config import NUMERIC_COLS, CATEGORICAL_COLS, MODELS_DIR


# --- Streamlit page setup ---
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    return load(MODELS_DIR / "credit_model.joblib")

model = load_model()

st.title("💳 Credit Risk Predictor")
st.caption("Demo machine learning app for finance/data science portfolio")

with st.form("input_form"):
    st.subheader("Applicant Information")

    # Numeric inputs
    num_inputs = {}
    for col in NUMERIC_COLS:
        num_inputs[col] = st.number_input(
            col.replace("_", " ").title(),
            value=0.0,
            step=1.0,
            format="%.2f"
        )

    # Categorical inputs (free text for now)
    cat_inputs = {}
    for col in CATEGORICAL_COLS:
        cat_inputs[col] = st.text_input(
            col.replace("_", " ").title(),
            value="unknown"
        )

    submitted = st.form_submit_button("Predict Default Probability")

if submitted:
    # Combine into one dataframe
    payload = {**num_inputs, **cat_inputs}
    X = pd.DataFrame([payload])

    try:
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= 0.5)

        st.metric("Default Probability", f"{proba:.2%}")
        st.write("Prediction:", "**Default**" if pred == 1 else "**No Default**")

    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")

st.info("Note: This demo is for educational purposes only, not real credit decisions.")
