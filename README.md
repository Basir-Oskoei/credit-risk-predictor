💳 Credit Risk Predictor (Finance DS Portfolio Project)

A clean, production-style demo of a credit risk scoring workflow using unsupervised anomaly detection.

Data → EDA → Preprocessing → Training → Evaluation → Streamlit App

Built with scikit-learn, deployed locally via Streamlit.

🚀 Features

Isolation Forest–based anomaly detection for unlabeled credit data

End-to-end modular ML pipeline (train, evaluate, serve)

Encodes categorical data + scales numeric fields automatically

Generates saved model artifacts (.joblib) + metrics JSON

Streamlit UI for interactive single-applicant scoring

Clean folder structure following DS best practices

🧾 Dataset

German Credit Data (UCI Statlog Project) — 1,000 anonymized credit applications from a German bank (1980s).
Each record includes demographic + financial info such as:

Feature	Description
Age	Applicant’s age (years)
Sex	Male / Female
Job	Employment level (0–3)
Housing	rent / own / free
Saving accounts	little / moderate / rich
Checking account	level of funds
Credit amount	Loan size requested
Duration	Repayment period (months)
Purpose	Reason for the loan

📚 Source: UCI ML Repository — German Credit Data

(This version omits the original Risk label — the model learns patterns without supervision.)

⚙️ Model Overview

Algorithm: IsolationForest
Goal: Detect unusual credit applicants (“high-risk anomalies”)
Logic: Outliers are isolated faster → flagged as riskier
Output:

Anomaly score (higher = safer)

Prediction: “Low Risk” or “High Risk”

📂 Project Structure
credit-risk-predictor/
│
├─ data/
│   ├─ raw/                  # raw dataset (german_credit.csv)
│   └─ processed/            # train/test splits
│
├─ models/                   # trained model + metrics
│   ├─ credit_model.joblib
│   └─ metrics.json
│
├─ src/
│   ├─ config.py
│   ├─ train_unsupervised.py
│   ├─ evaluate_unsupervised.py
│   ├─ app_unsupervised.py
│   └─ __init__.py
│
├─ notebooks/                # optional EDA
│   └─ 01_eda.ipynb
│
├─ requirements.txt
└─ README.md

🛠️ Setup
# Clone repo
git clone https://github.com/<yourusername>/credit-risk-predictor.git
cd credit-risk-predictor

# Create virtual env
python -m venv .venv
.\.venv\Scripts\Activate

# Install deps
pip install -r requirements.txt

🧮 Train & Evaluate
# Train Isolation Forest model
python -m src.train_unsupervised

# Evaluate & save metrics
python -m src.evaluate_unsupervised


Output:

✅ Trained unsupervised model → models/credit_model.joblib
✅ Saved metrics to models/metrics.json

🖥️ Run the App
python -m streamlit run src/app_unsupervised.py


Then open http://localhost:8501
.

🧍 Example Inputs
Scenario	Age	Credit amount	Duration	Sex	Job	Housing	Saving accounts	Checking account	Purpose
Low-risk (typical worker)	40	2500	24	male	2	own	moderate	moderate	radio/TV
High-risk (young, no job)	19	15000	60	female	0	rent	little	little	business
Basir (student)	20	1200	12	male	0	free	little	little	education
📊 Metrics Example
{
  "n_test": 200,
  "anomaly_rate": 0.09,
  "score_mean": 0.051,
  "score_p10": -0.037,
  "score_p50": 0.028,
  "score_p90": 0.104
}

🧠 Key Takeaways

Unsupervised ML can flag risky credit applicants even without labels

Clean modular code mirrors production ML workflows

Streamlit provides an intuitive way to visualize AI decision logic

🧱 Built With

Python 3.11

Pandas / NumPy


scikit-learn

Streamlit

Joblib