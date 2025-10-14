from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "german_credit.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# No labels in your file
TARGET_COL = None
ID_COL = "Unnamed: 0"   # drop this index column

# Match YOUR exact column names (with spaces)
NUMERIC_COLS = [
    "Age", "Credit amount", "Duration"
]

CATEGORICAL_COLS = [
    "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"
]

# Unsupervised controls
TEST_SIZE = 0.2           # still used for scoring split
RANDOM_STATE = 42
CONTAMINATION = 0.10      # ~10% flagged as high-risk anomalies
