# config.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API settings ──────────────────────────────────────────────────────────────
BASE_URL  = "https://api.sallinggroup.com/v1/food-waste"
ZIP_CODES = ["9000"]   # Add more zip codes here when ready to expand

# ── MySQL connection settings ─────────────────────────────────────────────────
# Read from environment variables — never hardcode credentials!
# Locally: add them to your .env file
# UCloud: set directly here since we connect via sql-net internally
DB_HOST     = os.getenv("DB_HOST",     "sql-net")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_NAME     = os.getenv("DB_NAME",     "food_waste")
DB_USER     = os.getenv("DB_USER",     "food_waste_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "food_waste_alex")

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths are absolute and derived from the project root.
# config.py sits at the root, so Path(__file__).parent is the project root.
# Import these in other scripts instead of recomputing them each time.
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
DATASET_DIR  = DATA_DIR / "dataset"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
MLRUNS_DIR   = PROJECT_ROOT / "mlruns"

# ── ML pipeline settings ──────────────────────────────────────────────────────
# Target — an offer "sells" if sell_through_rate >= this threshold
SELL_THRESHOLD = 0.8

# Evaluation — retrain champion if PR AUC drops below this
PR_AUC_THRESHOLD = 0.85

# Classification — probability above this is predicted as will_sell = 1
# Default 0.5, but likely needs tuning down given ~2.5% positive rate
# Used in both evaluate.py (F1/recall/precision metrics) and predict.py (serving)
PREDICTION_THRESHOLD = 0.5

# Train/test split
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# Logistic Regression
MAX_ITER = 5000

# LightGBM
N_ESTIMATORS  = 500
LEARNING_RATE = 0.05

# ── Feature definitions ───────────────────────────────────────────────────────
# Column groups used in preprocessing.py and predict.py.
# Keeping them here ensures both scripts always use the same feature set.

DROP_COLS = [
    # Identifiers — no predictive signal
    "unique_id", "store_id", "store_name", "store_street", "store_zip",
    "product_ean", "offer_ean", "product_description",
    # Directly leaks the target
    "final_stock", "sell_through_rate",
    # Requires parsing — handled later by feature_engineering.py
    "store_customer_flow_today",
]

NUMERIC_COLS = [
    "store_lat", "store_lng",
    "offer_original_price", "offer_new_price",
    "offer_discount", "offer_percent_discount",
    "initial_stock", "hours_on_clearance", "n_snapshots",
]

CATEGORICAL_COLS = [
    "store_brand", "store_city", "offer_stock_unit",
    "product_category_en", "product_category_da",
]

DATETIME_COLS = [
    "offer_start_time", "offer_end_time",
    "first_seen", "last_seen",
]

BOOLEAN_COLS = [
    "had_overnight_gap", "potential_relabelling",
]