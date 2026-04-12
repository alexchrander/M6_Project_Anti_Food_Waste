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
PROJECT_ROOT    = Path(__file__).parent
DATA_DIR        = PROJECT_ROOT / "data"
DATASET_DIR     = DATA_DIR / "dataset"
FEATURES_DIR    = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR      = PROJECT_ROOT / "models"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
MLRUNS_DIR      = PROJECT_ROOT / "mlruns"

# ── ML pipeline settings ──────────────────────────────────────────────────────
# Target — an offer "sells" if sell_through_rate >= this threshold
SELL_THRESHOLD = 0.8

# Evaluation — retrain champion if PR AUC drops below this
PR_AUC_THRESHOLD = 0.5

# Classification — probability above this is predicted as will_sell = 1
# Default 0.5, but likely needs tuning down given ~2.5% positive rate
# Used in both evaluate.py (F1/recall/precision metrics) and predict.py (serving)
PREDICTION_THRESHOLD = 0.5

# Train/test split
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# Logistic Regression
MAX_ITER = 1000

# LightGBM
N_ESTIMATORS  = 500
LEARNING_RATE = 0.05

# ── Column definitions ────────────────────────────────────────────────────────
# Single source of truth for all columns across the pipeline.
#
# type:        how the column is treated during preprocessing
#              "numeric"     → StandardScaler
#              "categorical" → LabelEncoder
#              "onehot"      → pd.get_dummies (no ordinal assumption)
#              "boolean"     → cast to int (0/1)
#              "datetime"    → extract hour + dayofweek, then drop
#              "passthrough"  → not used by model (display, joining, source for engineered features, or training-only labels)
#
# model:       whether the column is passed to the model as a feature
# app:         whether the column is written to the app MySQL table (app.py)
# predictions: whether the column is saved in the predictions parquet (audit/debug)

COLUMNS = {

    # ── API ───────────────────────────────────────────────────────────────────
    "unique_id":                    {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "product_ean":                  {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "product_description":          {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "product_image":                {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "product_category_en":          {"type": "passthrough",  "model": False, "app": False, "predictions": False},
    "product_category_da":          {"type": "passthrough",  "model": False, "app": False, "predictions": False},
    "offer_ean":                    {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "offer_new_price":              {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "offer_original_price":         {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "offer_discount":               {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "offer_percent_discount":       {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "offer_stock_unit":             {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "offer_start_time":             {"type": "datetime",    "model": True,  "app": True,  "predictions": True},
    "offer_end_time":               {"type": "datetime",    "model": True,  "app": True,  "predictions": True},
    "initial_stock":                {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "store_id":                     {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "store_name":                   {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "store_brand":                  {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "store_city":                   {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "store_street":                 {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "store_zip":                    {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "store_country":                {"type": "passthrough",  "model": False, "app": False, "predictions": True},
    "store_lat":                    {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "store_lng":                    {"type": "passthrough",  "model": False, "app": True,  "predictions": True},
    "store_hours_today":            {"type": "passthrough",  "model": False, "app": True, "predictions": True},
    "store_hours_tomorrow":         {"type": "passthrough",  "model": False, "app": True, "predictions": True},
    "store_customer_flow_today":    {"type": "passthrough",  "model": False, "app": False, "predictions": False},
    "store_customer_flow_tomorrow": {"type": "passthrough",  "model": False, "app": False, "predictions": False},
    # Training-only labels — never present at inference
    "final_stock":                  {"type": "passthrough",       "model": False, "app": False, "predictions": False},
    "sell_through_rate":            {"type": "passthrough",       "model": False, "app": False, "predictions": False},

    # ── Feature Engineered ────────────────────────────────────────────────────
    # Lifecycle — computed in build_dataset.py
    "first_seen":                   {"type": "datetime",    "model": True,  "app": True,  "predictions": True},
    "last_seen":                    {"type": "datetime",    "model": True,  "app": False, "predictions": True},
    "hours_on_clearance":           {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "n_snapshots":                  {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "had_overnight_gap":            {"type": "boolean",     "model": True,  "app": True,  "predictions": True},
    "potential_relabelling":        {"type": "boolean",     "model": True,  "app": False, "predictions": True},
    # Category — computed in build_features.py from product_category_en/da
    "category_level1_en":           {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "category_level2_en":           {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "category_level1_da":           {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    "category_level2_da":           {"type": "categorical", "model": True,  "app": True,  "predictions": True},
    # Stock — computed in build_features.py from offer_stock_unit
    "is_weight_based":              {"type": "boolean",     "model": True,  "app": True,  "predictions": True},
    # Customer flow — computed in build_features.py from store_customer_flow_today
    "flow_peak_hour":               {"type": "onehot",      "model": True,  "app": True,  "predictions": True},
    "flow_peak_value":              {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "flow_avg":                     {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "flow_at_offer_start":          {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "flow_evening_share":           {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    # Store hours — computed in build_features.py from store_hours_today/tomorrow
    "store_open_hours":             {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "hours_until_close":            {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
    "is_closed_tomorrow":           {"type": "boolean",     "model": True,  "app": True,  "predictions": True},
    # Time — computed in build_features.py from offer_start_time / offer_end_time
    "offer_start_dayofweek":        {"type": "onehot",      "model": True,  "app": True,  "predictions": True},
    "offer_start_hour_cest":        {"type": "onehot",      "model": True,  "app": True,  "predictions": True},
    "hours_until_offer_end":        {"type": "numeric",     "model": True,  "app": True,  "predictions": True},
}

# ── Derived column lists ──────────────────────────────────────────────────────
# All lists are derived from COLUMNS — never edit these directly.
# To change what goes where, update COLUMNS above.

DROP_COLS        = [col for col, m in COLUMNS.items() if not m["model"]]
NUMERIC_COLS     = [col for col, m in COLUMNS.items() if m["type"] == "numeric"]
CATEGORICAL_COLS = [col for col, m in COLUMNS.items() if m["type"] == "categorical"]
ONEHOT_COLS      = [col for col, m in COLUMNS.items() if m["type"] == "onehot"]
DATETIME_COLS    = [col for col, m in COLUMNS.items() if m["type"] == "datetime"]
BOOLEAN_COLS     = [col for col, m in COLUMNS.items() if m["type"] == "boolean"]

APP_COLS                 = [col for col, m in COLUMNS.items() if m["app"]]
PREDICTIONS_PARQUET_COLS = [col for col, m in COLUMNS.items() if m["predictions"]]