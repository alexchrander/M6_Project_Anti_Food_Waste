# config.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── API settings ─────────────────────────────────────────────────────────────
# Base URL for the Salling Anti Food Waste API
BASE_URL = "https://api.sallinggroup.com/v1/food-waste"

# Zip code to query — change this to fetch from a different area
ZIP_CODE = os.getenv("ZIP_CODE", "9000")

# ── Storage settings ─────────────────────────────────────────────────────────
# Single SQLite database file that holds both tables:
#   - history:  every fetch appended (used for ML training)
#   - current:  only the latest fetch (overwritten each run)
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "food_waste.db"))