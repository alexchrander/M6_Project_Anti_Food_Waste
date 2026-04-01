# config.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API settings ─────────────────────────────────────────────────────────────
BASE_URL = "https://api.sallinggroup.com/v1/food-waste"
ZIP_CODE = os.getenv("ZIP_CODE", "9000")

# ── MySQL connection settings ─────────────────────────────────────────────────
# These are read from environment variables — never hardcode credentials!
# Locally: add them to your .env file
# UCloud: they are set directly here since we connect via sql-net internally
DB_HOST     = os.getenv("DB_HOST",     "sql-net")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_NAME     = os.getenv("DB_NAME",     "food_waste")
DB_USER     = os.getenv("DB_USER",     "food_waste_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "food_waste_alex")