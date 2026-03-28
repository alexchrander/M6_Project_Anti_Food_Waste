# store_sql.py
from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DB_PATH


def init_db() -> sqlite3.Connection:
    """
    Initialise the SQLite database and create both tables if they don't exist.
    Returns an open connection.

    Two tables are created:
      - history: append-only, one row per offer per fetch — used for ML training.
                 unique_id + fetched_at together prevent exact duplicate rows.
      - current: replaced on every fetch — always reflects live offers right now.
    """
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ── History table ─────────────────────────────────────────────────────────
    # INSERT OR IGNORE on (unique_id, fetched_at) means if the exact same offer
    # is fetched twice within the same minute it won't be duplicated.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,

            -- unique_id is built from: store_id + offer_ean + offer_start_time
            -- It identifies one specific clearance deal across all fetches
            unique_id             TEXT NOT NULL,
            fetched_at            TEXT NOT NULL,

            -- Product
            product_ean           TEXT,
            product_description   TEXT,
            product_image         TEXT,

            -- Offer
            offer_ean             TEXT,
            offer_currency        TEXT,
            offer_original_price  REAL,
            offer_new_price       REAL,
            offer_discount        REAL,
            offer_percent_discount REAL,
            offer_stock           REAL,
            offer_stock_unit      TEXT,
            offer_start_time      TEXT,
            offer_end_time        TEXT,
            offer_last_update     TEXT,

            -- Store
            store_id              TEXT,
            store_name            TEXT,
            store_brand           TEXT,
            store_lat             REAL,
            store_lng             REAL,
            store_street          TEXT,
            store_city            TEXT,
            store_zip             TEXT,
            store_country         TEXT,
            store_hours_today     TEXT,
            store_hours_tomorrow  TEXT,

            -- Prevent inserting the exact same offer snapshot twice
            UNIQUE(unique_id, fetched_at)
        )
    """)

    # ── Current table ─────────────────────────────────────────────────────────
    # Same columns as history but without fetched_at — it's always just now.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS current (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,

            unique_id             TEXT NOT NULL,

            product_ean           TEXT,
            product_description   TEXT,
            product_image         TEXT,

            offer_ean             TEXT,
            offer_currency        TEXT,
            offer_original_price  REAL,
            offer_new_price       REAL,
            offer_discount        REAL,
            offer_percent_discount REAL,
            offer_stock           REAL,
            offer_stock_unit      TEXT,
            offer_start_time      TEXT,
            offer_end_time        TEXT,
            offer_last_update     TEXT,

            store_id              TEXT,
            store_name            TEXT,
            store_brand           TEXT,
            store_lat             REAL,
            store_lng             REAL,
            store_street          TEXT,
            store_city            TEXT,
            store_zip             TEXT,
            store_country         TEXT,
            store_hours_today     TEXT,
            store_hours_tomorrow  TEXT
        )
    """)

    conn.commit()
    return conn


def store_history(conn: sqlite3.Connection, rows: list[dict], fetched_at: str) -> int:
    """
    Append the current fetch to the history table.
    INSERT OR IGNORE skips rows where (unique_id, fetched_at) already exists,
    so repeated runs within the same minute won't create duplicates.
    Returns the number of newly inserted rows.
    """
    cursor = conn.cursor()
    inserted = 0

    sql = """
        INSERT OR IGNORE INTO history (
            unique_id, fetched_at,
            product_ean, product_description, product_image,
            offer_ean, offer_currency, offer_original_price, offer_new_price,
            offer_discount, offer_percent_discount, offer_stock, offer_stock_unit,
            offer_start_time, offer_end_time, offer_last_update,
            store_id, store_name, store_brand, store_lat, store_lng,
            store_street, store_city, store_zip, store_country,
            store_hours_today, store_hours_tomorrow
        ) VALUES (
            :unique_id, :fetched_at,
            :product_ean, :product_description, :product_image,
            :offer_ean, :offer_currency, :offer_original_price, :offer_new_price,
            :offer_discount, :offer_percent_discount, :offer_stock, :offer_stock_unit,
            :offer_start_time, :offer_end_time, :offer_last_update,
            :store_id, :store_name, :store_brand, :store_lat, :store_lng,
            :store_street, :store_city, :store_zip, :store_country,
            :store_hours_today, :store_hours_tomorrow
        )
    """

    for row in rows:
        cursor.execute(sql, {**row, "fetched_at": fetched_at})
        inserted += cursor.rowcount

    conn.commit()
    return inserted


def store_current(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """
    Replace the current table with the latest fetch.
    Clears the table first, then inserts all fresh rows.
    Returns the number of inserted rows.
    """
    cursor = conn.cursor()

    # Wipe the current table so it only ever holds the latest snapshot
    cursor.execute("DELETE FROM current")

    sql = """
        INSERT INTO current (
            unique_id,
            product_ean, product_description, product_image,
            offer_ean, offer_currency, offer_original_price, offer_new_price,
            offer_discount, offer_percent_discount, offer_stock, offer_stock_unit,
            offer_start_time, offer_end_time, offer_last_update,
            store_id, store_name, store_brand, store_lat, store_lng,
            store_street, store_city, store_zip, store_country,
            store_hours_today, store_hours_tomorrow
        ) VALUES (
            :unique_id,
            :product_ean, :product_description, :product_image,
            :offer_ean, :offer_currency, :offer_original_price, :offer_new_price,
            :offer_discount, :offer_percent_discount, :offer_stock, :offer_stock_unit,
            :offer_start_time, :offer_end_time, :offer_last_update,
            :store_id, :store_name, :store_brand, :store_lat, :store_lng,
            :store_street, :store_city, :store_zip, :store_country,
            :store_hours_today, :store_hours_tomorrow
        )
    """

    for row in rows:
        cursor.execute(sql, row)

    conn.commit()
    return len(rows)