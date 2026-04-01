# store_sql.py
from __future__ import annotations

import mysql.connector
from mysql.connector import connection

from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection() -> connection.MySQLConnection:
    """
    Create and return a new MySQL connection using credentials from config.
    """
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def init_db() -> connection.MySQLConnection:
    """
    Initialise the MySQL database and create both tables if they don't exist.
    Returns an open connection.

    Two tables are created:
      - history: append-only, one row per offer per fetch — used for ML training.
      - current: replaced on every fetch — always reflects live offers right now.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── History table ─────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            unique_id              TEXT NOT NULL,
            fetched_at             DATETIME NOT NULL,
            product_ean            TEXT,
            product_description    TEXT,
            product_image          TEXT,
            product_category_da    TEXT,
            product_category_en    TEXT,
            offer_ean              TEXT,
            offer_currency         TEXT,
            offer_original_price   FLOAT,
            offer_new_price        FLOAT,
            offer_discount         FLOAT,
            offer_percent_discount FLOAT,
            offer_stock            FLOAT,
            offer_stock_unit       TEXT,
            offer_start_time       DATETIME,
            offer_end_time         DATETIME,
            offer_last_update      DATETIME,
            store_id               TEXT,
            store_name             TEXT,
            store_brand            TEXT,
            store_lat              FLOAT,
            store_lng              FLOAT,
            store_street           TEXT,
            store_city             TEXT,
            store_zip              TEXT,
            store_country          TEXT,
            store_hours_today      TEXT,
            store_hours_tomorrow   TEXT,
            store_customer_flow_today    TEXT,
            store_customer_flow_tomorrow TEXT
        )
    """)

    # ── Current table ─────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS current (
            unique_id              TEXT NOT NULL,
            fetched_at             DATETIME,
            product_ean            TEXT,
            product_description    TEXT,
            product_image          TEXT,
            product_category_da    TEXT,
            product_category_en    TEXT,
            offer_ean              TEXT,
            offer_currency         TEXT,
            offer_original_price   FLOAT,
            offer_new_price        FLOAT,
            offer_discount         FLOAT,
            offer_percent_discount FLOAT,
            offer_stock            FLOAT,
            offer_stock_unit       TEXT,
            offer_start_time       DATETIME,
            offer_end_time         DATETIME,
            offer_last_update      DATETIME,
            store_id               TEXT,
            store_name             TEXT,
            store_brand            TEXT,
            store_lat              FLOAT,
            store_lng              FLOAT,
            store_street           TEXT,
            store_city             TEXT,
            store_zip              TEXT,
            store_country          TEXT,
            store_hours_today      TEXT,
            store_hours_tomorrow   TEXT,
            store_customer_flow_today    TEXT,
            store_customer_flow_tomorrow TEXT
        )
    """)

    conn.commit()
    cursor.close()
    return conn


def store_history(conn: connection.MySQLConnection, rows: list[dict], fetched_at: str) -> int:
    """
    Append the current fetch to the history table.
    Returns the number of inserted rows.
    """
    cursor = conn.cursor()

    sql = """
        INSERT INTO history (
            unique_id, fetched_at,
            product_ean, product_description, product_image,
            product_category_da, product_category_en,
            offer_ean, offer_currency, offer_original_price, offer_new_price,
            offer_discount, offer_percent_discount, offer_stock, offer_stock_unit,
            offer_start_time, offer_end_time, offer_last_update,
            store_id, store_name, store_brand, store_lat, store_lng,
            store_street, store_city, store_zip, store_country,
            store_hours_today, store_hours_tomorrow,
            store_customer_flow_today, store_customer_flow_tomorrow
        ) VALUES (
            %(unique_id)s, %(fetched_at)s,
            %(product_ean)s, %(product_description)s, %(product_image)s,
            %(product_category_da)s, %(product_category_en)s,
            %(offer_ean)s, %(offer_currency)s, %(offer_original_price)s, %(offer_new_price)s,
            %(offer_discount)s, %(offer_percent_discount)s, %(offer_stock)s, %(offer_stock_unit)s,
            %(offer_start_time)s, %(offer_end_time)s, %(offer_last_update)s,
            %(store_id)s, %(store_name)s, %(store_brand)s, %(store_lat)s, %(store_lng)s,
            %(store_street)s, %(store_city)s, %(store_zip)s, %(store_country)s,
            %(store_hours_today)s, %(store_hours_tomorrow)s,
            %(store_customer_flow_today)s, %(store_customer_flow_tomorrow)s
        )
    """

    for row in rows:
        cursor.execute(sql, {**row, "fetched_at": fetched_at})

    conn.commit()
    cursor.close()
    return len(rows)


def store_current(conn: connection.MySQLConnection, rows: list[dict], fetched_at: str) -> int:
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
            unique_id, fetched_at,
            product_ean, product_description, product_image,
            product_category_da, product_category_en,
            offer_ean, offer_currency, offer_original_price, offer_new_price,
            offer_discount, offer_percent_discount, offer_stock, offer_stock_unit,
            offer_start_time, offer_end_time, offer_last_update,
            store_id, store_name, store_brand, store_lat, store_lng,
            store_street, store_city, store_zip, store_country,
            store_hours_today, store_hours_tomorrow,
            store_customer_flow_today, store_customer_flow_tomorrow
        ) VALUES (
            %(unique_id)s, %(fetched_at)s,
            %(product_ean)s, %(product_description)s, %(product_image)s,
            %(product_category_da)s, %(product_category_en)s,
            %(offer_ean)s, %(offer_currency)s, %(offer_original_price)s, %(offer_new_price)s,
            %(offer_discount)s, %(offer_percent_discount)s, %(offer_stock)s, %(offer_stock_unit)s,
            %(offer_start_time)s, %(offer_end_time)s, %(offer_last_update)s,
            %(store_id)s, %(store_name)s, %(store_brand)s, %(store_lat)s, %(store_lng)s,
            %(store_street)s, %(store_city)s, %(store_zip)s, %(store_country)s,
            %(store_hours_today)s, %(store_hours_tomorrow)s,
            %(store_customer_flow_today)s, %(store_customer_flow_tomorrow)s
        )
    """

    for row in rows:
        cursor.execute(sql, {**row, "fetched_at": fetched_at})

    conn.commit()
    cursor.close()
    return len(rows)
