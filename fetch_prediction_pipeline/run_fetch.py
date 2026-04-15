from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so config.py is found
sys.path.append(str(Path(__file__).parent.parent))

import csv
import time
from datetime import datetime, timezone

from fetch import fetch_food_waste
from store_sql import init_db, store_history, store_current
from config import ZIP_CODES, OUTPUTS_DIR


def add_unique_id(rows: list[dict]) -> list[dict]:
    """
    Add a unique_id as the first column to every row.
    unique_id is composed of: store_id + offer_ean + offer_start_time.
    Example: "da2957d5-67ec-4f24-9c49-235b6712e063_20028992_2026-03-28 07:00:00"
    """
    enriched = []
    for row in rows:
        unique_id = f"{row['store_id']}_{row['offer_ean']}_{row['offer_start_time']}"
        enriched.append({"unique_id": unique_id, **row})
    return enriched


def log_run(summary: dict) -> None:
    """
    Append one row to outputs/fetch_log.csv after every pipeline run.
    The file is created with a header on the first run.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    log_path    = OUTPUTS_DIR / "fetch_log.csv"
    file_exists = log_path.is_file()

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    print("=" * 60)
    print("Anti Food Waste — Fetch Pipeline")
    print("=" * 60)

    start_time = time.time()

    # ── 1. Fetch ───────────────────────────────────────────────────────────────
    print(f"\n[1/3] Fetching offers from Salling API for zip codes: {ZIP_CODES}...")
    rows = fetch_food_waste()
    fetched_count = len(rows)

    if not rows:
        print("No offers fetched — exiting.")
        return

    print(f"Fetched {fetched_count} clearance offer(s)")

    # ── 2. Enrich ──────────────────────────────────────────────────────────────
    print("\n[2/3] Adding unique_id to all rows...")
    rows       = add_unique_id(rows)
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 3. Store ───────────────────────────────────────────────────────────────
    print("\n[3/3] Storing data in MySQL...")
    conn = init_db()

    history_inserted = store_history(conn, rows, fetched_at)
    print(f"  [history]  Inserted {history_inserted} new row(s)")

    current_inserted = store_current(conn, rows, fetched_at)
    print(f"  [current]  Replaced with {current_inserted} row(s)")

    conn.close()

    # ── 4. Log the run ─────────────────────────────────────────────────────────
    summary = {
        "timestamp":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "zip_codes":            ",".join(ZIP_CODES),
        "run_duration_seconds": round(time.time() - start_time, 2),
        "offers_fetched":       fetched_count,
        "history_inserted":     history_inserted,
        "current_replaced":     current_inserted,
    }
    log_run(summary)
    print(f"  [run_log]  Row appended to outputs/fetch_log.csv")

    print("\n" + "=" * 60)
    print(f"Fetch pipeline complete — {fetched_at}")
    print("=" * 60)


if __name__ == "__main__":
    main()