# main.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fetch import fetch_food_waste
from store_sql import init_db, store_history, store_current
from config import ZIP_CODE


def add_unique_id(rows: list[dict]) -> list[dict]:
    """
    Add a unique_id as the first column to every row.

    unique_id is composed of: store_id + offer_ean + offer_start_time
    - store_id:         identifies which store the offer is in
    - offer_ean:        identifies which product is on offer
    - offer_start_time: identifies when this specific clearance deal started
    Together these three fields pinpoint one unique clearance offer —
    no two offers will share the same combination.

    Example: "da2957d5-67ec-4f24-9c49-235b6712e063_20028992_2026-03-28 07:00:00"
    """
    enriched = []
    for row in rows:
        unique_id = f"{row['store_id']}_{row['offer_ean']}_{row['offer_start_time']}"
        enriched.append({"unique_id": unique_id, **row})
    return enriched


def main() -> None:
    print("=" * 60)
    print("Anti Food Waste Pipeline")
    print("=" * 60)

    # ── 1. Fetch ──────────────────────────────────────────────────
    print("\n[1/3] Fetching offers from Salling API...")
    rows = fetch_food_waste()

    fetched_count = len(rows)

    if not rows:
        print("No offers fetched — exiting.")
        return

    print(f"Fetched {fetched_count} clearance offer(s)")

    # ── 2. Enrich ─────────────────────────────────────────────────
    print("\n[2/3] Adding unique_id to all rows...")
    rows = add_unique_id(rows)

    # Capture the fetch timestamp once so all rows in this batch share it
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 3. Store ──────────────────────────────────────────────────
    print("\n[3/3] Storing data in SQLite...")
    conn = init_db()

    history_inserted = store_history(conn, rows, fetched_at)
    print(f"  [history]  Inserted {history_inserted} new row(s)")

    current_inserted = store_current(conn, rows)
    print(f"  [current]  Replaced with {current_inserted} row(s)")

    conn.close()

    # ── 4. Write run summary ──────────────────────────────────────
    # Creates outputs/run_summary.json after every fetch so you can
    # inspect the last run at a glance without opening the database
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    summary = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "zip_code":         ZIP_CODE,
        "offers_fetched":   fetched_count,
        "history_inserted": history_inserted,
        "current_replaced": current_inserted,
    }

    summary_path = outputs_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\n  [summary]  Written to {summary_path}")

    print("\n" + "=" * 60)
    print(f"Pipeline complete — {fetched_at}")
    print("=" * 60)


if __name__ == "__main__":
    main()