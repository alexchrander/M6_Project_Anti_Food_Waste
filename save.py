from __future__ import annotations

import csv
import os
from datetime import datetime

from fetch import fetch_food_waste


def add_unique_id(rows: list[dict]) -> list[dict]:
    """
    Add a unique_id as the first column to every row.

    unique_id is composed of: store_id + offer_ean + offer_start_time
    - store_id:         identifies which store the offer is in
    - offer_ean:        identifies which product is on offer
    - offer_start_time: identifies when this specific clearance deal started
    Together these three fields pinpoint one unique clearance offer across
    all stores, products, and time — no two offers will share the same combination.

    Example: "da2957d5-67ec-4f24-9c49-235b6712e063_20028992_2026-03-28 07:00:00"
    """
    enriched = []
    for row in rows:
        unique_id = f"{row['store_id']}_{row['offer_ean']}_{row['offer_start_time']}"

        # Insert unique_id at the front by building a new dict with it first
        enriched.append({"unique_id": unique_id, **row})

    return enriched


def save_current(rows: list[dict], output_dir: str = "data") -> None:
    """
    Save the current snapshot of offers to current.csv.
    This file is completely overwritten on every fetch — it always
    reflects only what is on offer right now.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "current.csv")

    if not rows:
        print("No data to save — the API returned 0 clearance offers.")
        return

    fieldnames = list(rows[0].keys())

    # Mode "w" overwrites the file completely on every run
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[current.csv]  Saved {len(rows)} rows → {filepath}")


def save_history(rows: list[dict], output_dir: str = "data") -> None:
    """
    Append the current fetch to history.csv.
    This file grows with every run and is the training database for the ML model.
    A fetched_at timestamp is added to each row so the ML model knows
    exactly when each snapshot was taken.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "history.csv")

    if not rows:
        print("No data to save — the API returned 0 clearance offers.")
        return

    # Stamp every row with the time this fetch was executed
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        row["fetched_at"] = fetched_at

    fieldnames = list(rows[0].keys())

    # If the file doesn't exist yet, create it and write the header
    # If it already exists, append rows only (no header repeated)
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[history.csv]  Appended {len(rows)} rows → {filepath}")


if __name__ == "__main__":
    # Fetch live data from the API
    rows = fetch_food_waste()

    # Add unique_id as the first column to all rows
    rows = add_unique_id(rows)

    # Save current snapshot (overwrite) and append to history
    save_current(rows)
    save_history(rows)