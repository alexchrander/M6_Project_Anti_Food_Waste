import sqlite3
import logging
from pathlib import Path
from datetime import date

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)


def load_data(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    history = pd.read_sql("SELECT * FROM history", conn)
    current = pd.read_sql("SELECT * FROM current", conn)
    conn.close()
    log.info(f"Loaded {len(history)} history rows and {len(current)} current rows")
    return history, current


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["fetched_at", "offer_start_time", "offer_end_time"]:
        df[col] = pd.to_datetime(df[col])
    return df


def exclude_active(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    active_ids = set(current["unique_id"])
    completed = history[~history["unique_id"].isin(active_ids)].copy()
    log.info(f"Total unique offers: {history['unique_id'].nunique()}")
    log.info(f"Excluded (still active): {len(active_ids)}")
    log.info(f"Completed offers: {completed['unique_id'].nunique()}")
    return completed


def aggregate_lifecycles(completed: pd.DataFrame) -> pd.DataFrame:
    completed = completed.sort_values(["unique_id", "fetched_at"])

    # First and last snapshot per offer
    first = completed.groupby("unique_id").first().reset_index()
    last = completed.groupby("unique_id").last().reset_index()

    # Overnight gap — any consecutive gap > 4 hours between snapshots
    completed["time_diff_hours"] = (
        completed.groupby("unique_id")["fetched_at"]
        .diff()
        .dt.total_seconds() / 3600
    )
    overnight_gap = (
        completed.groupby("unique_id")["time_diff_hours"]
        .max()
        .gt(4)
        .reset_index()
        .rename(columns={"time_diff_hours": "had_overnight_gap"})
    )

    # Snapshot count per offer
    n_snapshots = (
        completed.groupby("unique_id")["fetched_at"]
        .count()
        .reset_index()
        .rename(columns={"fetched_at": "n_snapshots"})
    )

    # Build base dataset from first snapshot
    dataset = first[[
        "unique_id", "store_id", "store_name", "store_brand", "store_city",
        "store_lat", "store_lng", "store_street", "store_zip",
        "product_ean", "product_description", "offer_ean",
        "offer_original_price", "offer_new_price", "offer_discount",
        "offer_percent_discount", "offer_stock_unit", "offer_start_time",
        "offer_end_time", "store_customer_flow_today",
        "product_category_da", "product_category_en",
    ]].copy()

    dataset["initial_stock"] = first["offer_stock"]
    dataset["final_stock"] = last["offer_stock"]
    dataset["first_seen"] = first["fetched_at"]
    dataset["last_seen"] = last["fetched_at"]

    # Labels
    dataset["sell_through_rate"] = (
        1 - (dataset["final_stock"] / dataset["initial_stock"])
    ).clip(0, 1)

    # hours_on_clearance: use offer_start_time if the offer appeared within one
    # polling interval (15 min) of our first snapshot — meaning we caught it from
    # the start. Otherwise fall back to first_seen to avoid inflating the value.
    POLL_INTERVAL = pd.Timedelta(minutes=15)
    use_offer_start = (dataset["first_seen"] - dataset["offer_start_time"]) <= POLL_INTERVAL
    dataset["hours_on_clearance"] = (
        (dataset["last_seen"] - dataset["offer_start_time"].where(use_offer_start, dataset["first_seen"]))
        .dt.total_seconds() / 3600
    )

    # Merge metadata
    dataset = dataset.merge(n_snapshots, on="unique_id")
    dataset = dataset.merge(overnight_gap, on="unique_id")

    # Detect potential relabelling — same product_ean at same store reappears within 2 hours
    offer_timeline = dataset[["unique_id", "store_id", "product_ean", "first_seen", "last_seen"]].copy()
    merged = offer_timeline.merge(offer_timeline, on=["store_id", "product_ean"], suffixes=("_prev", "_next"))
    relabelled = merged[
        (merged["unique_id_prev"] != merged["unique_id_next"]) &
        (merged["first_seen_next"] > merged["last_seen_prev"]) &
        (merged["first_seen_next"] - merged["last_seen_prev"] < pd.Timedelta(hours=2))
    ]
    relabelled_ids = set(relabelled["unique_id_prev"].unique())
    dataset["potential_relabelling"] = dataset["unique_id"].isin(relabelled_ids)

    log.info(f"Aggregated dataset: {len(dataset)} rows, {dataset.columns.nunique()} columns")
    log.info(f"Sell-through rate — mean: {dataset['sell_through_rate'].mean():.3f}, zero: {(dataset['sell_through_rate'] == 0).mean():.1%}")
    log.info(f"Hours on clearance — mean: {dataset['hours_on_clearance'].mean():.1f}, max: {dataset['hours_on_clearance'].max():.1f}")
    log.info(f"Overnight gap: {dataset['had_overnight_gap'].mean():.1%} of offers")
    log.info(f"Potential relabelling: {dataset['potential_relabelling'].sum()} ({dataset['potential_relabelling'].mean():.1%})")

    return dataset




def save_dataset(dataset: pd.DataFrame, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    output_path = f"{output_dir}/labelled_offers_{today}.parquet"
    dataset.to_parquet(output_path, index=False)
    log.info(f"Saved dataset to {output_path}")
    return output_path


def main():
    DB_PATH = "data/food_waste.db"
    OUTPUT_DIR = "data/datasets"

    history, current = load_data(DB_PATH)
    history = parse_timestamps(history)
    completed = exclude_active(history, current)
    dataset = aggregate_lifecycles(completed)
    save_dataset(dataset, OUTPUT_DIR)


if __name__ == "__main__":
    main()