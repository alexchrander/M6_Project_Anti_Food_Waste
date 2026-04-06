# Always run this from the project root:
# python ml_pipeline/build_dataset.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "fetch_pipeline"))

from store_sql import get_connection
from config import DATASET_DIR

import logging
from datetime import date
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn    = get_connection()
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
    completed  = history[~history["unique_id"].isin(active_ids)].copy()
    log.info(f"Total unique offers: {history['unique_id'].nunique()}")
    log.info(f"Excluded (still active): {len(active_ids)}")
    log.info(f"Completed offers: {completed['unique_id'].nunique()}")
    return completed


def compute_lifecycle_features(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lifecycle features that are observable for both completed and live offers.
    Used by both build_dataset.py (training) and predict.py (inference).

    Features computed:
    - first_seen          → timestamp of first history snapshot
    - last_seen           → timestamp of last history snapshot
    - initial_stock       → offer_stock at first snapshot
    - n_snapshots         → total number of snapshots
    - hours_on_clearance  → time from offer_start_time (or first_seen) to last_seen
    - had_overnight_gap   → whether any gap > 4 hours exists between snapshots
    """
    completed = completed.sort_values(["unique_id", "fetched_at"])

    first = completed.groupby("unique_id").first().reset_index()
    last  = completed.groupby("unique_id").last().reset_index()

    # ── n_snapshots ───────────────────────────────────────────────────────────
    n_snapshots = (
        completed.groupby("unique_id")["fetched_at"]
        .count()
        .reset_index()
        .rename(columns={"fetched_at": "n_snapshots"})
    )

    # ── had_overnight_gap ─────────────────────────────────────────────────────
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

    # ── Build base dataset from first snapshot ────────────────────────────────
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
    dataset["first_seen"]    = first["fetched_at"]
    dataset["last_seen"]     = last["fetched_at"].values  # moved from compute_labels

    # ── hours_on_clearance ────────────────────────────────────────────────────
    # Use offer_start_time if the offer appeared within one polling interval
    # (15 min) of our first snapshot. Otherwise fall back to first_seen.
    POLL_INTERVAL   = pd.Timedelta(minutes=15)
    use_offer_start = (dataset["first_seen"] - dataset["offer_start_time"]) <= POLL_INTERVAL
    dataset["hours_on_clearance"] = (
        (last["fetched_at"].values - dataset["offer_start_time"].where(use_offer_start, dataset["first_seen"]))
        .dt.total_seconds() / 3600
    )

    # ── Merge metadata ────────────────────────────────────────────────────────
    dataset = dataset.merge(n_snapshots,   on="unique_id")
    dataset = dataset.merge(overnight_gap, on="unique_id")

    return dataset, last


def compute_labels(dataset: pd.DataFrame, last: pd.DataFrame) -> pd.DataFrame:
    """
    Compute training-only labels that require completed offer data.
    Not used by predict.py since live offers haven't completed yet.

    Labels computed:
    - final_stock           → offer_stock at last snapshot
    - sell_through_rate     → how much of the stock sold
    - potential_relabelling → same product reappears at same store within 2 hours
    """
    dataset = dataset.copy()

    dataset["final_stock"] = last["offer_stock"].values

    # ── sell_through_rate ─────────────────────────────────────────────────────
    dataset["sell_through_rate"] = (
        1 - (dataset["final_stock"] / dataset["initial_stock"])
    ).clip(0, 1)

    # ── potential_relabelling ─────────────────────────────────────────────────
    # Detects if the same product_ean at the same store reappears within 2 hours
    # of a previous offer ending — requires completed offer data to compare against
    offer_timeline = dataset[["unique_id", "store_id", "product_ean", "first_seen", "last_seen"]].copy()
    merged = offer_timeline.merge(offer_timeline, on=["store_id", "product_ean"], suffixes=("_prev", "_next"))
    relabelled = merged[
        (merged["unique_id_prev"] != merged["unique_id_next"]) &
        (merged["first_seen_next"] > merged["last_seen_prev"]) &
        (merged["first_seen_next"] - merged["last_seen_prev"] < pd.Timedelta(hours=2))
    ]
    relabelled_ids = set(relabelled["unique_id_prev"].unique())
    dataset["potential_relabelling"] = dataset["unique_id"].isin(relabelled_ids)

    return dataset


def save_dataset(dataset: pd.DataFrame) -> str:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    today       = date.today().strftime("%Y%m%d")
    output_path = DATASET_DIR / f"labelled_offers_{today}.parquet"
    dataset.to_parquet(output_path, index=False)
    log.info(f"Saved dataset to {output_path}")
    return str(output_path)


def main():
    history, current = load_data()
    history          = parse_timestamps(history)
    completed        = exclude_active(history, current)

    dataset, last = compute_lifecycle_features(completed)
    dataset       = compute_labels(dataset, last)

    log.info(f"Aggregated dataset: {len(dataset)} rows, {dataset.columns.nunique()} columns")
    log.info(f"Sell-through rate — mean: {dataset['sell_through_rate'].mean():.3f}, zero: {(dataset['sell_through_rate'] == 0).mean():.1%}")
    log.info(f"Hours on clearance — mean: {dataset['hours_on_clearance'].mean():.1f}, max: {dataset['hours_on_clearance'].max():.1f}")
    log.info(f"Overnight gap: {dataset['had_overnight_gap'].mean():.1%} of offers")
    log.info(f"Potential relabelling: {dataset['potential_relabelling'].sum()} ({dataset['potential_relabelling'].mean():.1%})")

    save_dataset(dataset)


if __name__ == "__main__":
    main()