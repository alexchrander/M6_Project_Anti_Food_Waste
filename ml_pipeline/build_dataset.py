# Always run this from the project root:
# python ml_pipeline/build_dataset.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "fetch_prediction_pipeline"))

from store_sql import get_connection
from config import DATASET_DIR, SELL_THRESHOLD

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
    # offer_start_time, offer_end_time, offer_last_update come from the Salling API in UTC.
    # fetched_at is stored in CEST (+2h). Shift the API timestamps +2h so everything is CEST.
    # This also corrects hours_on_clearance, which compares offer_start_time to fetched_at.
    for col in ["offer_start_time", "offer_end_time", "offer_last_update"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]) + pd.Timedelta(hours=2)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return df


def exclude_active(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    active_ids = set(current["unique_id"])
    completed  = history[~history["unique_id"].isin(active_ids)].copy()
    log.info(f"Total unique offers: {history['unique_id'].nunique()}")
    log.info(f"Excluded (still active): {len(active_ids)}")
    log.info(f"Completed offers: {completed['unique_id'].nunique()}")
    return completed


def compute_snapshot_features(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one row per snapshot with per-snapshot features.
    Joins offer-level aggregates (initial_stock, final_stock, first_seen)
    back to all snapshot rows, then computes snapshot-level features.
    """
    completed = completed.sort_values(["unique_id", "fetched_at"]).copy()

    # Offer-level aggregates joined back to every snapshot row
    agg = completed.groupby("unique_id").agg(
        initial_stock=("offer_stock", "first"),
        final_stock=("offer_stock", "last"),
        first_seen=("fetched_at", "first"),
    ).reset_index()

    df = completed.merge(agg, on="unique_id", how="left")

    # sell_through_rate is offer-level — used only for labeling, not as a model feature
    df["sell_through_rate"] = (
        1 - (df["final_stock"] / df["initial_stock"].clip(lower=1))
    ).clip(0, 1)

    # ── Per-snapshot features ─────────────────────────────────────────────────
    df["offer_total_duration"] = (
        (df["offer_end_time"] - df["offer_start_time"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    df["hours_since_start"] = (
        (df["fetched_at"] - df["first_seen"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    df["hours_until_offer_end"] = (
        (df["offer_end_time"] - df["fetched_at"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    df["pct_time_elapsed"] = (
        df["hours_since_start"] / df["offer_total_duration"].clip(lower=0.25)
    ).clip(0, 1)

    df["stock_drop_so_far"] = (df["initial_stock"] - df["offer_stock"]).clip(lower=0)

    df["pct_stock_drop_so_far"] = (
        df["stock_drop_so_far"] / df["initial_stock"].clip(lower=1)
    ).clip(0, 1)

    df["stock_drop_per_hour"] = (
        df["stock_drop_so_far"] / df["hours_since_start"].clip(lower=0.25)
    )

    log.info(f"Snapshot features computed: {len(df)} rows from {df['unique_id'].nunique()} offers")
    return df


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dual-threshold labeling to all snapshot rows.
    Rows that cannot be labeled are set to NaN and dropped before training.

    Exclusions (will_sell = NaN → dropped):
    - offer_stock_unit == "kg"  (stock is a weight, not an item count)
    - initial_stock <= 2        (too few items for a reliable label)

    Labels:
    - initial_stock 3-5:  will_sell = 1 if final_stock <= 1
    - initial_stock > 5:  will_sell = 1 if sell_through_rate >= SELL_THRESHOLD
    """
    df = df.copy()
    df["will_sell"] = pd.NA

    eligible   = (df["offer_stock_unit"] != "kg") & (df["initial_stock"] > 2)
    low_stock  = eligible & (df["initial_stock"] <= 5)
    high_stock = eligible & (df["initial_stock"] > 5)

    df.loc[low_stock,  "will_sell"] = (df.loc[low_stock,  "final_stock"] <= 1).astype(int)
    df.loc[high_stock, "will_sell"] = (df.loc[high_stock, "sell_through_rate"] >= SELL_THRESHOLD).astype(int)

    before = len(df)
    df = df.dropna(subset=["will_sell"]).copy()
    df["will_sell"] = df["will_sell"].astype(int)

    excluded = before - len(df)
    log.info(f"Dropped {excluded} rows (kg or initial_stock ≤ 2) — {len(df)} rows remain")
    log.info(f"will_sell — sells: {df['will_sell'].sum()} ({df['will_sell'].mean():.1%}), no sell: {(df['will_sell'] == 0).sum()}")
    return df


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

    df = compute_snapshot_features(completed)
    df = compute_labels(df)

    log.info(f"Snapshot dataset: {len(df)} rows, {len(df.columns)} columns, {df['unique_id'].nunique()} offers")
    log.info(f"Positive rate: {df['will_sell'].mean():.1%}")

    save_dataset(df)


if __name__ == "__main__":
    main()
