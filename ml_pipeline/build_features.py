# Always run this from the project root:
# python ml_pipeline/build_features.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import glob
import logging
import numpy as np
import pandas as pd
from datetime import date

from config import DATASET_DIR, FEATURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    files = glob.glob(str(DATASET_DIR / "labelled_offers_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No dataset parquet files found in {DATASET_DIR}")
    latest = sorted(files)[-1]
    log.info(f"Loading dataset: {latest}")
    return pd.read_parquet(latest)


def engineer_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split hierarchical category strings into two levels for both English and Danish.

    Example: "Dairy And Cold Storage>Ready To Eat Meals>Dinner Meals"
      → category_level1_en = "Dairy And Cold Storage"
      → category_level2_en = "Ready To Eat Meals"

    Products with no category (e.g. butcher/meat items) → "Unknown" at all levels.
    Replaces the raw product_category_en / product_category_da columns.
    """
    df = df.copy()

    def split_level(series: pd.Series, level: int) -> pd.Series:
        def get_level(val):
            if pd.isna(val) or str(val).strip() == "":
                return "Unknown"
            parts = str(val).split(">")
            return parts[level].strip() if len(parts) > level else "Unknown"
        return series.apply(get_level)

    df["category_level1_en"] = split_level(df["product_category_en"], 0)
    df["category_level2_en"] = split_level(df["product_category_en"], 1)
    df["category_level1_da"] = split_level(df["product_category_da"], 0)
    df["category_level2_da"] = split_level(df["product_category_da"], 1)

    n_unknown = (df["category_level1_en"] == "Unknown").sum()
    log.info(
        f"category_level1_en: {df['category_level1_en'].nunique()} unique values "
        f"({n_unknown} Unknown = {n_unknown / len(df):.1%})"
    )
    return df


def engineer_stock_unit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_weight_based to distinguish kg products from count-based products.

    For kg products (meat dept.), offer_stock is a weight in kg (e.g. 0.977) — not
    a count of items. This flag lets the model handle the two scales separately.
    offer_stock_unit is kept as a categorical feature alongside this boolean.
    """
    df = df.copy()
    df["is_weight_based"] = df["offer_stock_unit"] == "kg"
    log.info(
        f"is_weight_based: {df['is_weight_based'].sum()} kg-based "
        f"({df['is_weight_based'].mean():.1%})"
    )
    return df


def _parse_flow_string(flow_str) -> list:
    """Parse comma-separated customer flow string into a list of 24 floats."""
    try:
        if pd.isna(flow_str) or str(flow_str).strip() == "":
            return [0.0] * 24
        values = [float(v) for v in str(flow_str).split(",")]
        if len(values) < 24:
            values += [0.0] * (24 - len(values))
        return values[:24]
    except Exception:
        return [0.0] * 24


def engineer_customer_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse store_customer_flow_today (24 comma-separated hourly traffic values) into
    numeric features. offer_start_time must already be a datetime column.

    Features:
    - flow_peak_value:      highest traffic value across all hours
    - flow_peak_hour:       hour of day with highest traffic (0-23)
    - flow_avg:             mean traffic during open hours (non-zero hours only)
    - flow_at_offer_start:  traffic value at the hour the offer went live
    - flow_evening_share:   share of daily traffic occurring between 17:00-21:00
    """
    df = df.copy()

    flow_array = np.array(df["store_customer_flow_today"].apply(_parse_flow_string).tolist())

    df["flow_peak_value"] = flow_array.max(axis=1)
    df["flow_peak_hour"]  = flow_array.argmax(axis=1)

    nonzero_mask   = flow_array > 0
    nonzero_counts = nonzero_mask.sum(axis=1).clip(1)
    df["flow_avg"] = np.where(
        nonzero_mask.any(axis=1),
        (flow_array * nonzero_mask).sum(axis=1) / nonzero_counts,
        0.0,
    )

    start_hours = pd.to_datetime(df["offer_start_time"]).dt.hour.clip(0, 23).values
    df["flow_at_offer_start"] = flow_array[np.arange(len(df)), start_hours]

    evening_flow = flow_array[:, 17:22].sum(axis=1)  # hours 17, 18, 19, 20, 21
    total_flow   = flow_array.sum(axis=1)
    df["flow_evening_share"] = np.where(total_flow > 0, evening_flow / total_flow, 0.0)

    log.info("Engineered customer flow features (flow_peak_value, flow_peak_hour, flow_avg, flow_at_offer_start, flow_evening_share)")
    return df


def _parse_store_hours(hours_str):
    """
    Parse "HH:MM-HH:MM" into (open_decimal_hour, close_decimal_hour).
    Returns (None, None) for "closed" or unparseable strings.
    """
    try:
        s = str(hours_str).strip().lower()
        if s in ("closed", "", "nan", "none"):
            return None, None
        open_str, close_str = s.split("-")
        def to_decimal(t):
            h, m = t.strip().split(":")
            return int(h) + int(m) / 60
        return to_decimal(open_str), to_decimal(close_str)
    except Exception:
        return None, None


def engineer_store_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse store_hours_today ("HH:MM-HH:MM") into numeric features.

    Features:
    - store_open_hours:   total hours the store is open today
    - hours_until_close:  hours from offer_start_time to store closing time
                          (how much selling time remains when the offer went live)
    - is_closed_tomorrow: whether the store is closed the following day
    """
    df = df.copy()

    parsed      = df["store_hours_today"].apply(_parse_store_hours)
    open_hours  = parsed.apply(lambda x: x[0])
    close_hours = parsed.apply(lambda x: x[1])

    df["store_open_hours"] = (close_hours - open_hours).clip(lower=0)

    start_decimal = (
        pd.to_datetime(df["offer_start_time"]).dt.hour
        + pd.to_datetime(df["offer_start_time"]).dt.minute / 60
    )
    df["hours_until_close"] = (close_hours - start_decimal).clip(lower=0)

    df["is_closed_tomorrow"] = df["store_hours_tomorrow"].apply(
        lambda x: str(x).strip().lower() == "closed"
    )

    log.info("Engineered store hours features (store_open_hours, hours_until_close, is_closed_tomorrow)")
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit time-based features using timezone-corrected timestamps.
    build_dataset.parse_timestamps() shifts offer_start_time/offer_end_time from
    UTC to CEST (+2h), so the hour extracted here is the correct local hour.

    Features:
    - hours_until_offer_end:  hours from first observation until offer_end_time
    - offer_start_dayofweek:  0=Monday ... 6=Sunday
    - offer_start_hour_cest:  local hour of day when the clearance offer started
    """
    df = df.copy()

    offer_end  = pd.to_datetime(df["offer_end_time"])
    first_seen = pd.to_datetime(df["first_seen"])
    df["hours_until_offer_end"] = (
        (offer_end - first_seen).dt.total_seconds() / 3600
    ).clip(lower=0)

    offer_start = pd.to_datetime(df["offer_start_time"])
    df["offer_start_dayofweek"] = offer_start.dt.dayofweek
    df["offer_start_hour_cest"] = offer_start.dt.hour

    log.info("Engineered time features (hours_until_offer_end, offer_start_dayofweek, offer_start_hour_cest)")
    return df


def apply_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.
    Importable by predict.py so inference uses the same transformations as training.
    """
    df = engineer_category(df)
    df = engineer_stock_unit(df)
    df = engineer_customer_flow(df)
    df = engineer_store_hours(df)
    df = engineer_time_features(df)
    return df


def save_features(df: pd.DataFrame) -> str:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    today       = date.today().strftime("%Y%m%d")
    output_path = FEATURES_DIR / f"features_{today}.parquet"
    df.to_parquet(output_path, index=False)
    log.info(f"Saved features to {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    return str(output_path)


def main():
    df = load_dataset()
    df = apply_all(df)

    new_cols = [
        c for c in df.columns if c.startswith((
            "category_", "flow_", "store_open", "hours_until",
            "is_weight", "is_closed", "offer_start_day", "offer_start_hour",
        ))
    ]
    log.info(f"New feature columns ({len(new_cols)}): {new_cols}")

    save_features(df)


if __name__ == "__main__":
    main()
