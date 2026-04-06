import sys
from pathlib import Path

# Add project root to path so config.py is found
sys.path.append(str(Path(__file__).parent.parent))

import logging
import glob
import joblib
import pandas as pd
from datetime import date
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from config import (
    FEATURES_DIR, MODELS_DIR,
    SELL_THRESHOLD, TEST_SIZE, RANDOM_STATE,
    DROP_COLS, NUMERIC_COLS, CATEGORICAL_COLS, DATETIME_COLS, BOOLEAN_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_latest_features() -> pd.DataFrame:
    """
    Load the most recent features parquet from data/features/.

    TODO: Once feature_engineering.py exists, this will load its output directly.
    For now, falls back to loading from data/dataset/.
    """
    files = glob.glob(str(FEATURES_DIR / "features_*.parquet"))

    if files:
        latest = sorted(files)[-1]
        log.info(f"Loading features: {latest}")
        return pd.read_parquet(latest)

    # Fallback — load raw dataset if no features file exists yet
    log.warning("No features parquet found — falling back to raw dataset")
    dataset_dir = FEATURES_DIR.parent / "dataset"
    files = glob.glob(str(dataset_dir / "labelled_offers_*.parquet"))
    if not files:
        raise FileNotFoundError("No dataset or features parquet files found")
    latest = sorted(files)[-1]
    log.info(f"Loading raw dataset as fallback: {latest}")
    return pd.read_parquet(latest)

# ── Preprocessing steps ────────────────────────────────────────────────────────

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful as features.
    Used by both preprocessing.py (training) and predict.py (inference).
    Only drops columns that are actually present — safe for both contexts.
    """
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def create_target(df: pd.DataFrame) -> pd.Series:
    """
    Create binary target from sell_through_rate.
    Training only — predict.py does not use this.
    """
    y = (df["sell_through_rate"] >= SELL_THRESHOLD).astype(int)
    log.info(f"Target — sells: {y.sum()} ({y.mean():.1%}), no sell: {(1-y).sum()} ({(1-y.mean()):.1%})")
    return y


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all non-numeric feature types.
    Used by both preprocessing.py (training) and predict.py (inference).
    - Datetime → hour + dayofweek
    - Categorical → label encoded integer
    - Boolean → integer (0/1)
    """
    X = X.copy()

    # ── Datetime: extract hour + day of week, drop originals ──────────────────
    for col in DATETIME_COLS:
        if col in X.columns:
            X[col]                = pd.to_datetime(X[col])
            X[f"{col}_hour"]      = X[col].dt.hour
            X[f"{col}_dayofweek"] = X[col].dt.dayofweek
    X = X.drop(columns=[c for c in DATETIME_COLS if c in X.columns])

    # ── Categorical: fill nulls, label encode ─────────────────────────────────
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown")
            le     = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # ── Boolean: convert to int (0/1) ─────────────────────────────────────────
    for col in BOOLEAN_COLS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    return X


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit StandardScaler on train split only, apply to both splits.
    Saves fitted scaler to models/scaler.joblib for use by predict.py and the API.
    Training only — predict.py loads the saved scaler instead of fitting a new one.
    """
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    log.info(f"Saved scaler to {MODELS_DIR / 'scaler.joblib'}")

    return X_train, X_test

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today = date.today().strftime("%Y%m%d")

    df = load_latest_features()

    # Create target and drop unwanted columns
    y = create_target(df)
    X = drop_columns(df)
    X = encode_features(X)

    log.info(f"Feature matrix shape after encoding: {X.shape}")
    log.info(f"Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    X_train, X_test = scale_features(X_train, X_test)

    # Add target column so train.py and evaluate.py can load it directly
    train_df              = X_train.copy()
    train_df["will_sell"] = y_train.values

    test_df              = X_test.copy()
    test_df["will_sell"] = y_test.values

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    train_path = FEATURES_DIR / f"train_{today}.parquet"
    test_path  = FEATURES_DIR / f"test_{today}.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    log.info(f"Saved train split to {train_path} ({len(train_df)} rows)")
    log.info(f"Saved test split  to {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()