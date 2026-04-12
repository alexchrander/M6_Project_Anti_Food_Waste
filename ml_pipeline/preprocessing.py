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

from config import (
    FEATURES_DIR, MODELS_DIR,
    SELL_THRESHOLD,
    DROP_COLS, NUMERIC_COLS, CATEGORICAL_COLS, ONEHOT_COLS, DATETIME_COLS, BOOLEAN_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

ENCODER_PATH       = MODELS_DIR / "encoder.joblib"
ONEHOT_COLS_PATH   = MODELS_DIR / "onehot.joblib"
SCALER_PATH        = MODELS_DIR / "scaler.joblib"

# ── Data loading ───────────────────────────────────────────────────────────────

def load_latest_features() -> pd.DataFrame:
    """Load the most recent features parquet from data/features/."""
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

# ── Target + drop ──────────────────────────────────────────────────────────────

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

# ── Encoding ───────────────────────────────────────────────────────────────────

def save_label_encoders(encoders: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, ENCODER_PATH)
    log.info(f"Saved {len(encoders)} label encoders to {ENCODER_PATH}")


def load_label_encoders() -> dict:
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"encoder.joblib not found in {MODELS_DIR} — run preprocessing.py first"
        )
    return joblib.load(ENCODER_PATH)


def save_onehot_columns(onehot_columns: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(onehot_columns, ONEHOT_COLS_PATH)
    log.info(f"Saved {len(onehot_columns)} one-hot column names to {ONEHOT_COLS_PATH}")


def load_onehot_columns() -> list:
    if not ONEHOT_COLS_PATH.exists():
        raise FileNotFoundError(
            f"onehot.joblib not found in {MODELS_DIR} — run preprocessing.py first"
        )
    return joblib.load(ONEHOT_COLS_PATH)


def encode_features(
    X: pd.DataFrame,
    encoders: dict | None = None,
    onehot_columns: list | None = None,
) -> tuple[pd.DataFrame, dict, list]:
    """
    Encode all non-numeric feature types.
    Used by both preprocessing.py (training) and predict.py (inference).

    - Datetime    → extract hour + dayofweek, drop originals
    - Categorical → LabelEncoder (fit on training, applied at inference)
    - One-hot     → pd.get_dummies (no ordinal assumption — for cyclic/time features)
    - Boolean     → cast to int (0/1)

    Training   (encoders=None):     fits encoders, generates dummies, saves column names
    Inference  (encoders=dict):     applies fitted encoders, aligns dummy columns to training layout

    Returns (X_encoded, encoders_dict, onehot_columns_list).
    """
    X = X.copy()
    fitted_encoders = encoders or {}

    # ── Datetime: extract hour + day of week, drop originals ──────────────────
    for col in DATETIME_COLS:
        if col in X.columns:
            X[col]                = pd.to_datetime(X[col])
            X[f"{col}_hour"]      = X[col].dt.hour
            X[f"{col}_dayofweek"] = X[col].dt.dayofweek
    X = X.drop(columns=[c for c in DATETIME_COLS if c in X.columns])

    # ── Categorical: fill nulls, label encode ─────────────────────────────────
    for col in CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        X[col] = X[col].fillna("Unknown").astype(str)

        if col in fitted_encoders:
            # Inference: map unseen categories to "Unknown", then encode
            le    = fitted_encoders[col]
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else "Unknown")
            X[col] = le.transform(X[col])
        else:
            # Training: always include "Unknown" so inference can use it
            all_values = sorted(set(X[col].unique()) | {"Unknown"})
            le = LabelEncoder()
            le.fit(all_values)
            X[col] = le.transform(X[col])
            fitted_encoders[col] = le

    # ── One-hot: pd.get_dummies for cyclic/time features ─────────────────────
    onehot_cols_present = [c for c in ONEHOT_COLS if c in X.columns]
    if onehot_cols_present:
        X = pd.get_dummies(X, columns=onehot_cols_present, prefix=onehot_cols_present, dtype=int)

        if onehot_columns is not None:
            # Inference: align to training columns — add missing as 0, drop extras
            for col in onehot_columns:
                if col not in X.columns:
                    X[col] = 0
            generated = [c for c in X.columns if any(c.startswith(f"{b}_") for b in onehot_cols_present)]
            extra     = [c for c in generated if c not in onehot_columns]
            if extra:
                X = X.drop(columns=extra)
            fitted_onehot_columns = onehot_columns
        else:
            # Training: save the generated column names for inference alignment
            fitted_onehot_columns = [
                c for c in X.columns
                if any(c.startswith(f"{b}_") for b in onehot_cols_present)
            ]
            log.info(f"One-hot encoded {len(onehot_cols_present)} columns → {len(fitted_onehot_columns)} dummy columns")
    else:
        fitted_onehot_columns = onehot_columns or []

    # ── Boolean: convert to int (0/1) ─────────────────────────────────────────
    for col in BOOLEAN_COLS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    return X, fitted_encoders, fitted_onehot_columns

# ── Scaling ────────────────────────────────────────────────────────────────────

def save_scaler(scaler: StandardScaler) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Saved scaler to {SCALER_PATH}")


def load_scaler() -> StandardScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"scaler.joblib not found in {MODELS_DIR} — run preprocessing.py first"
        )
    return joblib.load(SCALER_PATH)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit StandardScaler on train split only, apply to both splits.
    Saves fitted scaler to models/scaler.joblib for use at inference.
    Training only — inference uses preprocess_for_inference() instead.
    """
    cols_to_scale = [c for c in NUMERIC_COLS if c in X_train.columns]

    X_train = X_train.copy()
    X_test  = X_test.copy()

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    save_scaler(scaler)
    return X_train, X_test


def preprocess_for_inference(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all fitted preprocessing artifacts to a single DataFrame at inference time.
    Loads encoder.joblib, onehot.joblib, and scaler.joblib — never refits.
    Call this in predict.py instead of encoding and scaling separately.
    """
    encoders, onehot_columns = load_label_encoders(), load_onehot_columns()
    X, _, _ = encode_features(X, encoders=encoders, onehot_columns=onehot_columns)

    scaler = load_scaler()
    cols_to_scale = [c for c in NUMERIC_COLS if c in X.columns]
    if cols_to_scale:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X

# ── Train/test split ───────────────────────────────────────────────────────────

def split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based train/test split: oldest 80% → train, newest 20% → test.
    Assumes X is already sorted by last_seen ascending.
    No random state needed — split is fully deterministic via sort order.
    """
    cutoff  = int(len(X) * 0.8)
    X_train = X.iloc[:cutoff].copy()
    X_test  = X.iloc[cutoff:].copy()
    y_train = y.iloc[:cutoff].copy()
    y_test  = y.iloc[cutoff:].copy()

    log.info(f"Time-based split — Train: {len(X_train)}, Test: {len(X_test)}")
    log.info(f"Train positive rate: {y_train.mean():.1%}, Test positive rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,   y_test: pd.Series,
    today: str,
) -> None:
    """Save train and test splits to data/features/ as parquet."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df              = X_train.copy()
    train_df["will_sell"] = y_train.values
    test_df               = X_test.copy()
    test_df["will_sell"]  = y_test.values

    train_path = FEATURES_DIR / f"train_{today}.parquet"
    test_path  = FEATURES_DIR / f"test_{today}.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,   index=False)

    log.info(f"Saved train split to {train_path} ({len(train_df)} rows)")
    log.info(f"Saved test split  to {test_path}  ({len(test_df)} rows)")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today = date.today().strftime("%Y%m%d")

    df = load_latest_features()

    # Sort by last_seen ascending — temporal anchor for time-based split
    df = df.sort_values("last_seen").reset_index(drop=True)

    y = create_target(df)
    X = drop_columns(df)

    # Encode — fit and save label encoders + one-hot column names
    X, encoders, onehot_columns = encode_features(X)
    save_label_encoders(encoders)
    save_onehot_columns(onehot_columns)

    log.info(f"Feature matrix shape after encoding: {X.shape}")
    log.info(f"Features: {list(X.columns)}")

    # Split — time-based, deterministic
    X_train, X_test, y_train, y_test = split(X, y)

    # Scale — fit on train only, apply to both, save scaler
    X_train, X_test = scale_features(X_train, X_test)

    # Save splits
    save_splits(X_train, X_test, y_train, y_test, today)


if __name__ == "__main__":
    main()