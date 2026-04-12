import sys
from pathlib import Path

# Add project root to path so config.py is found
sys.path.append(str(Path(__file__).parent.parent))

import logging
import glob
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from datetime import date
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from config import FEATURES_DIR, MLRUNS_DIR, N_ESTIMATORS, LEARNING_RATE, MAX_ITER, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_train_split() -> tuple[pd.DataFrame, pd.Series, str]:
    """Load the most recent train split from data/features/."""
    files = glob.glob(str(FEATURES_DIR / "train_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train parquet files found in {FEATURES_DIR}")

    latest = sorted(files)[-1]
    log.info(f"Loading train split: {latest}")
    df = pd.read_parquet(latest)

    X = df.drop(columns=["will_sell"])
    y = df["will_sell"]

    log.info(f"Train size: {len(X)}, Positive rate: {y.mean():.1%}")
    return X, y, latest

# ── Training ───────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train) -> LogisticRegression:
    """Train a simple Logistic Regression as our baseline model."""
    model = LogisticRegression(
        solver="saga",
        max_iter=MAX_ITER,
        class_weight="balanced",  # handles class imbalance
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_lgbm(X_train, y_train) -> LGBMClassifier:
    """Train a LightGBM classifier as our main model."""
    model = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        class_weight="balanced",  # handles class imbalance
        random_state=RANDOM_STATE,
        verbose=-1,               # suppress LightGBM output
    )
    model.fit(X_train, y_train)
    return model

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today = date.today().strftime("%Y%m%d")

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment("sell_through_prediction")

    X_train, y_train, train_path = load_train_split()

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    with mlflow.start_run(run_name=f"logistic_regression_{today}"):

        baseline = train_baseline(X_train, y_train)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_iter", MAX_ITER)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_artifact(train_path)
        mlflow.sklearn.log_model(baseline, "model")

        log.info(f"Logistic Regression logged to MLflow as logistic_regression_{today}")

    # ── Main model: LightGBM ───────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"lightgbm_{today}"):

        lgbm = train_lgbm(X_train, y_train)

        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_artifact(train_path)
        mlflow.lightgbm.log_model(lgbm, "model")

        log.info(f"LightGBM logged to MLflow as lightgbm_{today}")

    log.info("Training complete — run evaluate.py to pick the champion")


if __name__ == "__main__":
    main()