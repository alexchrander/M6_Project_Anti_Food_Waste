import sys
from pathlib import Path

# predict.py lives in fetch_prediction_pipeline/ — project root is one level up.
# All three source directories must be on the path explicitly.
sys.path.append(str(Path(__file__).parent.parent))                                # project root → config.py
sys.path.append(str(Path(__file__).parent.parent / "fetch_prediction_pipeline")) # store_sql.py
sys.path.append(str(Path(__file__).parent.parent / "ml_pipeline"))               # build_dataset, build_features, preprocessing

import csv
import json
import logging
import pandas as pd
from datetime import datetime
import time

from store_sql import get_connection, init_app_table, store_app_table
from build_dataset import parse_timestamps, compute_lifecycle_features
from build_features import apply_all as apply_feature_engineering
from preprocessing import drop_columns, preprocess_for_inference
from config import (
    MODELS_DIR, OUTPUTS_DIR, PREDICTIONS_DIR,
    APP_COLS, PREDICTIONS_PARQUET_COLS,
    SELL_THRESHOLD, PR_AUC_THRESHOLD, PREDICTION_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

PREDICT_LOG = OUTPUTS_DIR / "predictions_log.csv"

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both current and history tables from MySQL.
    Current gives us the live offers to predict on.
    History gives us their snapshot history to compute lifecycle features.
    """
    conn    = get_connection()
    current = pd.read_sql("SELECT * FROM current", conn)
    history = pd.read_sql("SELECT * FROM history", conn)
    conn.close()
    log.info(f"Loaded {len(current)} current offers and {len(history)} history rows")
    return current, history

# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(X: pd.DataFrame) -> tuple[list[int], list[float], str, str]:
    """
    Load the champion model and score all current offers.
    Uses PREDICTION_THRESHOLD from config for binary classification.
    Returns binary predictions, raw probabilities, model type and trained_on date.
    """
    import joblib

    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"model.joblib not found in {MODELS_DIR} — run evaluate.py first"
        )

    model = joblib.load(model_path)
    log.info(f"Loaded champion model from {model_path}")

    # Read champion metadata for logging
    meta_path     = MODELS_DIR / "champion.json"
    champion_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model_type    = champion_meta.get("model_type", "unknown")
    run_id        = champion_meta.get("mlflow_run_id", "")
    trained_on    = champion_meta.get("trained_on", "unknown")

    # Align columns to what the model was trained on
    expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else X.columns
    missing       = [c for c in expected_cols if c not in X.columns]
    if missing:
        log.warning(f"Missing features — filling with 0: {missing}")
        for c in missing:
            X[c] = 0
    X = X[expected_cols]

    y_prob = model.predict_proba(X)[:, 1]                  # probability of selling
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)  # uses config threshold

    log.info(f"Using prediction threshold: {PREDICTION_THRESHOLD}")
    return y_pred.tolist(), y_prob.tolist(), model_type, run_id, trained_on

# ── Output ─────────────────────────────────────────────────────────────────────

def build_app_table(df: pd.DataFrame, y_pred: list, y_prob: list, predicted_at: str) -> pd.DataFrame:
    """
    Build the app table for MySQL (app.py serving layer).
    Columns are defined by APP_COLS in config.py.
    """
    available = [c for c in APP_COLS if c in df.columns]
    missing   = [c for c in APP_COLS if c not in df.columns]
    if missing:
        log.warning(f"APP_COLS — missing columns, skipping: {missing}")

    results                     = df[available].copy()
    results["will_sell"]        = y_pred
    results["sell_probability"] = [round(p, 4) for p in y_prob]
    results["predicted_at"]     = predicted_at

    # Sort by sell probability descending — most likely to sell first
    results = results.sort_values("sell_probability", ascending=False).reset_index(drop=True)
    return results


def build_predictions_parquet(df: pd.DataFrame, y_pred: list, y_prob: list, predicted_at: str) -> pd.DataFrame:
    """
    Build the full predictions snapshot for parquet (audit trail + debugging).
    Columns are defined by PREDICTIONS_PARQUET_COLS in config.py.
    """
    available = [c for c in PREDICTIONS_PARQUET_COLS if c in df.columns]
    missing   = [c for c in PREDICTIONS_PARQUET_COLS if c not in df.columns]
    if missing:
        log.warning(f"PREDICTIONS_PARQUET_COLS — missing columns, skipping: {missing}")

    snapshot                     = df[available].copy()
    snapshot["will_sell"]        = y_pred
    snapshot["sell_probability"] = [round(p, 4) for p in y_prob]
    snapshot["predicted_at"]     = predicted_at
    return snapshot


def save_predictions_parquet(snapshot: pd.DataFrame, predicted_at: str) -> Path:
    """Save predictions snapshot to data/predictions/scored_YYYYMMDD_HHMM.parquet."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts          = predicted_at.replace("-", "").replace(":", "").replace(" ", "_")[:13]
    output_path = PREDICTIONS_DIR / f"scored_{ts}.parquet"
    snapshot.to_parquet(output_path, index=False)
    log.info(f"Saved predictions parquet to {output_path}")
    return output_path

# ── Logging ────────────────────────────────────────────────────────────────────

def log_predict_run(run: dict) -> None:
    """
    Append one row to outputs/predict_log.csv after every prediction run.
    Creates the file with headers if it doesn't exist yet.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "duration_seconds",
        "sell_threshold",
        "pr_auc_threshold",
        "prediction_threshold",
        "n_offers_scored",
        "n_will_sell",
        "pct_will_sell",
        "avg_sell_probability",
        "champion_model",
        "champion_run_id",
        "champion_trained_on",
        "status",
        "failed_reason",
    ]

    write_header = not PREDICT_LOG.exists()
    with open(PREDICT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(run)

    log.info(f"Prediction run logged to {PREDICT_LOG}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting predict.py")
    start_time   = time.time()
    predicted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        current, history = load_data()

        if current.empty:
            log.warning("No current offers found — exiting")
            log_predict_run({
                "timestamp":            predicted_at,
                "duration_seconds":     round(time.time() - start_time, 2),
                "sell_threshold":       SELL_THRESHOLD,
                "pr_auc_threshold":     PR_AUC_THRESHOLD,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "n_offers_scored":      0,
                "n_will_sell":          0,
                "pct_will_sell":        0,
                "avg_sell_probability": 0,
                "champion_model":       "",
                "champion_run_id":      "",
                "champion_trained_on":  "",
                "status":               "success",
                "failed_reason":        "",
            })
            return

        history = parse_timestamps(history)

        active_ids     = set(current["unique_id"])
        active_history = history[history["unique_id"].isin(active_ids)].copy()

        df = compute_lifecycle_features(active_history)

        # Bring fetched_at and offer_stock from current into df (df is built from history, not current)
        df = df.merge(current[["unique_id", "fetched_at", "offer_stock"]], on="unique_id", how="left")

        # potential_relabelling — default to False for live offers
        # (can't detect until offer completes — handled in compute_labels in build_dataset.py)
        df["potential_relabelling"] = False

        # Apply the same feature engineering used during training
        df = apply_feature_engineering(df)

        # Preprocess — encode and scale using fitted artifacts from preprocessing.py
        X = drop_columns(df)
        X = preprocess_for_inference(X)

        y_pred, y_prob, model_type, run_id, trained_on = predict(X)

        # ── Build outputs ──────────────────────────────────────────────────────
        app_table = build_app_table(df, y_pred, y_prob, predicted_at)
        snapshot  = build_predictions_parquet(df, y_pred, y_prob, predicted_at)

        # ── Save predictions parquet ───────────────────────────────────────────
        save_predictions_parquet(snapshot, predicted_at)

        n_will_sell   = int(app_table["will_sell"].sum())
        pct_will_sell = round(app_table["will_sell"].mean(), 4)
        avg_sell_prob = round(app_table["sell_probability"].mean(), 4)

        log.info(f"Predicted {len(app_table)} offers")
        log.info(f"Will sell:  {n_will_sell} ({pct_will_sell:.1%})")
        log.info(f"Won't sell: {len(app_table) - n_will_sell} ({1 - pct_will_sell:.1%})")
        log.info(f"Avg sell probability: {avg_sell_prob:.4f}")

        # ── Write to MySQL app table ───────────────────────────────────────────
        conn = get_connection()
        init_app_table(conn)
        store_app_table(conn, app_table)
        conn.close()
        log.info(f"Wrote {len(app_table)} rows to MySQL app table")

        log_predict_run({
            "timestamp":            predicted_at,
            "duration_seconds":     round(time.time() - start_time, 2),
            "sell_threshold":       SELL_THRESHOLD,
            "pr_auc_threshold":     PR_AUC_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "n_offers_scored":      len(app_table),
            "n_will_sell":          n_will_sell,
            "pct_will_sell":        pct_will_sell,
            "avg_sell_probability": avg_sell_prob,
            "champion_model":       model_type,
            "champion_run_id":      run_id,
            "champion_trained_on":  trained_on,
            "status":               "success",
            "failed_reason":        "",
        })

        log.info("predict.py complete")

    except Exception as e:
        log.error(f"predict.py failed: {e}")
        log_predict_run({
            "timestamp":            predicted_at,
            "duration_seconds":     round(time.time() - start_time, 2),
            "sell_threshold":       SELL_THRESHOLD,
            "pr_auc_threshold":     PR_AUC_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "n_offers_scored":      "",
            "n_will_sell":          "",
            "pct_will_sell":        "",
            "avg_sell_probability": "",
            "champion_model":       "",
            "champion_run_id":      "",
            "champion_trained_on":  "",
            "status":               "failed",
            "failed_reason":        str(e),
        })
        raise


if __name__ == "__main__":
    main()
