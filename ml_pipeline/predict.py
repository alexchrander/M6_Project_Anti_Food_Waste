import sys
from pathlib import Path

# Add project root to path so config.py is found
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "fetch_pipeline"))

import csv
import json
import logging
import joblib
import pandas as pd
from datetime import datetime
import time

from store_sql import get_connection
from build_dataset import parse_timestamps, compute_lifecycle_features
from preprocessing import drop_columns, encode_features
from config import (
    MODELS_DIR, NUMERIC_COLS, OUTPUTS_DIR,
    SELL_THRESHOLD, PR_AUC_THRESHOLD, PREDICTION_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

PREDICT_LOG = OUTPUTS_DIR / "predict_log.csv"

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

# ── Scaling ────────────────────────────────────────────────────────────────────

def scale(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the saved scaler to numeric columns.
    Uses the same scaler fitted during preprocessing.py — do NOT refit.
    """
    scaler_path = MODELS_DIR / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"scaler.joblib not found in {MODELS_DIR} — run preprocessing.py first"
        )

    scaler        = joblib.load(scaler_path)
    cols_to_scale = [c for c in NUMERIC_COLS if c in X.columns]
    if cols_to_scale:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X

# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(X: pd.DataFrame) -> tuple[list[int], list[float], str, str]:
    """
    Load the champion model and score all current offers.
    Uses PREDICTION_THRESHOLD from config for binary classification.
    Returns binary predictions, raw probabilities, model type and trained_on date.
    """
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
    return y_pred.tolist(), y_prob.tolist(), model_type, trained_on

# ── Output ─────────────────────────────────────────────────────────────────────

def build_results(display_cols: pd.DataFrame, y_pred: list, y_prob: list) -> pd.DataFrame:
    """
    Join predictions back to display columns to produce the final results table.
    This is what will eventually be written to the predictions MySQL table.
    """
    results                     = display_cols.copy()
    results["will_sell"]        = y_pred
    results["sell_probability"] = [round(p, 4) for p in y_prob]
    results["predicted_at"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sort by sell probability descending — most likely to sell first
    results = results.sort_values("sell_probability", ascending=False).reset_index(drop=True)

    return results

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
    start_time = time.time()
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        current, history = load_data()

        if current.empty:
            log.warning("No current offers found — exiting")
            log_predict_run({
                "timestamp":            timestamp,
                "duration_seconds":     round(time.time() - start_time, 2),
                "sell_threshold":       SELL_THRESHOLD,
                "pr_auc_threshold":     PR_AUC_THRESHOLD,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "n_offers_scored":      0,
                "n_will_sell":          0,
                "pct_will_sell":        0,
                "avg_sell_probability": 0,
                "champion_model":       "",
                "champion_trained_on":  "",
                "status":               "success",
                "failed_reason":        "",
            })
            return

        history = parse_timestamps(history)

        active_ids     = set(current["unique_id"])
        active_history = history[history["unique_id"].isin(active_ids)].copy()

        df, _ = compute_lifecycle_features(active_history)

        # potential_relabelling — default to False for live offers
        # (can't detect until offer completes — handled in compute_labels in build_dataset.py)
        df["potential_relabelling"] = False

        display_cols = df[[
            "unique_id", "store_name", "store_brand", "store_city",
            "store_lat", "store_lng",
            "product_description", "product_category_en",
            "offer_new_price", "offer_original_price", "offer_percent_discount",
            "offer_stock_unit", "offer_end_time",
        ]].copy()

        X = drop_columns(df)
        X = encode_features(X)

        non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            log.warning(f"Dropping non-numeric columns: {non_numeric}")
            X = X.drop(columns=non_numeric)

        X = scale(X)

        y_pred, y_prob, model_type, trained_on = predict(X)

        results = build_results(display_cols, y_pred, y_prob)

        n_will_sell   = int(results["will_sell"].sum())
        pct_will_sell = round(results["will_sell"].mean(), 4)
        avg_sell_prob = round(results["sell_probability"].mean(), 4)

        log.info(f"Predicted {len(results)} offers")
        log.info(f"Will sell:  {n_will_sell} ({pct_will_sell:.1%})")
        log.info(f"Won't sell: {len(results) - n_will_sell} ({1 - pct_will_sell:.1%})")
        log.info(f"Avg sell probability: {avg_sell_prob:.4f}")

        output_path = Path(__file__).parent.parent / "test_data" / "predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        log.info(f"Saved predictions to {output_path}")

        # ══════════════════════════════════════════════════════════════════════
        # TODO: Write results to MySQL predictions table once schema is agreed
        # Something like:
        #   store_predictions(conn, results)
        # ══════════════════════════════════════════════════════════════════════

        log_predict_run({
            "timestamp":            timestamp,
            "duration_seconds":     round(time.time() - start_time, 2),
            "sell_threshold":       SELL_THRESHOLD,
            "pr_auc_threshold":     PR_AUC_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "n_offers_scored":      len(results),
            "n_will_sell":          n_will_sell,
            "pct_will_sell":        pct_will_sell,
            "avg_sell_probability": avg_sell_prob,
            "champion_model":       model_type,
            "champion_trained_on":  trained_on,
            "status":               "success",
            "failed_reason":        "",
        })

        log.info("predict.py complete")

    except Exception as e:
        log.error(f"predict.py failed: {e}")
        log_predict_run({
            "timestamp":            timestamp,
            "duration_seconds":     round(time.time() - start_time, 2),
            "sell_threshold":       SELL_THRESHOLD,
            "pr_auc_threshold":     PR_AUC_THRESHOLD,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "n_offers_scored":      "",
            "n_will_sell":          "",
            "pct_will_sell":        "",
            "avg_sell_probability": "",
            "champion_model":       "",
            "champion_trained_on":  "",
            "status":               "failed",
            "failed_reason":        str(e),
        })
        raise


if __name__ == "__main__":
    main()