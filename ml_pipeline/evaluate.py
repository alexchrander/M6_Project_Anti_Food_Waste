import sys
from pathlib import Path

# Add project root to path so config.py is found
sys.path.append(str(Path(__file__).parent.parent))

import logging
import glob
import json
import joblib
import argparse
import pandas as pd
import mlflow
from datetime import date
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    log_loss,
)

from config import (
    FEATURES_DIR, MODELS_DIR, MLRUNS_DIR,
    SELL_THRESHOLD, PR_AUC_THRESHOLD, PREDICTION_THRESHOLD,
)
from preprocessing import promote_candidate_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    """Load the most recent test split from data/features/."""
    files = glob.glob(str(FEATURES_DIR / "test_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No test parquet files found in {FEATURES_DIR}")

    latest = sorted(files)[-1]
    log.info(f"Loading test split: {latest}")
    df = pd.read_parquet(latest)

    X = df.drop(columns=["will_sell"])
    y = df["will_sell"]

    log.info(f"Test size: {len(X)}, Positive rate: {y.mean():.1%}")
    return X, y

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """
    Score a model on the test set.
    PR AUC is the primary metric — all others are informational only.
    Uses PREDICTION_THRESHOLD from config for binary classification metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)

    metrics = {
        "pr_auc":    average_precision_score(y_test, y_prob),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "log_loss":  log_loss(y_test, y_prob),
    }

    log.info(f"\n{model_name} metrics (threshold={PREDICTION_THRESHOLD}):")
    log.info(f"  PR AUC    (primary):  {metrics['pr_auc']:.4f}")
    log.info(f"  F1                :  {metrics['f1']:.4f}")
    log.info(f"  Recall            :  {metrics['recall']:.4f}")
    log.info(f"  Precision         :  {metrics['precision']:.4f}")
    log.info(f"  Log Loss          :  {metrics['log_loss']:.4f}")

    return metrics


def load_champion() -> tuple:
    """
    Load the current champion model and its metadata.
    Returns (model, meta) or (None, None) if no champion exists yet.
    """
    model_path = MODELS_DIR / "model.joblib"
    meta_path  = MODELS_DIR / "champion.json"

    if not model_path.exists() or not meta_path.exists():
        log.info("No champion model found yet")
        return None, None

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    log.info(f"Loaded champion: {meta['model_type']} | run {meta.get('mlflow_run_id', 'unknown')[:8]}... | trained on {meta['trained_on']} (PR AUC: {meta['pr_auc']})")
    return model, meta


def load_latest_mlflow_runs() -> dict:
    """
    Load today's trained models from MLflow artifacts.
    Returns a dict keyed by model_type, each containing the model and its run ID.
    """
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("sell_through_prediction")
    if not experiment:
        raise ValueError("MLflow experiment 'sell_through_prediction' not found — run train.py first")

    today = date.today().strftime("%Y%m%d")
    runs  = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name LIKE '%{today}%'",
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise ValueError(f"No MLflow runs found for today ({today}) — run train.py first")

    models = {}
    for run in runs:
        model_type   = run.data.params.get("model_type")
        run_id       = run.info.run_id
        artifact_uri = f"runs:/{run_id}/model"

        if model_type == "logistic_regression" and "logistic_regression" not in models:
            models["logistic_regression"] = {
                "model":  mlflow.sklearn.load_model(artifact_uri),
                "run_id": run_id,
            }
            log.info(f"Loaded logistic_regression from run {run_id[:8]}...")

        elif model_type == "lightgbm" and "lightgbm" not in models:
            models["lightgbm"] = {
                "model":  mlflow.lightgbm.load_model(artifact_uri),
                "run_id": run_id,
            }
            log.info(f"Loaded lightgbm from run {run_id[:8]}...")

    if not models:
        raise ValueError("Could not load any models from MLflow — check that train.py logged model_type param")

    return models


def _threshold_meta() -> dict:
    """Return all three thresholds as a dict — added to every champion.json write."""
    return {
        "sell_threshold":       SELL_THRESHOLD,
        "pr_auc_threshold":     PR_AUC_THRESHOLD,
        "prediction_threshold": PREDICTION_THRESHOLD,
    }


def save_champion(model, model_type: str, run_id: str, metrics: dict):
    """Save a new champion model, promote candidate preprocessing artifacts, and update champion.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "model.joblib")
    log.info(f"Saved champion model to {MODELS_DIR / 'model.joblib'}")

    promote_candidate_artifacts()

    champion_meta = {
        "model_type":    model_type,
        "mlflow_run_id": run_id,
        "trained_on":    str(date.today()),
        **_threshold_meta(),
        "pr_auc":        round(metrics["pr_auc"], 4),
        "f1":            round(metrics["f1"], 4),
        "recall":        round(metrics["recall"], 4),
        "precision":     round(metrics["precision"], 4),
        "log_loss":      round(metrics["log_loss"], 4),
    }
    with open(MODELS_DIR / "champion.json", "w") as f:
        json.dump(champion_meta, f, indent=4)
    log.info(f"Updated champion.json: {champion_meta}")


def refresh_champion_score(meta: dict, metrics: dict):
    """
    Update champion.json with today's evaluated scores without replacing the model.
    Called when the current champion wins so scores always reflect today's evaluation.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    updated_meta = {
        **meta,
        **_threshold_meta(),
        "pr_auc":        round(metrics["pr_auc"], 4),
        "f1":            round(metrics["f1"], 4),
        "recall":        round(metrics["recall"], 4),
        "precision":     round(metrics["precision"], 4),
        "log_loss":      round(metrics["log_loss"], 4),
        "evaluated_on":  str(date.today()),
    }
    with open(MODELS_DIR / "champion.json", "w") as f:
        json.dump(updated_meta, f, indent=4)
    log.info(f"Refreshed champion.json with today's winning scores")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["check", "compare"],
        required=True,
        help="check: test current champion. compare: evaluate new models and promote best."
    )
    args = parser.parse_args()

    X_test, y_test = load_test_split()

    # ── Mode 1: check ──────────────────────────────────────────────────────────
    if args.mode == "check":
        champion, meta = load_champion()

        if champion is None:
            log.info("No champion exists yet — retraining needed")
            sys.exit(1)

        metrics = evaluate_model(champion, X_test, y_test, "Champion")
        refresh_champion_score(meta, metrics)

        if metrics["pr_auc"] < PR_AUC_THRESHOLD:
            log.info(f"Champion PR AUC {metrics['pr_auc']:.4f} below threshold {PR_AUC_THRESHOLD} — retraining needed")
            sys.exit(1)
        else:
            log.info(f"Champion PR AUC {metrics['pr_auc']:.4f} is above threshold — no retraining needed")
            sys.exit(0)

    # ── Mode 2: compare ────────────────────────────────────────────────────────
    if args.mode == "compare":
        champion, champion_meta = load_champion()

        is_bootstrap    = champion is None
        champion_pr_auc = champion_meta["pr_auc"] if champion_meta else 0.0

        new_models = load_latest_mlflow_runs()
        results    = {}
        for model_type, entry in new_models.items():
            metrics = evaluate_model(entry["model"], X_test, y_test, model_type)
            results[model_type] = {
                "model":   entry["model"],
                "run_id":  entry["run_id"],
                "metrics": metrics,
            }

        best_name   = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
        best        = results[best_name]
        best_pr_auc = best["metrics"]["pr_auc"]

        log.info(f"\nBest new model: {best_name} (PR AUC: {best_pr_auc:.4f})")

        if is_bootstrap:
            log.info("No previous champion — promoting best new model unconditionally")
            save_champion(best["model"], best_name, best["run_id"], best["metrics"])

        elif best_pr_auc > champion_pr_auc:
            log.info(f"New model (PR AUC: {best_pr_auc:.4f}) beats champion (PR AUC: {champion_pr_auc:.4f}) — promoting")
            save_champion(best["model"], best_name, best["run_id"], best["metrics"])

        else:
            log.info(f"Champion (PR AUC: {champion_pr_auc:.4f}) still best — keeping current model.joblib")


if __name__ == "__main__":
    main()