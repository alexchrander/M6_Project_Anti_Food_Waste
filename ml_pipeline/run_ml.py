import sys
from pathlib import Path

# Add project root to path so config.py is found
sys.path.append(str(Path(__file__).parent.parent))

import logging
import subprocess
import csv
import json
import glob
import pandas as pd
from datetime import datetime

from config import (
    MODELS_DIR, DATASET_DIR, FEATURES_DIR, OUTPUTS_DIR,
    SELL_THRESHOLD, PR_AUC_THRESHOLD, PREDICTION_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_LOG = OUTPUTS_DIR / "pipeline_log.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_step(script: str, args: list = []) -> int:
    """
    Run a pipeline step as a subprocess.
    Returns the exit code — 0 means success, anything else means failure.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "ml_pipeline" / script)] + args
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    log.info(f"Exit code: {result.returncode}")
    return result.returncode


def read_champion_meta() -> dict:
    """Read current champion.json if it exists, otherwise return empty dict."""
    meta_path = MODELS_DIR / "champion.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def read_dataset_stats() -> dict:
    """Read dataset size and positive rate from the latest train + test parquets."""
    stats = {
        "dataset_date":  "",
        "dataset_size":  "",
        "positive_rate": "",
    }

    dataset_files = glob.glob(str(DATASET_DIR / "labelled_offers_*.parquet"))
    if dataset_files:
        latest = sorted(dataset_files)[-1]
        stats["dataset_date"] = Path(latest).stem.replace("labelled_offers_", "")

    train_files = glob.glob(str(FEATURES_DIR / "train_*.parquet"))
    test_files  = glob.glob(str(FEATURES_DIR / "test_*.parquet"))

    if train_files and test_files:
        train_df = pd.read_parquet(sorted(train_files)[-1])
        test_df  = pd.read_parquet(sorted(test_files)[-1])
        combined = pd.concat([train_df, test_df])

        stats["dataset_size"]  = len(combined)
        stats["positive_rate"] = round(combined["will_sell"].mean(), 4)

    return stats


def log_pipeline_run(run: dict):
    """
    Append one row to pipeline_log.csv.
    Creates the file with headers if it doesn't exist yet.
    """
    PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "pipeline_duration_seconds",
        "sell_threshold",
        "pr_auc_threshold",
        "prediction_threshold",
        "dataset_date",
        "dataset_size",
        "positive_rate",
        "retrain_triggered",
        "champion_model",
        "champion_run_id",
        "champion_pr_auc",
        "champion_f1",
        "champion_precision",
        "champion_recall",
        "champion_log_loss",
        "champion_trained_on",
        "pipeline_status",
        "failed_step",
    ]

    write_header = not PIPELINE_LOG.exists()
    with open(PIPELINE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(run)

    log.info(f"Pipeline run logged to {PIPELINE_LOG}")


def abort(timestamp: str, start_time: datetime, retrain_triggered: bool, failed_step: str):
    """Log a failed run and exit."""
    log.error(f"{failed_step} failed — aborting pipeline")
    duration = round((datetime.now() - start_time).total_seconds())
    log_pipeline_run({
        "timestamp":                 timestamp,
        "pipeline_duration_seconds": duration,
        "sell_threshold":            SELL_THRESHOLD,
        "pr_auc_threshold":          PR_AUC_THRESHOLD,
        "prediction_threshold":      PREDICTION_THRESHOLD,
        "dataset_date":              "",
        "dataset_size":              "",
        "positive_rate":             "",
        "retrain_triggered":         retrain_triggered,
        "champion_model":            "",
        "champion_run_id":           "",
        "champion_pr_auc":           "",
        "champion_f1":               "",
        "champion_precision":        "",
        "champion_recall":           "",
        "champion_log_loss":         "",
        "champion_trained_on":       "",
        "pipeline_status":           "failed",
        "failed_step":               failed_step,
    })
    sys.exit(1)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    start_time        = datetime.now()
    timestamp         = start_time.strftime("%Y-%m-%d %H:%M:%S")
    retrain_triggered = False

    # ── Step 1: build_dataset.py ───────────────────────────────────────────────
    if run_step("build_dataset.py") != 0:
        abort(timestamp, start_time, retrain_triggered, "build_dataset")

    # ── Step 1.5: build_features.py ────────────────────────────────────────────
    if run_step("build_features.py") != 0:
        abort(timestamp, start_time, retrain_triggered, "build_features")

    # ── Step 2: preprocessing.py ───────────────────────────────────────────────
    if run_step("preprocessing.py") != 0:
        abort(timestamp, start_time, retrain_triggered, "preprocessing")

    # ── Step 3: evaluate.py --mode check ──────────────────────────────────────
    code = run_step("evaluate.py", ["--mode", "check"])

    if code == 0:
        log.info("Champion is performing well — skipping retraining")

    elif code == 1:
        log.info("Retraining triggered")
        retrain_triggered = True

        # ── Step 4: train.py ───────────────────────────────────────────────────
        if run_step("train.py") != 0:
            abort(timestamp, start_time, retrain_triggered, "train")

        # ── Step 5: evaluate.py --mode compare ────────────────────────────────
        if run_step("evaluate.py", ["--mode", "compare"]) != 0:
            abort(timestamp, start_time, retrain_triggered, "evaluate_compare")

    else:
        abort(timestamp, start_time, retrain_triggered, "evaluate_check")

    # ── Log outcome ────────────────────────────────────────────────────────────
    meta     = read_champion_meta()
    stats    = read_dataset_stats()
    duration = round((datetime.now() - start_time).total_seconds())

    log_pipeline_run({
        "timestamp":                 timestamp,
        "pipeline_duration_seconds": duration,
        "sell_threshold":            SELL_THRESHOLD,
        "pr_auc_threshold":          PR_AUC_THRESHOLD,
        "prediction_threshold":      PREDICTION_THRESHOLD,
        "dataset_date":              stats["dataset_date"],
        "dataset_size":              stats["dataset_size"],
        "positive_rate":             stats["positive_rate"],
        "retrain_triggered":         retrain_triggered,
        "champion_model":            meta.get("model_type", ""),
        "champion_run_id":           meta.get("mlflow_run_id", ""),
        "champion_pr_auc":           meta.get("pr_auc", ""),
        "champion_f1":               meta.get("f1", ""),
        "champion_precision":        meta.get("precision", ""),
        "champion_recall":           meta.get("recall", ""),
        "champion_log_loss":         meta.get("log_loss", ""),
        "champion_trained_on":       meta.get("trained_on", ""),
        "pipeline_status":           "success",
        "failed_step":               "",
    })

    log.info(f"Pipeline completed successfully in {duration}s")


if __name__ == "__main__":
    main()