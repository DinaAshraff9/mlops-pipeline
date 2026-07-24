"""
MLOps Pipeline - Model Evaluation & Drift Detection
Uses Evidently AI to detect data drift and trigger retraining.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR        = Path(os.getenv("MODEL_OUTPUT_DIR", "/app/model/artifacts"))
DATA_PATH        = os.getenv("DATA_PATH", "/app/data/dataset.csv")
REFERENCE_PATH   = os.getenv("REFERENCE_DATA_PATH", "/app/data/reference.csv")
DRIFT_THRESHOLD  = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
REPORT_OUTPUT    = Path(os.getenv("REPORT_OUTPUT", "/app/reports"))

REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)


def load_reference_data() -> pd.DataFrame:
    if Path(REFERENCE_PATH).exists():
        return pd.read_csv(REFERENCE_PATH)
    # Use first 20% of training data as reference
    df = pd.read_csv(DATA_PATH)
    ref = df.head(int(len(df) * 0.2))
    ref.to_csv(REFERENCE_PATH, index=False)
    return ref


def run_drift_detection(current_data: pd.DataFrame) -> dict:
    """Detect data drift between reference and current data."""
    reference_data = load_reference_data()

    # Drop target column for drift detection
    ref = reference_data.iloc[:, :-1]
    cur = current_data.iloc[:, :-1]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    report_path = REPORT_OUTPUT / "drift_report.html"
    report.save_html(str(report_path))
    logger.info(f"Drift report saved to {report_path}")

    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    logger.info(f"Drift share: {drift_share:.2%}")

    return {
        "drift_share": drift_share,
        "drift_detected": drift_share > DRIFT_THRESHOLD,
        "report_path": str(report_path),
    }


def evaluate_model_performance(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate current model on new data."""
    model  = joblib.load(MODEL_DIR / "model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)

    from sklearn.metrics import accuracy_score, f1_score
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred, average="weighted"),
    }

    logger.info(f"Current model performance: {metrics}")

    # Save metrics
    with open(REPORT_OUTPUT / "performance.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    logger.info("Running evaluation pipeline...")
    df = pd.read_csv(DATA_PATH)

    drift_result = run_drift_detection(df)
    logger.info(f"Drift result: {drift_result}")

    if drift_result["drift_detected"]:
        logger.warning("⚠️  Data drift detected! Retraining may be required.")
        # In CI/CD, this triggers a new training run
        exit(1)
    else:
        logger.info("✅ No significant drift detected.")
        exit(0)
