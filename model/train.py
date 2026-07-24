"""
MLOps Pipeline - Model Training Script
Trains a classification model with experiment tracking via MLflow
"""

import os
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import optuna
import joblib
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-pipeline")
MODEL_NAME          = os.getenv("MODEL_NAME", "fraud-detector")
DATA_PATH           = os.getenv("DATA_PATH", "/app/data/dataset.csv")
MODEL_OUTPUT_DIR    = Path(os.getenv("MODEL_OUTPUT_DIR", "/app/model/artifacts"))
N_TRIALS            = int(os.getenv("OPTUNA_TRIALS", "30"))

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(path: str) -> tuple:
    """Load and preprocess dataset."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # Drop rows with nulls
    df.dropna(inplace=True)

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Dataset shape: {X.shape}, Target distribution:\n{y.value_counts()}")
    return X, y


# ─── Optuna Objective ─────────────────────────────────────────────────────────
def objective(trial, X_train, y_train):
    """Optuna objective for hyperparameter tuning."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),
    }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1)
    return score.mean()


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    X, y = load_data(DATA_PATH)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)

    best_params = study.best_params
    logger.info(f"Best params: {best_params}")

    # ── MLflow run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="rf-optimized") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Train final model
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall":    recall_score(y_test, y_pred, average="weighted"),
            "f1":        f1_score(y_test, y_pred, average="weighted"),
        }
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

        logger.info(f"Metrics: {metrics}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        # Log everything to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_optuna_trials", N_TRIALS)
        mlflow.log_param("dataset_rows", len(X))

        # Save artifacts
        joblib.dump(model,  MODEL_OUTPUT_DIR / "model.joblib")
        joblib.dump(scaler, MODEL_OUTPUT_DIR / "scaler.joblib")

        with open(MODEL_OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(MODEL_OUTPUT_DIR / "params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        mlflow.log_artifact(str(MODEL_OUTPUT_DIR / "metrics.json"))
        mlflow.log_artifact(str(MODEL_OUTPUT_DIR / "params.json"))

        # Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        logger.info(f"Model registered as '{MODEL_NAME}' in MLflow Registry.")

    # Promote best model to Production if F1 > 0.80
    if metrics["f1"] >= 0.80:
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(MODEL_NAME, stages=["None"])
        if latest:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest[0].version,
                stage="Production",
            )
            logger.info(f"Model v{latest[0].version} promoted to Production.")
    else:
        logger.warning(f"F1={metrics['f1']:.4f} < 0.80 — model NOT promoted.")

    return metrics


if __name__ == "__main__":
    metrics = train()
    logger.info("Training complete.")
