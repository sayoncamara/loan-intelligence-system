"""
Loan Intelligence System — Training Pipeline
Orchestrated with Prefect. Implements:
- Structured logging at every step
- Retries with exponential backoff on external calls
- Evaluation gate: stops deployment if AUC < threshold
- Model versioning via timestamped artifacts
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier

# --- Configuration ---
AUC_THRESHOLD = 0.75
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


@task(name="load_data", retries=2, retry_delay_seconds=5)
def load_data(path: str = "data/lending_club.csv") -> pd.DataFrame:
    """Load raw lending club data with validation."""
    logger = get_run_logger()
    start = time.time()

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # Basic validation
    assert len(df) > 0, "Dataset is empty"
    assert "loan_amnt" in df.columns, "Missing required column: loan_amnt"

    latency = time.time() - start
    logger.info(f"Data loading completed in {latency:.2f}s")
    return df


@task(name="prepare_features")
def prepare_features(df: pd.DataFrame) -> tuple:
    """Clean data and engineer features. Mirrors dbt transformations."""
    logger = get_run_logger()
    start = time.time()

    initial_rows = len(df)

    # Filter to completed loans only
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    logger.info(f"Filtered to {len(df)} completed loans (dropped {initial_rows - len(df)})")

    # Target
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

    # Term
    df["term"] = df["term"].apply(lambda x: 36 if "36" in str(x) else 60)

    # Grade encoding
    grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
    df["grade"] = df["grade"].map(grade_map).fillna(4)

    # Sub-grade encoding
    sub_grades = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
    sub_grade_map = {sg: i for i, sg in enumerate(reversed(sub_grades), 1)}
    df["sub_grade"] = df["sub_grade"].map(sub_grade_map).fillna(18)

    # Employment length
    emp_map = {
        "< 1 year": 0.5, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10,
    }
    df["emp_length"] = df["emp_length"].map(emp_map).fillna(0)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["home_ownership", "purpose"], prefix=["home_ownership", "purpose"])

    # Fill nulls
    for col in ["pub_rec", "mort_acc", "pub_rec_bankruptcies", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Select feature columns
    feature_cols = [c for c in df.columns if c not in ["loan_status", "target"]]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols]
    y = df["target"]

    logger.info(f"Features prepared: {X.shape[1]} columns, {X.shape[0]} rows")
    latency = time.time() - start
    logger.info(f"Feature engineering completed in {latency:.2f}s")

    return X, y


@task(name="train_model")
def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
    """Train XGBoost model and return model + metrics."""
    logger = get_run_logger()
    start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "n_features": X_train.shape[1],
    }

    logger.info(f"Model metrics: AUC={metrics['auc_roc']}, Accuracy={metrics['accuracy']}")

    latency = time.time() - start
    logger.info(f"Training completed in {latency:.2f}s")

    return {"model": model, "metrics": metrics, "feature_names": X_train.columns.tolist()}


@task(name="evaluation_gate")
def evaluation_gate(result: dict) -> bool:
    """
    CRITICAL: Stop pipeline if model quality drops below threshold.
    This prevents deploying a bad model to production.
    """
    logger = get_run_logger()
    auc = result["metrics"]["auc_roc"]

    if auc < AUC_THRESHOLD:
        logger.error(
            f"EVALUATION GATE FAILED: AUC {auc} < threshold {AUC_THRESHOLD}. "
            f"Model will NOT be deployed. Investigate data quality or feature drift."
        )
        return False

    logger.info(f"Evaluation gate PASSED: AUC {auc} >= {AUC_THRESHOLD}")
    return True


@task(name="save_model")
def save_model(result: dict) -> str:
    """Save model artifact with timestamp and metrics."""
    logger = get_run_logger()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"xgboost_loan_model_{timestamp}.json"
    metrics_path = MODEL_DIR / f"metrics_{timestamp}.json"

    # Save model
    result["model"].save_model(str(model_path))

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)

    # Also save as "latest" for the app to use
    result["model"].save_model("xgboost_loan_model.json")

    logger.info(f"Model saved: {model_path}")
    logger.info(f"Metrics saved: {metrics_path}")
    logger.info(f"Latest model updated: xgboost_loan_model.json")

    return str(model_path)


# --- Main Flow ---

@flow(name="loan_training_pipeline", log_prints=True)
def training_pipeline(data_path: str = "data/lending_club.csv"):
    """
    End-to-end training pipeline:
    1. Load data
    2. Prepare features
    3. Train model
    4. Evaluation gate (stop if AUC < threshold)
    5. Save model artifact
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("LOAN INTELLIGENCE SYSTEM — Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Load
    df = load_data(data_path)

    # Step 2: Features
    X, y = prepare_features(df)

    # Step 3: Train
    result = train_model(X, y)

    # Step 4: Evaluation gate
    passed = evaluation_gate(result)

    if not passed:
        logger.error("Pipeline STOPPED — model did not pass evaluation gate.")
        return {"status": "FAILED", "metrics": result["metrics"]}

    # Step 5: Save
    model_path = save_model(result)

    logger.info("=" * 60)
    logger.info(f"Pipeline COMPLETE — Model deployed: {model_path}")
    logger.info("=" * 60)

    return {"status": "SUCCESS", "metrics": result["metrics"], "model_path": model_path}


if __name__ == "__main__":
    training_pipeline()
