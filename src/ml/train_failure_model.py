
"""
FactoryPulse - Machine Failure Prediction Baseline

Trains a Random Forest classifier on the real AI4I 2020 dataset.

Target:
    Machine failure

Inputs:
    Type and six genuine machine measurements.

Important:
    Failure-mode labels are excluded from the inputs to prevent leakage.
    This model predicts benchmark dataset failures; it is not yet a
    validated real-time industrial failure prediction system.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Resolve the project root regardless of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "failure_classifier.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.csv"


def main():
    # Create output folders if they do not already exist.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stop with a clear message if the dataset is missing.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Load the benchmark dataset.
    df = pd.read_csv(DATA_PATH)

    # Use only genuine sensor measurements and machine type.
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    target_column = "Machine failure"

    # Check that all expected columns exist before training.
    required_columns = feature_columns + [target_column]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Features contain measurements, not failure outcomes or identifiers.
    X = df[feature_columns].copy()
    y = df[target_column].astype(int)

    if y.nunique() != 2:
        raise ValueError("Expected a binary Machine failure target.")

    # Split before fitting any preprocessing or model components.
    # Stratification preserves the failure/non-failure ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Encode the categorical machine Type; keep numeric features unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "machine_type",
                OneHotEncoder(handle_unknown="ignore"),
                ["Type"],
            ),
        ],
        remainder="passthrough",
    )

    # Balanced class weights help the model account for rare failures.
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Bundle preprocessing and classification into one saved pipeline.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("Training Random Forest baseline...")
    model.fit(X_train, y_train)

    # Evaluate once on the held-out test set.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "RandomForestClassifier",
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "random_state": 42,
        "test_size": 0.20,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_failures": int(y_train.sum()),
        "testing_failures": int(y_test.sum()),
        "failure_rate_test_percent": round(float(y_test.mean() * 100), 4),
        "precision": round(float(precision_score(
            y_test, predictions, zero_division=0
        )), 4),
        "recall": round(float(recall_score(
            y_test, predictions, zero_division=0
        )), 4),
        "f1_score": round(float(f1_score(
            y_test, predictions, zero_division=0
        )), 4),
        "roc_auc": round(float(roc_auc_score(
            y_test, probabilities
        )), 4),
        "average_precision": round(float(average_precision_score(
            y_test, probabilities
        )), 4),
    }

    # Store confusion matrix with rows = actual, columns = predicted.
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual_No_Failure", "Actual_Failure"],
        columns=["Predicted_No_Failure", "Predicted_Failure"],
    )

    # Save the full fitted pipeline so inference uses identical preprocessing.
    joblib.dump(model, MODEL_PATH)

    # Save reproducible metrics and the confusion matrix.
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
    confusion_df.to_csv(CONFUSION_PATH)

    print("\n=== FactoryPulse: Test Results ===")
    print(json.dumps(metrics, indent=4))

    print("\n=== Classification Report ===")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0,
    ))

    print("\n=== Confusion Matrix ===")
    print(confusion_df.to_string())

    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved confusion matrix:", CONFUSION_PATH)


if __name__ == "__main__":
    main()
