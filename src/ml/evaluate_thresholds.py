
"""
FactoryPulse - Failure Prediction Threshold Evaluation

Compares different probability thresholds for the saved Random Forest.

Important:
- Uses the same fixed test split as the baseline.
- This is exploratory threshold analysis, not independent validation.
- Thresholds must ultimately be selected on validation data, not the
  final test set, to avoid optimistic performance estimates.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "failure_classifier.joblib"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "threshold_comparison.csv"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"


def main():
    # Confirm the required files exist before starting.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    # Reproduce the exact split used in train_failure_model.py.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Load the previously trained pipeline.
    model = joblib.load(MODEL_PATH)

    # Get continuous failure probabilities once.
    probabilities = model.predict_proba(X_test)[:, 1]

    # ROC-AUC and average precision do not depend on the chosen threshold.
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(
        "Average precision: "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    results = []

    # Explore thresholds; these results are for analysis only.
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "false_alarms": int(
                ((y_test == 0) & (predictions == 1)).sum()
            ),
            "missed_failures": int(
                ((y_test == 1) & (predictions == 0)).sum()
            ),
            "detected_failures": int(
                ((y_test == 1) & (predictions == 1)).sum()
            ),
        })

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    # Round only for display; the CSV retains full precision.
    print("\n=== Threshold Comparison ===")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
