
"""
FactoryPulse - Leakage-aware model validation

60% training, 20% validation, 20% final test.
The validation set selects the threshold; the test set evaluates it.

The existing baseline artifacts are not overwritten.
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


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "failure_classifier_validated.joblib"
RESULT_PATH = ROOT / "outputs" / "validated_model_metrics.json"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def binary_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(
            float(precision_score(y_true, predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, predictions, zero_division=0)), 4
        ),
        "f1_score": round(
            float(f1_score(y_true, predictions, zero_division=0)), 4
        ),
        "false_alarms": int(((y_true == 0) & (predictions == 1)).sum()),
        "missed_failures": int(((y_true == 1) & (predictions == 0)).sum()),
        "detected_failures": int(((y_true == 1) & (predictions == 1)).sum()),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # First reserve 20% as the final test set.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Split the remaining 80% into 60% train and 20% validation overall.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        random_state=43,
        stratify=y_dev,
    )

    preprocessor = ColumnTransformer([
        (
            "machine_type",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"],
        ),
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training on 60% of the dataset...")
    model.fit(X_train, y_train)

    # Select a threshold on validation data only.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    validation_results = [
        binary_metrics(y_val, val_probabilities, threshold)
        for threshold in thresholds
    ]

    # Select the threshold with the highest validation F1.
    # Ties resolve to the first threshold in the list.
    best = max(validation_results, key=lambda row: row["f1_score"])
    selected_threshold = best["threshold"]

    print("\nValidation results:")
    print(pd.DataFrame(validation_results).to_string(index=False))
    print(f"\nSelected threshold by validation F1: {selected_threshold}")

    # Evaluate the selected threshold once on the held-out test set.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    test_results = binary_metrics(
        y_test, test_probabilities, selected_threshold
    )
    test_results["roc_auc"] = round(
        float(roc_auc_score(y_test, test_probabilities)), 4
    )
    test_results["average_precision"] = round(
        float(average_precision_score(y_test, test_probabilities)), 4
    )

    print("\nFinal test results:")
    print(json.dumps(test_results, indent=4))

    print("\nTest classification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0,
    ))

    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

    # Save a separate artifact so the original baseline remains intact.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    results = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "split_percentages": {
            "training": 60,
            "validation": 20,
            "test": 20,
        },
        "row_counts": {
            "training": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "threshold_selection": "Highest validation F1-score",
        "selected_threshold": selected_threshold,
        "validation_results": validation_results,
        "final_test_results": test_results,
        "test_confusion_matrix": confusion_matrix(
            y_test, test_predictions, labels=[0, 1]
        ).tolist(),
    }

    RESULT_PATH.write_text(json.dumps(results, indent=4), encoding="utf-8")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
