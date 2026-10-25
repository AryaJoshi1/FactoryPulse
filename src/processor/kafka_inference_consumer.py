
import json
from pathlib import Path

import pandas as pd
import joblib
from kafka import KafkaConsumer
from mongo_storage import save_prediction


# Find the project root regardless of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "failure_classifier_validated.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "validated_model_metrics.json"

KAFKA_SERVER = "127.0.0.1:29092"
KAFKA_TOPIC = "sensor.raw"


def main():
    # Stop with a clear error if the model or its metrics are missing.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Model metrics not found: {METRICS_PATH}")

    # Load the validated model and the threshold selected during validation.
    model = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    threshold = float(metrics["selected_threshold"])

    print(f"Loaded model: {MODEL_PATH.name}")
    print(f"Selected failure threshold: {threshold:.2f}")
    print(f"Listening to Kafka topic: {KAFKA_TOPIC}")
    print("Waiting for new sensor messages...\n")

    # Read only new messages, avoiding replay of older demo messages.
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="factorypulse-ml-inference",
    )

    try:
        for record in consumer:
            message = record.value

            # Convert the producer's field names to the model's training schema.
            features = pd.DataFrame([{
                "Type": message["machine_type"],
                "Air temperature [K]": message["air_temp_K"],
                "Process temperature [K]": message["process_temp_K"],
                "Rotational speed [rpm]": message["rotational_speed_rpm"],
                "Torque [Nm]": message["torque_nm"],
                "Tool wear [min]": message["tool_wear_min"],
            }])

            # Get the model's estimated probability of machine failure.
            failure_probability = float(model.predict_proba(features)[0][1])
            predicted_failure = failure_probability >= threshold

            alert_level = "WARNING" if predicted_failure else "NORMAL"

            save_prediction(
                message=message,
                probability=failure_probability,
                threshold=threshold,
                alert_level=alert_level,
                            )

            print(
                f"Machine: {message.get('machine_id', 'unknown')} | "
                f"Failure risk: {failure_probability:.1%} | "
                f"Alert: {alert_level}"
            )

    except KeyboardInterrupt:
        print("\nInference consumer stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
