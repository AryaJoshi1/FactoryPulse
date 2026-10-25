
import os
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError


# Connect to the existing local MongoDB server.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DATABASE_NAME = "factorypulse"
COLLECTION_NAME = "machine_predictions"


def get_collection():
    """Connect to MongoDB and return the predictions collection."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    # Confirm that MongoDB is reachable before returning the collection.
    client.admin.command("ping")

    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # Index timestamps to make recent-record queries faster.
    collection.create_index("timestamp")

    return client, collection


def save_prediction(message, probability, threshold, alert_level):
    """Save one sensor message together with its ML prediction."""
    client = None

    try:
        client, collection = get_collection()

        document = {
            "machine_id": message.get("machine_id", "unknown"),
            "timestamp": message.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            ),
            "machine_type": message.get("machine_type", "unknown"),
            "sensor_data": {
                "air_temp_K": message.get("air_temp_K"),
                "process_temp_K": message.get("process_temp_K"),
                "rotational_speed_rpm": message.get("rotational_speed_rpm"),
                "torque_nm": message.get("torque_nm"),
                "tool_wear_min": message.get("tool_wear_min"),
                "vibration_mm_s": message.get("vibration_mm_s"),
                "hydraulic_pressure_bar": message.get("hydraulic_pressure_bar"),
            },
            "failure_probability": float(probability),
            "threshold": float(threshold),
            "predicted_failure": bool(probability >= threshold),
            "alert_level": alert_level,
            "prediction_created_at": datetime.now(timezone.utc),
            "model_name": "failure_classifier_validated",
        }

        result = collection.insert_one(document)
        print(f"MongoDB saved prediction: {result.inserted_id}")

    except PyMongoError as error:
        print(f"MongoDB storage error: {error}")

    finally:
        if client is not None:
            client.close()
