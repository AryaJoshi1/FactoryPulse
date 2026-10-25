
from src.processor.mongo_storage import get_collection, save_prediction


sample_message = {
    "machine_id": "machine_test",
    "machine_type": "M",
    "timestamp": "2026-10-26T10:00:00+00:00",
    "air_temp_K": 300.5,
    "process_temp_K": 310.2,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40.0,
    "tool_wear_min": 100,
    "vibration_mm_s": 2.5,
    "hydraulic_pressure_bar": 5.0,
}

# Insert a clearly labelled test prediction.
save_prediction(
    message=sample_message,
    probability=0.72,
    threshold=0.40,
    alert_level="WARNING",
)

# Read the record back to confirm that it was stored.
client, collection = get_collection()

try:
    saved = collection.find_one({"machine_id": "machine_test"})
    assert saved is not None, "Test record was not found in MongoDB"
    assert saved["alert_level"] == "WARNING"
    print("MongoDB storage test passed!")
    print("Saved alert:", saved["alert_level"])
    print("Saved failure probability:", saved["failure_probability"])
finally:
    client.close()
