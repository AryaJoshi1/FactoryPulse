# FactoryPulse
### Real-Time Industrial Equipment Health Monitoring & Predictive Analytics

FactoryPulse is a local industrial monitoring prototype that combines streaming telemetry, machine learning inference, database storage, and a live dashboard to demonstrate predictive maintenance workflows.

The system generates machine sensor readings, publishes them to Apache Kafka, evaluates failure risk using a trained machine learning classifier, stores predictions in MongoDB, and visualizes results through Streamlit.

> **Project status:** Local working prototype. Telemetry includes simulated sensor values, and model evaluation uses the AI4I 2020 Predictive Maintenance dataset. This is an experimental demonstration, not a production-validated industrial safety system.

## Architecture

```text
Python Sensor Producer
        |
        v
Apache Kafka (sensor.raw)
        |
        v
ML Inference Consumer
        |
        v
Failure Probability + Alert
        |
        v
MongoDB (factorypulse)
        |
        v
Streamlit Dashboard
```

## Key Features

- Continuous synthetic machine telemetry generation.
- Kafka-based asynchronous message ingestion.
- Machine failure-risk estimation using a trained machine learning classifier.
- Configurable classification threshold loaded from evaluation metrics.
- MongoDB storage of sensor readings and prediction results.
- Streamlit dashboard with KPIs, risk visualizations, sensor trends, and recent predictions.
- Model evaluation reports, including a confusion matrix and threshold comparison.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Streaming Ingestion | Apache Kafka |
| Machine Learning | scikit-learn-compatible classifier, joblib, pandas |
| Database | MongoDB |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker Compose |
| Dataset | AI4I 2020 Predictive Maintenance |

## Project Structure

```text
FactoryPulse/
├── config/
├── data/
│   └── raw/
│       └── ai4i2020.csv
├── models/
│   └── failure_classifier_validated.joblib
├── outputs/
│   ├── validated_model_metrics.json
│   ├── confusion_matrix.csv
│   └── threshold_comparison.csv
├── src/
│   ├── producer/
│   │   └── sensor_producer.py
│   ├── processor/
│   │   ├── kafka_inference_consumer.py
│   │   └── mongo_storage.py
│   ├── dashboard/
│   │   └── app.py
│   └── ml/
├── tests/
├── compose.yaml
├── requirements.txt
└── README.md
```

Some files, including the trained model and dataset, may be excluded from Git. Ensure the required local files are available before running the corresponding components.

## Prerequisites

- Python 3.11
- Docker Desktop
- Git
- MongoDB accessible at `127.0.0.1:27017`, unless configured otherwise

## Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/sampatel23/FacotoryPulse.git
cd FacotoryPulse
```

### 2. Create and Activate a Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, configure an appropriate execution policy for your user or use another supported activation method.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `requirements.txt` exists and lists all required project dependencies.

### 4. Start Kafka

Run the following commands from the project root:

```powershell
docker compose up -d
docker compose ps
```

The current local configuration exposes Kafka at `127.0.0.1:29092`. Ensure the producer and consumer use the same bootstrap server.

### 5. Start MongoDB

Ensure MongoDB is running and accessible at:

```text
mongodb://127.0.0.1:27017
```

The default database is `factorypulse`, and prediction records are stored in the `machine_predictions` collection.

If MongoDB uses a different address, update the database connection configuration accordingly.

## Run the Pipeline

Open three separate PowerShell terminals. In each terminal, navigate to the project root and activate the virtual environment.

### Terminal 1: Start the Sensor Producer

```powershell
python src/producer/sensor_producer.py
```

The producer generates machine telemetry and publishes messages to the Kafka topic `sensor.raw`.

### Terminal 2: Start ML Inference

```powershell
python src/processor/kafka_inference_consumer.py
```

The consumer reads Kafka messages, prepares the model features, calculates failure probability, applies the selected threshold, and stores the predictions in MongoDB.

The trained model and evaluation-metrics JSON must exist in their expected local paths.

### Terminal 3: Start the Dashboard

```powershell
streamlit run src/dashboard/app.py
```

Open the local URL printed by Streamlit, usually:

`http://localhost:8501`

Keep the producer, inference consumer, Kafka, MongoDB, and dashboard running to observe the end-to-end pipeline. The dashboard refreshes its view, but new predictions require the producer and consumer to be running.

## Machine Learning Evaluation

The classifier was evaluated on a held-out test split from the AI4I 2020 Predictive Maintenance dataset.

| Metric | Test Result |
|---|---:|
| Precision | 0.5747 |
| Recall | 0.7353 |
| F1-score | 0.6452 |
| ROC-AUC | 0.9659 |
| Average Precision | 0.7207 |
| Selected Threshold | 0.40 |
| False Alarms | 37 |
| Missed Failures | 18 |
| Detected Failures | 50 |

These results describe the offline evaluation on the selected test split. They do not establish equivalent performance on real factory equipment.

The inference consumer labels a prediction `WARNING` when the estimated failure probability is at least the selected threshold of `0.40`. Otherwise, it labels the prediction `NORMAL`.

## Data and Limitations

- The AI4I 2020 Predictive Maintenance dataset is used for model development and evaluation.
- Some generated telemetry fields, such as vibration and hydraulic pressure, are synthetic proxies rather than direct AI4I measurements.
- Offline model metrics should not be interpreted as proof of real-world predictive maintenance performance.
- The current pipeline uses a Kafka consumer for inference. Apache Spark Structured Streaming should be considered a future extension unless separately implemented and tested.
- The system is intended for learning, experimentation, and demonstration, not autonomous equipment control or safety-critical decisions.

## Future Improvements

- Integrate Apache Spark Structured Streaming for distributed stream processing.
- Add automated tests for producer, inference, and database components.
- Monitor message-processing latency and consumer failures.
- Improve model calibration and evaluate performance across additional datasets.
- Add reproducible model artifacts, deployment configuration, and pipeline health checks.

## Author

Team Project: FactoryPulse was developed collaboratively as a group project. Contributors - Arya Joshi, Samarth Patel, Mandar Deshmukh, Pramdhan Kumar.

## License

Add a license before redistributing the project as open source.
