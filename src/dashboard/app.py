
import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="FactoryPulse",
    page_icon="🏭",
    layout="wide",
)

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://127.0.0.1:27017",
)

DATABASE_NAME = "factorypulse"
COLLECTION_NAME = "machine_predictions"


# --------------------------------------------------
# 2. Automatic refresh every 5 seconds
# --------------------------------------------------
st_autorefresh(
    interval=5000,
    key="factorypulse_refresh",
)

# Visible proof that Streamlit has rerun.
st.session_state["refresh_count"] = (
    st.session_state.get("refresh_count", 0) + 1
)


# --------------------------------------------------
# 3. Load the latest records from MongoDB
# --------------------------------------------------
def load_records():
    """Fetch recent predictions directly from MongoDB."""
    client = None

    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
        )

        # Check whether MongoDB is reachable.
        client.admin.command("ping")

        records = list(
            client[DATABASE_NAME][COLLECTION_NAME]
            .find({}, {"_id": 0})
            .sort("_id", -1)
            .limit(2000)
        )

        return records, None

    except PyMongoError as exc:
        return [], str(exc)

    finally:
        if client is not None:
            client.close()


# --------------------------------------------------
# 4. Dashboard heading and manual refresh
# --------------------------------------------------
st.title("🏭 FactoryPulse")
st.caption(
    "Industrial Equipment Health Monitoring & "
    "Predictive Analytics"
)

top_left, top_right = st.columns([4, 1])

with top_left:
    st.caption(
        f"Dashboard refresh #{st.session_state['refresh_count']}"
    )

with top_right:
    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()


records, error = load_records()

if error:
    st.error(f"MongoDB connection failed: {error}")
    st.info(
        "Check that MongoDB is running on "
        "127.0.0.1:27017."
    )
    st.stop()

if not records:
    st.warning(
        "No predictions found. Start the Kafka producer "
        "and inference consumer, then wait for new records."
    )
    st.caption(
        "This page will check MongoDB again automatically."
    )
    st.stop()


# --------------------------------------------------
# 5. Prepare database records
# --------------------------------------------------
df = pd.json_normalize(records, sep="_")

df["machine_id"] = (
    df.get("machine_id", pd.Series("unknown", index=df.index))
    .fillna("unknown")
    .astype(str)
)

df["alert_level"] = (
    df.get("alert_level", pd.Series("UNKNOWN", index=df.index))
    .fillna("UNKNOWN")
    .astype(str)
    .str.upper()
)

df["failure_probability"] = pd.to_numeric(
    df.get(
        "failure_probability",
        pd.Series(index=df.index, dtype=float),
    ),
    errors="coerce",
)

df["timestamp_parsed"] = pd.to_datetime(
    df.get(
        "timestamp",
        pd.Series(index=df.index, dtype=object),
    ),
    errors="coerce",
    utc=True,
)


# --------------------------------------------------
# 6. Sidebar filters
# --------------------------------------------------
st.sidebar.header("Dashboard filters")

machines = sorted(df["machine_id"].unique().tolist())
alerts = sorted(df["alert_level"].unique().tolist())

selected_machines = st.sidebar.multiselect(
    "Machines",
    machines,
    default=machines,
)

selected_alerts = st.sidebar.multiselect(
    "Alert levels",
    alerts,
    default=alerts,
)

data = df[
    df["machine_id"].isin(selected_machines)
    & df["alert_level"].isin(selected_alerts)
].copy()


# --------------------------------------------------
# 7. Key performance indicators
# --------------------------------------------------
latest_risks = data.dropna(
    subset=["failure_probability"]
)

latest_risk = (
    float(latest_risks.iloc[0]["failure_probability"])
    if not latest_risks.empty
    else None
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Readings loaded",
    f"{len(data):,}",
)

col2.metric(
    "Machines",
    data["machine_id"].nunique(),
)

col3.metric(
    "Warning alerts",
    int((data["alert_level"] == "WARNING").sum()),
)

col4.metric(
    "Latest displayed risk",
    f"{latest_risk:.1%}" if latest_risk is not None else "N/A",
)

st.caption(
    "Last loaded: "
    + datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
)


# --------------------------------------------------
# 8. Failure-risk trend
# --------------------------------------------------
st.subheader("Failure Risk Over Time")

trend = data.dropna(
    subset=["failure_probability", "timestamp_parsed"]
).copy()

trend = trend.sort_values("timestamp_parsed")

if not trend.empty:
    fig = px.line(
        trend,
        x="timestamp_parsed",
        y="failure_probability",
        color="machine_id",
        markers=True,
        labels={
            "timestamp_parsed": "Timestamp (UTC)",
            "failure_probability": "Estimated failure probability",
            "machine_id": "Machine",
        },
    )

    fig.update_yaxes(
        tickformat=".1%",
        rangemode="tozero",
        title="Estimated failure probability",
    )

    fig.update_layout(
        legend_title_text="Machine",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )
else:
    st.info(
        "No timestamped risk data matches the selected filters."
    )


# --------------------------------------------------
# 9. Alert distribution and sensor trends
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Alert Distribution")

    alert_counts = (
        data["alert_level"]
        .value_counts()
        .rename_axis("Alert")
        .reset_index(name="Readings")
    )

    if not alert_counts.empty:
        alert_fig = px.bar(
            alert_counts,
            x="Alert",
            y="Readings",
            color="Alert",
        )

        st.plotly_chart(
            alert_fig,
            use_container_width=True,
        )
    else:
        st.info("No alerts match these filters.")


with right:
    st.subheader("Sensor Trends")

    sensors = {
        "sensor_data_air_temp_K": "Air Temperature (K)",
        "sensor_data_process_temp_K": "Process Temperature (K)",
        "sensor_data_rotational_speed_rpm": "Rotational Speed (rpm)",
        "sensor_data_torque_nm": "Torque (Nm)",
        "sensor_data_tool_wear_min": "Tool Wear (min)",
    }

    available_sensors = {
        key: label
        for key, label in sensors.items()
        if key in data.columns
    }

    if available_sensors:
        sensor = st.selectbox(
            "Choose sensor",
            list(available_sensors.keys()),
            format_func=lambda key: available_sensors[key],
        )

        sensor_data = data.copy()

        sensor_data[sensor] = pd.to_numeric(
            sensor_data[sensor],
            errors="coerce",
        )

        sensor_data = sensor_data.dropna(
            subset=[sensor, "timestamp_parsed"]
        ).sort_values("timestamp_parsed")

        if not sensor_data.empty:
            sensor_fig = px.line(
                sensor_data,
                x="timestamp_parsed",
                y=sensor,
                color="machine_id",
                labels={
                    "timestamp_parsed": "Timestamp (UTC)",
                    sensor: available_sensors[sensor],
                    "machine_id": "Machine",
                },
            )

            st.plotly_chart(
                sensor_fig,
                use_container_width=True,
            )
        else:
            st.info("No sensor values available for this selection.")
    else:
        st.info("No sensor fields found in the database records.")


# --------------------------------------------------
# 10. Recent predictions table
# --------------------------------------------------
st.subheader("Recent Predictions")

display_columns = [
    "machine_id",
    "timestamp",
    "alert_level",
    "failure_probability",
    "threshold",
    "predicted_failure",
    "sensor_data_air_temp_K",
    "sensor_data_process_temp_K",
    "sensor_data_rotational_speed_rpm",
    "sensor_data_torque_nm",
    "sensor_data_tool_wear_min",
]

display_columns = [
    column
    for column in display_columns
    if column in data.columns
]

recent = data[display_columns].head(100).copy()

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 11. Model limitation notice
# --------------------------------------------------
st.warning(
    "Failure probabilities are model estimates, not confirmed faults. "
    "The classifier was trained on the AI4I 2020 benchmark dataset. "
    "Predictions on synthetic telemetry are for integration demonstration "
    "and are not verified real-world fault diagnoses."
)
