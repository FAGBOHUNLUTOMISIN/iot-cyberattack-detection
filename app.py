
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="IoT Cyberattack Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ IoT Cyberattack Detection System")
st.write(
    "Big-data framework for identifying suspicious IoT network traffic "
    "using a Spark MLlib-trained model."
)

MODEL_PATH = Path("models/iot_model.json")
DATA_PATH = Path("dataset/combined_binary.csv")
MAX_ROWS = 10000


# -----------------------------
# 2. LOAD TRAINED MODEL
# -----------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "r") as file:
        return json.load(file)


if not MODEL_PATH.exists():
    st.error("Model file not found. Expected: models/iot_model.json")
    st.stop()

model_data = load_model()

features = model_data["features"]
coefficients = np.array(model_data["coefficients"], dtype=float)
intercept = float(model_data["intercept"])


# -----------------------------
# 3. SELECT DATASET
# -----------------------------
st.sidebar.header("Detection Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload IoT traffic CSV",
    type=["csv"]
)

if uploaded_file is not None:
    st.sidebar.success("Uploaded dataset selected.")
    source_name = uploaded_file.name
else:
    st.sidebar.info("Using the project's CICIoT2023 dataset.")
    source_name = "combined_binary.csv"

st.sidebar.caption(
    f"Maximum records processed per run: {MAX_ROWS:,}"
)


# -----------------------------
# 4. DETECTION FUNCTION
# -----------------------------
def detect_attacks(data):

    missing_features = [
        feature for feature in features
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "The uploaded CSV is missing required model features: "
            + ", ".join(missing_features[:10])
        )

    original_labels = (
        data["Label"].copy()
        if "Label" in data.columns
        else None
    )

    X = data[features].copy()

    for column in features:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # Logistic regression prediction
    logits = X.to_numpy(dtype=float) @ coefficients + intercept

    logits = np.clip(logits, -500, 500)

    probabilities = 1 / (1 + np.exp(-logits))

    predictions = (probabilities >= 0.5).astype(int)

    results = data.copy()

    results["Attack_Probability"] = probabilities
    results["Prediction"] = np.where(
        predictions == 1,
        "Potential Attack",
        "Benign"
    )

    return results, original_labels


# -----------------------------
# 5. RUN DETECTION
# -----------------------------
st.header("Network Traffic Analysis")

if st.button("Run Attack Detection", type="primary"):

    try:
        with st.spinner("Loading and analysing network traffic..."):

            if uploaded_file is not None:
                data = pd.read_csv(
                    uploaded_file,
                    nrows=MAX_ROWS
                )
            else:
                if not DATA_PATH.exists():
                    st.error("Project dataset not found.")
                    st.stop()

                data = pd.read_csv(
                    DATA_PATH,
                    nrows=MAX_ROWS
                )

            if data.empty:
                st.error("The selected CSV contains no records.")
                st.stop()

            results, actual_labels = detect_attacks(data)

        st.success("Detection completed successfully.")

        total = len(results)

        attack_count = (
            results["Prediction"] == "Potential Attack"
        ).sum()

        benign_count = total - attack_count

        attack_percentage = (attack_count / total) * 100

        # -----------------------------
        # 6. DISPLAY METRICS
        # -----------------------------
        st.subheader("Detection Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Records Analysed", f"{total:,}")
        col2.metric("Benign Traffic", f"{benign_count:,}")
        col3.metric("Potential Attacks", f"{attack_count:,}")
        col4.metric("Attack Percentage", f"{attack_percentage:.2f}%")

        # -----------------------------
        # 7. DISPLAY CHART
        # -----------------------------
        st.subheader("Traffic Classification")

        chart_data = pd.DataFrame({
            "Traffic Type": ["Benign", "Potential Attack"],
            "Records": [benign_count, attack_count]
        })

        st.bar_chart(
            chart_data,
            x="Traffic Type",
            y="Records"
        )

        # -----------------------------
        # 8. DISPLAY RESULTS
        # -----------------------------
        st.subheader("Prediction Results")

        display_data = results.copy()

        display_data["Attack_Probability"] = (
            display_data["Attack_Probability"] * 100
        ).round(2)

        st.dataframe(
            display_data[
                ["Prediction", "Attack_Probability"]
            ].join(
                results.drop(
                    columns=["Prediction", "Attack_Probability"]
                ).head(0)
            ) if False else display_data,
            use_container_width=True
        )

        # -----------------------------
        # 9. OPTIONAL EVALUATION
        # -----------------------------
        if actual_labels is not None:

            st.subheader("Comparison With Dataset Labels")

            actual_binary = (
                actual_labels.astype(str).str.strip().str.lower()
                != "benign"
            ).astype(int)

            predicted_binary = (
                results["Prediction"] == "Potential Attack"
            ).astype(int)

            correct = (actual_binary == predicted_binary).sum()
            evaluation_accuracy = correct / total

            st.metric(
                "Accuracy on these labelled records",
                f"{evaluation_accuracy * 100:.2f}%"
            )

            st.caption(
                "This is an evaluation against the labels in the "
                "selected records, not an independent test result."
            )

        # -----------------------------
        # 10. DOWNLOAD RESULTS
        # -----------------------------
        csv_output = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Detection Results",
            data=csv_output,
            file_name="iot_detection_results.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Detection failed: {error}")


# -----------------------------
# 11. MODEL INFORMATION
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("Model Information")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Framework: Apache Spark MLlib")
st.sidebar.write("Training sample: 5,000 records")
st.sidebar.write("Classification: Benign / DDoS-like attack")
st.sidebar.caption(
    "This prototype was trained on Benign and DDoS-UDP_Flood "
    "traffic only. It has not been validated for all CICIoT2023 "
    "attack categories."
)
