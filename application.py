"""
app.py
------
Streamlit frontend for the Predictive Maintenance project.

Pages (sidebar navigation):
1. Predict          -> input form, live prediction, failure probability, risk level,
                        maintenance recommendation
2. Prediction History -> table of past predictions pulled from SQLite
3. Dashboard         -> interactive charts (bar, pie, line, histogram) over history
4. About             -> project info

Run with:
    streamlit run app.py
"""

import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st




MODEL_DIR = "models"
DB_PATH = "predictions.db"

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🛠️",
    layout="wide",
)




@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    le_machine = joblib.load(f"{MODEL_DIR}/le_machine.pkl")
    le_status = joblib.load(f"{MODEL_DIR}/le_status.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    model_name = joblib.load(f"{MODEL_DIR}/model_name.pkl")
    return model, scaler, le_machine, le_status, feature_cols, model_name


ARTIFACTS_OK = os.path.exists(f"{MODEL_DIR}/best_model.pkl")




def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            machine_type TEXT,
            temperature REAL,
            vibration REAL,
            rpm REAL,
            voltage REAL,
            current REAL,
            pressure REAL,
            predicted_status TEXT,
            failure_probability REAL,
            risk_level TEXT,
            recommendation TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_prediction(record: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (
            timestamp, machine_type, temperature, vibration, rpm,
            voltage, current, pressure, predicted_status,
            failure_probability, risk_level, recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["timestamp"], record["machine_type"], record["temperature"],
            record["vibration"], record["rpm"], record["voltage"],
            record["current"], record["pressure"], record["predicted_status"],
            record["failure_probability"], record["risk_level"], record["recommendation"],
        ),
    )
    conn.commit()
    conn.close()


def fetch_history() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


init_db()




def get_risk_and_recommendation(status: str, failure_prob: float):
    if status == "Healthy":
        risk = "Low"
        rec = "No action needed. Continue routine monitoring and scheduled maintenance."
    elif status == "Warning":
        risk = "Medium"
        rec = (
            "Schedule an inspection within the next few days. Check vibration "
            "damping, lubrication, and cooling systems."
        )
    else:
        risk = "High"
        rec = (
            "Immediate maintenance required. Stop the machine if safe to do so "
            "and inspect for mechanical failure, overheating, or electrical faults."
        )

    
    if failure_prob >= 80 and risk != "High":
        risk = "High"
    elif failure_prob >= 50 and risk == "Low":
        risk = "Medium"

    return risk, rec





st.sidebar.title("🛠️ Predictive Maintenance")
page = st.sidebar.radio(
    "Navigate",
    ["Predict", "Prediction History", "Dashboard", "About"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "AI-powered predictive maintenance system. "
    "Enter live sensor readings to estimate machine health."
)




if page == "Predict":
    st.title("🔧 Machine Health Prediction")
    st.write(
        "Enter the current sensor readings below to predict machine status, "
        "failure probability, and get a maintenance recommendation."
    )

    if not ARTIFACTS_OK:
        st.error(
            "Model artifacts not found. Please run `python generate_dataset.py` "
            "and then `python train_model.py` first to train and save the model."
        )
    else:
        model, scaler, le_machine, le_status, feature_cols, model_name = load_artifacts()
        st.caption(f"Model in use: **{model_name}**")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                machine_type = st.selectbox(
                    "Machine Type", list(le_machine.classes_)
                )
                temperature = st.number_input(
                    "Temperature (°C)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
                )
                vibration = st.number_input(
                    "Vibration (mm/s)", min_value=0.0, max_value=50.0, value=2.5, step=0.1
                )
                rpm = st.number_input(
                    "RPM (Rotational Speed)", min_value=0.0, max_value=10000.0, value=1500.0, step=10.0
                )

            with col2:
                voltage = st.number_input(
                    "Voltage (V)", min_value=0.0, max_value=1000.0, value=220.0, step=1.0
                )
                current = st.number_input(
                    "Current (A)", min_value=0.0, max_value=200.0, value=10.0, step=0.5
                )
                pressure = st.number_input(
                    "Pressure (bar)", min_value=0.0, max_value=50.0, value=5.0, step=0.1
                )

            submitted = st.form_submit_button("Predict Machine Status")

        if submitted:
            machine_enc = le_machine.transform([machine_type])[0]
            input_df = pd.DataFrame(
                [[machine_enc, temperature, vibration, rpm, voltage, current, pressure]],
                columns=feature_cols,
            )
            input_scaled = scaler.transform(input_df)

            pred_enc = model.predict(input_scaled)[0]
            pred_status = le_status.inverse_transform([pred_enc])[0]

            proba = model.predict_proba(input_scaled)[0]
            classes = le_status.inverse_transform(np.arange(len(proba)))
            proba_map = dict(zip(classes, proba))

            
            failure_prob = round(
                (proba_map.get("Warning", 0) + proba_map.get("Faulty", 0)) * 100, 2
            )

            risk_level, recommendation = get_risk_and_recommendation(pred_status, failure_prob)

            
            st.markdown("
            r1, r2, r3 = st.columns(3)

            status_color = {"Healthy": "green", "Warning": "orange", "Faulty": "red"}[pred_status]
            r1.markdown(
                f"**Machine Status:** <span style='color:{status_color}; font-weight:bold'>{pred_status}</span>",
                unsafe_allow_html=True,
            )
            r2.metric("Failure Probability", f"{failure_prob}%")

            risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_level]
            r3.markdown(
                f"**Risk Level:** <span style='color:{risk_color}; font-weight:bold'>{risk_level}</span>",
                unsafe_allow_html=True,
            )

            st.info(f"🔧 **Maintenance Recommendation:** {recommendation}")

            
            proba_df = pd.DataFrame(
                {"Status": list(proba_map.keys()), "Probability": [v * 100 for v in proba_map.values()]}
            )
            fig = px.bar(
                proba_df, x="Status", y="Probability", color="Status",
                color_discrete_map={"Healthy": "green", "Warning": "orange", "Faulty": "red"},
                title="Class Probability Breakdown (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

            
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "machine_type": machine_type,
                "temperature": temperature,
                "vibration": vibration,
                "rpm": rpm,
                "voltage": voltage,
                "current": current,
                "pressure": pressure,
                "predicted_status": pred_status,
                "failure_probability": failure_prob,
                "risk_level": risk_level,
                "recommendation": recommendation,
            }
            insert_prediction(record)
            st.success("Prediction saved to history.")




elif page == "Prediction History":
    st.title("📜 Prediction History")
    history_df = fetch_history()

    if history_df.empty:
        st.warning("No predictions yet. Go to the Predict page to run one.")
    else:
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history as CSV", csv, "prediction_history.csv", "text/csv"
        )

        if st.button("Clear History"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.rerun()




elif page == "Dashboard":
    st.title("📊 Maintenance Dashboard")
    history_df = fetch_history()

    if history_df.empty:
        st.warning("No prediction data yet. Run some predictions first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", len(history_df))
        c2.metric("Healthy", int((history_df["predicted_status"] == "Healthy").sum()))
        c3.metric("Warning", int((history_df["predicted_status"] == "Warning").sum()))
        c4.metric("Faulty", int((history_df["predicted_status"] == "Faulty").sum()))

        col1, col2 = st.columns(2)

        with col1:
            status_counts = history_df["predicted_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig1 = px.pie(
                status_counts, names="Status", values="Count",
                title="Status Distribution",
                color="Status",
                color_discrete_map={"Healthy": "green", "Warning": "orange", "Faulty": "red"},
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                status_counts, x="Status", y="Count", color="Status",
                color_discrete_map={"Healthy": "green", "Warning": "orange", "Faulty": "red"},
                title="Predictions by Status",
            )
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.histogram(
                history_df, x="failure_probability", nbins=20,
                title="Failure Probability Distribution",
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            history_df_sorted = history_df.sort_values("id")
            fig4 = px.line(
                history_df_sorted, x="id", y="failure_probability",
                title="Failure Probability Over Time (by prediction order)",
                markers=True,
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("
        numeric_cols = ["temperature", "vibration", "rpm", "voltage", "current", "pressure"]
        fig5 = px.imshow(
            history_df[numeric_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
            title="Sensor Correlation Heatmap",
        )
        st.plotly_chart(fig5, use_container_width=True)




else:
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        **Predictive Maintenance using AI** is a machine learning system that
        predicts machine health status from live sensor readings
        (temperature, vibration, RPM, voltage, current, pressure).

        **Pipeline:**
        1. Data cleaning & EDA (`train_model.py`)
        2. Feature engineering, encoding, scaling
        3. Model training & comparison (Logistic Regression, Decision Tree,
           Random Forest, SVM, optional XGBoost)
        4. Best model saved with `joblib`
        5. Streamlit app for live predictions + SQLite history + dashboard

        **Outputs:**
        - Machine Status: Healthy / Warning / Faulty
        - Failure Probability (%)
        - Risk Level: Low / Medium / High
        - Maintenance Recommendation

        Built with Python, Scikit-learn, Streamlit, SQLite, and Plotly.
        """
    )