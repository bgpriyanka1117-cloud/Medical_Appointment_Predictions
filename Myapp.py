import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Medical AI System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Appointment AI System")

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    models = {}
    models["logistic"]      = joblib.load(os.path.join(BASE_DIR, "..", "models", "logistic_regression.pkl"))
    models["random_forest"] = joblib.load(os.path.join(BASE_DIR, "..", "models", "random_forest.pkl"))
    models["xgboost"]       = joblib.load(os.path.join(BASE_DIR, "..", "models", "xgboost.pkl"))
    models["arima"]         = joblib.load(os.path.join(BASE_DIR, "..", "models", "forcasting_Regration", "arima_model.pkl"))
    models["sarima"]        = joblib.load(os.path.join(BASE_DIR, "..", "models", "forcasting_Regration", "sarima_model.pkl"))
    models["forecast_xgb"]  = joblib.load(os.path.join(BASE_DIR, "..", "models", "forcasting_Regration", "forecast_xgb_model.pkl"))
    return models

models = load_models()

# ==============================
# SAFE INPUT FUNCTION
# ==============================
def build_input(features, feature_cols):
    df = pd.DataFrame(0, index=[0], columns=feature_cols)
    for k, v in features.items():
        if k in df.columns:
            df[k] = v
    return df

# ==============================
# SIDEBAR
# ==============================
module = st.sidebar.radio(
    "Select Module",
    ["No-show Prediction", "Demand Forecasting"]
)

# =====================================================
# 1. NO SHOW PREDICTION
# =====================================================
if module == "No-show Prediction":

    model_name = st.selectbox("Model", ["logistic", "random_forest", "xgboost"])
    model = models[model_name]

    st.subheader("👤 Patient Info")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Basic Details**")
        age         = st.number_input("Age", 0, 100, 30)
        gender      = st.selectbox("Gender", ["Male", "Female"])
        specialty   = st.selectbox("Specialty", [
            "sem especialidade", "enf", "physiotherapy", "psychotherapy"
        ])
        disability  = st.selectbox("Disability", ["none", "motor"])
        appointment_shift = st.selectbox(
            "Appointment Shift", [0, 1],
            format_func=lambda x: "Morning (0)" if x == 0 else "Afternoon (1)"
        )

    with col2:
        st.markdown("**Age Group & Companion**")
        under_12            = st.selectbox("Under 12 Years Old", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        over_60             = st.selectbox("Over 60 Years Old",  [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        needs_companion     = st.selectbox("Patient Needs Companion", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        st.markdown("**Medical Conditions**")
        hipertension = st.selectbox("Hipertension",  ["No", "Yes"])   # ➕ ADDED
        diabetes     = st.selectbox("Diabetes",      ["No", "Yes"])
        alcohol      = st.selectbox("Alcoholism",    ["No", "Yes"])
        handcap      = st.selectbox("Handicap",      ["No", "Yes"])
        scholarship  = st.selectbox("Scholarship",   ["No", "Yes"])   # ➕ ADDED
        sms          = st.selectbox("SMS Received",  ["No", "Yes"])

    with col3:
        st.markdown("**Weather on Appointment Day**")
        temp         = st.number_input("Avg Temp Day",  -5.0, 50.0, 25.0)
        rain         = st.number_input("Avg Rain Day",   0.0, 200.0,  0.0)
        max_temp     = st.number_input("Max Temp Day",  -5.0, 50.0,  30.0)   # ➕ ADDED
        max_rain     = st.number_input("Max Rain Day",   0.0, 300.0,  0.0)   # ➕ ADDED
        rain_intensity = st.slider("Rain Intensity (0–5)",  0, 5, 2)         # ➕ ADDED
        heat_intensity = st.slider("Heat Intensity (0–5)",  0, 5, 2)         # ➕ ADDED

        st.markdown("**Previous Day Weather**")
        rainy_day_before = st.selectbox("Rainy Day Before?",  [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  # ➕ ADDED
        storm_day_before = st.selectbox("Storm Day Before?",  [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  # ➕ ADDED

    yn = lambda x: 1 if x == "Yes" else 0

    feature_cols = model.feature_names_in_

    features = {
        # Original
        "age":                    age,
        "average_temp_day":       temp,
        "average_rain_day":       rain,
        "SMS_received":           yn(sms),
        "Diabetes":               yn(diabetes),
        "Alcoholism":             yn(alcohol),
        "Handcap":                yn(handcap),
        f"specialty_{specialty}": 1,
        f"disability_{disability}": 1,
        "gender_M":               1 if gender == "Male" else 0,

        # ➕ Newly Added
        "appointment_shift":      appointment_shift,
        "under_12_years_old":     under_12,
        "over_60_years_old":      over_60,
        "patient_needs_companion": needs_companion,
        "max_temp_day":           max_temp,
        "max_rain_day":           max_rain,
        "rain_intensity":         rain_intensity,
        "heat_intensity":         heat_intensity,
        "rainy_day_before":       rainy_day_before,
        "storm_day_before":       storm_day_before,
        "Hipertension":           yn(hipertension),
        "Scholarship":            yn(scholarship),
    }

    input_df = build_input(features, feature_cols)

    # --- Show entered row as preview ---
    with st.expander("🔍 Input Row Preview (Raw Data)"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    if st.button("Predict", use_container_width=True):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error("⚠️ The patient is unlikely to show up")
        else:
            st.success("✅ SHOW — The patient is highly likely to show up")

        col_a, col_b = st.columns(2)
        col_a.metric("No-show Probability", f"{prob:.2%}")
        col_b.metric("Show Probability",    f"{1 - prob:.2%}")

# =====================================================
# 2. FORECASTING
# =====================================================
if module == "Demand Forecasting":

    model_type = st.selectbox("Model", ["ARIMA", "SARIMA", "XGBoost"])
    days = st.slider("Days", 1, 30, 7)

    if st.button("Forecast"):

        future_dates = pd.date_range(datetime.today(), periods=days)

        if model_type == "ARIMA":
            preds = models["arima"].forecast(days)

        elif model_type == "SARIMA":
            preds = models["sarima"].forecast(days)

        else:
            model = models["forecast_xgb"]
            feature_cols = model.feature_names_in_

            future_df = pd.DataFrame(0, index=future_dates, columns=feature_cols)
            future_df["day_of_week"]    = future_dates.dayofweek
            future_df["month"]          = future_dates.month
            future_df["week_of_year"]   = future_dates.isocalendar().week.values
            future_df["day_of_month"]   = future_dates.day
            future_df["lag_1"]          = 0
            future_df["lag_7"]          = 0
            future_df["lag_14"]         = 0
            future_df["rolling_mean_7"] = 0
            future_df["rolling_mean_14"]= 0

            preds = model.predict(future_df)

        df = pd.DataFrame({
            "Date":      future_dates,
            "Predicted": np.maximum(preds, 0)
        })

        st.line_chart(df.set_index("Date"))
        st.dataframe(df)

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "forecast.csv"
        )