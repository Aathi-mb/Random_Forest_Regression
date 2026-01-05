import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Taxi Trip Price Prediction",
    page_icon="🚖",
    layout="centered"
)

st.title("🚖 Taxi Trip Price Prediction")
st.write("Enter trip details to predict the taxi fare")

# -------------------------------------------------
# Load model (JOBLIB ONLY)
# -------------------------------------------------
model = joblib.load("taxi_rf_model_joblib.pkl")

# -------------------------------------------------
# Load dataset (RELATIVE PATH ONLY)
# -------------------------------------------------
df = pd.read_csv("taxi_trip_pricingR.csv")

# -------------------------------------------------
# Input section
# -------------------------------------------------
st.header("Trip Details")

trip_distance = st.number_input(
    "Trip Distance (km)",
    min_value=0.0,
    value=float(df["Trip_Distance_km"].mean())
)

passenger_count = st.number_input(
    "Passenger Count",
    min_value=1,
    max_value=10,
    value=1
)

trip_duration = st.number_input(
    "Trip Duration (Minutes)",
    min_value=0.0,
    value=float(df["Trip_Duration_Minutes"].mean())
)

base_fare = st.number_input(
    "Base Fare",
    min_value=0.0,
    value=float(df["Base_Fare"].mean())
)

per_km_rate = st.number_input(
    "Per Km Rate",
    min_value=0.0,
    value=float(df["Per_Km_Rate"].mean())
)

per_minute_rate = st.number_input(
    "Per Minute Rate",
    min_value=0.0,
    value=float(df["Per_Minute_Rate"].mean())
)

time_of_day = st.selectbox(
    "Time of Day",
    sorted(df["Time_of_Day"].dropna().astype(str).unique())
)

day_of_week = st.selectbox(
    "Day of Week",
    sorted(df["Day_of_Week"].dropna().astype(str).unique())
)

traffic = st.selectbox(
    "Traffic Conditions",
    sorted(df["Traffic_Conditions"].dropna().astype(str).unique())
)

weather = st.selectbox(
    "Weather",
    sorted(df["Weather"].dropna().astype(str).unique())
)

# -------------------------------------------------
# Prepare input data
# -------------------------------------------------
input_data = {
    "Trip_Distance_km": trip_distance,
    "Passenger_Count": passenger_count,
    "Trip_Duration_Minutes": trip_duration,
    "Base_Fare": base_fare,
    "Per_Km_Rate": per_km_rate,
    "Per_Minute_Rate": per_minute_rate,
    "Time_of_Day": time_of_day,
    "Day_of_Week": day_of_week,
    "Traffic_Conditions": traffic,
    "Weather": weather
}

input_df = pd.DataFrame([input_data])

# One-hot encode categorical columns
input_df = pd.get_dummies(input_df)

# Align with training features
input_df = input_df.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Trip Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Trip Price: ₹ {prediction:.2f}")

# -------------------------------------------------
# Optional: Show dataset
# -------------------------------------------------
if st.checkbox("Show dataset preview"):
    st.dataframe(df.head())
