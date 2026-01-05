# taxi_price_app.py
import streamlit as st
import pandas as pd
import pickle




df = pd.read_csv(r'h:\SpeclizationTraingClass\taxi_trip_pricingR.csv')

st.set_page_config(page_title="Taxi Trip Price Prediction", page_icon="🚖", layout="centered")
st.title("🚖 Taxi Trip Price Prediction App")

st.header("Enter Trip Details:")

# --------------------------
# Step 2: Input Fields
# --------------------------
# Numeric features
trip_distance = st.number_input("Trip Distance (km)", min_value=0.0, max_value=float(df['Trip_Distance_km'].max()), value=float(df['Trip_Distance_km'].mean()))
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
base_fare = st.number_input("Base Fare", min_value=0.0, max_value=float(df['Base_Fare'].max()), value=float(df['Base_Fare'].mean()))
per_km_rate = st.number_input("Per Km Rate", min_value=0.0, max_value=float(df['Per_Km_Rate'].max()), value=float(df['Per_Km_Rate'].mean()))
per_minute_rate = st.number_input("Per Minute Rate", min_value=0.0, max_value=float(df['Per_Minute_Rate'].max()), value=float(df['Per_Minute_Rate'].mean()))
trip_duration = st.number_input("Trip Duration (Minutes)", min_value=0.0, max_value=float(df['Trip_Duration_Minutes'].max()), value=float(df['Trip_Duration_Minutes'].mean()))

# Categorical features
time_of_day = st.selectbox("Time of Day", df['Time_of_Day'].unique())
day_of_week = st.selectbox("Day of Week", df['Day_of_Week'].unique())
traffic = st.selectbox("Traffic Conditions", df['Traffic_Conditions'].unique())
weather = st.selectbox("Weather", df['Weather'].unique())

# --------------------------
# Step 3: Prepare Input for Model
# --------------------------
input_dict = {
    'Trip_Distance_km': trip_distance,
    'Time_of_Day': time_of_day,
    'Day_of_Week': day_of_week,
    'Passenger_Count': passenger_count,
    'Traffic_Conditions': traffic,
    'Weather': weather,
    'Base_Fare': base_fare,
    'Per_Km_Rate': per_km_rate,
    'Per_Minute_Rate': per_minute_rate,
    'Trip_Duration_Minutes': trip_duration
}

# Convert to DataFrame and one-hot encode categorical variables (same as training)
input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
# Align with model features
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# --------------------------
# Step 4: Prediction
# --------------------------
if st.button("Predict Trip Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Trip Price: ${prediction:,.2f}")

# Optional: Show raw data
if st.checkbox("Show Dataset Preview"):
    st.write(df.head())
