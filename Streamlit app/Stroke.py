import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os



# Load saved model and scaler
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# Page title
st.set_page_config(page_title="Stroke Prediction App", layout="centered")
st.title("🧠 Stroke Risk Prediction")
st.markdown("Predict the likelihood of a stroke based on patient health data.")

# Sidebar for user input
st.sidebar.header("Input Patient Data")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 0, 100, 45)
    hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.number_input("Avg Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["Formerly Smoked", "Never Smoked", "Smokes", "Unknown"])

    # Create DataFrame
    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    return pd.DataFrame([data])

input_df = user_input()

# Preprocess input data to match training
def preprocess(df):
    # Match training preprocessing steps exactly
    df_encoded = pd.get_dummies(df)
    
    # Reindex to match model's input
    columns_path = os.path.join(os.path.dirname(__file__), "model_columns.pkl")

    expected_cols = joblib.load(columns_path)  # Save this during training
    for col in expected_cols:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_cols]
    
    # Scale numerical features
    
    return df_encoded

# Predict button
if st.button("Predict Stroke Risk"):
    processed = preprocess(input_df)
    result = model.predict(processed)
    proba = model.predict_proba(processed)[0][1]

    st.subheader("🔍 Prediction Result:")
    if result[0] == 1:
        st.error(f"🔴 High Risk of Stroke (Chances of Stroke: {proba*100:.2f}%)")
    else:
        st.success(f"🟢 Low Risk of Stroke (Chances of Stroke: {proba*100:.2f}%)")
    
    st.markdown("Note: This is a machine learning prediction and should not replace professional medical advice.")
