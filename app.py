import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests

# ------------------- Page Setup -------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",  # Pink ribbon
    layout="centered",
)

st.title("🎗️ Breast Cancer Risk Predictor")
st.write("Enter the details below to estimate the risk of breast cancer.")

# ------------------- Model Loading -------------------
model_path = "breastcancer.joblib"
model_url = "https://your-storage-service.com/breastcancer.joblib"  # Replace with actual link

# Download model if not available locally
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

model = joblib.load(model_path)

# ------------------- Input Fields -------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=40)
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    exercise_numeric = st.number_input("Weekly Exercise (hours)", min_value=0.0, max_value=50.0, value=3.0)
    nrelbc_clean = st.selectbox("Family History of Breast Cancer", ["None", "Multiple", "Other"])

with col2:
    menarche = st.slider("Age at First Menstruation (Menarche)", 8, 20, 12)
    tobacco = st.selectbox("Tobacco Use", ["No", "Yes"])
    biopsies = st.number_input("Number of Prior Biopsies", min_value=0, max_value=10, value=1)

# Mapping inputs
alcohol_binary = 1 if alcohol == "Yes" else 0
tobacco_binary = 1 if tobacco == "Yes" else 0
nrelbc_encoded = {"Other": 0, "None": 1, "Multiple": 2}[nrelbc_clean]

# ------------------- Predict Button -------------------
if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, menarche, biopsies, alcohol_binary,
        tobacco_binary, exercise_numeric, nrelbc_encoded
    ]], columns=[
        'age', 'menarche', 'biopsies', 'alcohol_binary',
        'tobacco_binary', 'exercise_numeric', 'nrelbc_encoded'
    ])

    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.markdown(
                "<h3 style='color:red; text-align:center;'>Prediction: High Risk of Breast Cancer</h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='color:green; text-align:center;'>Prediction: Low Risk of Breast Cancer</h3>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
