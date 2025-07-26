import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests

# ------------------- Page Setup -------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="centered"
)

st.title("🎗️ Breast Cancer Risk Predictor")
st.write("Enter the following information to predict the likelihood of breast cancer.")

# ------------------- Load Model -------------------
model_path = "breastcancer.joblib"
model_url = "https://your-model-storage.com/breastcancer.joblib"  # 🔁 Replace with actual URL

if not os.path.exists(model_path):
    st.warning("Downloading model file...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    st.success("Model downloaded successfully.")

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------- User Inputs -------------------
st.header("🔎 Input Features")

age = st.number_input("Age", min_value=20, max_value=100, value=45)
menarche = st.slider("Age at First Menstruation (Menarche)", 8, 20, 12)
biopsies = st.number_input("Number of Prior Biopsies", min_value=0, max_value=10, value=1)

alcohol = st.radio("Do you consume alcohol?", ["No", "Yes"])
alcohol_binary = 1 if alcohol == "Yes" else 0

tobacco = st.radio("Do you use tobacco?", ["No", "Yes"])
tobacco_binary = 1 if tobacco == "Yes" else 0

exercise_numeric = st.number_input("Weekly Exercise (hours)", min_value=0.0, max_value=50.0, value=3.0)

nrelbc_clean = st.selectbox("Family History of Breast Cancer", ["None", "Multiple", "Other"])
nrelbc_encoded = {"Other": 0, "None": 1, "Multiple": 2}[nrelbc_clean]

# ------------------- Preprocessing -------------------
alcohol_binary = 1 if alcohol == "Yes" else 0
tobacco_binary = 1 if tobacco == "Yes" else 0
nrelbc_encoded = {"Other": 0, "None": 1, "Multiple": 2}[nrelbc_clean]

# ------------------- Prediction -------------------
if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[
        age, menarche, biopsies,
        alcohol_binary, tobacco_binary,
        exercise_numeric, nrelbc_encoded
    ]], columns=[
        'age', 'menarche', 'biopsies',
        'alcohol_binary', 'tobacco_binary',
        'exercise_numeric', 'nrelbc_encoded'
    ])

    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.markdown(
                "<h3 style='color:red; text-align:center;'>🎗️ Prediction: High Risk of Breast Cancer</h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='color:green; text-align:center;'>🎗️ Prediction: Low Risk of Breast Cancer</h3>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
