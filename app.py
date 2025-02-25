import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load("best_breast_cancer_model.pkl")

# Title
st.title("🔬 Breast Cancer Prediction App")

# Sidebar Information
st.sidebar.header("Input Features")
st.sidebar.write("Enter the values below to predict if the tumor is Malignant or Benign.")

# Feature Input Fields
def user_input_features():
    feature_names = [
        "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
        "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
        "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
        "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
        "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
        "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
    ]
    
    input_data = []
    for feature in feature_names:
        value = st.sidebar.number_input(f"{feature}", min_value=0.0, format="%.4f")
        input_data.append(value)
    
    return np.array(input_data).reshape(1, -1)

# Get user input
user_data = user_input_features()

# Predict button
if st.sidebar.button("Predict"):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)

    # Display results
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ The tumor is **Malignant (Cancerous)**.")
    else:
        st.success("✅ The tumor is **Benign (Non-Cancerous)**.")

    st.subheader("Prediction Probability:")
    st.write(f"Benign: {prediction_proba[0][0] * 100:.2f}%")
    st.write(f"Malignant: {prediction_proba[0][1] * 100:.2f}%")
