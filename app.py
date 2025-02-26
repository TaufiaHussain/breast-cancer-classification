import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/data.csv")

# Drop unnecessary columns
df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')

# Convert 'diagnosis' column: M → 1 (Malignant), B → 0 (Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load the trained model
model = joblib.load("best_breast_cancer_model.pkl")

# Define feature names
feature_names = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
    "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
]

# Streamlit App Title
st.title("🔬 Breast Cancer Prediction App")

# Sidebar Input
st.sidebar.header("📂 Upload Patient Biopsy Report (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Function to collect user input manually
def user_input_features():
    input_data = []
    for feature in feature_names:
        value = st.sidebar.number_input(f"{feature}", min_value=0.0, format="%.4f")
        input_data.append(value)
    return np.array(input_data).reshape(1, -1)

# Check if file is uploaded
if uploaded_file:
    patient_df = pd.read_csv(uploaded_file)

    # Ensure the file contains required features
    if set(feature_names).issubset(patient_df.columns):
        user_data = patient_df.iloc[0][feature_names].values.reshape(1, -1)
        st.sidebar.success("✅ Patient biopsy data loaded successfully!")
    else:
        st.sidebar.error("❌ Incorrect file format. Ensure it has the required features.")
        user_data = user_input_features()  # Fallback to manual input if file is incorrect
else:
    user_data = user_input_features()  # Default to manual input

# Predict Button
if st.sidebar.button("Predict"):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)

    # Display Prediction
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ The tumor is **Malignant (Cancerous)**.")
    else:
        st.success("✅ The tumor is **Benign (Non-Cancerous)**.")

    # Display Probability
    st.subheader("Prediction Probability:")
    st.write(f"Benign: {prediction_proba[0][0] * 100:.2f}%")
    st.write(f"Malignant: {prediction_proba[0][1] * 100:.2f}%")

    # Function to generate PDF Report
    def generate_pdf(prediction, probability, user_data):
        temp_dir = tempfile.mkdtemp()
        pdf_filename = os.path.join(temp_dir, "Breast_Cancer_Prediction_Report.pdf")

        c = canvas.Canvas(pdf_filename, pagesize=letter)
        c.setFont("Helvetica", 12)

        c.drawString(200, 750, "Breast Cancer Prediction Report")
        c.line(200, 745, 420, 745)
        c.drawString(50, 700, f"Prediction: {'Malignant (Cancerous)' if prediction == 1 else 'Benign (Non-Cancerous)'}")
        c.drawString(50, 680, f"Probability: {probability * 100:.2f}%")

        c.save()
        return pdf_filename

    # PDF Download Button
    if st.button("Download PDF Report"):
        pdf_file = generate_pdf(prediction[0], prediction_proba[0][prediction[0]], user_data)
        with open(pdf_file, "rb") as file:
            st.download_button(
                label="📄 Download Your Prediction Report",
                data=file,
                file_name="Breast_Cancer_Prediction_Report.pdf",
                mime="application/pdf"
            )

    # Visualization of Tumor Features
    st.subheader("📊 Visualization of Tumor Features")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=feature_names, y=user_data[0], palette="viridis", ax=ax)
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_title("Tumor Feature Values Entered by User")
    ax.set_ylabel("Feature Value")
    st.pyplot(fig)

    # SHAP Explainability
    st.subheader("🔍 Explainability: Which Features Influenced This Prediction?")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(user_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(shap_values[0], max_display=10)
    st.pyplot(fig)
