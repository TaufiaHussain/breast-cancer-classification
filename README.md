# 🔬 Breast Cancer Prediction App

## 🏥 About the Project
This is a **machine learning-powered web app** that predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using **30 tumor features** from the **Breast Cancer Wisconsin Dataset**.

🚀 **Live App:** [Click here to try it](https://breast-cancer-classification-okiloekrwntap2raz7rq6e.streamlit.app/)  

---


## 🏥 **Key Features**
✅ **Upload Biopsy Data (CSV)** – Doctors can upload **real biopsy reports**  
✅ **Instant Prediction** – Classifies tumor as **Benign or Malignant**  
✅ **PDF Report Generation** – Downloadable medical-style reports  
✅ **Data Visualization** – Bar charts to understand tumor features  
✅ **Explainable AI (SHAP)** – See which tumor features influenced the prediction  

---

## 📊 Dataset Information
- **Dataset:** [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
- **Samples:** 569  
- **Features:** 30 (tumor size, texture, symmetry, etc.)  
- **Target Variable:**  
  - `1` = Malignant (Cancerous)  
  - `0` = Benign (Non-Cancerous)  

---

## 🖥️ How to Run Locally
### 1️⃣ Clone this Repository
```bash
git clone https://github.com/TaufiaHussain/breast-cancer-classification.git
cd breast-cancer-classification

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py

🏗️ Model Training & Accuracy
Algorithm Used: Logistic Regression (Tuned with Grid Search)
Baseline Accuracy: 97.37%
Optimized Accuracy: 🚀 99.12%

📜 Disclaimer
🛑 This app is not a substitute for professional medical advice.
🛑 Always consult a licensed medical professional for diagnosis.

📬 Contact
📧 Taufia Hussain
🔗 LinkedIn Profile : https://www.linkedin.com/in/taufia-hussain-52300015/
📌 GitHub Repo: View Code


---

### **3️⃣ Replace These Parts**
📌 **Replace `https://your-streamlit-app-link.streamlit.app`** with your actual Streamlit app link.  
📌 **Replace `YOUR_GITHUB_USERNAME`** with your GitHub username.  
📌 **Replace `Your Name`** with your actual name.  
📌 **Replace `LinkedIn Profile`** if you want to share your LinkedIn.  

---

### **4️⃣ Push README to GitHub**
Run these commands in **GitHub Codespaces terminal**:

```bash
git add README.md
git commit -m "Added README file"
git push origin main




