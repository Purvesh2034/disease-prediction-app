import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Disease Prediction App", layout="wide")

# Title
st.title("🩺 Disease Prediction System")
st.markdown("### Enter Patient Details Below")

# Sidebar inputs
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 1, 100, 25)

sex = st.sidebar.selectbox(
    "Sex",
    ["Male", "Female"]
)

# Convert sex to numeric
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])

trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)

chol = st.sidebar.slider("Cholesterol", 100, 400, 200)

fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])

restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])

thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)

slope = st.sidebar.selectbox("Slope", [0, 1, 2])

ca = st.sidebar.selectbox("Major Vessels (ca)", [0, 1, 2, 3])

thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

# Predict button
if st.sidebar.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Disease\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Disease\nProbability: {probability:.2f}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Machine Learning")
