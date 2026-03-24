import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Disease Prediction App", layout="centered")

# Title
st.title("💓 AI-Based Disease Risk Prediction System")
st.markdown("Enter patient details to predict heart disease risk.")

# Load dataset
df = pd.read_csv("heart.csv")

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Convert target to binary
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Features & Target
X = df.drop('num', axis=1)
y = df['num']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Input section
st.header("🧾 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 40)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes)", [0, 1])
    restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes)", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

# Extra features (fixed values for simplicity)
slope = 1
ca = 0
thal = 2

# Prediction
if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {probability:.2f}")

    st.progress(int(probability * 100))