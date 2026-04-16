import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Disease Prediction", layout="wide")

# Title
st.title("💓 AI-Based Heart Disease Prediction System")
st.markdown("### Smart Healthcare using Machine Learning")

# Load dataset
df = pd.read_csv("heart.csv")

# Drop unnecessary columns
df = df.drop(columns=['id'], errors='ignore')

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('num', axis=1)
y = df['num']

# Fill missing values
X = X.fillna(X.mean())

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# ---------------- SIDEBAR INPUT ---------------- #

st.sidebar.header("🧾 Enter Patient Details")

age = st.sidebar.slider("Age", 20, 80, 40)
sex_label = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_label == "Male" else 0
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting BP", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Angina", [0, 1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)

# Fixed values
slope = 1
ca = 0
thal = 2

# ---------------- MAIN OUTPUT ---------------- #

st.subheader("📊 Prediction Result")

if st.button("🔍 Predict"):

    input_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    input_data = pd.DataFrame([input_dict])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Layout columns
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🔴 HIGH RISK")
        else:
            st.success("🟢 LOW RISK")

    with col2:
        st.metric(label="Risk Probability", value=f"{probability*100:.2f}%")

    # Progress bar
    st.progress(int(probability * 100))

    # ---------------- RECOMMENDATIONS ---------------- #

    st.subheader("🩺 Health Recommendation")

    if prediction == 1:
        st.warning("""
        ⚠️ High risk detected. Please consider:
        - Consult a cardiologist immediately  
        - Reduce cholesterol intake  
        - Maintain regular exercise  
        - Monitor blood pressure regularly  
        """)
    else:
        st.info("""
        ✅ Low risk. Maintain healthy lifestyle:
        - Regular physical activity  
        - Balanced diet  
        - Routine health checkups  
        """)
