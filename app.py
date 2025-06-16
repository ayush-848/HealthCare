import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model, scaler, and feature order
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

# ---------- Page Config ----------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="centered",
    page_icon="🩺"
)

# ---------- CSS Styling ----------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f5f6fa;
        color: #333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.6em 1.2em;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 1.5em;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Get an instant health risk analysis using Machine Learning based on your details.")

# ---------- Input Form ----------
with st.form("input_form"):
    st.subheader("📋 Enter your health details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 1, 100, 30)
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 24.0)
        hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5)
        glucose = st.number_input("Blood Glucose (mg/dL)", 50, 300, 120)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
        heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
        smoking = st.selectbox("Smoking History", [
            "never", "former", "current", "ever", "not current", "No Info"
        ])
    submitted = st.form_submit_button("🔍 Predict")

# ---------- Input Encoding ----------
def encode_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=["gender", "smoking_history"], drop_first=True)

    # Fill missing dummy cols
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    return df[feature_order]

# ---------- Prediction ----------
if submitted:
    user_input = {
        "age": age,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "gender": gender,
        "smoking_history": smoking
    }

    df_input = encode_input(user_input)
    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.markdown("### 🧾 Prediction Result")
    with st.container():
        if prediction == 1:
            st.error(f"⚠️ **You are likely at risk of Diabetes.**\n\n🔬 *Confidence: {prob:.2%}*")
        else:
            st.success(f"✅ **You are unlikely to have Diabetes.**\n\n🧘 *Confidence: {(1 - prob):.2%}*")

    # ---------- Health Tips Based on Risk Factors ----------
    st.markdown("### 💡 Personalized Health Suggestions")

    if age > 50:
        st.info("👴 **Age over 50** increases diabetes risk. Annual screenings are recommended.")

    if bmi >= 30:
        st.info("⚖️ **High BMI** (≥ 30) is a major risk factor. Consider lifestyle changes.")

    if hba1c > 6.4:
        st.info("🧪 **HbA1c over 6.4%** suggests pre-diabetes or diabetes. Seek medical consultation.")

    if glucose >= 126:
        st.info("🩸 **Blood glucose over 125 mg/dL** is concerning. Follow up with a test.")

    if smoking in ['current', 'ever']:
        st.info("🚭 **Smoking history** increases risk. Consider quitting if you haven’t.")

    if hypertension:
        st.info("💓 **Hypertension** is closely linked with diabetes. Monitor regularly.")

    if heart_disease:
        st.info("❤️ **Heart conditions** can co-exist with diabetes. Manage both with care.")
    else:
        st.info("❤️ **No heart disease** reported. Maintain a healthy lifestyle to keep it that way.")
# ---------- Footer ----------
st.markdown("---")
st.caption("📊 Model trained with real medical records. Predictions are estimations, not diagnoses.")
