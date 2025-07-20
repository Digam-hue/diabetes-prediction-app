import streamlit as st
import pandas as pd
import joblib

#load trained model and scaler
try:
    model = joblib.load('saved_models/final_model.pkl')
    scaler = joblib.load('saved_models/scaler.pkl')
except FileNotFoundError:
    st.error("Required model files not found. Make sure they exist in the 'saved_models' folder.")
    st.stop()

#ensure the input order matches model training
model_columns = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_M']

#Page setup
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Prediction App")
st.write("""
This app predicts the likelihood of diabetes based on patient health metrics. 
Fill in the details below to get an instant analysis.
""")
st.markdown("---")

# Input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Biometric Data")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=28.0, format="%.1f")
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    st.subheader("Lab Results")
    hba1c = st.number_input("HbA1c (Glycated Hemoglobin)", min_value=0.0, max_value=20.0, value=6.0, format="%.1f")
    chol = st.number_input("Cholesterol", min_value=0.0, max_value=15.0, value=4.5, format="%.1f")
    tg = st.number_input("Triglycerides (TG)", min_value=0.0, max_value=15.0, value=2.0, format="%.1f")

st.markdown("---")
st.subheader("Additional Lab Results")
col3, col4, col5 = st.columns(3)

with col3:
    urea = st.number_input("Urea", min_value=0.0, max_value=40.0, value=5.0, format="%.1f")
    hdl = st.number_input("HDL", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")

with col4:
    cr = st.number_input("Cr (Creatinine)", min_value=1.0, max_value=800.0, value=70.0, format="%.0f")
    ldl = st.number_input("LDL", min_value=0.0, max_value=10.0, value=2.5, format="%.1f")

with col5:
    vldl = st.number_input("VLDL", min_value=0.0, max_value=40.0, value=0.8, format="%.1f")

# Run prediction
if st.button("Analyze and Predict", type="primary"):
    gender_m = 1 if gender == "Male" else 0

    input_data = {
        'AGE': age,
        'Urea': urea,
        'Cr': cr,
        'HbA1c': hba1c,
        'Chol': chol,
        'TG': tg,
        'HDL': hdl,
        'LDL': ldl,
        'VLDL': vldl,
        'BMI': bmi,
        'Gender_M': gender_m
    }

    input_df = pd.DataFrame([input_data])[model_columns]

    st.write("---")
    st.subheader("Patient Data Overview")
    st.dataframe(input_df)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    prediction_labels = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
    result = prediction_labels[prediction[0]]

    st.write("---")
    st.subheader("Prediction Result")

    if result == 'Diabetic':
        st.error(f"**Status: {result}**")
        st.warning("High risk of diabetes. Please consult a doctor.")
    elif result == 'Pre-Diabetic':
        st.warning(f"**Status: {result}**")
        st.info("Slight risk. Consider lifestyle changes and a medical consultation.")
    else:
        st.success(f"**Status: {result}**")
        st.info("Low risk. Keep up the healthy lifestyle.")

    st.subheader("Prediction Confidence")
    proba_df = pd.DataFrame({
        'Class': ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic'],
        'Probability': prediction_proba[0]
    })
    st.bar_chart(proba_df.set_index('Class'))
