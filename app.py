import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Saved Model and Scaler ---
# Make sure the paths are correct, pointing to where you saved your files.
try:
    model = joblib.load('saved_models/final_model.pkl')
    scaler = joblib.load('saved_models/scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Make sure 'final_model.pkl' and 'scaler.pkl' are in the 'saved_models' directory.")
    st.stop() # Stop the app from running further if files are missing.

# --- 2. Define the Column Order (CRITICAL) ---
# This must be the exact same order of columns that your model was trained on.
# You can get this from the training notebook (e.g., from X_train.columns)
model_columns = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender_M']

# --- 3. Set Up the Streamlit Page ---
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Prediction App")
st.write("""
This app predicts the likelihood of a patient having diabetes based on their medical information. 
Please enter the patient's details below.
""")
st.markdown("---")

# --- 4. Create the User Input Fields in Two Columns for a Cleaner Layout ---
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

# --- 5. The Predict Button and Logic ---
if st.button("Analyze and Predict", type="primary"):
    # Step A: Preprocess the inputs to match the model's training data
    
    # Convert gender to the one-hot encoded format (Gender_M)
    gender_m = 1 if gender == "Male" else 0
    
    # Create a dictionary with the user's input
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
    
    # Convert to a pandas DataFrame with the correct column order
    input_df = pd.DataFrame([input_data])[model_columns]
    
    st.write("---")
    st.subheader("Patient Data Overview")
    st.dataframe(input_df)

    # Step B: Scale the features using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # Step C: Make the prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Map the numeric prediction back to a human-readable label
    # This mapping must match your LabelEncoder from the notebook
    # 0: Non-Diabetic, 1: Pre-Diabetic, 2: Diabetic
    prediction_labels = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
    result = prediction_labels[prediction[0]]
    
    # --- 6. Display the Result ---
    st.write("---")
    st.subheader("Prediction Result")
    
    if result == 'Diabetic':
        st.error(f"**Status: {result}**")
        st.warning("The model predicts a high risk of diabetes. Please consult a healthcare professional.")
    elif result == 'Pre-Diabetic':
        st.warning(f"**Status: {result}**")
        st.info("The model indicates a potential risk of developing diabetes. Lifestyle changes and a follow-up with a doctor are recommended.")
    else:
        st.success(f"**Status: {result}**")
        st.info("The model predicts a low risk of diabetes. Maintain a healthy lifestyle.")
        
    # Display the prediction probabilities in a more visual way
    st.subheader("Prediction Confidence")
    proba_df = pd.DataFrame({
        'Class': ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic'],
        'Probability': prediction_proba[0]
    })
    st.bar_chart(proba_df.set_index('Class'))