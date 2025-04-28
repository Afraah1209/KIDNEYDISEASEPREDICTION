import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("kidney_disease_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Kidney Disease Prediction App")

st.header("Enter Patient Details:")

# Numerical Inputs with step
bp = st.number_input("Blood Pressure (Bp)", min_value=0.0, max_value=200.0, step=0.1)
sg = st.number_input("Specific Gravity (Sg)", min_value=1.0, max_value=1.030, step=0.001)
al = st.number_input("Albumin (Al)", min_value=0.0, max_value=5.0, step=0.1)
su = st.number_input("Sugar (Su)", min_value=0.0, max_value=5.0, step=0.1)
bu = st.number_input("Blood Urea (Bu)", min_value=0.0, max_value=400.0, step=1.0)
sc = st.number_input("Serum Creatinine (Sc)", min_value=0.0, max_value=30.0, step=0.1)
sod = st.number_input("Sodium (Sod)", min_value=100.0, max_value=170.0, step=0.1)
pot = st.number_input("Potassium (Pot)", min_value=2.0, max_value=10.0, step=0.1)
hemo = st.number_input("Hemoglobin (Hemo)", min_value=3.0, max_value=20.0, step=0.1)
wbcc = st.number_input("White Blood Cell Count (Wbcc)", min_value=3000.0, max_value=18000.0, step=10.0)
rbcc = st.number_input("Red Blood Cell Count (Rbcc)", min_value=2.0, max_value=7.0, step=0.1)

# Categorical Inputs
rbc = st.selectbox("Red Blood Cells (Rbc)", ("Normal", "Abnormal"))
htn = st.selectbox("Hypertension (Htn)", ("Yes", "No"))

# Map categorical inputs manually
rbc_mapping = {"Normal": 0, "Abnormal": 1}
htn_mapping = {"Yes": 1, "No": 0}

# Convert categorical to numerical
rbc_value = rbc_mapping[rbc]
htn_value = htn_mapping[htn]

# Prepare feature array
features = np.array([[bp, sg, al, su, rbc_value, bu, sc, sod, pot, hemo, wbcc, rbcc, htn_value]])

# Apply scaling
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict"):
    prediction = model['model'].predict(features_scaled)   # <-- Notice model['model']
    if prediction[0] == 1:
        st.success("The patient is likely to have Chronic Kidney Disease (CKD).")
    else:
        st.success("The patient is NOT likely to have Chronic Kidney Disease (CKD).")
