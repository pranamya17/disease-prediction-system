import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)


# -------------------------------------------------
# LOAD SAVED MODEL AND PREPROCESSING FILES
# -------------------------------------------------

model = joblib.load("disease_prediction_naive_bayes.pkl")
label_encoder = joblib.load("disease_label_encoder.pkl")
scaler = joblib.load("disease_scaler.pkl")
feature_columns = joblib.load("disease_feature_columns.pkl")
numerical_columns = joblib.load("disease_numerical_columns.pkl")


# -------------------------------------------------
# SYMPTOMS
# -------------------------------------------------

all_symptoms = [
    "Unknown",
    "abdominal pain",
    "blurred vision",
    "body ache",
    "chest pain",
    "cough",
    "diarrhea",
    "dizziness",
    "fatigue",
    "fever",
    "frequent urination",
    "headache",
    "increased thirst",
    "nausea",
    "sensitivity to light",
    "shortness of breath",
    "sore throat",
    "vomiting"
]


# -------------------------------------------------
# AGE GROUP FUNCTION
# -------------------------------------------------

def get_age_group(age):

    if 18 <= age <= 30:
        return "18-30"

    elif 31 <= age <= 45:
        return "31-45"

    elif 46 <= age <= 60:
        return "46-60"

    elif 61 <= age <= 75:
        return "61-75"

    elif age >= 76:
        return "76+"

    else:
        return "Unknown"


# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------

def predict_disease(patient):

    patient = patient.copy()

    # Create symptom binary features
    for symptom in all_symptoms:

        patient[symptom] = patient["Symptoms"].apply(
            lambda x: 1
            if symptom in [s.strip() for s in x.split(",")]
            else 0
        )

    # Remove original Symptoms column
    patient = patient.drop("Symptoms", axis=1)

    # Create Age Group
    patient["Age_Group"] = patient["Age"].apply(get_age_group)

    # One-hot encoding
    patient = pd.get_dummies(
        patient,
        columns=[
            "Gender",
            "Medical_History",
            "Age_Group"
        ],
        drop_first=True,
        dtype=int
    )

    # Make exactly the same 41 features
    patient = patient.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale numerical columns
    patient[numerical_columns] = scaler.transform(
        patient[numerical_columns]
    )

    # Prediction
    prediction = model.predict(patient)

    # Convert encoded prediction to disease name
    disease = label_encoder.inverse_transform(prediction)[0]

    # Prediction probability
    probability = np.max(
        model.predict_proba(patient)
    ) * 100

    return disease, probability


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🩺 Disease Prediction System")

st.write(
    "Enter the patient's health information below "
    "to generate a machine-learning prediction."
)


# -------------------------------------------------
# PATIENT INPUT
# -------------------------------------------------

st.subheader("Patient Information")


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=45
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)


sugar_level = st.number_input(
    "Sugar Level",
    min_value=50,
    max_value=500,
    value=100
)


cholesterol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=500,
    value=180
)


symptoms = st.multiselect(
    "Symptoms",
    all_symptoms
)


medical_history = st.selectbox(
    "Medical History",
    [
        "None",
        "Diabetes",
        "Hypertension",
        "Asthma",
        "Heart Disease"
    ]
)


# -------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------

if st.button("🔮 Predict Disease"):

    # Make sure at least one symptom is selected
    if len(symptoms) == 0:

        st.warning(
            "Please select at least one symptom."
        )

    else:

        # Convert selected symptoms into comma-separated text
        symptoms_text = ", ".join(symptoms)

        # Create patient DataFrame
        patient = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Symptoms": [symptoms_text],
            "Blood_Pressure": [blood_pressure],
            "Sugar_Level": [sugar_level],
            "Cholesterol": [cholesterol],
            "Medical_History": [medical_history]
        })

        # Get prediction
        disease, confidence = predict_disease(patient)

        # Display result
        st.success("Prediction Completed!")

        st.subheader("Prediction Result")

        st.write(
            f"### 🩺 Predicted Disease: **{disease}**"
        )

        st.write(
            f"### 📊 Prediction Probability: **{confidence:.2f}%**"
        )

        st.info(
            "This is a machine-learning prediction and "
            "should not be considered a medical diagnosis."
        )