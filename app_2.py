import streamlit as st
import pandas as pd
import pickle
xgb_model = pickle.load(open("xgb_model_2.sav", "rb"))
st.title("Daibetics Prediction")
pregnancies=st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bloodPressure = st.number_input("BloodPressure")
skinThickness = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
diabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
age = st.number_input("Age")

start = st.button("Predict")

if start:
    d={

        'Pregnancies':0,
        'Glucose':0,
        'BloodPressure':0,
        'SkinThickness':0,
        'Insulin':0,
        'BMI':0,
        'DiabetesPedigreeFunction':0,
        'Age':0,
        'Outcome':0,

    }
    d["Pregnancies"] = pregnancies
    d["Glucose"] = glucose
    d["BloodPressure"] = bloodPressure
    d["SkinThickness"] = skinThickness
    d["Insulin"] = insulin
    d["BMI"] = bmi
    d["DiabetesPedigreeFunction"] = diabetesPedigreeFunction
    d["Age"] = age
    diabetes_df_copy = pd.DataFrame(columns=d.keys())
    diabetes_df_copy = diabetes_df_copy.append(d, ignore_index=True)

    xgb_pred = xgb_model.predict(diabetes_df_copy)[0]
    xgb_pred = round(xgb_pred)

    if xgb_pred == 0:
        result = "not diabetic"
    else:
        result = "diabetic"

    st.text(f"The prediction is: {result}")
