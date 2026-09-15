from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("model.pkl")


@st.cache_resource
def load_model():
	return joblib.load(MODEL_PATH)


def build_input(
	gender: str,
	age: float,
	hypertension: int,
	heart_disease: int,
	smoking_history: str,
	bmi: float,
	hba1c_level: float,
	blood_glucose_level: int,
) -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"gender": gender,
				"age": age,
				"hypertension": hypertension,
				"heart_disease": heart_disease,
				"smoking_history": smoking_history,
				"bmi": bmi,
				"HbA1c_level": hba1c_level,
				"blood_glucose_level": blood_glucose_level,
			}
		]
	)


st.set_page_config(
	page_title="Diabetes Risk Check",
	page_icon="🩺",
	layout="centered",
)

st.title("Diabetes Risk Check")
st.write("Enter the patient measurements to get a prediction from the saved model.")

try:
	model = load_model()
except Exception as error:
	st.error(f"Could not load the model from {MODEL_PATH}: {error}")
	st.stop()

with st.form("diabetes_prediction_form"):
	st.subheader("Patient information")
	first_column, second_column = st.columns(2)

	with first_column:
		gender = st.selectbox("Gender", ["Female", "Male", "Other"])
		age = st.number_input("Age", min_value=0.0, max_value=120.0, value=40.0, step=1.0)
		hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda value: "No" if value == 0 else "Yes")
		heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda value: "No" if value == 0 else "Yes")

	with second_column:
		smoking_history = st.selectbox(
			"Smoking history",
			["No Info", "never", "current", "former", "ever", "not current"],
		)
		bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.19, step=0.01)
		hba1c_level = st.number_input("HbA1c level", min_value=0.0, max_value=20.0, value=5.7, step=0.1)
		blood_glucose_level = st.number_input(
			"Blood glucose level",
			min_value=0,
			max_value=1000,
			value=100,
			step=1,
		)

	submitted = st.form_submit_button("Check diabetes risk", type="primary", use_container_width=True)

if submitted:
	patient = build_input(
		gender,
		age,
		hypertension,
		heart_disease,
		smoking_history,
		bmi,
		hba1c_level,
		blood_glucose_level,
	)
	prediction = int(model.predict(patient)[0])
	probability = None
	if hasattr(model, "predict_proba"):
		classes = list(model.classes_)
		probability = float(model.predict_proba(patient)[0][classes.index(1)])

	st.divider()
	if prediction == 1:
		st.error("The model predicts a higher likelihood of diabetes.")
	else:
		st.success("The model predicts a lower likelihood of diabetes.")

	if probability is not None:
		st.metric("Estimated diabetes probability", f"{probability:.1%}")
		st.progress(probability)

	st.caption("This prediction is for educational use and is not a medical diagnosis.")
