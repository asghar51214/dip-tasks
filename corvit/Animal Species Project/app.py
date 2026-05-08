import streamlit as st
import pickle
import numpy as np

# ==============================
# LOAD MODELS
# ==============================
linear_model = pickle.load(open("linear_model.pkl", "rb"))
poly_model = pickle.load(open("poly_model.pkl", "rb"))
poly = pickle.load(open("poly_converter.pkl", "rb"))

# ==============================
# UI
# ==============================
st.title("🐾 Animal Lifespan Predictor")

st.write("Enter animal features:")

speed = st.number_input("Speed (km/h)", min_value=0.0)
weight = st.number_input("Weight (kg)", min_value=0.0)

model_choice = st.radio("Choose Model", ["Linear", "Polynomial"])

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Lifespan"):

    input_data = np.array([[speed, weight]])

    if model_choice == "Linear":
        prediction = linear_model.predict(input_data)[0]
    else:
        input_poly = poly.transform(input_data)
        prediction = poly_model.predict(input_poly)[0]

    st.success(f"Predicted Lifespan: {prediction:.2f} years")