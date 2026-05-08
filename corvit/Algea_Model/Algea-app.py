import streamlit as st  # Web app framework
import pandas as pd     # Data handling
import pickle           # For loading our saved "brain" (model)

# --- 1. PAGE SETUP ---
# Configure the page title and center the layout
st.set_page_config(page_title="Algae Bloom Predictor", layout="centered")

# --- 2. LOAD THE SAVED MODEL ---
# We use st.cache_resource so the model stays in memory for speed
@st.cache_resource
def load_algae_model():
    # 'rb' stands for Read Binary (standard for pickle files)
    with open('Algea_Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Try to load the model; show an error if the file is missing
try:
    model = load_algae_model()
except FileNotFoundError:
    st.error("Error: 'algae_model.pkl' not found. Please save your model first!")

# --- 3. USER INTERFACE (White Background Style) ---
st.title("🌿 Algae Bloom Prediction System")
st.write("Adjust the environmental factors below to analyze the risk of a bloom.")

# Organize inputs into two columns for a professional look
col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Factors")
    light = st.slider("Light Intensity (lux)", 0.0, 500.0, 150.0)
    temp = st.slider("Temperature (°C)", 0.0, 45.0, 22.0)
    ph = st.slider("pH Level", 0.0, 14.0, 7.0)

with col2:
    st.subheader("Chemical Factors")
    nitrate = st.number_input("Nitrate (mg/L)", value=5.0)
    iron = st.number_input("Iron (mg/L)", value=0.1)
    phosphate = st.number_input("Phosphate (mg/L)", value=1.0)
    co2 = st.number_input("CO2 (mg/L)", value=10.0)

# --- 4. PREDICTION LOGIC ---
if st.button("Run Environmental Analysis"):
    # Create a DataFrame with the exact column names the model learned
    input_data = pd.DataFrame([[light, nitrate, iron, phosphate, temp, ph, co2]], 
                               columns=['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2'])
    
    # Get the 0/1 prediction and the probability
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # --- 5. RESULTS DISPLAY ---
    st.divider()
    
    if prediction == 1:
        # Show a red warning box
        st.error(f"### ⚠️ RESULT: ALGAE BLOOM DETECTED")
        st.write(f"Confidence Level: **{probabilities[1]*100:.2f}%**")
        st.warning("Action Required: High levels of nutrients detected. Consider reducing runoff.")
    else:
        # Show a green success box
        st.success(f"### ✅ RESULT: NO BLOOM PREDICTED")
        st.write(f"Confidence Level: **{probabilities[0]*100:.2f}%**")
        st.info("The current chemical balance appears stable for the ecosystem.")