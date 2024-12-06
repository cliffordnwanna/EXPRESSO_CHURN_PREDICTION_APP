import streamlit as st
import pandas as pd
import pickle
import requests
import io

# Function to download and load models from URLs
def download_file(url):
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

# URLs where the files are hosted
MODEL_URL = "https://drive.google.com/file/d/1Tm3_Hccyj7QwNU2S938MiG97FgR9KtCm"
FEATURES_URL = "https://drive.google.com/file/d/1GpUMFk6_z9S5GIMepaHGyF9SwPEjyg9V"
ENCODERS_URL = "https://drive.google.com/file/d/1kT2AgFSyS9-9EiQPUT2VadWH_oEOzGY7"

# Download and load files from URLs
clf = download_file(MODEL_URL)
selected_features = download_file(FEATURES_URL)
label_encoders = download_file(ENCODERS_URL)

# App UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn:")

# Generate input fields dynamically based on selected features
input_data = {}
for feature in selected_features:
    if feature in label_encoders:  # Categorical columns
        category_options = label_encoders[feature].classes_
        input_data[feature] = st.selectbox(f"{feature}:", options=category_options)
    else:  # Numeric columns
        input_data[feature] = st.text_input(f"{feature}:")

# Predict
if st.button("Predict"):
    try:
        # Handle categorical data encoding
        for feature in selected_features:
            if feature in label_encoders:
                input_data[feature] = label_encoders[feature].transform([input_data[feature]])[0]
            else:
                # Ensure numeric fields are converted correctly
                input_data[feature] = float(input_data[feature]) if input_data[feature].replace('.', '', 1).isdigit() else 0

        input_df = pd.DataFrame([input_data])
        prediction = clf.predict(input_df)
        result = "Churn" if prediction[0] == 1 else "No Churn"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
