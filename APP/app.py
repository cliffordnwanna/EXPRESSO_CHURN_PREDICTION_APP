import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gdown
import os

# Function to download the model from Google Drive
def download_model_from_gdrive(gdrive_url, output_path):
    if not os.path.exists(output_path):
        st.info("Downloading model from Google Drive...")
        gdown.download(gdrive_url, output_path, quiet=False)
        st.success("Model downloaded successfully!")
    else:
        st.info("Model already exists locally. Skipping download.")

# Google Drive URL (replace with your actual file ID)
gdrive_url = "https://drive.google.com/uc?id=FILE_ID"
model_path = "models/main_trained_model_1.sav"

# Download model
download_model_from_gdrive(gdrive_url, model_path)

# Load your trained model
loaded_model = pickle.load(open(model_path, 'rb'))

# Prediction function
def make_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'NO CHURN'
    else:
        return 'CHURN'

# Streamlit main function
def main():
    st.title('EXPRESSO CHURN PREDICTION')
    
    # Input fields for features
    user_id = st.number_input('Enter user_id:')
    REGION = st.number_input('Enter REGION:')
    TENURE = st.number_input('Enter TENURE:')
    # Add all other fields similarly
    
    if st.button('Make Prediction'):
        input_data = [user_id, REGION, TENURE]  # Add all other input fields here
        result = make_prediction(input_data)
        st.success(f'The predicted result is: {result}')

if __name__ == "__main__":
    main()
