import streamlit as st
import pickle
import numpy as np
import pandas as pd # for data frame
import matplotlib.pyplot as plt # for plots and graphs
import seaborn as sns # for plots
import sklearn.datasets # sklearn is machine learning libraby, we will import the BOSTON dataset from sklearn.dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics # for evaluating our model

# Load the model and scaler
regmodel =  pickle.load(open('regmodel.pkl', 'rb'))
# Load the scaler dictionary
@st.cache_data
def load_scalers():
    scaler = pickle.load(open('scaling.pkl', 'rb'))
    return scaler['scaler_X'], scaler['scaler_Y']

# Load scalers correctly
scaler_X, scaler_Y = load_scalers()

# Set the title of the app
st.title("House Price Prediction")
# Project description
st.markdown("### 🏡 Data Science Project by **Aritra Mandal**")
st.write("📊 Empowering buyers and sellers with data-driven insights for smarter real estate decisions.")
st.write("🏠 Enter the area-specific details below, and the model will estimate the property value for that location.")


# Create input fields for user data
st.header("Input Features")
feature_1 = st.number_input("crime: value between 0 and 100")
feature_2 = st.number_input("airport: value 0 or 1")
feature_3= st.number_input("air_pollution: value between 0.4 and 1")
feature_4 = st.number_input("avg_rooms: value between 0 and 100")
feature_5 = st.number_input("highway_access: value between 1 and 25")
feature_6 = st.number_input("tax: value between 150 and 1000")
feature_7 = st.number_input("education: value between 15 and 50")
feature_8 = st.number_input("lstat: value between 1.5 and 50")
# Add more input fields as needed...

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Gather input data
    data = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
    # Convert to a NumPy array and scale
    new_data = scaler_X.transform(np.array(data).reshape(1, -1))
    # Add intercept (1) at the beginning
    new_data = np.insert(new_data, 0, 1, axis=1)
    
    # Make the prediction
    output = regmodel.predict(new_data)[0]

    # Step 1: Reverse Natural Log (exp to undo log)
    output_shifted = np.exp(output)  # This reverses np.log()

    # Step 2: Subtract 2 (Undo Shift)
    output_standard = output_shifted - 2

    output_final = scaler_Y.inverse_transform(np.array(output_standard).reshape(-1, 1))[0][0]

    
    # Display the result
    st.success(f"🏡 Estimated House Price: **{output_final:,.2f} Lakh rupees**")

