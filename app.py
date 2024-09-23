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
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Set the title of the app
st.title("House Price Prediction")

# Create input fields for user data
st.header("Input Features")
feature_1 = st.number_input("crim")
feature_2 = st.number_input("zn")
feature_3 = st.number_input("indus")
feature_4 = st.number_input("chas")
feature_5= st.number_input("nox")
feature_6 = st.number_input("rm")
feature_7 = st.number_input("age")
feature_8 = st.number_input("dis")
feature_9 = st.number_input("rad")
feature_10 = st.number_input("tax")
feature_11 = st.number_input("ptratio")
feature_12 = st.number_input("b")
feature_13 = st.number_input("lstat")
# Add more input fields as needed...

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Gather input data
    data = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13]
    # Convert to a NumPy array and scale
    new_data = scaler.transform(np.array(data).reshape(1, -1))
    
    # Make the prediction
    output = regmodel.predict(new_data)[0]
    
    # Display the result
    st.success(f"The Linear regression House Price Prediction is: {output}")
