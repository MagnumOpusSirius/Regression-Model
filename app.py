import streamlit as st
import pickle
import pandas as pd

# Load model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Salary Prediction App")
# Input fields
city = st.text_input("City")
position = st.text_input("Position")
gp = st.number_input("GP", min_value=0, step=1)

# Convert categorical inputs to numerical
input_data = pd.DataFrame([[city, position, gp]], columns=['City', 'Position', 'GP'])
input_data['City'] = input_data['City'].astype('category').cat.codes
input_data['Position'] = input_data['Position'].astype('category').cat.codes

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Salary: {prediction}")
