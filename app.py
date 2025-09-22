import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

## Loading the trained model
model = tf.keras.models.load_model('model.h5', compile=False)


## load all the pickle files
with open('label_encoder_gender.pk1','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pk1', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scalar.pk1', 'rb') as file:
    scalar = pickle.load(file)

    ## streamlit app
    st.title("Customer Churn Predection")

    # User Inputs
CreditScore = st.number_input("Credit Score", min_value=0, max_value=1500, value=600)
Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100, value=40)
Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
Balance = st.number_input("Balance", min_value=0.0, value=60000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
HasCrCard = st.selectbox("Has Credit Card?", [1, 0])
IsActiveMember = st.selectbox("Is Active Member?", [1, 0])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)


# Create input dictionary with correct variable names
input_data = {
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Gender': Gender,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}

# Preprocessing
input_df = pd.DataFrame([input_data])

# Encode Gender
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# One-hot encode Geography
geography_encoded = onehot_encoder_geography.transform(input_df[['Geography']]).toarray()
geography_df = pd.DataFrame(
    geography_encoded,
    columns=onehot_encoder_geography.get_feature_names_out(['Geography'])
)

# Drop original Geography and concatenate
input_df = pd.concat([input_df.drop('Geography', axis=1), geography_df], axis=1)

## Reorder columns to match training data
# Ensure columns match scaler input
expected_columns = scalar.feature_names_in_
input_df = input_df[expected_columns]

## Scale the input
input_scaled = scalar.transform(input_df)
input_scaled


prediction= model.predict(input_scaled)
prediction

# Predict when button is clicked
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    # Get prediction probability
    prediction_prob = model.predict(input_scaled)[0][0]

    # Display result
    if prediction_prob > 0.5:
        st.write("Customer is likely to churn!")
    else:
        st.write("Customer is not likely to churn.")
