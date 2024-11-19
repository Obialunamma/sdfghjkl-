import streamlit as st
import numpy as np
import pandas as pd
import joblib

# load trained model and encode
model = joblib.load("model/model.pkl")
encoders = joblib.load("model/encoders.pkl")

#title
st.title("Customer Churn Prediction")


# image display 
st.image("https://gorilladesk.com/wp-content/uploads/2020/02/customer-churn.png", use_container_width=True)


# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.radio("Are you a Senior Citizen?", [0, 1])
partner = st.radio("Do you have a partner?", ["Yes", "No"])
dependents = st.radio("Do you have dependents?", ["Yes", "No"])

# Numeric Inputs
tenure = st.slider("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = float(st.text_input("Total Charges (enter as a number)", "200.0"))



# More Categorical Inputs
phone_service = st.radio("Do you have phone service?", ["Yes", "No"])
multiple_lines = st.radio("Do you have multiple lines?", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Type of Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.radio("Do you have online security?", ["Yes", "No", "No internet service"])
online_backup = st.radio("Do you have online backup?", ["Yes", "No", "No internet service"])
device_protection = st.radio("Do you have device protection?", ["Yes", "No", "No internet service"])
tech_support = st.radio("Do you have tech support?", ["Yes", "No", "No internet service"])
streaming_tv = st.radio("Do you stream TV?", ["Yes", "No", "No internet service"])
streaming_movies = st.radio("Do you stream movies?", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.radio("Do you have paperless billing?", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Preview of input features
st.write("Your selected inputs:")
st.write({
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure": tenure,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
})


# Prepare the input data for prediction
input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Transform categorical features using the same encoders used during training
encoded_features = []
for col, value in input_data.items():
    if col in encoders:  # If the column was encoded during training
        encoded_value = encoders[col].transform([value])[0]
        encoded_features.append(encoded_value)
    else:  # For numerical columns, append the value directly
        encoded_features.append(value)

# Convert to NumPy array for prediction
input_features = np.array(encoded_features).reshape(1, -1)

# Prediction
prediction = model.predict(input_features)

# Show prediction result
if prediction[0] == 0:
    st.write("The model predicts: **Customer will not churn**")
else:
    st.write("The model predicts: **Customer will churn**")
