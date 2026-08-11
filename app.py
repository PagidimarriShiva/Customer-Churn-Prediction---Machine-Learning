import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information to predict the likelihood of customer churn."
)

st.divider()

# Customer information
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

with col3:
    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    has_cr_card = st.selectbox(
        "Has Credit Card?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_active_member = st.selectbox(
        "Is Active Member?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

st.divider()

# Prediction button
if st.button("🔍 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    st.progress(float(probability))

    if probability >= 0.70:
        st.warning("High churn risk")

    elif probability >= 0.40:
        st.info("Medium churn risk")

    else:
        st.success("Low churn risk")