import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species using Machine Learning."
)

st.divider()

# Input section
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

st.divider()

# Prediction
if st.button("🔍 Predict Flower Species", use_container_width=True):

    input_data = pd.DataFrame([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]], columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"🌸 Predicted Species: **{prediction}**")

st.divider()

st.caption("Built using Python, Scikit-learn, Random Forest and Streamlit")