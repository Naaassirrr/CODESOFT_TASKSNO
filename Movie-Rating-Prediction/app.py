import streamlit as st
import joblib
import pandas as pd

model = joblib.load("movie_rating_model.pkl")
preprocessor = joblib.load("movie_rating_preprocessor.pkl")

st.title("🎬 Movie Rating Prediction")

st.write("Enter movie details to predict its rating.")

year = st.number_input("Release Year", min_value=1900, max_value=2026, value=2020)

duration = st.number_input("Duration (minutes)", min_value=1, max_value=500, value=120)

genre = st.text_input("Genre", "Drama")

votes = st.number_input("Number of Votes", min_value=0, value=1000)

director = st.text_input("Director", "Unknown")

actor1 = st.text_input("Actor 1", "Unknown")

actor2 = st.text_input("Actor 2", "Unknown")

actor3 = st.text_input("Actor 3", "Unknown")

if st.button("Predict Rating"):

    movie_data = pd.DataFrame({
        "Year": [year],
        "Duration": [duration],
        "Genre": [genre],
        "Votes": [votes],
        "Director": [director],
        "Actor 1": [actor1],
        "Actor 2": [actor2],
        "Actor 3": [actor3]
    })

    movie_data_transformed = preprocessor.transform(movie_data)

    prediction = model.predict(movie_data_transformed)

    st.success(f"Predicted Movie Rating: {prediction[0]:.1f} ⭐")