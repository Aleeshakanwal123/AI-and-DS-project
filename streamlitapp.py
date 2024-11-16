import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('TestReviews.csv')
vectorizer = joblib.load('TestReviews.csv')

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

# Add a title and description to the app
st.title("Sentiment Analysis on Reviews")
st.markdown("Enter a review text, and the model will predict its sentiment (positive or negative).")

# Create a text input box for the user to enter a review
review_text = st.text_area("Enter Review Text", height=150)

# Define the action when the button is pressed
if st.button("Predict Sentiment"):
    if review_text:
        # Vectorize the input review text
        vect = vectorizer.transform([review_text])

        # Predict the sentiment using the model
        prediction = model.predict(vect)

        # Display the prediction result
        if prediction[0] == 1:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        # Show the predicted sentiment
        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.warning("Please enter some text to analyze.")
