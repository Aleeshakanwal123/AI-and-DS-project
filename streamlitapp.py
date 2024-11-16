import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

# Add a title and description
st.title("Sentiment Analysis of Product Reviews")
st.markdown("Enter a product review, and the model will predict whether the sentiment is positive, negative, or neutral.")

# Create a text input box for the user to enter a review
review_text = st.text_area("Enter Review Text", height=150)

# Define the action when the button is pressed
if st.button("Predict Sentiment"):
    if review_text:
        # Vectorize the input review text
        vect = vectorizer.transform([review_text])

        # Predict the sentiment using the model
        prediction = model.predict(vect)

        # Show the predicted sentiment
        if prediction[0] == 1:
            sentiment = "Positive"
        elif prediction[0] == 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.warning("Please enter some text to analyze.")
