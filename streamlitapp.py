import streamlit as st
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib  # To save/load models

# Streamlit app interface
st.title("Text Prediction App")

# File upload to upload a dataset for training
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())
    
# Prediction section
st.subheader("Make Predictions")
model = load_model()  # Load the model (or None if not trained)

if model is not None:
    # Input text for prediction
    user_input = st.text_area("Enter text for prediction:")
    
    if st.button("Predict"):
        if user_input:
            prediction = model.predict([user_input])
            st.write(f"Prediction: {prediction[0]}")
        else:
            st.write("Please enter some text to predict.")
