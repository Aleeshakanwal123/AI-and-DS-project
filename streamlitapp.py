import streamlit as st
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib  # To save/load models

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to train a text classification model
def train_model(df):
    # Extract features (text) and labels (target)
    X = df['text']
    y = df['label']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline: vectorizer + Naive Bayes model
    model = make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB())
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Show accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Save the trained model for later use
    joblib.dump(model, 'model.pkl')
    st.write("Model saved successfully.")

    return model

# Function to load a pre-trained model
def load_model():
    try:
        model = joblib.load('model.pkl')
        st.write("Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.write("No pre-trained model found. Please train a model first.")
        return None

# Streamlit app interface
st.title("Text Prediction App")

# File upload to upload a dataset for training
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())
    
    # Check for required columns
    if 'text' in df.columns and 'label' in df.columns:
        # Train model on uploaded data
        model = train_model(df)
        
    else:
        st.error("Dataset must contain 'text' and 'label' columns.")

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
