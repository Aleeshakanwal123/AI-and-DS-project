import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st

# Load the dataset
dataset_path = 'TestReviews.csv'

    # Display dataset preview in Streamlit
    st.write("Dataset preview:")
    st.write(data.head())

    # Preprocessing
    X = data['text']  # assuming the review text is in a column named 'text'
    y = data['label']  # assuming the sentiment labels are in a column named 'label'

    # Split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_vect, y_train)

    # Save model and vectorizer for future use
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # Display success message
    st.success("Model trained and saved successfully!")
