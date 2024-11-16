import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Predict using saved model
def predict(new_data):
    model = joblib.load("model.pkl")
    predictions = model.predict(new_data)
    return predictions

# Main execution
if __name__ == "__main__":
    # Example workflow
    data_file = "TestReviews.csv"  

    
