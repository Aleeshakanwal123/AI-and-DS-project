import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y

# Train model (for demonstration purposes)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    joblib.dump(model, "model.pkl")

# Predict using saved model
def predict(new_data):
    model = joblib.load("model.pkl")
    predictions = model.predict(new_data)
    return predictions

# Main execution
if __name__ == "__main__":
    # Example workflow
    data_file = "your_dataset.csv"  # Replace with your file
    X, y = load_data(data_file)

    # Train and save the model
    train_model(X, y)

    
