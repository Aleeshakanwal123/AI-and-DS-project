import pandas as pd

# Load the CSV file
def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        print("CSV Data Loaded Successfully:")
        print(data.head())
    except Exception as e:
        print(f"Error loading CSV file: {e}")

if __name__ == "__main__":
    # Specify your file path
    file_path = "TestReviews.csv"
    load_csv(file_path)
