import pandas as pd
from sklearn.model_selection import train_test_split

# Simulated data for demonstration
def load_data():
    # Loading sample data
    data = pd.read_csv('customer_data.csv')
    return data

def preprocess_data(data):
    # Cleaning and preprocessing steps
    data.dropna(inplace=True)
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    return data

# Load and preprocess
raw_data = load_data()
preprocessed_data = preprocess_data(raw_data)
