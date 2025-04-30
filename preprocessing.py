import pandas as pd

def clean_data(path='Churn_data.csv'):
    """
    Loads and cleans the Telco Customer Churn dataset.

    Steps:
    - Reads the raw CSV file
    - Converts 'TotalCharges' to numeric
    - Drops rows with missing 'TotalCharges'
    - Converts 'Churn' to binary (Yes -> 1, No -> 0)
    - Drops 'customerID' column

    Returns:
    --------
    pd.DataFrame: Cleaned DataFrame ready for modeling
    """
    # Load raw data
    df = pd.read_csv("Churn_data.csv")

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Normalize and map Churn to binary
    df['Churn'] = df['Churn'].str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Drop customerID column
    df = df.drop('customerID', axis=1)

    return df
