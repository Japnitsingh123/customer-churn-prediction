import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "telco_customer_churn.csv")

DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
]

def ensure_dataset(data_path: str = DATA_FILE) -> str:
    """Ensures the Telco customer churn dataset exists locally; downloads if missing."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if os.path.exists(data_path) and os.path.getsize(data_path) > 10000:
        return data_path

    for url in DATA_URLS:
        try:
            print(f"Downloading Telco Churn dataset from {url}...")
            urllib.request.urlretrieve(url, data_path)
            if os.path.exists(data_path) and os.path.getsize(data_path) > 10000:
                print(f"Saved dataset to {data_path}")
                return data_path
        except Exception as err:
            print(f"Warning: could not retrieve from {url}: {err}")

    raise FileNotFoundError("Could not download IBM Telco Churn dataset from remote sources.")


def load_and_clean_data(data_path: str = DATA_FILE) -> pd.DataFrame:
    """
    Loads raw CSV, performs essential cleaning:
    - Strips whitespace
    - Coerces TotalCharges to float (imputes tenure==0 cases with 0.0)
    - Maps Churn to numeric 1/0
    - Converts SeniorCitizen to object/category if desired
    """
    path = ensure_dataset(data_path)
    df = pd.read_csv(path)

    # 1. TotalCharges blank string cleaning
    # Telco dataset has 11 records with empty spaces " " where tenure is 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # 2. Target mapping: Yes -> 1, No -> 0
    if 'Churn' in df.columns:
        if set(df['Churn'].unique()).intersection({'Yes', 'No'}):
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # 3. Clean string columns whitespace
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    print("Dataset shape:", df.shape)
    print("Missing TotalCharges remaining:", df['TotalCharges'].isna().sum())
    print("Churn distribution:\n", df['Churn'].value_counts(normalize=True))
