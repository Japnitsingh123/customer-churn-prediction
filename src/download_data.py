import urllib.request
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "telco_customer_churn.csv")

URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
]

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 10000:
        print(f"Dataset already exists at {DATA_FILE}")
        return DATA_FILE

    for url in URLS:
        try:
            print(f"Downloading dataset from: {url}")
            urllib.request.urlretrieve(url, DATA_FILE)
            df = pd.read_csv(DATA_FILE)
            if len(df) > 1000 and "Churn" in df.columns:
                print(f"Successfully downloaded dataset with {len(df)} rows and {len(df.columns)} columns.")
                return DATA_FILE
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    raise RuntimeError("Failed to download dataset from all sources.")

if __name__ == "__main__":
    download_data()
