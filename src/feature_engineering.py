import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICES_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

NUMERIC_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'num_services', 'avg_monthly_charges', 'charge_difference'
]

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'tenure_cohort', 'is_alone', 'has_internet'
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates informative domain-specific features for customer churn analysis.
    Works for both training datasets and single inference payloads.
    """
    data = df.copy()

    # 1. Total service count
    num_services = pd.Series(0, index=data.index)
    for col in SERVICES_COLS:
        if col in data.columns:
            num_services += (data[col].astype(str) == 'Yes').astype(int)
    data['num_services'] = num_services

    # 2. Tenure cohort / bucket
    if 'tenure' in data.columns:
        bins = [-1, 12, 24, 48, 60, 100]
        labels = ['0-12m', '13-24m', '25-48m', '49-60m', '>60m']
        data['tenure_cohort'] = pd.cut(data['tenure'], bins=bins, labels=labels).astype(str)
    else:
        data['tenure_cohort'] = '0-12m'

    # 3. Solitary customer indicator
    partner_no = (data['Partner'].astype(str) == 'No') if 'Partner' in data.columns else False
    dep_no = (data['Dependents'].astype(str) == 'No') if 'Dependents' in data.columns else False
    data['is_alone'] = (partner_no & dep_no).astype(str)

    # 4. Internet presence indicator
    if 'InternetService' in data.columns:
        data['has_internet'] = (data['InternetService'].astype(str) != 'No').astype(str)
    else:
        data['has_internet'] = 'No'

    # 5. Average monthly charge and difference
    if 'TotalCharges' in data.columns and 'tenure' in data.columns and 'MonthlyCharges' in data.columns:
        data['avg_monthly_charges'] = data['TotalCharges'] / (data['tenure'] + 1)
        data['charge_difference'] = data['MonthlyCharges'] - data['avg_monthly_charges']
    else:
        data['avg_monthly_charges'] = data.get('MonthlyCharges', 0.0)
        data['charge_difference'] = 0.0

    # Ensure SeniorCitizen is treated as categorical string
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales numeric features
    and one-hot encodes categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor
