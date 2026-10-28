# 📉 Telco Customer Churn Prediction (End-to-End ML & Streamlit Cloud)

An end-to-end Machine Learning web application to predict customer churn in subscription-based telecommunication services, compare classification models, explain top churn drivers, and deliver personalized retention strategies.

Built with **Python**, **Scikit-Learn**, **XGBoost**, and **Streamlit**.

---

## 🌟 Key Features

1. **Exploratory Data Analysis (EDA) Dashboard**:
   - Executive KPIs: Total Customers, Churn Rate (~26%), Revenue at Risk, Average Monthly Bill.
   - Interactive visualizations: Churn by Contract Type, Internet Service (Fiber vs. DSL), Customer Tenure density, and Payment Methods.
2. **Multi-Model Benchmark & Comparison**:
   - Side-by-side evaluation of:
     - **Logistic Regression** (Interpretable baseline, class-weighted)
     - **Random Forest Classifier** (Non-linear ensemble)
     - **XGBoost Classifier** (Gradient-boosted decision trees)
   - Evaluated across **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** (~0.81 – 0.835).
   - Interactive Confusion Matrices and ROC curves.
3. **Live Single-Customer Churn Predictor**:
   - Input customer demographics, contract status, charges, and subscribed services.
   - Real-time churn probability gauge & Risk categorization (`Low`, `Medium`, `High Risk`).
   - Automated, rule-based **Personalized Retention Playbook** (e.g. 1-year contract discount, complimentary tech support).
4. **Feature Importance & Business Playbook**:
   - Ranked predictive weights for top churn drivers (month-to-month contracts, fiber optic pricing, electronic check billing, tenure).
   - 4 executive retention strategies with estimated business ROI.
5. **Batch CSV Scoring & Export**:
   - Upload bulk customer CSV files.
   - Run batch inference to generate predicted probabilities and risk tiers.
   - 1-click download of the enriched CSV with retention recommendations.

---

## 📁 Project Architecture

```
customer-churn-prediction/
│
├── .streamlit/
│   └── config.toml                  # Streamlit visual styling and theme
│
├── data/
│   └── telco_customer_churn.csv     # IBM Telco Customer Churn dataset
│
├── models/                          # Pre-trained models and cached metrics
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── preprocessor.pkl
│   ├── model_metrics.json
│   ├── roc_curves.json
│   └── feature_importance.json
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and cleaning
│   ├── feature_engineering.py       # Domain feature engineering & preprocessing
│   ├── train.py                     # ML training pipeline for 3 classifiers
│   └── evaluate.py                  # Classification metrics & ROC curves
│
├── app.py                           # 5-Tab Streamlit Web Application
├── requirements.txt                 # Dependencies for Streamlit Cloud
└── README.md                        # Documentation & deployment guide
```

---

## 🚀 Quickstart: Running Locally

### 1. Clone or Navigate to the Project
```bash
cd C:\Users\kaurk\.gemini\antigravity\scratch\customer-churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain Models
All models are pre-trained and ready to run immediately. If you want to retrain from scratch:
```bash
python -m src.train
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Cloud (Step-by-Step)

Deploying this app to [Streamlit Cloud](https://share.streamlit.io/) is 100% free and takes less than 2 minutes:

### Step 1: Initialize Git and Push to GitHub
1. Create a new public repository on [GitHub](https://github.com/new) (e.g., `telco-churn-prediction`).
2. Run the following commands in your project directory:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Telco Churn Prediction End-to-End"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/telco-churn-prediction.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app**.
3. Select:
   - **Repository**: `<YOUR_GITHUB_USERNAME>/telco-churn-prediction`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

Streamlit Cloud will read `requirements.txt`, install dependencies, load the bundled dataset and pre-trained models, and launch your live web app with a shareable public URL!

---

## 📊 Model Evaluation Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **78.60%** | **56.79%** | **73.02%** | **63.89%** | **0.8345** |
| **Logistic Regression** | 76.13% | 52.75% | 76.19% | 62.34% | 0.8235 |
| **XGBoost** | 78.19% | 56.58% | 68.25% | 61.87% | 0.8140 |

> **Business Context**: High recall (~73–76%) is prioritized because the business cost of a missed churned customer (false negative) significantly exceeds the minor outreach cost of a false positive retention promo.
