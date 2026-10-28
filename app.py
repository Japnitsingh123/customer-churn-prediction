import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Local imports
from src.data_loader import load_and_clean_data
from src.feature_engineering import add_engineered_features
from src.train import train_and_evaluate_models

# Directory constants
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "telco_customer_churn.csv")

# Streamlit Page Config
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cleaned_data():
    """Loads and caches cleaned IBM Telco dataset."""
    return load_and_clean_data(DATA_FILE)


def ensure_models_trained():
    """Checks if model artifacts exist; if not, triggers training."""
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    preproc_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if not (os.path.exists(best_model_path) and os.path.exists(preproc_path) and os.path.exists(metrics_path)):
        with st.spinner("Initializing models and calculating metrics... This happens only once."):
            train_and_evaluate_models()


def load_artifacts():
    """Loads preprocessor, metrics, and models."""
    ensure_models_trained()
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))

    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "r") as f:
        metrics_meta = json.load(f)

    with open(os.path.join(MODELS_DIR, "roc_curves.json"), "r") as f:
        roc_curves = json.load(f)

    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "r") as f:
        feature_importances = json.load(f)

    available_models = {}
    for name in metrics_meta['metrics'].keys():
        slug = name.lower().replace(" ", "_")
        path = os.path.join(MODELS_DIR, f"{slug}.pkl")
        if os.path.exists(path):
            available_models[name] = joblib.load(path)

    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    return preprocessor, metrics_meta, roc_curves, feature_importances, available_models, best_model


# Load data and models
try:
    df_raw = get_cleaned_data()
    preprocessor, metrics_meta, roc_curves, feat_importances, available_models, best_model = load_artifacts()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("https://raw.githubusercontent.com/streamlit/brand/main/logo/streamlit-logo-primary-colormark-darktext.png", width=180)
st.sidebar.markdown("### 📊 Customer Churn Intelligence")
st.sidebar.markdown("Predict and mitigate customer churn using Machine Learning.")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📈 Overview & EDA",
        "⚖️ Model Benchmark & Comparison",
        "🎯 Real-Time Churn Predictor",
        "💡 Feature Importance & Strategy",
        "📁 Batch CSV Prediction"
    ]
)

st.sidebar.divider()
selected_model_name = st.sidebar.selectbox(
    "Active Prediction Engine",
    list(available_models.keys()),
    index=list(available_models.keys()).index(metrics_meta.get('best_model', list(available_models.keys())[0]))
)
active_model = available_models[selected_model_name]

if st.sidebar.button("🔄 Retrain All Models"):
    with st.spinner("Retraining Logistic Regression, Random Forest, and XGBoost..."):
        train_and_evaluate_models()
        st.cache_data.clear()
        st.success("Models retrained successfully! Refreshing...")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: IBM Telco Customer Churn (7,043 records)")
st.sidebar.caption("Deployed with Streamlit Cloud ready configuration.")


# ==========================================
# TAB 1: OVERVIEW & EDA
# ==========================================
if nav_choice == "📈 Overview & EDA":
    st.markdown('<div class="main-header">📈 Executive Overview & Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding churn distribution, contract dynamics, and customer longevity drivers.</div>', unsafe_allow_html=True)

    # Top KPI Metrics
    total_customers = len(df_raw)
    churn_count = (df_raw['Churn'] == 1).sum()
    retained_count = total_customers - churn_count
    churn_rate = (churn_count / total_customers) * 100
    avg_monthly = df_raw['MonthlyCharges'].mean()
    total_monthly_risk = df_raw[df_raw['Churn'] == 1]['MonthlyCharges'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    with col3:
        st.metric("Active / Retained", f"{retained_count:,}")
    with col4:
        st.metric("Avg Monthly Bill", f"${avg_monthly:.2f}")
    with col5:
        st.metric("Monthly Revenue at Risk", f"${total_monthly_risk:,.0f}")

    st.markdown("---")

    # EDA Visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Churn Rate by Contract Type")
        contract_churn = df_raw.groupby('Contract')['Churn'].mean().reset_index()
        contract_churn['Churn_Rate'] = contract_churn['Churn'] * 100

        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(contract_churn['Contract'], contract_churn['Churn_Rate'], color=['#EF4444', '#3B82F6', '#10B981'], width=0.55)
        ax.set_ylabel("Churn Rate (%)", fontsize=11)
        ax.set_ylim(0, 55)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.caption("💡 Month-to-month contracts have an alarming ~42.7% churn rate compared to only ~2.8% for 2-year contracts.")

    with col_right:
        st.subheader("2. Churn Rate by Internet Service")
        net_churn = df_raw.groupby('InternetService')['Churn'].mean().reset_index()
        net_churn['Churn_Rate'] = net_churn['Churn'] * 100

        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(net_churn['InternetService'], net_churn['Churn_Rate'], color=['#3B82F6', '#EF4444', '#10B981'], width=0.55)
        ax.set_ylabel("Churn Rate (%)", fontsize=11)
        ax.set_ylim(0, 50)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.caption("💡 Fiber optic customers experience significantly higher churn (~41.9%), driven by high monthly fees and lack of technical support.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. Customer Tenure vs Churn")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.histplot(data=df_raw, x='tenure', hue='Churn', kde=True, bins=30,
                     palette={0: '#10B981', 1: '#EF4444'}, ax=ax, alpha=0.4)
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Number of Customers")
        ax.legend(labels=['Churned (1)', 'Retained (0)'])
        st.pyplot(fig)
        st.caption("💡 Churn heavily clusters in the first 0-12 months. Early customer onboarding is the highest-leverage retention window.")

    with col4:
        st.subheader("4. Payment Method vs Churn")
        pay_churn = df_raw.groupby('PaymentMethod')['Churn'].mean().reset_index()
        pay_churn['Churn_Rate'] = pay_churn['Churn'] * 100
        pay_churn = pay_churn.sort_values(by='Churn_Rate', ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.barh(pay_churn['PaymentMethod'], pay_churn['Churn_Rate'], color='#6366F1', height=0.55)
        ax.set_xlabel("Churn Rate (%)")
        ax.set_xlim(0, 55)
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.1f}%",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha='left', va='center', fontweight='bold')
        st.pyplot(fig)
        st.caption("💡 Electronic check users churn at ~45.3%, whereas automatic bank/credit card deduction users churn at ~15-18%.")

    # Dataset Explorer
    with st.expander("🔍 Explore Raw Dataset Samples"):
        st.dataframe(df_raw.head(100), use_container_width=True)


# ==========================================
# TAB 2: MODEL BENCHMARK & COMPARISON
# ==========================================
elif nav_choice == "⚖️ Model Benchmark & Comparison":
    st.markdown('<div class="main-header">⚖️ Model Performance & Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparing Logistic Regression, Random Forest, and XGBoost on test holdout (20%).</div>', unsafe_allow_html=True)

    metrics_dict = metrics_meta['metrics']
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1_score'] * 100:.2f}%",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "Status": "⭐ Top Benchmark" if model_name == metrics_meta.get('best_model') else "Competitive"
        })
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_chart, col_cm = st.columns([1.1, 0.9])

    with col_chart:
        st.subheader("Metric Comparison Across Models")
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        models_list = list(metrics_dict.keys())

        x = np.arange(len(metric_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 4.8))
        colors = ['#3B82F6', '#10B981', '#8B5CF6']
        for i, m_name in enumerate(models_list):
            vals = [
                metrics_dict[m_name]['accuracy'],
                metrics_dict[m_name]['precision'],
                metrics_dict[m_name]['recall'],
                metrics_dict[m_name]['f1_score']
            ]
            ax.bar(x + (i - len(models_list)/2 + 0.5) * width, vals, width, label=m_name, color=colors[i % len(colors)])

        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with col_cm:
        st.subheader(f"Confusion Matrix: {selected_model_name}")
        cm_data = metrics_dict[selected_model_name]['confusion_matrix']
        cm_matrix = np.array([
            [cm_data['tn'], cm_data['fp']],
            [cm_data['fn'], cm_data['tp']]
        ])

        fig, ax = plt.subplots(figsize=(6, 4.8))
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Retain', 'Predicted Churn'],
                    yticklabels=['Actual Retain', 'Actual Churn'],
                    annot_kws={"size": 13, "weight": "bold"}, ax=ax)
        st.pyplot(fig)
        st.caption(f"Recall on Churned Customers: **{metrics_dict[selected_model_name]['recall'] * 100:.1f}%** | Precision: **{metrics_dict[selected_model_name]['precision'] * 100:.1f}%**")

    # ROC Curves
    st.subheader("Receiver Operating Characteristic (ROC) Curves")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot([0, 1], [0, 1], linestyle='--', color='#94A3B8', label='Random Guess (AUC = 0.50)')

    for m_name in models_list:
        if m_name in roc_curves:
            fpr = roc_curves[m_name]['fpr']
            tpr = roc_curves[m_name]['tpr']
            auc_val = metrics_dict[m_name]['roc_auc']
            ax.plot(fpr, tpr, lw=2, label=f"{m_name} (AUC = {auc_val:.3f})")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)


# ==========================================
# TAB 3: REAL-TIME CHURN PREDICTOR
# ==========================================
elif nav_choice == "🎯 Real-Time Churn Predictor":
    st.markdown('<div class="main-header">🎯 Real-Time Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter customer profile details below to get instant churn probability and retention interventions.</div>', unsafe_allow_html=True)

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen?", ["0 (No)", "1 (Yes)"])
            senior_val = "1" if senior_citizen.startswith("1") else "0"
            partner = st.selectbox("Has Partner?", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

        with col2:
            st.markdown("#### 💳 Account Details")
            tenure = st.slider("Tenure with Telco (Months)", min_value=0, max_value=72, value=6)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col3:
            st.markdown("#### 💰 Charges")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=75.0, step=1.0)
            auto_calc_total = tenure * monthly_charges
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(auto_calc_total), step=10.0)

        st.markdown("#### 🌐 Telecom & Value-Added Services")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with s_col2:
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        with s_col3:
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with s_col4:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submit_btn = st.form_submit_button("⚡ Predict Customer Churn Risk", use_container_width=True)

    if submit_btn:
        # Construct input payload
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior_val,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        # Add engineered features
        engineered_input = add_engineered_features(input_data)
        processed_input = preprocessor.transform(engineered_input)

        # Predict probability
        prob = active_model.predict_proba(processed_input)[0, 1]
        churn_pct = prob * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        r_col1, r_col2 = st.columns([1, 1.5])
        with r_col1:
            st.metric("Predicted Churn Probability", f"{churn_pct:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))

            if churn_pct >= 60:
                st.markdown('<div class="risk-badge-high">🚨 HIGH RISK OF CHURN</div>', unsafe_allow_html=True)
            elif churn_pct >= 35:
                st.markdown('<div class="risk-badge-medium">⚠️ MEDIUM RISK OF CHURN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-badge-low">✅ LOW RISK / STABLE CUSTOMER</div>', unsafe_allow_html=True)

        with r_col2:
            st.markdown("#### 🎯 Prescribed Retention Strategies")
            actions = []
            if contract == "Month-to-month":
                actions.append("🏷️ **Incentivize 1-Year Contract**: Customer is on month-to-month contract. Propose a 15% discount for upgrading to an annual contract.")
            if internet_service == "Fiber optic" and tech_support != "Yes":
                actions.append("🛠️ **Bundle Free Tech Support**: Fiber optic subscribers without tech support exhibit 4x higher cancellation rates. Offer 3 months of complimentary support.")
            if tenure <= 12:
                actions.append("📞 **Proactive Onboarding Check-in**: Customer is in high-risk early tenure (<= 12 months). Assign a dedicated customer success outreach.")
            if payment == "Electronic check":
                actions.append("💳 **Automate Payment Incentive**: Give a one-time $10 credit to switch from Electronic Check to Auto-Pay via Bank/Credit Card.")
            if monthly_charges > 85.0:
                actions.append("📦 **Plan Optimization**: High monthly charges ($>85). Offer plan bundle restructuring to reduce perceived cost while retaining core services.")

            if not actions:
                actions.append("🎉 **Customer in Loyalty Tier**: Profile indicates high satisfaction and strong retention signals. Keep engaged with periodic loyalty appreciation perks.")

            for act in actions:
                st.write(act)


# ==========================================
# TAB 4: FEATURE IMPORTANCE & STRATEGY
# ==========================================
elif nav_choice == "💡 Feature Importance & Strategy":
    st.markdown('<div class="main-header">💡 Feature Importance & Business Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-driven drivers of attrition and actionable strategies for executive decision-makers.</div>', unsafe_allow_html=True)

    if selected_model_name in feat_importances:
        top_feats = feat_importances[selected_model_name]
        feat_df = pd.DataFrame(top_feats).head(12)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(feat_df['feature'][::-1], feat_df['importance'][::-1], color='#3B82F6', height=0.6)
        ax.set_xlabel("Relative Predictive Weight", fontsize=11)
        ax.set_title(f"Top 12 Predictive Drivers ({selected_model_name})", fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📋 Executive Churn Mitigation Playbook")

    strat_col1, strat_col2 = st.columns(2)

    with strat_col1:
        st.markdown("""
        ### 1. Contract Migration Campaign
        - **Insight**: Month-to-month contracts account for **>85%** of churned users.
        - **Action**: Offer an attractive 10-15% discount or bundled streaming perk for migrating to a 12-month or 24-month contract.
        - **Expected Impact**: Reducing month-to-month contracts by 20% can decrease overall churn by ~7-8%.

        ### 2. High-Tech Support for Fiber Optic Lines
        - **Insight**: Fiber optic internet churn is disproportionately high due to installation friction and unresolved tech complaints.
        - **Action**: Bundle 6 months of VIP Tech Support with all Fiber optic installs. Proactively monitor latency and speed dropouts.
        - **Expected Impact**: Elevates NPS and reduces Fiber optic churn by up to 12%.
        """)

    with strat_col2:
        st.markdown("""
        ### 3. Early-Lifecycle Onboarding Shield (Months 1–6)
        - **Insight**: Churn risk is highest during months 1 through 12, after which retention sharply stabilizes.
        - **Action**: Implement a "Welcome Concierge" sequence: Day 7 setup confirmation, Day 30 satisfaction call, and Day 60 value review.
        - **Expected Impact**: Curtails early defection by up to 18%.

        ### 4. Auto-Pay & Paperless Enrollment Perk
        - **Insight**: Customers paying via Electronic Check churn at **45.3%**, compared to 15-18% for automated payment methods.
        - **Action**: Offer a one-off $10 billing credit to enroll in Automatic Credit Card or Bank Transfer payments.
        - **Expected Impact**: Builds billing frictionlessness and creates persistent customer inertia.
        """)


# ==========================================
# TAB 5: BATCH CSV PREDICTION
# ==========================================
elif nav_choice == "📁 Batch CSV Prediction":
    st.markdown('<div class="main-header">📁 Batch Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file containing multiple customer records to evaluate and rank churn risk in bulk.</div>', unsafe_allow_html=True)

    # Provide download sample button
    sample_df = df_raw.drop(columns=['Churn'], errors='ignore').head(50)
    sample_csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample Customer CSV Template",
        data=sample_csv,
        file_name="telco_sample_customers.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(batch_df)} customer records.")

            # Clean and engineer
            if 'TotalCharges' in batch_df.columns:
                batch_df['TotalCharges'] = pd.to_numeric(batch_df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)

            engineered_batch = add_engineered_features(batch_df)
            processed_batch = preprocessor.transform(engineered_batch)

            # Predict probabilities
            probs = active_model.predict_proba(processed_batch)[:, 1]
            batch_df['Churn_Probability'] = np.round(probs * 100, 1)
            batch_df['Risk_Level'] = pd.cut(
                batch_df['Churn_Probability'],
                bins=[-1, 35, 60, 100],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )

            # Quick batch metrics
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                st.metric("Total Customers Evaluated", len(batch_df))
            with b_col2:
                high_risk_n = (batch_df['Risk_Level'] == 'High Risk').sum()
                st.metric("High Risk Customers Flagged", f"{high_risk_n} ({(high_risk_n / len(batch_df))*100:.1f}%)")
            with b_col3:
                avg_risk = batch_df['Churn_Probability'].mean()
                st.metric("Average Churn Risk", f"{avg_risk:.1f}%")

            # Show scored table
            st.dataframe(batch_df[['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability', 'Risk_Level']].sort_values(by='Churn_Probability', ascending=False), use_container_width=True)

            # Download scored CSV
            scored_csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Customer List (CSV)",
                data=scored_csv,
                file_name="churn_predictions_scored.csv",
                mime="text/csv"
            )

        except Exception as err:
            st.error(f"Error processing uploaded CSV: {err}")
