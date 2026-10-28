import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Local imports
from src.data_loader import load_and_clean_data
from src.feature_engineering import add_engineered_features, build_preprocessor
from src.evaluate import evaluate_predictions, compute_roc_curve

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def train_and_evaluate_models():
    """
    End-to-end training pipeline for Logistic Regression, Random Forest, and XGBoost.
    Saves trained models, preprocessor, and evaluation metrics for Streamlit app consumption.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("1. Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"   Loaded {len(df)} customer records.")

    print("2. Engineering features...")
    df_feat = add_engineered_features(df)

    # Separate target
    y = df_feat['Churn']
    if set(y.unique()).intersection({'Yes', 'No'}):
        y = y.map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    else:
        y = y.astype(int)
    X = df_feat.drop(columns=['Churn', 'customerID'], errors='ignore')

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # Fit preprocessor on training data only
    print("3. Fitting preprocessing pipeline...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names for readability (remove "num__" and "cat__" prefixes)
    clean_feature_names = [
        f.replace("num__", "").replace("cat__", "") for f in feature_names
    ]

    # Save preprocessor artifact
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    print("   Saved preprocessor.pkl")

    # Define model dictionary
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
    }

    # Add XGBoost if available
    try:
        from xgboost import XGBClassifier
        # Calculate scale_pos_weight = count(negative) / count(positive)
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos = neg_count / max(pos_count, 1)

        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42
        )
        print("   XGBoost module available.")
    except ImportError:
        print("   Notice: XGBoost not yet installed, continuing with scikit-learn models.")

    all_metrics = {}
    roc_curves = {}
    feature_importances = {}
    best_model_name = None
    best_roc_auc = -1.0

    print("\n4. Training and evaluating models...")
    for name, model in models.items():
        print(f"   --> Training {name}...")
        model.fit(X_train_proc, y_train)

        # Predictions
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        all_metrics[name] = metrics
        roc_curves[name] = compute_roc_curve(y_test, y_prob)

        # Save individual model
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{slug}.pkl"))

        # Extract feature importance / coefficients
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'feature': clean_feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            feature_importances[name] = feat_imp_df.head(15).to_dict(orient='records')
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            feat_imp_df = pd.DataFrame({
                'feature': clean_feature_names,
                'importance': coefs
            }).sort_values(by='importance', ascending=False)
            feature_importances[name] = feat_imp_df.head(15).to_dict(orient='records')

        print(f"       Accuracy:  {metrics['accuracy']:.4f}")
        print(f"       Precision: {metrics['precision']:.4f}")
        print(f"       Recall:    {metrics['recall']:.4f}")
        print(f"       F1-Score:  {metrics['f1_score']:.4f}")
        print(f"       ROC-AUC:   {metrics['roc_auc']:.4f}")

        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_model_name = name

    print(f"\nBest Performing Model: {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")
    best_slug = best_model_name.lower().replace(" ", "_")
    best_model_obj = models[best_model_name]
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_model.pkl"))

    # Save summary metadata
    meta = {
        'best_model': best_model_name,
        'best_roc_auc': best_roc_auc,
        'metrics': all_metrics,
        'feature_names': clean_feature_names
    }
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(MODELS_DIR, "roc_curves.json"), "w") as f:
        json.dump(roc_curves, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(feature_importances, f, indent=2)

    print("5. Successfully saved all model artifacts into models/ directory.")
    return meta

if __name__ == "__main__":
    train_and_evaluate_models()
