"""
Machine Learning - Assignment 2: Classification Models & Streamlit Deployment
Dataset: UCI Spambase Dataset (4,601 instances, 57 features, Binary Classification)

This script:
1. Downloads and prepares the UCI Spambase dataset.
2. Performs 80/20 stratified train-test split.
3. Saves test_data.csv (root) and data artifacts.
4. Fits and persists the StandardScaler preprocessing artifact.
5. Trains and evaluates all 5 classification models:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbor (KNN) Classifier
   - Naive Bayes Classifier (GaussianNB)
   - Support Vector Machine (SVM with RBF Kernel)
6. Computes all 6 required evaluation metrics:
   - Accuracy, AUC Score, Precision, Recall, F1 Score, MCC Score.
7. Saves all trained model artifacts (.joblib) and metric summaries.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def run_pipeline():
    # Resolve directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    model_dir = current_dir

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("Machine Learning - Assignment 2: Classification Models & Evaluation")
    print("Dataset: UCI Spambase Classification Benchmark")
    print("=" * 70)

    # 1. Fetch Dataset
    print("\n[Step 1/5] Fetching UCI Spambase dataset from OpenML repository...")
    raw_data = fetch_openml(name='spambase', version=1, as_frame=True)
    df = raw_data.frame.copy()

    # Format column names to standard lower snake_case
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Target column check
    target_col = 'class'
    df[target_col] = df[target_col].astype(int)

    total_instances, total_features = df.shape[0], df.shape[1] - 1
    print(f"Dataset shape: {df.shape}")
    print(f"Total instances: {total_instances}, Features: {total_features}")
    print(f"Class distribution: {dict(df[target_col].value_counts())}")

    # Save full dataset
    full_csv_path = os.path.join(data_dir, "spambase_full.csv")
    df.to_csv(full_csv_path, index=False)
    print(f"Saved full dataset to: {full_csv_path}")

    # 2. Train-Test Split (80% Train, 20% Test, Stratified)
    print("\n[Step 2/5] Performing stratified 80/20 train-test split...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Combine test data with target column
    test_df = X_test.copy()
    test_df[target_col] = y_test

    # Combine train data with target column
    train_df = X_train.copy()
    train_df[target_col] = y_train

    # Save test_data.csv to project root and data directory
    root_test_csv = os.path.join(project_root, "test_data.csv")
    test_df.to_csv(root_test_csv, index=False)
    train_df.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    print(f"Saved test_data.csv to root: {root_test_csv} ({test_df.shape[0]} rows, {test_df.shape[1]} cols)")

    # 3. Feature Preprocessing (StandardScaler)
    print("\n[Step 3/5] Fitting StandardScaler on training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler artifact
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler preprocessing artifact to: {scaler_path}")

    # 4. Define and Train the 5 Required Classifiers
    print("\n[Step 4/5] Training all 5 classification models...")
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'logistic_regression.joblib'
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=7, min_samples_split=5, random_state=42),
            'decision_tree.joblib'
        ),
        'KNN': (
            KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
            'knn.joblib'
        ),
        'Naive Bayes': (
            GaussianNB(var_smoothing=1e-9),
            'naive_bayes.joblib'
        ),
        'Support Vector Machine': (
            SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            'svm.joblib'
        )
    }

    metrics_list = []
    detailed_metrics = {}

    for model_name, (clf, file_name) in models.items():
        print(f" -> Training {model_name}...")
        clf.fit(X_train_scaled, y_train)

        # Model persistence
        save_path = os.path.join(model_dir, file_name)
        joblib.dump(clf, save_path)

        # Predictions & Probability
        y_pred = clf.predict(X_test_scaled)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = clf.decision_function(X_test_scaled)

        # Evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics_list.append({
            'ML Model Name': model_name,
            'Accuracy': round(float(acc), 4),
            'AUC': round(float(auc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1': round(float(f1), 4),
            'MCC': round(float(mcc), 4)
        })

        detailed_metrics[model_name] = {
            'Accuracy': round(float(acc), 4),
            'AUC': round(float(auc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1': round(float(f1), 4),
            'MCC': round(float(mcc), 4),
            'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),
            'Classification Report': classification_report(y_test, y_pred, output_dict=True)
        }

    # Save summary metrics to JSON
    json_path = os.path.join(model_dir, "model_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=4)
    print(f"Saved model metrics JSON to: {json_path}")

    # 5. Display Evaluation Metrics Comparison Table
    print("\n[Step 5/5] Model Evaluation Metrics Comparison Table:")
    print("=" * 85)
    results_df = pd.DataFrame(metrics_list)
    print(results_df.to_string(index=False))
    print("=" * 85)
    print("\nModel training and evaluation complete.")

if __name__ == '__main__':
    run_pipeline()
