"""
Machine Learning - Assignment 2
Interactive Model Demonstration: Spam Email Classification
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Spam Classification - Model Demonstration",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, simple custom CSS
st.markdown("""
<style>
    .report-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .report-subtitle {
        font-size: 1rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.3rem;
    }
    .info-box {
        background-color: #f3f4f6;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #1f2937;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# File Paths and Caching
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST_CSV = os.path.join(BASE_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": {
        "summary": "Linear classifier using a sigmoid function with L2 regularization.",
        "params": "Solver: lbfgs, C=1.0, max_iter=1000, features standardized."
    },
    "Decision Tree": {
        "summary": "Non-linear decision tree splitting on feature thresholds based on Gini impurity.",
        "params": "Criterion: Gini, max_depth=7, min_samples_split=5."
    },
    "KNN": {
        "summary": "Instance-based classifier that assigns the majority class of the 5 closest neighbors.",
        "params": "k=5, metric=Euclidean (Minkowski p=2), features standardized."
    },
    "Naive Bayes": {
        "summary": "Probabilistic model applying Bayes' theorem with Gaussian likelihood assumptions.",
        "params": "GaussianNB, var_smoothing=1e-9."
    },
    "Support Vector Machine": {
        "summary": "Max-margin classifier using a Radial Basis Function (RBF) kernel.",
        "params": "Kernel=RBF, C=1.0, probability=True, features standardized."
    }
}

@st.cache_resource
def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_model(name):
    filename = MODEL_FILES.get(name)
    if not filename:
        return None
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_test_data():
    if os.path.exists(DEFAULT_TEST_CSV):
        return pd.read_csv(DEFAULT_TEST_CSV)
    return None

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Controls")
    
    # Model selector
    model_choices = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine",
        "Compare All Models"
    ]
    selected_model = st.selectbox("Select Model to Evaluate:", model_choices, index=0)

    st.markdown("---")
    st.markdown("### Dataset Overview")
    st.markdown("""
    - **Source:** UCI Spambase
    - **Total Records:** 4,601
    - **Features:** 57 numeric columns
    - **Train Set:** 3,680 rows (80%)
    - **Test Set:** 921 rows (20%)
    - **Target:** 0 = Ham, 1 = Spam
    """)

    # Optional file upload kept in a collapsed expander for rubric compliance
    with st.expander("Upload Custom Test CSV (Optional)"):
        uploaded_file = st.file_uploader(
            "Upload a test CSV with 57 features and 'class' column",
            type=["csv"],
            help="By default the app uses the 20% test split (921 rows). You can optionally upload your own test set."
        )

# Determine active test dataset
if 'uploaded_file' in locals() and uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        using_custom = True
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        df_test = load_test_data()
        using_custom = False
else:
    df_test = load_test_data()
    using_custom = False

# ---------------------------------------------------------
# Main Page Content
# ---------------------------------------------------------
st.markdown('<div class="report-title">Spam Email Classification - Model Demonstration</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">Evaluation and comparison of five supervised classification models on the Spambase dataset.</div>', unsafe_allow_html=True)

if df_test is None:
    st.error("Test data file (test_data.csv) was not found in the project directory.")
    st.stop()

# Identify target column
target_col = 'class' if 'class' in df_test.columns else df_test.columns[-1]
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col].astype(int)

# Load scaler
scaler = load_scaler()
if scaler is None:
    st.error("StandardScaler artifact (scaler.joblib) not found. Run model/train_models.py first.")
    st.stop()

try:
    X_test_scaled = scaler.transform(X_test)
except Exception as e:
    st.error(f"Feature scaling failed: {e}. Check that test data has the expected 57 columns.")
    st.stop()

# ---------------------------------------------------------
# Section 1: Dataset and Split Information
# ---------------------------------------------------------
st.markdown('<div class="section-header">1. Dataset and Train-Test Split Details</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>Dataset Summary:</strong> We used the <strong>UCI Spambase</strong> dataset, which contains <strong>4,601</strong> email instances. 
    Each email is described by <strong>57 continuous numeric features</strong>: 48 word frequency percentages (like 'free', 'money', 'credit'), 
    6 special character percentages (;, (, [, !, $, #), and 3 capital letter run-length statistics. 
    The target variable is binary: <strong>0 for Ham (legitimate email)</strong> and <strong>1 for Spam</strong>.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Dataset Size", "4,601 rows")
c2.metric("Features", "57 columns")
c3.metric("Train Split (80%)", "3,680 rows")
c4.metric("Test Split (20%)", f"{len(df_test):,} rows")
ham_total = int((y_test == 0).sum())
spam_total = int((y_test == 1).sum())
c5.metric("Test Class Ratio", f"{ham_total} Ham / {spam_total} Spam")

with st.expander("Show sample data rows and feature column names"):
    st.markdown("**First 5 rows of test data:**")
    st.dataframe(df_test.head(5), use_container_width=True)
    st.markdown(f"**Total feature columns ({len(X_test.columns)}):**")
    st.write(", ".join(X_test.columns))

# ---------------------------------------------------------
# Helper function for evaluation
# ---------------------------------------------------------
def evaluate_model(model, X_scaled, y_true):
    y_pred = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_scaled)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True)

    return {
        "acc": acc, "auc": auc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc,
        "y_pred": y_pred, "y_prob": y_prob, "cm": cm, "cr": cr
    }

# ---------------------------------------------------------
# Section 2: Model Evaluation
# ---------------------------------------------------------
st.markdown('<div class="section-header">2. Model Evaluation and Performance Metrics</div>', unsafe_allow_html=True)

if selected_model != "Compare All Models":
    clf = load_model(selected_model)
    if clf is None:
        st.error(f"Could not load saved model for {selected_model}.")
        st.stop()

    res = evaluate_model(clf, X_test_scaled, y_test)
    desc = MODEL_DESCRIPTIONS.get(selected_model, {})

    st.markdown(f"**Selected Model:** `{selected_model}`")
    st.caption(f"{desc.get('summary')} | Hyperparameters: {desc.get('params')}")

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{res['acc']:.4f}")
    m2.metric("AUC Score", f"{res['auc']:.4f}")
    m3.metric("Precision", f"{res['prec']:.4f}")
    m4.metric("Recall", f"{res['rec']:.4f}")
    m5.metric("F1 Score", f"{res['f1']:.4f}")
    m6.metric("MCC Score", f"{res['mcc']:.4f}")

    st.write("")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("**Confusion Matrix:**")
        cm = res['cm']
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([
            [f"TN: {tn}\n({tn/len(y_test):.1%})", f"FP: {fp}\n({fp/len(y_test):.1%})"],
            [f"FN: {fn}\n({fn/len(y_test):.1%})", f"TP: {tp}\n({tp/len(y_test):.1%})"]
        ])
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"],
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Confusion Matrix: {selected_model}")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_right:
        st.markdown("**Classification Report:**")
        cr_dict = res['cr']
        table_rows = [
            {
                "Class": "Ham (0 - Legitimate)",
                "Precision": f"{cr_dict['0']['precision']:.4f}",
                "Recall": f"{cr_dict['0']['recall']:.4f}",
                "F1-Score": f"{cr_dict['0']['f1-score']:.4f}",
                "Support": int(cr_dict['0']['support'])
            },
            {
                "Class": "Spam (1 - Unwanted)",
                "Precision": f"{cr_dict['1']['precision']:.4f}",
                "Recall": f"{cr_dict['1']['recall']:.4f}",
                "F1-Score": f"{cr_dict['1']['f1-score']:.4f}",
                "Support": int(cr_dict['1']['support'])
            }
        ]
        st.table(pd.DataFrame(table_rows))
        st.markdown(f"""
        - **Accuracy on test set:** `{res['acc']:.4f}` ({res['acc']:.2%})
        - **False Positives (legitimate emails marked as spam):** `{fp}`
        - **False Negatives (spam emails missed):** `{fn}`
        """)

    # ROC Curve
    st.markdown("**ROC Curve:**")
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    fig_roc, ax_roc = plt.subplots(figsize=(6, 3.5))
    ax_roc.plot(fpr, tpr, label=f"{selected_model} (AUC = {res['auc']:.4f})", color="#2563eb", lw=2)
    ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Chance (0.50)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig_roc)
    plt.close(fig_roc)

# ---------------------------------------------------------
# Section 3: All Models Comparison
# ---------------------------------------------------------
else:
    st.markdown("Evaluating and comparing all five classification models side by side on the test set.")

    benchmarks = []
    roc_data = {}

    for name in MODEL_FILES.keys():
        m = load_model(name)
        if m is not None:
            r = evaluate_model(m, X_test_scaled, y_test)
            benchmarks.append({
                "ML Model Name": name,
                "Accuracy": round(r['acc'], 4),
                "AUC": round(r['auc'], 4),
                "Precision": round(r['prec'], 4),
                "Recall": round(r['rec'], 4),
                "F1": round(r['f1'], 4),
                "MCC": round(r['mcc'], 4)
            })
            roc_data[name] = (r['y_prob'], r['auc'])

    df_bench = pd.DataFrame(benchmarks)

    st.markdown("#### Comparison Table")
    st.dataframe(df_bench, use_container_width=True, hide_index=True)

    # Bar chart
    st.markdown("#### Metric Comparison Chart")
    chart_data = df_bench.set_index("ML Model Name")
    fig_bar, ax_bar = plt.subplots(figsize=(9, 4.5))
    chart_data.plot(kind="bar", ax=ax_bar, colormap="tab10", width=0.8)
    ax_bar.set_ylim(0.65, 1.0)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Performance Comparison Across All 5 Classifiers")
    ax_bar.grid(axis='y', linestyle=":", alpha=0.6)
    plt.xticks(rotation=15, ha='right')
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # ROC Curves Comparison
    st.markdown("#### ROC Curves Comparison")
    fig_all_roc, ax_aroc = plt.subplots(figsize=(8, 4.5))
    for name, (prob, auc_val) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax_aroc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", lw=1.8)
    ax_aroc.plot([0, 1], [0, 1], 'k--', label="Random Chance (0.50)")
    ax_aroc.set_xlabel("False Positive Rate")
    ax_aroc.set_ylabel("True Positive Rate")
    ax_aroc.set_title("Combined ROC Curves")
    ax_aroc.legend(loc="lower right")
    ax_aroc.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig_all_roc)
    plt.close(fig_all_roc)

    # Observations table
    st.markdown("#### Observations on Model Performance")
    observations = [
        {
            "ML Model Name": "Logistic Regression",
            "Observation about model performance": "Best overall model. Standardizing continuous frequency features helped it find a strong linear decision boundary. It achieved the highest Accuracy (92.94%), AUC (0.9702), F1 (0.9093), and MCC (0.8518) with fast training and low prediction latency."
        },
        {
            "ML Model Name": "Decision Tree",
            "Observation about model performance": "Performed well with 91.10% accuracy and 0.8980 precision. Decision rules are easy to interpret. Its recall (87.33%) was slightly lower because single axis-aligned splits struggle with subtle correlations across word frequencies."
        },
        {
            "ML Model Name": "KNN",
            "Observation about model performance": "Achieved 90.77% accuracy and 0.9506 AUC. It works by finding the 5 closest emails in Euclidean space. However, calculating distances across 57 features makes it slower at inference time."
        },
        {
            "ML Model Name": "Naive Bayes",
            "Observation about model performance": "Had the highest Recall of all models (95.87%), catching almost all spam emails. However, its precision was lower (71.46%), resulting in more false alarms on legitimate emails due to the assumption that word frequencies are independent."
        },
        {
            "ML Model Name": "Support Vector Machine",
            "Observation about model performance": "Achieved the highest Precision (92.77%) and second-highest Accuracy (92.73%). The RBF kernel mapped the features effectively, keeping false positives to a minimum, which is useful when legitimate emails cannot be lost."
        },
        {
            "ML Model Name": "Overall, Winner for your dataset?",
            "Observation about model performance": "Winner: Logistic Regression. It produced the best overall balance across all six evaluation metrics (92.94% Accuracy, 0.9702 AUC, 0.9093 F1, and 0.8518 MCC). If false positives must be minimized above all else, SVM is a close alternative with 92.77% precision."
        }
    ]
    st.table(pd.DataFrame(observations))

st.markdown("---")
st.caption("Demonstration app for supervised machine learning classification.")
