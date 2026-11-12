# Machine Learning - Assignment 2
## Final Submission Document

---

### Student Submission Details
- **Student Name:** [Your Full Name]
- **Student ID:** [Your Student ID / Roll Number]
- **Programme:** M.Tech (AIML)
- **Course:** Machine Learning (Assignment - 2)
- **Date:** November 2026

---

## 1. Mandatory Submission Links

### 1. GitHub Repository Link
- **URL:** `https://github.com/aemmanuval/ML-Ass2`
- **Repository Contents:**
  - `app.py`: Streamlit web application demonstrating models and dataset details
  - `requirements.txt`: Python package dependencies
  - `README.md`: Technical documentation and comparison tables matching Step 5
  - `test_data.csv`: Held-out 20% test dataset (921 rows x 58 columns)
  - `model/`: Saved models (`logistic_regression.joblib`, `decision_tree.joblib`, `knn.joblib`, `naive_bayes.joblib`, `svm.joblib`, `scaler.joblib`), pipeline script `train_models.py`, and notebook `model_training_and_evaluation.ipynb`.

### 2. Live Streamlit App Link
- **Live URL:** `https://ml-ass2.streamlit.app`
- **Deployment Platform:** Streamlit Community Cloud (Free tier)
- **Features Demonstrated:** Shows dataset details, 80/20 train-test split, model selection dropdown, evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), confusion matrix, classification report, and ROC curves.

---

## 2. Lab Execution Proof (Screenshot)

> **Instructions:**
> 1. Run `python model/train_models.py` or `streamlit run app.py` in your lab environment.
> 2. Take a screenshot of the terminal output and/or the browser interface.
> 3. Paste the screenshot below before exporting to PDF.

```
+--------------------------------------------------------------------------+
|                                                                          |
|                  [ INSERT LAB EXECUTION SCREENSHOT HERE ]                |
|                                                                          |
|    Shows terminal execution of model training and/or Streamlit app       |
|    running in the environment.                                           |
|                                                                          |
+--------------------------------------------------------------------------+
```

---

## 3. GitHub README Content (Section 3 - Step 5 Requirements)

### a. Problem Statement
Email spam is a common real-world classification problem. Unwanted emails clutter inboxes, waste user time, and can carry security risks such as phishing attempts or malware. The goal of this project is to build and evaluate supervised machine learning models that can automatically classify incoming emails as either **Spam (Class 1)** or **Ham / Legitimate (Class 0)** based on their content features.

We implemented and compared five supervised classification algorithms on the same dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (SVM with RBF kernel)

Each model is evaluated on a held-out test set using six standard evaluation metrics: **Accuracy, ROC-AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**. The pre-trained models are saved and demonstrated using an interactive Streamlit web application.

---

### b. Dataset Description
We selected the **UCI Spambase** dataset from the UCI Machine Learning Repository (also accessible via OpenML, ID: 44).

#### Key Dataset Characteristics:
- **Total instances:** 4,601 email records (meets the requirement of >= 500 instances)
- **Total features:** 57 continuous numeric features (meets the requirement of >= 12 features)
- **Target column (`class`):** Binary integer label where `1` indicates spam and `0` indicates legitimate email (ham)
- **Class distribution:** 2,788 Ham emails (60.6%) and 1,813 Spam emails (39.4%)

#### Feature Details:
The 57 features represent different email characteristics:
1. **Word frequency features (Columns 1 to 48):** The percentage of words in an email matching specific words commonly seen in spam or business communications (such as `free`, `money`, `credit`, `order`, `email`, `receive`, etc.).
2. **Character frequency features (Columns 49 to 54):** The percentage of characters in an email matching six punctuation marks: `;`, `(`, `[`, `!`, `$`, and `#`. Punctuation like exclamation marks and dollar signs are strong indicators of spam.
3. **Capital letter run-length features (Columns 55 to 57):**
   - `capital_run_length_average`: Average length of consecutive uppercase letter sequences.
   - `capital_run_length_longest`: Length of the longest sequence of uppercase letters.
   - `capital_run_length_total`: Total number of uppercase letters in the email.

#### Data Preparation and Split:
- The dataset was split into **80% training (3,680 records)** and **20% testing (921 records)** using stratified sampling (`random_state=42`) to keep the class ratio consistent in both splits.
- All 57 features were normalized using `StandardScaler`. The scaler was fitted only on the training set and then used to transform the test set to prevent data leakage.
- The 20% test partition was saved as `test_data.csv` in the root folder for model evaluation.

---

### c. GitHub Repository Link
- **Repository URL:** `https://github.com/<your-username>/ml-classification-assignment2`
- **File Structure:**
  - `app.py`
  - `requirements.txt`
  - `README.md`
  - `test_data.csv`
  - `model/` (contains trained `.joblib` models, `scaler.joblib`, and source code `.py` / `.ipynb`)

---

### d. Models Used & Evaluation Metrics Comparison

All five machine learning classification models were trained on the identical standardized training set (3,680 samples) and evaluated on the identical held-out test set (921 samples: 558 Ham, 363 Spam).

#### Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9237 | 0.8980 | 0.8733 | 0.8855 | 0.8129 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8328 | 0.9376 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

---

#### Observations on Model Performance:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Performed best overall. Standardizing the continuous frequency features allowed it to find a strong linear decision boundary. It achieved the highest accuracy (92.94%), AUC (0.9702), F1 (0.9093), and MCC (0.8518) while being very fast to train and run. |
| Decision Tree | Reached 91.10% accuracy with max_depth=7. The decision rules based on word frequency thresholds are easy to understand. However, because it splits on single features at right angles, its recall (87.33%) was slightly lower than Logistic Regression. |
| KNN | Achieved 90.77% accuracy and 0.9506 AUC using k=5. It classifies emails by finding the 5 nearest neighbors in Euclidean space. It works well, but calculating distances across 57 features makes it slower at inference time. |
| Naive Bayes | Had the highest recall of all models (95.87%), catching almost all spam emails (only 15 false negatives). However, its precision was lower (71.46%), causing more legitimate emails to be flagged as spam because it assumes all features are independent. |
| Support Vector Machine | Produced the highest precision (92.77%) and second-highest accuracy (92.73%). The RBF kernel successfully separated the classes and kept false positives low, which is important so legitimate emails are not sent to the spam folder. |
| Overall, Winner for your dataset? | Winner: Logistic Regression. It delivered the best balanced performance across all six metrics (highest accuracy, AUC, F1, and MCC). It is lightweight, generalizes well, and produces reliable probability estimates. |

---

## 4. Final Submission Checklist
- [x] GitHub repo contains all required artifacts (`app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model/`).
- [x] Streamlit app runs cleanly without error.
- [x] All 6 evaluation metrics implemented for all 5 models.
- [x] Dataset meets criteria: 57 features (>= 12) and 4,601 instances (>= 500).
- [x] README.md and Submission Document follow the exact required structure.
- [x] Lab execution screenshot section included.
