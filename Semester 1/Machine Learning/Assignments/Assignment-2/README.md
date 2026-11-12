# Machine Learning - Assignment 2
## End-to-End Classification Benchmark & Streamlit Demonstration

---

## a. Problem Statement

Email spam is a common real-world classification problem. Unwanted emails clutter inboxes, waste user time, and can carry security risks such as phishing attempts or malware. The goal of this project is to build and evaluate supervised machine learning models that can automatically classify incoming emails as either **Spam (Class 1)** or **Ham / Legitimate (Class 0)** based on their content features.

We implemented and compared five supervised classification algorithms on the same dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (SVM with RBF kernel)

Each model is evaluated on a held-out test set using six standard evaluation metrics: **Accuracy, ROC-AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**. The pre-trained models are saved and demonstrated using an interactive Streamlit web application.

---

## b. Dataset Description

We selected the **UCI Spambase** dataset from the UCI Machine Learning Repository (also accessible via OpenML, ID: 44).

### Key Dataset Characteristics:
- **Total instances:** 4,601 email records (meets the requirement of >= 500 instances)
- **Total features:** 57 continuous numeric features (meets the requirement of >= 12 features)
- **Target column (`class`):** Binary integer label where `1` indicates spam and `0` indicates legitimate email (ham)
- **Class distribution:** 2,788 Ham emails (60.6%) and 1,813 Spam emails (39.4%)

### Feature Details:
The 57 features represent different email characteristics:
1. **Word frequency features (Columns 1 to 48):** The percentage of words in an email matching specific words commonly seen in spam or business communications (such as `free`, `money`, `credit`, `order`, `email`, `receive`, etc.).
2. **Character frequency features (Columns 49 to 54):** The percentage of characters in an email matching six punctuation marks: `;`, `(`, `[`, `!`, `$`, and `#`. Punctuation like exclamation marks and dollar signs are strong indicators of spam.
3. **Capital letter run-length features (Columns 55 to 57):**
   - `capital_run_length_average`: Average length of consecutive uppercase letter sequences.
   - `capital_run_length_longest`: Length of the longest sequence of uppercase letters.
   - `capital_run_length_total`: Total number of uppercase letters in the email.

### Data Preparation and Split:
- The dataset was split into **80% training (3,680 records)** and **20% testing (921 records)** using stratified sampling (`random_state=42`) to keep the class ratio consistent in both splits.
- All 57 features were normalized using `StandardScaler`. The scaler was fitted only on the training set and then used to transform the test set to prevent data leakage.
- The 20% test partition was saved as `test_data.csv` in the root folder for model evaluation.

---

## c. GitHub Repository Link

- **Repository Link:** `https://github.com/aemmanuval/ML-Ass2`
- **Streamlit App Link:** `https://ml-ass2.streamlit.app`

### Repository Structure:
```
project-folder/
|-- app.py                 # Streamlit web application demonstrating models
|-- requirements.txt       # Dependencies for deployment
|-- README.md              # Project report and evaluation tables
|-- test_data.csv          # Test dataset (921 rows, 58 columns)
|-- data/
|   |-- spambase_full.csv  # Full Spambase dataset
|   `-- train_data.csv     # Training partition
`-- model/
    |-- train_models.py    # Training script
    |-- model_training_and_evaluation.ipynb # Jupyter notebook
    |-- scaler.joblib      # Saved StandardScaler
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    `-- svm.joblib
```

---

## d. Models Used

We trained all 5 classification models on the standardized training data and evaluated them on the held-out test data (921 rows: 558 legitimate, 363 spam).

### Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9237 | 0.8980 | 0.8733 | 0.8855 | 0.8129 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8328 | 0.9376 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

---

### Observations on Model Performance:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Performed best overall. Standardizing the continuous frequency features allowed it to find a strong linear decision boundary. It achieved the highest accuracy (92.94%), AUC (0.9702), F1 (0.9093), and MCC (0.8518) while being very fast to train and run. |
| Decision Tree | Reached 91.10% accuracy with max_depth=7. The decision rules based on word frequency thresholds are easy to understand. However, because it splits on single features at right angles, its recall (87.33%) was slightly lower than Logistic Regression. |
| KNN | Achieved 90.77% accuracy and 0.9506 AUC using k=5. It classifies emails by finding the 5 nearest neighbors in Euclidean space. It works well, but calculating distances across 57 features makes it slower at inference time. |
| Naive Bayes | Had the highest recall of all models (95.87%), catching almost all spam emails (only 15 false negatives). However, its precision was lower (71.46%), causing more legitimate emails to be flagged as spam because it assumes all features are independent. |
| Support Vector Machine | Produced the highest precision (92.77%) and second-highest accuracy (92.73%). The RBF kernel successfully separated the classes and kept false positives low, which is important so legitimate emails are not sent to the spam folder. |
| Overall, Winner for your dataset? | Winner: Logistic Regression. It delivered the best balanced performance across all six metrics (highest accuracy, AUC, F1, and MCC). It is lightweight, generalizes well, and produces reliable probability estimates. |

---

## e. Streamlit Demonstration App

The web application (`app.py`) is designed as a clean demonstration of the dataset, data split, and model evaluation:
- **Dataset and Split Summary:** Displays the total dataset size (4,601), number of features (57), training set size (3,680 rows, 80%), and testing set size (921 rows, 20%). Includes an expandable preview of the sample data rows.
- **Model Selector:** A dropdown menu allows selecting each model or viewing all models together.
- **Metrics Display:** Shows Accuracy, AUC, Precision, Recall, F1, and MCC for the selected model.
- **Visualizations:** Displays the confusion matrix (with true negatives, false positives, false negatives, and true positives), classification report, and ROC curve.
- **All Models View:** Presents the complete comparison table, a bar chart comparing metrics, and all five ROC curves plotted together.

---

## f. How to Run Locally and Deploy

### Local Execution:
```bash
# Clone the repository
git clone https://github.com/aemmanuval/ML-Ass2.git
cd ML-Ass2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

### Streamlit Community Cloud Deployment:
1. Push this project to your GitHub account.
2. Sign in at [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Click "New App", choose your repository, branch (`main`), and file (`app.py`).
4. Click "Deploy".
