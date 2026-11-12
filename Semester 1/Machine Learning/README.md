# BITS Pilani — AIML ZG565: Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Course](https://img.shields.io/badge/BITS%20Pilani-WILP%20M.Tech-red.svg)](https://bits-pilani-wilp.ac.in)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An academic study, revision, and examination preparation repository for **AIML ZG565: Machine Learning** (Semester 1, M.Tech in Artificial Intelligence & Machine Learning) at Birla Institute of Technology & Science (BITS), Pilani.

This repository features topic-wise modular descriptive notes, a 50-question examination question-answer bank with 6-marks model answers, a master formula sheet & oral exam guide, textbooks, faculty handwritten derivations, practice numerical sets, and an interactive **Streamlit web application**.

---

## 🌟 Interactive Streamlit Web Portal

Access and explore all course notes, mathematical derivations, solved numericals, and textbooks from any mobile or desktop browser via the built-in Streamlit portal.

### Run Locally

```bash
# 1. Clone repository
git clone https://github.com/aemmanuval/AIML-ZG565-Machine-Learning.git
cd AIML-ZG565-Machine-Learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run streamlit_app.py
```

### Deploy to Streamlit Community Cloud (Free with GitHub)
1. Fork or push this repository to your GitHub account (`aemmanuval/AIML-ZG565-Machine-Learning`).
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New App"** and select:
   - **Repository**: `aemmanuval/AIML-ZG565-Machine-Learning`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py` (or `app.py`)
4. Click **Deploy!** Your cloud study portal is now live with permanent 24/7 web access.

---

## 📂 Repository Structure

```
AIML-ZG565-Machine-Learning/
├── Syllabus.md                               # Human-readable markdown syllabus
├── AIML_ZG565_COURSE_HANDOUT.md              # Formal course handout markdown
├── Curriculum_Mapping_and_Faculty_Audit.md   # 9-faculty audit, MS Teams bug fix & curriculum map
├── streamlit_app.py                          # Multi-tab interactive Streamlit web application
├── app.py                                    # Streamlit entrypoint symlink
├── requirements.txt                          # Python dependencies for Streamlit cloud
├── Books/                                    # Prescribed textbooks (T1) & references (R1, R2, R3)
│   ├── T1_Tom_Mitchell_Machine_Learning.pdf
│   ├── R1_Christopher_Bishop_Pattern_Recognition_and_Machine_Learning.pdf
│   ├── R2_Tan_Steinbach_Kumar_Introduction_to_Data_Mining.pdf
│   ├── R3_Burges_Tutorial_on_Support_Vector_Machines_for_Pattern_Recognition.pdf
│   └── ...
├── Notes/
│   ├── AI Notes/
│   │   ├── 01_Descriptive_Notes/             # Topic-wise modular study guides (Modules 1–11)
│   │   │   ├── Module_01_Introduction_and_Designing_a_Learning_System.md
│   │   │   ├── Module_02_Mathematical_Preliminaries_and_ML_Workflow.md
│   │   │   ├── Module_03_Linear_Models_for_Regression.md
│   │   │   ├── Module_04_Linear_Models_for_Classification_and_Logistic_Regression.md
│   │   │   ├── Module_05_Decision_Trees_and_Information_Theory.md
│   │   │   ├── Module_06_Instance_Based_Learning.md
│   │   │   ├── Module_07_Support_Vector_Machines.md
│   │   │   ├── Module_08_Bayesian_Learning.md
│   │   │   ├── Module_09_Ensemble_Learning.md
│   │   │   ├── Module_10_Unsupervised_Learning_and_Clustering.md
│   │   │   └── Module_11_Machine_Learning_Model_Evaluation_and_Comparison.md
│   │   ├── 02_Question_Answer_Notes/
│   │   │   ├── 50_Exam_Question_Answer_Bank.md       # 50 Q&As (15 Easy, 25 Medium, 10 Tough)
│   │   │   └── Master_Formula_Sheet_and_Study_Guide.md # Master formulas, dimensions & viva questions
│   │   └── assets/                           # Extracted visual keyframes from lecture recordings
│   ├── Handwitten Notes/                     # 8 annotated session slide decks with tablet derivations
│   ├── Lecture Slides & Companions/          # Hands-on demo notebooks (6 .ipynb) & distance numericals
│   └── Question Papers & Solutions/          # BITS examination numerical worksheets & solutions
└── Assignments/
    └── Assignment-2/                         # Real-world ML model development assignment codebase
```

---

## 📑 Syllabus Coverage Overview

| Module | Core Topics | Primary Prescribed Reference |
| :---: | :--- | :--- |
| **01** | Introduction to ML, Designing a Learning System, LMS Algorithm | Mitchell (T1, Ch 1) |
| **02** | Matrix Calculus, Covariance, Normalization, Standardization, SMOTE | Tan et al. (R2, Ch 2-3) |
| **03** | Linear Regression, Normal Equation, Gradient Descent, Ridge & Lasso | Bishop (R1, Ch 3) |
| **04** | Fisher LDA, Sigmoid Link, Binary Cross-Entropy Derivation, Softmax | Bishop (R1, Ch 4) |
| **05** | Shannon Entropy, Gini Impurity, Information Gain, Gain Ratio, MDL | Mitchell (T1, Ch 3) |
| **06** | $k$-NN, Distance Metrics (Mahalanobis, Gower), Cover-Hart Bound, LWR | Mitchell (T1, Ch 8) |
| **07** | Support Vector Machines, Margin $\frac{2}{\|\mathbf{w}\|}$, Dual QP, KKT, RBF Kernel | Burges (R3), Tan (R2, Ch 4) |
| **08** | Bayes' Theorem, MLE vs MAP, Least Squares as Gaussian MLE, Naïve Bayes | Mitchell (T1, Ch 6) |
| **09** | Condorcet Jury Theorem, Bagging, Random Forests, AdaBoost, XGBoost | Tan et al. (R2, Ch 4) |
| **10** | $k$-Means, $k$-Means++ Bound, GMM Expectation-Maximization, Silhouette | Bishop (R1, Ch 9) |
| **11** | Confusion Matrix, ROC-AUC, PR Curves, McNemar's Test, Fairness, SHAP | Mitchell (T1, Ch 5) |

---

## 🎓 Faculty Stream Attribution

The synthesized study material is constructed using an audited multi-faculty synthesis:
- **Primary Faculty Followed**: **Prof. Pritee Narendra Raole (Parwekar)** — Selected for mathematical rigor, step-by-step derivations on tablet, and complete 1-to-1 syllabus adherence.
- **Companion Webinar & Code Faculty**: **Prof. Nivethitha Somu** — Lead Webinar Instructor, providing Jupyter demo code and distance calculation guides.
- **Companion Problem Set Faculty**: **Prof. Devanshu Tayal** — Author of 25 examination numerical problem sets with step-by-step solutions.
- **Lead Course Authors**: **Prof. Suresh Kumar Sanampudi & Swarna Chaudhary** (BITS Pilani WILP Division).

