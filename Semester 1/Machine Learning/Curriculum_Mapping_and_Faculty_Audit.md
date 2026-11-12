# AIML ZG565: Machine Learning — Master Curriculum Mapping & Faculty Stream Audit

**Course**: AIML ZG565 — Machine Learning (4 Credit Units, 3-1-0)  
**Institution**: Birla Institute of Technology & Science (BITS), Pilani — WILP Division  
**Lead Faculty**: Swarna Chaudhary | **Course Instructors**: Prof. Pritee Narendra Raole (Parwekar), Prof. Suresh Kumar Sanampudi, Prof. Nivethitha Somu, Prof. Devanshu Tayal, Prof. Sharon Christa, Prof. Amit Sharma, Prof. Anshid Aboobacker, Prof. Vishali Sivalenka, Prof. Honey Sharma

---

## 1. Multi-Faculty Stream Audit & Evaluation Report

In accordance with Phase 3 of the Course Notes Generator framework, all 9 parallel faculty lecture streams available in the `Inbox/` directory were audited against technical depth, syllabus adherence, administrative overhead, and stream continuity.

| Faculty Stream | Video Count | Mean Words / Session | Connection Drops | Tech-to-Admin Word Ratio | Available Materials & Artefacts | Audit Status & Anomalies |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Prof. Pritee Narendra Raole** | **14** | **15,396** | **1 (Recovered)** | **614.8** | **38 Annotated Session PDFs, Practice Problem Sets & Step-by-Step Solutions** | **⭐️ Selected Primary Stream (Highest Rigor)** |
| **Prof. Suresh Kumar Sanampudi** | 13 | 12,084 | 0 | 939.7 | Standard Class Presentation Slides | Complete Stream |
| **Prof. Amit Sharma** | 13 | 10,791 | 0 | 693.5 | Standard Presentation Slides | Complete Stream |
| **Prof. Anshid Aboobacker** | 13 | 9,905 | 0 | 581.6 | 13 Presentation PDFs, 2 Jupyter Notebooks | MS Teams Naming Bug |
| **Prof. Devanshu Tayal** | 14 | 9,011 | 0 | 684.7 | 25 Practice Question Sets & Solutions, Session Slides | Secondary Companion (Exam Problem Sets) |
| **Prof. Nivethitha Somu** | 13 | 8,641 | 0 | 738.0 | 25 PDFs (Gower Distance, Numericals), 6 Jupyter Notebooks, Textbooks | Secondary Companion (Code & Math Walkthroughs) |
| **Prof. Vishali Sivalenka** | 12 | 8,307 | 0 | 897.1 | 12 Lecture PDFs, 2 Jupyter Notebooks | MS Teams Naming Bug |
| **Prof. Honey Sharma** | 13 | 7,962 | 0 | 510.8 | 17 Annotated Slides, Ebooks, 2 Python Notebooks | MS Teams Naming Bug |
| **Prof. Sharon Christa** | 14 | 6,739 | 0 | 772.3 | 13 Session PPTs | Complete Stream |

---

## 2. Primary Faculty Selection Rationale

### ⭐️ Primary Faculty Selected: Prof. PRITEE NARENDRA RAOLE (PARWEKAR)
1. **Unrivaled Conceptual & Mathematical Depth**: Averaging **15,396 words per session**, Prof. Pritee provides exhaustive step-by-step mathematical derivations on screen, explains geometric intuition, and covers every edge case.
2. **Rich Handwritten Annotations**: 38 annotated session slide decks with explicit tablet derivations, vector dimensional tracking, and intermediate calculations.
3. **Rigorous Syllabus Adherence**: 100% 1-to-1 fidelity with the official BITS AIML ZG565 Course Handout schedule.

### Companion Secondary Faculty Streams:
- **Prof. Nivethitha Somu**: Provides structured numerical calculation worksheets (`Gower_Distance_Step_by_Step.docx`) and 6 Python implementation notebooks.
- **Prof. Devanshu Tayal**: Provides 25 categorized examination numerical problem sets with full step-by-step solutions (`Classification_Metrics_Full_Step_by_Step_Solutions.pdf`, `Linear_Regression_Step_by_Step_Solutions.pdf`).

---

## 3. The MS Teams Naming Anomaly & Chronological Reconciliation

In Microsoft Teams, ad-hoc meetings start with `Meeting in Prof...` whereas scheduled recurring channel meetings start with `Machine Learning (S1-26_AIMLZG565) - Prof...`. Naive alphabetical sorting puts `Meeting in...` ahead of scheduled classes, throwing chronological ordering out of alignment.

To establish ground-truth chronological sequencing, we inspected:
1. First-line markdown transcript timestamps (`# Transcript: ... YYYYMMDD_HHMMSS`).
2. Video container timestamps.
3. Prof. Pritee's opening date slides.

On **2026-10-04 (Session 9)**, a network disconnection occurred after 7 minutes (`20261004_081307`, 4.1 MB). Prof. Pritee immediately reconnected 7 minutes later (`20261004_082033`, 321.9 MB) and delivered the complete 2-hour lecture. Both fragments are cataloged, with the complete recording mapped into the curriculum.

---

## 4. Master 1-to-1 Curriculum Mapping Matrix

| Contact Session | Date (YYYY-MM-DD) | Canonical Video File (`Videos/`) | Raw Synchronized Recording (`Inbox/`) | Syllabus Module | Core Topics Covered & Endpoints | Slides & Reference Materials | Generated Note |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2026-07-26 | `Lecture_01_Intro_to_ML_and_Design_of_Learning_Systems.mp4` | `...20260726_082007-Meeting Recording.mp4` | **Module 1**: Intro to ML & Designing Learning System | What is ML, Task T, Performance P, Experience E, Checkers Learning System, Generalizer, LMS Update | `2026-26th July-ML_CS 1.pdf` | `Module_01_Introduction_and_Designing_a_Learning_System.md` |
| **2** | 2026-08-02 | `Lecture_02_Mathematical_Preliminaries_and_ML_Workflow.mp4` | `...20260802_082102-Meeting Recording.mp4` | **Module 2**: Math Preliminaries & ML Workflow | Linear Algebra, Calculus, Covariance, Shannon Entropy, Data Wrangling, Normalization vs Standardization | `ML_CS 2_09Aug2026.pdf` | `Module_02_Mathematical_Preliminaries_and_ML_Workflow.md` |
| **3** | 2026-08-09 | `Lecture_03_Linear_Models_for_Regression.mp4` | `...20260809_081851-Meeting Recording.mp4` | **Module 3**: Linear Models for Regression | Ordinary Least Squares, Normal Equation derivation, Gradient Descent (Batch, SGD), Basis functions, Ridge/Lasso | `ML_CS 3&4_9&16Aug2026.pdf` | `Module_03_Linear_Models_for_Regression.md` |
| **4** | 2026-08-16 | `Lecture_04_Linear_Models_for_Classification.mp4` | `...20260816_081121-Meeting Recording.mp4` | **Module 4**: Linear Models for Classification | Discriminant Functions, Hyperplanes, Fisher LDA derivation, Generative vs Discriminative classification | `ML_CS 3&4_9&16Aug2026.pdf` | `Module_04_Linear_Models_for_Classification_and_Logistic_Regression.md` |
| **5** | 2026-08-23 | `Lecture_05_Logistic_Regression_Logloss_Multiclass.mp4` | `...20260823_081138-Meeting Recording.mp4` | **Module 4**: Logistic Regression Deep-Dive | Sigmoid function, Log-Odds, Binary Cross-Entropy (Log Loss) derivation, Convexity, Multi-class Softmax | `23rd-30th Aug-ML_CS 5 & 6.pdf` | `Module_04_Linear_Models_for_Classification_and_Logistic_Regression.md` |
| **6** | 2026-08-30 | `Lecture_06_Decision_Trees_Entropy_InfoGain_ID3.mp4` | `...20260830_080935-Meeting Recording.mp4` | **Module 5**: Decision Trees & Information Theory | Shannon Entropy, Gini Impurity, Information Gain, Gain Ratio (C4.5), Continuous split thresholds, MDL Principle | `23rd-30th Aug-ML_CS 5 & 6.pdf` | `Module_05_Decision_Trees_and_Information_Theory.md` |
| **7** | 2026-09-06 | `Lecture_07_Instance_Based_Learning_KNN_LWR.mp4` | `...20260906_081300-Meeting Recording.mp4` | **Module 6**: Instance-Based Learning | Eager vs Lazy learning, $k$-NN, Euclidean/Manhattan/Mahalanobis/Gower distances, Cover-Hart Bound, LWR | `6th-13th Sept-ML_CS 7&8.pdf` | `Module_06_Instance_Based_Learning.md` |
| **8** | 2026-09-13 | `Lecture_08_Mid_Semester_Review_Numericals.mp4` | `...20260913_081151-Meeting Recording.mp4` | **Review**: Mid-Semester Exam Review | Comprehensive numerical problem walkthroughs on Regression, Classification, Decision Trees, and $k$-NN | `13th Sep26-Practice Numericals.pdf` | `50_Exam_Question_Answer_Bank.md` |
| **9** | 2026-10-04 | `Lecture_09_Support_Vector_Machines_Part1.mp4` | `...20261004_082033-Meeting Recording.mp4` | **Module 7**: Support Vector Machines (Linear) | Geometric Margin $\frac{2}{\|\mathbf{w}\|}$, Primal QP formulation, Lagrangian, KKT conditions, Support Vectors | `4th Oct 26-ML_CS 7&8.pdf` | `Module_07_Support_Vector_Machines.md` |
| **10** | 2026-10-11 | `Lecture_10_Support_Vector_Machines_Part2_Kernels.mp4` | `...20261011_081857-Meeting Recording.mp4` | **Module 7**: Soft-Margin SVM & Kernel Trick | Slack variables $\xi_i$, Penalty parameter $C$, Dual box constraints, Mercer's Theorem, Polynomial & RBF Kernels | `ML_CS 10_SVM_October 2026.pdf` | `Module_07_Support_Vector_Machines.md` |
| **11** | 2026-10-18 | `Lecture_11_Bayesian_Learning_MLE_MAP_Naive_Bayes.mp4` | `...20261018_081543-Meeting Recording.mp4` | **Module 8**: Bayesian Learning | Bayes' Theorem, MLE vs MAP, Least Squares as Gaussian MLE, Bayes Optimal Classifier, Naïve Bayes, Laplace smoothing | `ML_Bayes Classifier-CS 11.pdf` | `Module_08_Bayesian_Learning.md` |
| **12** | 2026-10-25 | `Lecture_12_Ensemble_Learning_Bagging_Random_Forest.mp4` | `...20261025_081930-Meeting Recording.mp4` | **Module 9**: Ensemble Learning (Bagging & RF) | Condorcet Jury Theorem, Variance reduction, Bootstrap aggregation, OOB error estimation, Random Forest feature bagging | `Extra Material/Ensemble` | `Module_09_Ensemble_Learning.md` |
| **13** | 2026-11-01 | `Lecture_13_Ensemble_Boosting_and_Unsupervised_Learning.mp4` | `...20261101_082050-Meeting Recording.mp4` | **Modules 9, 10, 11**: Boosting, Clustering & Evaluation | AdaBoost weight updating, XGBoost 2nd-order expansion, $k$-Means, $k$-Means++, GMM EM Algorithm, ROC-AUC, Fairness | `Extra Material/Clustering` | `Module_10_Unsupervised_Learning_and_Clustering.md` & `Module_11_...md` |

