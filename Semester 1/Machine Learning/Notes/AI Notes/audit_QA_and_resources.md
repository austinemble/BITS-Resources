# AIML ZG565: Machine Learning — Supplemental Resources & QA Bank Audit Report

**Course**: AIML ZG565 — Machine Learning  
**Target Document Audited**: 
- `02_Question_Answer_Notes/50_Exam_Question_Answer_Bank.md`
- `02_Question_Answer_Notes/Master_Formula_Sheet_and_Study_Guide.md`
**Supplemental Sources Audited**:
1. Prof. Pritee Narendra Raole Annotated Practice Questions (5 documents)
2. Prof. Pritee Narendra Raole Practice Question Solutions (10 documents)
3. Prof. Pritee Narendra Raole Extra Material Slide Decks & Handouts (11 documents)
4. Question Papers & Solutions Collection (19 documents)
5. Prof. Nivethitha Somu Worksheets & Prof. Devanshu Tayal Problem Sets
6. Lecture Transcripts & Curriculum Mapping Archive

---

## Executive Summary & Audit Overview

This audit comprehensively examines all practice question problem sets, step-by-step solution PDFs, extra lecture materials, and question papers provided across the semester for **AIML ZG565: Machine Learning** (BITS Pilani WILP), benchmarked against the existing `50_Exam_Question_Answer_Bank.md` and `Master_Formula_Sheet_and_Study_Guide.md`.

### Core Finding
The existing 50-Question Bank and Formula Sheet provide excellent theoretical rigor (derivations of Normal Equations, Ridge invertibility, Bias-Variance, Sigmoid derivative, Log-loss gradient, Cover-Hart bound, Soft-margin SVM, EM algorithm, etc.). However, **substantial, high-yield examination numerical archetypes and university-specific problem patterns** that Prof. Pritee Narendra Raole and the BITS examination committee actively test in Mid-Semester and Comprehensive Examinations are **either missing, incomplete, or insufficiently detailed**.

Specifically, the supplemental materials contain four distinct categories of content that must be integrated:
1. **BITS-Specific Complex Numericals**:
   - **Locally Weighted Regression (LWR) with Gradient Descent Updates**: Computing $1/d^2$ kernel weights, calculating weighted local predictions $\hat{y} = \frac{\sum K_i y_i}{\sum K_i}$, and executing one step of local gradient descent ($\Delta w_0, \Delta w_1$) to update the local regression line.
   - **Mixed Data Type Distance-Weighted $k$-NN**: Combining Euclidean distance over normalized continuous features, scaled ordinal ranks, and nominal attribute mismatch penalties ($\frac{\text{mismatches}}{k_{\text{nom}}}$), followed by inverse distance squared weighting that flips the majority vote.
   - **Full CART Decision Tree Construction on Continuous Attributes**: Step-by-step Gini Impurity evaluation across binary categorical, multi-categorical, and sorted continuous attributes (e.g., the 10-record Loan Default dataset testing threshold midpoints at class boundary transitions).
   - **Minimum Description Length (MDL) Bit Calculations**: Quantifying tree complexity $L_{C_1}(h)$ versus exception error cost $L_{C_2}(D \mid h)$ using Shannon bit encoding ($\log_2 k$) to prove why a simpler tree with errors is chosen over a zero-error overfitted tree.
   - **Reverse Confusion Matrix Algebraic Reconstruction**: Deriving complete 4-quadrant confusion matrices algebraically from summary metrics (e.g., given $N$, Prevalence, Recall, and Precision/Specificity).
2. **Missing Master Formulas & Operational Rules**:
   - Tukey's 1.5 $\times$ IQR Outlier Rule ($[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$).
   - Elastic Net Regularized Cost and L1/L2 Weight Decay Gradient Updates.
   - Simple Linear Regression closed-form slope ($m$) and intercept ($b$) using bivariate summation formulas.
   - Coefficient of Determination ($R^2 = 1 - \text{SS}_{\text{res}}/\text{SS}_{\text{tot}}$).
   - False Positive Rate (FPR) and False Negative Rate (FNR) explicit definitions.
   - Min-Max scaling to arbitrary bounded ranges $[a, b]$: $x' = a + (b-a)\frac{x - x_{\min}}{x_{\max} - x_{\min}}$.
3. **Critical Extra Lecture Material Insights**:
   - **The High Accuracy Paradox in Medical Diagnostics**: The 10,000-patient cancer diagnosis case study demonstrating 96.5% overall accuracy with a disastrous 30% recall (missing 70% of cancer cases).
   - **4 Scenarios Favoring High Precision vs. 4 Scenarios Favoring High Recall**: Clinical, legal, and security criteria governing model threshold selection.
   - **The Curse of Dimensionality 1D vs. 2D vs. 3D Concrete Case**: How adding irrelevant features (shoe size, favorite number) mathematically inverts $k$-NN distance ranking, and its dual remedies (attribute weighting $w^2 \Delta x^2$ vs. CV feature elimination).
   - **Algorithmic Scaling Invariance**: Why Decision Trees and tree ensembles (Random Forest, XGBoost) are monotonically invariant to feature scaling, whereas distance and gradient algorithms strictly require it.

---

## 1. Comprehensive Inventory of Audited Supplemental Materials

| Category | File Name | Pages / Format | Core Topic & Technical Depth |
| :--- | :--- | :---: | :--- |
| **Practice Questions** | `13th Sep26-Practice Numericals AML.pdf` | 13 Pages | 36 Exam Numericals across Modules 1–5 (Splits, CV, Preprocessing, Outliers, OLS, Ridge, Lasso, Logistic, Trees, MDL) |
| **Practice Questions** | `Linear_Regression_Questions.pdf` | 3 Pages | 12 Questions (Simple, Medium, Hard: Summation slope/intercept, Matrix OLS, SGD, Batch GD) |
| **Practice Questions** | `Vectorization_Questions.pdf` | 1 Page | 10 Matrix-Vector numericals ($X \cdot W$, $Y = XW + b$, dimension tracking) |
| **Practice Questions** | `regularization_cost_function_5_questions.pdf` | 1 Page | 5 Cost numericals (L2 Ridge, L1 Lasso, Elastic Net, Cost difference) |
| **Practice Questions** | `regularization_numerical_questions.pdf` | 2 Pages | 10 Regularization numericals (Loss penalty, L1/L2 weight update steps) |
| **Practice Solutions** | `Linear_Regression_Step_by_Step_Solutions.pdf` | 13 Pages | Full 12 step-by-step solutions with sums, matrix inversions, and GD updates |
| **Practice Solutions** | `regularization_15_solutions_line_by_line.pdf` | 6 Pages | 15 Complete line-by-line solutions (Loss, penalties, weight decay updates) |
| **Practice Solutions** | `KNN-BITS-Problme Type-1-2-doctor.pdf` | 5 Pages | Problem Type 1 (Distance-weighted KNN) & Problem Type 2 (Mixed data types locally weighted KNN) |
| **Practice Solutions** | `KNN-BITS-Problme-2-doctor.pdf` | 5 Pages | Mixed data type KNN with ordinal scaling and categorical mismatch penalty |
| **Practice Solutions** | `KNN-BITS-Problem3.pdf` | 4 Pages | Problem Type 3: Locally Weighted Regression (LWR) prediction + 1-step GD update |
| **Practice Solutions** | `KNN-BITS-Additional Problem.pdf` | 5 Pages | 7-NN Regression, Manhattan distance, $[0, 5]$ min-max scaling, GD divergence proof, Outlier effect |
| **Practice Solutions** | `1-4th Oct26-BITS-Gini Index-CART-Decision Tree.pdf` | 13 Pages | Complete CART Gini decision tree derivation on 10-record Defaulted Loan dataset |
| **Practice Solutions** | `Gini Inedx-long tennis.pdf` | 6 Pages | CART Gini impurity calculation on Play Tennis dataset (Outlook, Temp, Humidity, Wind) |
| **Extra Material** | `3-4th Oct26-BITS-Multiway-binarysplit.pdf` | 11 Pages | Multiway vs Binary splits on continuous attributes, threshold selection at class boundaries |
| **Extra Material** | `4-BITS-Gain Ratio .pdf` | 11 Pages | C4.5 Gain Ratio step-by-step derivation on continuous temperature splits |
| **Extra Material** | `5-4th Oct 26-BITS-Overfitting problem-Decision tree.pdf` | 11 Pages | Decision tree overfitting, memorization pitfalls, Pre-pruning vs Post-pruning |
| **Extra Material** | `6-4th Oct 26-BITS-Minimum Description Length (MDL) Principle.pdf` | 18 Pages | MDL principle, Occam's razor, bit description costs $L(h) + L(D \mid h)$, classroom numerical |
| **Extra Material** | `BITS-Linear Regression.pdf` | 7 Pages | Meal bill vs tip numerical ($m, b$), why divide cost by $2m$ (derivative cancellation) |
| **Extra Material** | `GD-Solved Numerical.pdf.pdf` | 7 Pages | 5-parameter Gradient Descent on Heart Disease Relative Risk ($X_1 \dots X_4$, $\alpha = 10^{-5}$) |
| **Extra Material** | `KNN-CURSE-slide-34.pdf` | 6 Pages | Curse of Dimensionality 1D vs 2D vs 3D numerical proof, feature weighting, CV selection |
| **Extra Material** | `Normalization-standardization.pdf.pdf` | 2 Pages | Invariant algorithms vs sensitive algorithms, outlier compression in min-max |
| **Extra Material** | `Classifier_Evaluation_Confusion_Matrix_Solution.pdf` | 5 Pages | 10,000-patient cancer diagnosis case study, High Accuracy Paradox, Precision vs Recall trade-offs |
| **Extra Material** | `Confustion-Matrix-Classification_Metrics_Step_by_Step_5_Examples.pdf` | 12 Pages | 5 Detailed classification examples with definitions, substitutions, interpretations |
| **QP & Solutions** | `decision_tree_5_questions_final.pdf` & `...answers.pdf` | 12 Pages | 5 Comprehensive exam questions (Customer purchase, Employee attrition, Play tennis, Loan approval with continuous age, Loan default) |
| **QP & Solutions** | `Classification_Metrics_Numerical_Questions.pdf` & `...Solutions.pdf` | 18 Pages | 10 Exam numericals (Fraud detection, Medical screening, Model A vs B, Cybersecurity imbalance, Reverse calculations, Threshold comparisons, Quality control, Phishing FPR/FNR, Algebraic reconstruction) |

---

## 2. Master List of ALL Exam-Pattern Question Types (Grouped by Module)

Below is the definitive catalog of all question types appearing across university practice sets, quizzes, and comprehensive examinations, including their typical marks, given parameters, and required solving steps.

### Module 1: Introduction & Designing a Learning System
1. **Mitchell $(T, P, E)$ Triplet Formulation** [2–4 Marks]:
   - *Given*: Real-world application description (autonomous vehicle, checkers player, medical diagnosis, loan approval, spam filter).
   - *Required*: Exact formulation of Task $T$, Performance measure $P$, and Training Experience $E$.
   - *Faculty Tip*: Always specify operational telemetry/data types for $E$ and quantifiable objective metrics for $P$.
2. **Train / Test / Validation Sizing** [2 Marks]:
   - *Given*: Dataset size $N$, split ratios (e.g., 80/20, 70/15/15).
   - *Required*: Number of training, validation, and test instances.
   - *Formula*: $N_{\text{train}} = \text{ratio} \times N$. Verification: $\sum N_i = N$.
3. **$k$-Fold Cross-Validation Sizing** [2–4 Marks]:
   - *Given*: Dataset size $N$, fold count $k$.
   - *Required*: Fold validation size ($N/k$), training size per run ($N - N/k$), total models trained ($k$).
4. **Accuracy to Error Count Conversion** [2 Marks]:
   - *Given*: Overall accuracy percentage, total test count $N$.
   - *Formula*: $\text{Errors} = (1 - \text{Accuracy}) \times N$.
5. **Least Mean Squares (LMS) Weight Update in Checkers System** [4–6 Marks]:
   - *Given*: Current board state features $\mathbf{x}$, current weights $\mathbf{w}$, target value $V_{\text{train}}(b)$ from Critic, learning rate $\eta$.
   - *Derivation*: Minimize squared error $\mathcal{L} = \frac{1}{2}(V_{\text{train}} - \hat{V})^2$.
   - *Update*: $w_j \leftarrow w_j + \eta(V_{\text{train}}(b) - \hat{V}(b))x_j$.

### Module 2: Mathematical Preliminaries, Preprocessing & Quality
6. **Vectorization & Shape Tracking** [2–4 Marks]:
   - *Given*: Feature vector/matrix $X$, weight vector $W$, bias $b$.
   - *Required*: Compute dot product $X \cdot W$, linear affine prediction $y = XW + b$, or evaluate dimensions for $N$ observations and $d$ features ($(N \times d) \times (d \times 1) + (1,) = (N \times 1)$).
7. **Min-Max Feature Normalization (Standard & Custom Range)** [2–4 Marks]:
   - *Standard $[0, 1]$*: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$.
   - *Custom Range $[a, b]$*: $x' = a + (b - a)\frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (e.g., scaling to $[0, 5]$).
8. **Z-Score Feature Standardization** [2–4 Marks]:
   - *Formula*: $z = \frac{x - \mu}{\sigma}$.
   - *Faculty Tip*: State the sign and meaning in words (e.g., "1.6 standard deviations above the mean").
9. **Tukey's Outlier Detection using the $1.5 \times \text{IQR}$ Rule** [4 Marks]:
   - *Given*: First quartile $Q_1$, third quartile $Q_3$, candidate observation $x$.
   - *Formulas*: $\text{IQR} = Q_3 - Q_1$, $\text{Lower Bound} = Q_1 - 1.5\text{IQR}$, $\text{Upper Bound} = Q_3 + 1.5\text{IQR}$.
   - *Faculty Tip*: Calculate BOTH bounds explicitly even if checking only one side to avoid losing partial credit.
10. **Missing Value Imputation (Mean Imputation)** [2 Marks]:
    - *Given*: Column array with missing cells (e.g., `[12, 15, NA, 18, 15]`).
    - *Formula*: $\text{Mean} = \frac{\sum \text{Observed}}{\text{Count of Observed}}$ (never include NA in denominator count).
11. **Deduplication Impact on Dataset Quality** [2 Marks]:
    - *Formula*: $\text{Remaining} = N - \text{Duplicates}$, $\text{Fraction} = \frac{\text{Duplicates}}{N}$.
12. **Algorithmic Requirements for Feature Scaling** [4–6 Marks]:
    - *Invariant Algorithms*: Decision Trees, Random Forests, XGBoost (splits depend only on rank order).
    - *Strictly Required*: $k$-NN, SVM, Ridge/Lasso, Gradient Descent, Neural Networks.
13. **Sample Covariance Matrix & PSD Proof** [6 Marks]:
    - *Formula*: $\mathbf{\Sigma} = \frac{1}{N}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$. Proof: $\mathbf{v}^T\mathbf{\Sigma}\mathbf{v} = \frac{1}{N}\|\tilde{\mathbf{X}}\mathbf{v}\|^2 \ge 0$.

### Module 3: Linear Models for Regression & Regularization
14. **Simple Bivariate Linear Regression (Summation Formulas)** [4–6 Marks]:
    - *Formulas*: Slope $m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2} = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sum(x - \bar{x})^2}$, Intercept $b = \bar{y} - m\bar{x}$.
    - *Prediction*: $\hat{y} = mx + b$.
15. **Ordinary Least Squares (OLS) Normal Equations (Matrix Form)** [6 Marks]:
    - *Matrix Setup*: $\mathbf{X} \in \mathbb{R}^{N \times 2}$ (with leading column of 1s), $\mathbf{y} \in \mathbb{R}^{N \times 1}$.
    - *Formulas*: $\mathbf{X}^T\mathbf{X}$, $\mathbf{X}^T\mathbf{y}$, $(\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{\det}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$, $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$.
16. **Cost Function (Mean Squared Error & SSE)** [2–4 Marks]:
    - *Formulas*: $\text{SSE} = \sum (y_i - \hat{y}_i)^2$, $\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$, $J(\theta) = \frac{1}{2m}\sum (\hat{y}_i - y_i)^2$.
    - *Theoretical Question*: Why divide by $2m$? (The factor of 2 cancels when taking the derivative $\frac{d}{dx}x^2 = 2x$, simplifying gradient updates).
17. **One Step of Gradient Descent Update (Single Point SGD)** [4 Marks]:
    - *Model*: $\hat{y} = wx + b$. Error: $e = \hat{y} - y$.
    - *Gradients*: $\frac{\partial J}{\partial w} = e \cdot x$, $\frac{\partial J}{\partial b} = e$.
    - *Updates*: $w \leftarrow w - \alpha \frac{\partial J}{\partial w}$, $b \leftarrow b - \alpha \frac{\partial J}{\partial b}$.
18. **One Step of Batch Gradient Descent (Multiple Points)** [6 Marks]:
    - *Gradients*: $\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^m (\hat{y}_i - y_i)x_i$, $\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^m (\hat{y}_i - y_i)$.
19. **Multi-Variable Gradient Descent (Clinical/Economic Models)** [6 Marks]:
    - Multiple features (e.g., Systolic, Diastolic, BMI, Waist $\to$ Relative Risk).
    - Simultaneous update across $\theta_0, \theta_1, \dots, \theta_d$ with small learning rates (e.g., $\alpha = 10^{-5}$).
20. **Learning Rate Behavior Analysis** [4 Marks]:
    - Small $\alpha$: Slow, steady convergence.
    - Large $\alpha$: Overshooting, oscillation, or catastrophic divergence.
21. **Goodness of Fit ($R^2$ Score)** [2–4 Marks]:
    - *Formula*: $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Interpretation: Percentage of variance explained.
22. **Bias vs. Variance Diagnosis from Train/Test Errors** [4 Marks]:
    - High Train Error + High Test Error $\implies$ High Bias (Underfitting).
    - Low Train Error + High Test Error (large gap) $\implies$ High Variance (Overfitting).
23. **Ridge Regression ($L_2$) Closed-Form Solution & Invertibility Proof** [6 Marks]:
    - *Derivation*: $\mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$.
    - *Proof*: $\mathbf{X}^T\mathbf{X} \succeq 0 \implies \mu_i \ge 0 \implies \lambda_i = \mu_i + \lambda > 0$ for $\lambda > 0$.
24. **Ridge & Lasso Cost Function Computations** [4 Marks]:
    - *Ridge*: $J(\theta) = \frac{\text{SSE}}{2m} + \frac{\lambda}{2m}\sum \theta_j^2$ or $\text{MSE} + \lambda \sum w_j^2$.
    - *Lasso*: $J(\theta) = \frac{\text{SSE}}{2m} + \frac{\lambda}{2m}\sum |\theta_j|$ or $\text{MSE} + \lambda \sum |w_j|$.
25. **Elastic Net Cost Function** [4 Marks]:
    - *Formula*: $J(\theta) = \frac{\text{SSE}}{2m} + \frac{\lambda}{2m}\left[ \alpha \sum |\theta_j| + (1 - \alpha)\sum \theta_j^2 \right]$.
26. **Regularized Gradient Descent Parameter Updates** [4 Marks]:
    - *L2 Update (Weight Decay)*: $w \leftarrow w - \eta\left( \frac{\partial \mathcal{L}}{\partial w} + 2\lambda w \right) = w(1 - 2\eta\lambda) - \eta \frac{\partial \mathcal{L}}{\partial w}$.
    - *L1 Update*: $w \leftarrow w - \eta\left( \frac{\partial \mathcal{L}}{\partial w} + \lambda \cdot \text{sign}(w) \right)$.

### Module 4: Linear Classification & Logistic Regression
27. **Odds, Log-Odds & Logit Transformation** [2–4 Marks]:
    - *Formulas*: $\text{Odds} = \frac{p}{1 - p}$, $\text{Logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T\mathbf{x} + b$.
28. **Sigmoid Probability & Decision Boundary Classification** [4 Marks]:
    - *Logit*: $z = \mathbf{w}^T\mathbf{x} + b$. Probability: $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$.
    - *Rule*: If $z \ge 0 \iff \hat{y} \ge 0.5$, predict class 1; else class 0.
29. **Binary Cross-Entropy (Log-Loss) Asymmetric Penalty** [4 Marks]:
    - *Formula*: $\mathcal{L} = -[y\ln \hat{y} + (1 - y)\ln(1 - \hat{y})]$.
    - Compare $\mathcal{L}$ for correctly-leaning prediction ($\hat{y} = 0.8, y=1 \implies 0.223$) vs. confidently wrong prediction ($\hat{y} = 0.1, y=1 \implies 2.303$).
30. **Derivative of Standard Logistic Sigmoid** [4–6 Marks]:
    - *Proof*: $\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$.
31. **Derivation of Log-Loss Gradient w.r.t. Weights** [6 Marks]:
    - *Multivariate Chain Rule*: $\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$.
32. **One Step of Gradient Descent for Logistic Regression** [4–6 Marks]:
    - $z = \mathbf{w}^T\mathbf{x}$, $\hat{y} = \sigma(z)$, $e = \hat{y} - y$, $\mathbf{w}^{(1)} = \mathbf{w}^{(0)} - \eta \cdot e \cdot \mathbf{x}$.
33. **Multi-Class Softmax Regression & Logit Invariance** [4–6 Marks]:
    - *Formula*: $P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum e^{z_j}}$.
    - *Proof*: Invariant to adding constant $c$: $\frac{e^{z_k+c}}{\sum e^{z_j+c}} = \frac{e^{z_k}}{\sum e^{z_j}}$.
34. **Fisher Linear Discriminant Analysis (LDA) Derivation** [6 Marks]:
    - Between-class scatter $\mathbf{S}_B$, within-class scatter $\mathbf{S}_W$.
    - Criterion: $J(\mathbf{w}) = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}} \implies \mathbf{w} \propto \mathbf{S}_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$.

### Module 5: Decision Trees & Information Theory
35. **Shannon Entropy & Node Impurity Calculation** [2–4 Marks]:
    - *Formula*: $H(S) = -\sum_{i=1}^C p_i \log_2 p_i$. Pure: $H=0$. 50/50: $H=1.0$.
36. **Classification Error Impurity Metric** [2 Marks]:
    - *Formula*: $I_E(S) = 1 - \max_i(p_i)$.
37. **Gini Impurity (CART)** [2–4 Marks]:
    - *Formula*: $I_G(S) = 1 - \sum_{i=1}^C p_i^2$. Binary maximum: $0.50$.
38. **Information Gain (ID3 Algorithm)** [4–6 Marks]:
    - *Formula*: $\text{Gain}(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|}H(S_v)$. Select attribute with maximum Gain.
39. **Weighted Gini Index for Candidate Splits (CART)** [4–6 Marks]:
    - *Formula*: $I_G(S, A) = \sum_{v} \frac{|S_v|}{|S|}I_G(S_v)$. Select split with MINIMUM weighted Gini.
40. **Gain Ratio (C4.5 Algorithm)** [4–6 Marks]:
    - *Formulas*: $\text{SplitInfo}(S, A) = -\sum \frac{|S_v|}{|S|}\log_2 \frac{|S_v|}{|S|}$, $\text{GainRatio} = \frac{\text{Gain}}{\text{SplitInfo}}$.
    - Explanation: Penalizes high-cardinality attributes.
41. **Continuous Attribute Binary Splitting (Thresholding)** [6 Marks]:
    - Sort attribute values: $a_1, a_2, \dots, a_m$.
    - Identify midpoints $c_j = \frac{a_j + a_{j+1}}{2}$ ONLY where target labels change.
    - Evaluate Information Gain or Gini Index for binary splits $A \le c_j$ vs. $A > c_j$.
42. **Multi-Way vs. Binary Splits on Continuous Attributes** [4 Marks]:
    - Categorical features can split multi-way into all categories; continuous features must be split into binary half-spaces.
43. **Decision Tree Pruning Mechanics (Pre-Pruning vs. Post-Pruning)** [4–6 Marks]:
    - *Pre-Pruning*: Stop splitting if Gain $< \theta_{\min}$ or sample count $< n_{\min}$.
    - *Post-Pruning*: Grow full tree, prune subtrees using held-out validation set.
44. **Minimum Description Length (MDL) Principle for Decision Trees** [6 Marks]:
    - *Formula*: $\text{MDL}(h) = L_{C_1}(h) + L_{C_2}(D \mid h)$.
    - Calculate bits for tree structure ($L(h) = \text{nodes} \times \text{bits}$) and bits for misclassified exceptions ($L(D \mid h) = \text{errors} \times \text{bits/error}$).
    - Select tree with minimum total bits, proving Occam's razor quantitatively.

### Module 6: Instance-Based Learning ($k$-NN & LWR)
45. **Distance Metrics (Euclidean, Manhattan, Minkowski, Mahalanobis)** [4 Marks]:
    - $d_{\text{Euclidean}} = \sqrt{\sum (x_i - z_i)^2}$, $d_{\text{Manhattan}} = \sum |x_i - z_i|$.
    - $d_{\text{Mahalanobis}} = \sqrt{(\mathbf{x} - \mathbf{z})^T\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{z})}$.
46. **Problem Type 1: Standard $k$-NN with Distance Weighting** [6 Marks]:
    - Calculate distances from query $\mathbf{x}_q$ to all training instances.
    - Evaluate unweighted majority voting across $K=3, 5, 7$.
    - Apply inverse distance squared weights $w_i = 1/d_i^2$, showing how distance weighting prevents distant points from flipping the decision.
47. **Problem Type 2: Mixed Data Type Distance-Weighted $k$-NN** [6 Marks]:
    - Continuous features normalized to $[0, 1]$.
    - Ordinal categories mapped to ranks and scaled to $[0, 1]$ via $(\text{rank} - 1)/(R_{\max} - 1)$.
    - Nominal categorical features evaluated via mismatch penalty divided by count of nominal features:
      $$d = \sqrt{(\Delta \text{Numeric})^2 + (\Delta \text{Ordinal})^2} + \frac{\text{Mismatches}}{k_{\text{nom}}}$$
    - Weighted voting with $w = 1/d^2$ across all instances.
48. **Problem Type 3: Locally Weighted Regression (LWR) Prediction + GD Step** [6 Marks]:
    - Identify $k$-nearest neighbors of query $x_q$.
    - Compute kernel weights $K_i = 1/d_i^2$.
    - Compute weighted local prediction: $\hat{y} = \frac{\sum K_i y_i}{\sum K_i}$.
    - Execute one step of local gradient descent on initial line $y = w_0 + w_1 x$:
      $$\Delta w_0 = \eta \sum K_i (y_i - \hat{y}_i), \qquad \Delta w_1 = \eta \sum K_i (y_i - \hat{y}_i) x_i$$
49. **Influence of Scaling on $k$-NN vs. Regression GD Convergence** [4–6 Marks]:
    - Min-Max scaling preserves Euclidean distance ranking and $k$-NN predictions, but prevents gradient descent from diverging violently.
50. **The Curse of Dimensionality & Asymptotic Distance Limit** [4–6 Marks]:
    - Mathematical limit: $\lim_{d \to \infty}\frac{d_{\max} - d_{\min}}{d_{\min}} = 0$.
    - Numerical walkthrough: How adding noise attributes (shoe size, favorite number) inverts 1D distance rankings.
    - Dual remedies: Attribute weighting ($w_j^2 \Delta x_j^2$) and cross-validation feature selection.
51. **Cover-Hart Theorem Asymptotic Error Bounds** [4–6 Marks]:
    - $R^* \le R_{1-\text{NN}} \le 2R^*(1 - R^*) \le 2R^*$.

### Module 7: Support Vector Machines
52. **Geometric Margin Derivation from Canonical Hyperplanes** [4–6 Marks]:
    - $\mathbf{w}^T\mathbf{x}^+ + b = +1, \mathbf{w}^T\mathbf{x}^- + b = -1 \implies \text{Margin } \gamma = \frac{2}{\|\mathbf{w}\|}$.
53. **Maximum Margin Hyperplane Calculation from Support Vectors** [6 Marks]:
    - Solve for $\mathbf{w}$ and $b$ using support vector equations $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$. Classify test points.
54. **Soft-Margin Primal Optimization Problem Formulation** [6 Marks]:
    - $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum \xi_i$ subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 - \xi_i, \xi_i \ge 0$.
    - Detailed role of slack $\xi_i$ and penalty hyperparameter $C$ (bias-variance trade-off).
55. **KKT Complementary Slackness Conditions** [4–6 Marks]:
    - $\alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 + \xi_i] = 0$. Proves only support vectors define boundary.
56. **Mercer's Theorem & The Kernel Trick** [6 Marks]:
    - Gram matrix $\mathbf{K} \succeq 0$. How Gaussian RBF kernel maps to infinite dimensions without explicit projection.

### Module 8: Bayesian Learning
57. **Bayes' Theorem & MLE vs. MAP Derivation** [4–6 Marks]:
    - Derivation proving Gaussian noise MLE yields Ordinary Least Squares (MSE).
    - Derivation proving Gaussian zero-mean prior yields Ridge Regression ($L_2$).
58. **Bayes Optimal Classifier Formulation** [4–6 Marks]:
    - $v_{\text{BOC}} = \arg\max_{v_j} \sum_{h_i \in H} P(v_j \mid h_i) P(h_i \mid D)$.
59. **Naïve Bayes Classification Numerical** [6 Marks]:
    - Class priors $P(C_j)$, feature conditional probabilities $\prod P(x_k \mid C_j)$, normalization, and class prediction.
60. **The Zero-Probability Problem & Laplace (Add-1) Smoothing** [4 Marks]:
    - $\hat{P}(x_k = c \mid v_j) = \frac{N_{jk} + 1}{N_j + K_k}$.

### Module 9: Ensemble Learning
61. **Condorcet's Jury Theorem Binomial Expansion** [4–6 Marks]:
    - Analytical proof that majority voting converges to $1.0$ as ensemble size $M \to \infty$ if base accuracy $p > 0.5$.
62. **Ensemble Variance Reduction Formula** [4–6 Marks]:
    - $\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{M}\sigma^2$. How Random Forest feature bagging reduces $\rho$.
63. **Out-of-Bag (OOB) Generalization Error Derivation** [4 Marks]:
    - Probability of sample omission: $\lim_{N \to \infty}(1 - 1/N)^N = e^{-1} \approx 36.8\%$.
64. **AdaBoost Exponential Loss Classifier Weight Derivation** [6 Marks]:
    - Minimize $\sum \exp(-y_i F_m(\mathbf{x}_i)) \implies \alpha_m = \frac{1}{2}\ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$.
65. **AdaBoost Round-by-Round Weight Updates** [6 Marks]:
    - Compute weighted error $\epsilon_m$, $\alpha_m$, unnormalized weights $w_i \exp(\pm \alpha_m)$, normalization factor $Z_m$, and updated weights.
66. **XGBoost 2nd-Order Taylor Expansion & Optimal Leaf Weights** [6 Marks]:
    - $w^* = -\frac{G}{H + \lambda}$, $\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma$.

### Module 10: Unsupervised Learning & Clustering
67. **$k$-Means WCSS Objective & Coordinate Descent Convergence Proof** [6 Marks]:
    - Distortion measure $J = \sum \sum r_{ik}\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$. Proof of finite step termination.
68. **$k$-Means Step-by-Step Iteration Numerical** [6 Marks]:
    - Assign points to nearest centroid, recompute mean centroids, compute inertia.
69. **$k$-Means++ Seeding Algorithm & Competitive Bound** [4–6 Marks]:
    - Sampling with $P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum D(\mathbf{x})^2}$; expected distortion bound $\mathcal{O}(\log K)J_{\text{opt}}$.
70. **Gaussian Mixture Models (GMM) EM Derivation** [6 Marks]:
    - E-step responsibilities $\gamma_{ik}$, M-step parameters ($\pi_k, \boldsymbol{\mu}_k, \mathbf{\Sigma}_k$), reduction to $k$-means when $\mathbf{\Sigma} \to 0$.
71. **Silhouette Coefficient Cluster Validation** [4 Marks]:
    - $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$. Interpretation of $-1, 0, +1$.

### Module 11: Classifier Evaluation, Comparison & Fairness
72. **Confusion Matrix Construction & Fundamental Metrics** [4–6 Marks]:
    - $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}}$, $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$, $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$, $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$, $\text{FPR} = 1 - \text{Specificity}$, $\text{FNR} = 1 - \text{Recall}$, $F_1 = \frac{2\text{PR}}{\text{P} + \text{R}}$.
73. **The High Accuracy Paradox in Imbalanced Classification** [4–6 Marks]:
    - Disease screening example ($N=10,000$, 300 positive): Model achieves 96.5% accuracy but misses 70% of positives (Recall 30%).
74. **Operational Trade-offs: When to Prioritize Precision vs. Recall** [4–6 Marks]:
    - Precision priority: Invasive follow-ups, scarce/expensive confirmatory tests ($10k PET scan), criminal justice, spam filtering.
    - Recall priority: Routine cancer screening, contagious epidemics, airport weapon detection, turbine fracture inspection.
75. **Threshold Shifting & Business Decision Trade-offs** [4–6 Marks]:
    - Evaluating model performance at $T_1$ vs. $T_2$ to optimize cost under unequal error consequences.
76. **Reverse Confusion Matrix Algebraic Reconstruction** [6 Marks]:
    - Given $N$, actual positives, Precision, and Recall (or Specificity), algebraically reconstruct TP, FP, FN, TN.
77. **ROC Curve Construction & Probabilistic Area Under Curve (ROC-AUC)** [6 Marks]:
    - Pairwise ranking comparisons: $\text{AUC} = P(\text{Score}(\mathbf{x}^+) > \text{Score}(\mathbf{x}^-))$.
78. **Statistical Comparison of Two Classifiers (McNemar's Test)** [6 Marks]:
    - Paired contingency table of discordant errors $e_{01}, e_{10}$; statistic $\chi^2 = \frac{(|e_{01} - e_{10}| - 1)^2}{e_{01} + e_{10}} \sim \chi^2_1$.
79. **Algorithmic Fairness: Demographic Parity vs. Equalized Odds** [4–6 Marks]:
    - Demographic Parity: $P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$.
    - Equalized Odds: Conditioning on ground truth $Y=y$.

---

## 3. Systematic Audit of the Existing 50-Question Answer Bank

### A. High-Yield Question Types Missing from the 50-QA Bank

| Missing Question Topic | Relevant Module | Source Document in Course Pack | Exam Weight / Importance |
| :--- | :---: | :--- | :---: |
| **Locally Weighted Regression (LWR) Numerical with Gradient Descent Update** | Module 6 | `KNN-BITS-Problem3.pdf`, `KNN-BITS-Additional Problem.pdf` | **6 Marks (High)**: Flagship BITS exam problem; combines local kernel weights with gradient updates. |
| **Mixed Data Type Distance-Weighted $k$-NN** | Module 6 | `KNN-BITS-Problme Type-1-2-doctor.pdf` | **6 Marks (High)**: Features Euclidean distance on scaled features + categorical mismatch penalty. |
| **Full CART Decision Tree Construction on Continuous Attributes (Defaulted Dataset)** | Module 5 | `1-4th Oct26-BITS-Gini Index-CART-Decision Tree.pdf` | **6 Marks (High)**: Standard multi-page exam problem testing continuous split midpoints and tree induction. |
| **Minimum Description Length (MDL) Bit Calculation Numerical** | Module 5 | `6-4th Oct 26-BITS-MDL Principle.pdf`, `13th Sep26 Practice Numericals (Q36)` | **6 Marks (Medium)**: Quantitative Occam's razor evaluation using leaf bits and error bits. |
| **Reverse Confusion Matrix Algebraic Reconstruction** | Module 11 | `Classification_Metrics_Solutions.pdf` (Q5, Q6, Q10) | **6 Marks (High)**: Solving for TP, FP, FN, TN given summary rates. |
| **In-Depth Operational Trade-off Scenarios (Precision vs. Recall)** | Module 11 | `Classifier_Evaluation_Confusion_Matrix_Solution.pdf` | **4–6 Marks (High)**: Tested in medical diagnostics, criminal justice, and security. |
| **Multi-Feature Gradient Descent Step (Heart Disease Relative Risk)** | Module 3 | `GD-Solved Numerical.pdf.pdf` | **6 Marks (Medium)**: 5-parameter gradient descent with small learning rate $\alpha = 10^{-5}$. |
| **Tukey's $1.5 \times \text{IQR}$ Outlier Detection & Data Cleansing Math** | Module 2 | `13th Sep26 Practice Numericals` (Q6, Q7, Q8) | **4 Marks (Medium)**: Core university numerical for data preprocessing. |
| **Elastic Net & Regularized Weight Decay Updates** | Module 3 | `regularization_numerical_questions.pdf`, `regularization_cost_function_5_questions.pdf` | **4–6 Marks (Medium)**: Elastic net cost and L1/L2 gradient descent updates. |
| **Matrix Vectorization & Shape Tracking** | Module 2 | `Vectorization_Questions.pdf` | **2–4 Marks (Medium)**: Matrix multiplication dimensions and vector affine calculations. |

### B. Questions in the 50-QA Bank Requiring Richer / Corrected Solutions

1. **Q13 & Q39 (Decision Trees & Impurity)**:
   - *Current State*: Q13 computes simple node Gini/Entropy; Q39 computes Information Gain for one attribute on Play Tennis.
   - *Enhancement Needed*: Add the step-by-step CART Gini derivation for continuous attributes from `1-4th Oct26-BITS-Gini Index-CART-Decision Tree.pdf`, showing why sorted midpoint thresholds at class transitions are evaluated.
2. **Q40 ($k$-NN Distance-Weighted Voting)**:
   - *Current State*: Solves a 3-point toy problem.
   - *Enhancement Needed*: Include the 8-point Problem Type 1 from `KNN-BITS-Problme Type-1-2-doctor.pdf`, explicitly demonstrating that at $K=7$ unweighted voting flips from Positive to Negative, while distance-weighted voting ($1/d^2$) remains stably Positive.
3. **Q45 (Confusion Matrix Evaluation)**:
   - *Current State*: Simple forward calculations from a given matrix.
   - *Enhancement Needed*: Augment with the **Accuracy Paradox** analysis (showing that 96.5% accuracy can conceal a 70% missed cancer rate) and include FPR ($1 - \text{Specificity}$) and FNR ($1 - \text{Recall}$).
4. **Q37 (Ridge Regression Parameter Calculation)**:
   - *Current State*: Applies regularization to slope $w_1$ only.
   - *Enhancement Needed*: Clarify the distinction between standard Ridge (penalizing only weights $w_1, \dots, w_d$) and regularizing all parameters, and provide the regularized gradient descent update formula ($w \leftarrow w(1 - 2\eta\lambda) - \eta \nabla \mathcal{L}_0$).

---

## 4. Specific Formulas Missing from the Master Formula Sheet

The following mathematical formulas and operational expressions appear prominently in course practice sets and lecture handouts but are currently absent from `Master_Formula_Sheet_and_Study_Guide.md`:

| Formula Name | Mathematical Expression | Context / Module | Exam Application |
| :--- | :--- | :---: | :--- |
| **Tukey's IQR Outlier Rule** | $\text{IQR} = Q_3 - Q_1$, $[Q_1 - 1.5\text{IQR}, \ Q_3 + 1.5\text{IQR}]$ | Module 2 | Detecting valid ranges vs outliers |
| **Min-Max Scaling to Custom Range $[a, b]$** | $x' = a + (b - a)\frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Module 2 / 6 | Scaling features to ranges like $[0, 5]$ |
| **Bivariate OLS Slope Formula** | $m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2} = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sum(x - \bar{x})^2}$ | Module 3 | Simple linear regression without matrices |
| **Bivariate OLS Intercept Formula** | $b = \bar{y} - m\bar{x}$ | Module 3 | Intercept from sample means |
| **Coefficient of Determination ($R^2$)** | $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Module 3 | Quantifying explained variance percentage |
| **Elastic Net Regularized Cost** | $J(\theta) = \frac{\text{SSE}}{2m} + \frac{\lambda}{2m}\left[ \alpha \sum |\theta_j| + (1 - \alpha)\sum \theta_j^2 \right]$ | Module 3 | Blended L1 and L2 regularization cost |
| **L2 Gradient Weight Decay Update** | $w \leftarrow w - \eta\left( \frac{\partial \mathcal{L}}{\partial w} + 2\lambda w \right) = w(1 - 2\eta\lambda) - \eta \frac{\partial \mathcal{L}}{\partial w}$ | Module 3 | Gradient descent step under Ridge penalty |
| **L1 Gradient Sign Update** | $w \leftarrow w - \eta\left( \frac{\partial \mathcal{L}}{\partial w} + \lambda \cdot \text{sign}(w) \right)$ | Module 3 | Gradient descent step under Lasso penalty |
| **Odds and Log-Odds (Logit)** | $\text{Odds} = \frac{p}{1 - p}, \quad \text{Logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T\mathbf{x} + b$ | Module 4 | Mapping probability to affine score |
| **Classification Error Impurity** | $I_E(S) = 1 - \max_i(p_i)$ | Module 5 | Alternative tree splitting/pruning criterion |
| **Continuous Split Midpoint Rule** | $c_j = \frac{a_j + a_{j+1}}{2} \quad \text{for } y(a_j) \neq y(a_{j+1})$ | Module 5 | Fayyad-Irani candidate threshold selection |
| **Minimum Description Length (MDL)** | $\text{MDL}(h) = L_{C_1}(h) + L_{C_2}(D \mid h)$ | Module 5 | Information-theoretic tree pruning criterion |
| **Attribute Bit Encoding Length** | $\text{Bits} \approx \log_2(\text{Number of Possibilities})$ | Module 5 | Coding cost per node/leaf in MDL |
| **Mixed-Attribute Distance (Prof. Pritee)** | $d = \sqrt{\sum_{\text{num, ord}} (\Delta x_j)^2} + \frac{\text{mismatches}_{\text{nom}}}{k_{\text{nom}}}$ | Module 6 | Distance with continuous, ordinal, nominal features |
| **LWR Gradient Descent Weight Updates** | $\Delta w_0 = \eta \sum K_i(y_i - \hat{y}_i), \quad \Delta w_1 = \eta \sum K_i(y_i - \hat{y}_i)x_i$ | Module 6 | Local line gradient update in LWR |
| **False Positive Rate (FPR)** | $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{Specificity}$ | Module 11 | Type I error rate |
| **False Negative Rate (FNR)** | $\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = 1 - \text{Recall}$ | Module 11 | Type II error rate / Miss rate |
| **$k$-Fold Cross Validation Sizing** | $\text{Fold Size} = N/k, \quad \text{Train Size} = N - N/k$ | Module 1 | Partitioning dataset across cross-validation runs |

---

## 5. Critical Insights from Extra Material Not Fully Reflected in Notes

### 1. Minimum Description Length (MDL) Principle (`6-4th Oct 26-BITS-MDL Principle.pdf`)
- **Core Concept**: Connects Occam's razor to algorithmic information theory. A decision tree that memorizes noise might achieve 100% training accuracy ($L(D \mid h) = 0$), but its description length $L(h)$ explodes.
- **Explicit Bit Calculation**:
  - Suppose describing each leaf costs 2 bits, and each split costs 1 bit.
  - Describing each misclassified exception costs $\log_2 N$ (or fixed 2 bits in classroom examples).
  - *Tree A (Simple)*: 3 leaves, 1 split $\implies L(h) = 2(3) + 1 = 7$ bits. Makes 3 errors $\implies L(D \mid h) = 3 \times 2 = 6$ bits. $\text{Total MDL} = 7 + 6 = 13$ bits.
  - *Tree B (Complex)*: 7 leaves, 6 splits $\implies L(h) = 15$ bits. Makes 1 error $\implies L(D \mid h) = 1 \times 2 = 2$ bits. $\text{Total MDL} = 15 + 2 = 17$ bits.
  - *Conclusion*: Tree A is chosen by MDL despite having 3 errors versus Tree B's 1 error.

### 2. Multi-Way vs. Binary Splits on Continuous Attributes (`3-4th Oct26-BITS-Multiway-binarysplit.pdf`)
- A categorical feature with $V$ values can naturally branch into $V$ child nodes (multi-way split).
- A continuous feature has an infinite number of possible values. Therefore, decision trees **must** discretize continuous attributes into binary predicates ($A < c$ vs. $A \ge c$).
- Candidate thresholds $c$ must be midpoints between adjacent distinct values where the target label changes. Midpoints between identical labels can never maximize Information Gain (proven by Fayyad & Irani).

### 3. Curse of Dimensionality Concrete Case Study (`KNN-CURSE-slide-34.pdf`)
- Prof. Pritee presents an everyday educational case study: predicting whether a student will Pass/Fail.
  - True predictive attribute: Study Hours.
  - Noise attributes: Shoe Size, Favorite Number.
  - Student A (Query): Study 8h, Shoe 6, Fav 3.
  - Student B (True peer): Study 8.5h, Shoe 10, Fav 9 $\to$ Pass.
  - Student C (Impostor): Study 2h, Shoe 6, Fav 4 $\to$ Fail.
  - In 1D (Study Hours only): $d(A, B) = 0.5 < d(A, C) = 6.0 \implies$ Predict Pass (Correct).
  - In 2D (+ Shoe Size): $d(A, B) = 4.03 < d(A, C) = 6.00 \implies$ Gap shrinks.
  - In 3D (+ Fav Number): $d(A, B) = 7.23 > d(A, C) = 6.08 \implies$ Predict Fail (Wrong!).
- **Dual Remedies**:
  1. *Attribute Weighting*: Multiply squared differences by $w_j^2$ (e.g., $w_{\text{study}} = 1.0, w_{\text{noise}} = 0.01$).
  2. *Attribute Elimination*: Use cross-validation to drop features that degrade validation performance.

### 4. Cancer Diagnostic Case Study & The High Accuracy Paradox (`Classifier_Evaluation_Confusion_Matrix_Solution.pdf`)
- Given $N = 10,000$ patients, 300 actual cancer, 9,700 healthy.
  - Model: $\text{TP} = 90, \text{FN} = 210, \text{FP} = 140, \text{TN} = 9,560$.
  - Accuracy: $\frac{90 + 9560}{10000} = 96.50\%$.
  - Recall (Sensitivity): $\frac{90}{300} = 30.00\%$ (Catastrophic: 70% of cancer patients missed!).
  - Precision: $\frac{90}{230} = 39.13\%$ (Over 60% false alarms).
- **Four Scenarios Requiring High Precision**:
  1. High-risk, invasive, or irreversible follow-up interventions (e.g., unnecessary organ resection, major surgery, intensive chemotherapy).
  2. Extremely expensive or scarce confirmatory diagnostic procedures (e.g., $10,000 PET scans or rare specialist biopsies).
  3. Severe psychological trauma, depression, or social stigma associated with false alarms.
  4. Non-medical analogs: Spam filtering (never delete legitimate emails) and criminal prosecution ("better that ten guilty escape than one innocent suffer").
- **Four Scenarios Requiring High Recall**:
  1. Routine, low-cost screening programs for life-threatening illnesses (mammography, Pap smears).
  2. Highly contagious epidemic outbreaks (COVID-19, Ebola, Tuberculosis) where missed positives trigger super-spreader events.
  3. Aviation and critical infrastructure security (airport weapons screening).
  4. Critical industrial turbine/machinery fracture monitoring before catastrophic failure.

### 5. Multi-Feature Gradient Descent Optimization (`GD-Solved Numerical.pdf.pdf`)
- Predicts Cardiovascular Relative Risk ($Y$) using 4 physiological features:
  - $X_1$: Systolic BP (140, 120, 130)
  - $X_2$: Diastolic BP (80, 80, 100)
  - $X_3$: BMI (35, 25, 30)
  - $X_4$: Waist circumference (100, 80, 60)
  - True risk $Y$: 1.81, 1.22, 1.71
- Illustrates vector matrix formulation across 5 parameters $\theta = [\theta_0, \theta_1, \theta_2, \theta_3, \theta_4]^T$, initialized at zero.
- Shows how raw feature magnitudes (e.g., BP 140 vs. BMI 35) generate wildly uneven gradients ($\frac{\partial J}{\partial \theta_1} = -207.36$ vs. $\frac{\partial J}{\partial \theta_0} = -1.58$), necessitating a tiny learning rate $\alpha = 10^{-5}$ to prevent gradient explosion—providing concrete numerical justification for feature standardization.

---

## 6. Actionable Recommendations for Course Notes & QA Bank

1. **Add Part VIII to `50_Exam_Question_Answer_Bank.md`**:
   - Expand the question bank from 50 to 60+ questions by adding the 10 missing high-yield exam archetypes (LWR with gradient descent, mixed-type KNN, full CART continuous tree, MDL bit calculations, reverse confusion matrix algebraic reconstruction, and clinical risk multi-variable gradient descent).
2. **Update `Master_Formula_Sheet_and_Study_Guide.md`**:
   - Insert the 18 missing formulas identified in Section 4 of this report into the Master Reference Table.
   - Add a dedicated subsection on "Operational Rules & Exam Shortcuts" covering Tukey's IQR bounds, learning rate stability bounds, and the 4 Precision vs. Recall decision rules.
3. **Enhance Module Notes**:
   - In `Module_05_Decision_Trees_and_Information_Theory.md`, include the MDL bit-calculation example and continuous threshold derivation.
   - In `Module_06_Instance_Based_Learning.md`, add Problem Type 1, 2, and 3 explicit walkthroughs.
   - In `Module_11_Machine_Learning_Model_Evaluation_and_Comparison.md`, add the Cancer Case Study and the 4-part Precision vs Recall operational framework.

---
*Audit completed by resources-auditor on 2026-11-16.*
