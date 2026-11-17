# AIML ZG565: Machine Learning — Master Formula Sheet & Study Guide

**Course**: AIML ZG565 — Machine Learning  
**Target Audience**: Mid-Semester & Comprehensive Examination Preparation, Viva / Oral Defense, Quick Revision  
**Primary Teacher Followed**: Prof. Pritee Narendra Raole (BITS Pilani WILP)

---

## 1. Master Formula Reference Table

| Module & Concept | Canonical Mathematical Formulation | Key Parameters & Dimensions | Operational Bounds & Remarks |
| :--- | :--- | :--- | :--- |
| **M1: Learning Criterion** | $P(T, E_{\text{new}}) > P(T, E_{\text{old}})$ | $T$: Task, $P$: Performance, $E$: Experience | Mitchell's formal operational definition |
| **M1: Arthur Samuel (1959)** | Learning without explicit programming | Foundational definition | Lecture Session 2 |
| **M2: Min-Max Scaling** | $x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $x_{\min}, x_{\max} \in \mathbb{R}$ | Bound: $x_{\text{norm}} \in [0, 1]$; sensitive to outliers |
| **M2: Z-Score Standardization** | $z = \frac{x - \mu}{\sigma}$ | $\mu \in \mathbb{R}, \sigma > 0$ | $\mu_z = 0, \sigma_z = 1$; preserves outlier presence |
| **M2: Tukey IQR Outliers** | $\text{IQR}=Q_3-Q_1$; bounds $Q_1-1.5\,\text{IQR},\ Q_3+1.5\,\text{IQR}$ | Quartiles from sorted data | Exam: flag points outside both bounds |
| **M2: Min-Max to $[a,b]$** | $x' = a + (b-a)\frac{x-x_{\min}}{x_{\max}-x_{\min}}$ | Custom range scaling | Used in mixed-type $k$-NN practice |
| **M3: Bivariate OLS Slope** | $m=\frac{\sum(x-\bar{x})(y-\bar{y})}{\sum(x-\bar{x})^2}$ | Simple regression | Meal-bill / tip numerics |
| **M3: $R^2$** | $R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Explained variance | Goodness of fit |
| **M3: Elastic Net Cost** | $J=\frac{\text{SSE}}{2m}+\frac{\lambda}{2m}\left[\alpha\sum|\theta_j|+(1-\alpha)\sum\theta_j^2\right]$ | $\alpha \in [0,1]$ | Blended L1/L2 |
| **M3: L2 Weight Decay GD** | $w\leftarrow w(1-2\eta\lambda)-\eta\frac{\partial\mathcal{L}_0}{\partial w}$ | Ridge gradient step | Exam regularization updates |
| **M5: Classification Error** | $I_E(S)=1-\max_i p_i$ | Tree impurity | Session 8 comparison |
| **M5: Continuous Split** | $c_j=\frac{a_j+a_{j+1}}{2}$ where labels change | CART thresholds | Income $c=97.5$ example |
| **M6: LWR GD Update** | $\Delta w_0=\eta\sum K_i(y_i-\hat{y}_i)$; $\Delta w_1=\eta\sum K_i(y_i-\hat{y}_i)x_i$ | $K_i=1/d_i^2$ common | BITS Problem Type 3 |
| **M6: Mixed-attribute distance** | $d=\sqrt{\sum(\Delta x_j)^2}+\frac{\text{nom mismatches}}{k_{\text{nom}}}$ | Scaled num/ord + nominal | Pritee KNN sheets |
| **M11: FPR / FNR** | $\text{FPR}=\frac{FP}{FP+TN}$; $\text{FNR}=\frac{FN}{FN+TP}$ | Type I / II rates | Cancer paradox worksheet |
| **M1: $k$-Fold Sizing** | Fold $=N/k$; train $=N-N/k$ | CV partition | Intro module numerics |
| **M2: Sample Covariance** | $\text{Cov}(X, Y) = \frac{1}{N} \sum_{i=1}^N (x_i - \mu_x)(y_i - \mu_y)$ | $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$ | Positive semi-definite matrix |
| **M2: Shannon Entropy** | $H(X) = - \sum_{k=1}^K p_k \log_2 p_k$ | $\sum p_k = 1, p_k \ge 0$ | Bound: $H(X) \in [0, \log_2 K]$; units: bits |
| **M2: KL Divergence** | $D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log \left(\frac{P(x)}{Q(x)}\right)$ | $P, Q$: probability distributions | Non-negative: $D_{\text{KL}} \ge 0$; asymmetric |
| **M3: OLS Normal Equation** | $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ | $\mathbf{X} \in \mathbb{R}^{N \times d}, \mathbf{y} \in \mathbb{R}^N, \mathbf{w} \in \mathbb{R}^d$ | Requires $\text{rank}(\mathbf{X}) = d$; complexity $\mathcal{O}(d^3 + Nd^2)$ |
| **M3: Ridge ($L_2$) Solution** | $\mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$ | $\lambda \ge 0$ (regularization strength) | Guaranteed invertible for all $\lambda > 0$; shrinkage |
| **M3: Lasso ($L_1$) Soft-Threshold** | $w_j^* = \text{sign}(w_j^{\text{OLS}}) \max(\|w_j^{\text{OLS}}\| - \lambda, 0)$ | $\lambda \ge 0$ (for orthonormal $\mathbf{X}$) | Induces exact parameter sparsity ($w_j = 0$) |
| **M3: Bias-Variance Decomposition**| $\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Variance} + \sigma_\epsilon^2$ | $\sigma_\epsilon^2$: irreducible noise variance | Fundamental generalization trade-off |
| **M4: Logistic Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$ | $z = \mathbf{w}^T \mathbf{x} + w_0 \in \mathbb{R}$ | Symmetry: $\sigma(-z) = 1 - \sigma(z)$; $\sigma' = \sigma(1-\sigma)$ |
| **M4: Binary Cross-Entropy** | $\mathcal{L}(\mathbf{w}) = - \sum_{i=1}^N [y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i)]$ | $y_i \in \{0, 1\}, \hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$ | Strictly convex; Hessian $\mathbf{H} = \mathbf{X}^T \mathbf{R} \mathbf{X} \succeq 0$ |
| **M4: Softmax Function** | $P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$ | $\mathbf{W} \in \mathbb{R}^{K \times d}, k \in \{1, \dots, K\}$ | $\sum_{k=1}^K P(y=k \mid \mathbf{x}) = 1.0$ |
| **M4: Fisher LDA Projection** | $\mathbf{w} \propto \mathbf{S}_W^{-1} (\mathbf{m}_1 - \mathbf{m}_2)$ | $\mathbf{S}_W$: within-class scatter matrix | Maximizes between-to-within scatter ratio |
| **M5: Gini Impurity (CART)** | $I_G(S) = 1 - \sum_{i=1}^C p_i^2$ | $C$: number of classes | Bound for binary: $I_G \in [0, 0.50]$ |
| **M5: Information Gain** | $\text{Gain}(S, A) = H(S) - \sum_{v} \frac{\|S_v\|}{\|S\|} H(S_v)$ | $A$: candidate splitting attribute | Expected entropy reduction |
| **M5: Gain Ratio (C4.5)** | $\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$ | $\text{SplitInfo} = - \sum \frac{\|S_v\|}{\|S\|} \log_2 \frac{\|S_v\|}{\|S\|}$ | Penalizes high-cardinality attributes |
| **M6: Minkowski Distance** | $d_p(\mathbf{x}, \mathbf{z}) = \left( \sum_{j=1}^d \|x_j - z_j\|^p \right)^{1/p}$ | $p \ge 1$ ($p=1$ Manhattan, $p=2$ Euclidean) | Metric space axiom compliance |
| **M6: Mahalanobis Distance** | $d_M(\mathbf{x}, \mathbf{z}) = \sqrt{(\mathbf{x} - \mathbf{z})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{z})}$ | $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$: covariance matrix | Invariant to scale and feature correlation |
| **M6: Cover-Hart Theorem** | $R^* \le R_{1-\text{NN}} \le 2 R^* (1 - R^*) \le 2 R^*$ | $R^*$: Bayes optimal minimum risk | 1-NN asymptotic error is at most $2 \times$ Bayes risk |
| **M6: LWR Kernel Weight** | $K(\mathbf{x}_i, \mathbf{x}_q) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_q\|_2^2}{2\tau^2}\right)$ | $\tau > 0$: bandwidth hyperparameter | Localized weighting around query $\mathbf{x}_q$ |
| **M7: Geometric Margin** | $\text{Margin } \gamma = \frac{2}{\|\mathbf{w}\|_2}$ | $\mathbf{w} \in \mathbb{R}^d$: hyperplane normal vector | Derived from canonical planes $\mathbf{w}^T \mathbf{x} + b = \pm 1$ |
| **M7: Soft-Margin Primal SVM**| $\min \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \xi_i$ | $\xi_i \ge 0$ (slack), $C > 0$ (penalty) | Trade-off: margin width vs. slack violations |
| **M7: Dual SVM Objective** | $\max_{\boldsymbol{\alpha}} \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$ | $0 \le \alpha_i \le C, \ \sum \alpha_i y_i = 0$ | Convex quadratic program; KKT complementary slackness |
| **M7: Gaussian RBF Kernel** | $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|_2^2)$ | $\gamma = \frac{1}{2\sigma^2} > 0$ | Infinite-dimensional Hilbert feature space |
| **M8: Bayes' Theorem** | $P(h \mid D) = \frac{P(D \mid h) P(h)}{P(D)}$ | Likelihood $\times$ Prior $/$ Evidence | Foundation of probabilistic inductive inference |
| **M8: MAP Hypothesis** | $h_{\text{MAP}} = \arg\max_{h} P(D \mid h) P(h)$ | Discards evidence $P(D)$ | Maximum A Posteriori estimator |
| **M8: Naïve Bayes Decision** | $v_{\text{NB}} = \arg\max_{v_j} P(v_j) \prod_{k=1}^d P(x_k \mid v_j)$ | Assumes conditional independence | Avoids exponential parameter explosion |
| **M8: Laplace Smoothing** | $\hat{P}(x_k = c \mid v_j) = \frac{N_{jk} + 1}{N_j + K_k}$ | $K_k$: distinct values of feature $k$ | Prevents zero-probability product collapse |
| **M9: Condorcet Jury Theorem** | $P_{\text{ens}} = \sum_{k=\lceil M/2 \rceil}^M \binom{M}{k} p^k (1-p)^{M-k}$ | $p > 0.5$: independent base accuracy | Convergence: $P_{\text{ens}} \to 1.0$ as $M \to \infty$ |
| **M9: Ensemble Variance** | $\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{M} \sigma^2$ | $\rho$: pairwise inter-tree correlation | RF feature bagging reduces correlation $\rho$ |
| **M9: OOB Sample Probability** | $P(\text{OOB}) = (1 - \frac{1}{N})^N \xrightarrow{N \to \infty} \frac{1}{e} \approx 36.8\%$ | $N$: training dataset sample size | Inherent cross-validation in Bagging/RF |
| **M9: AdaBoost Classifier Weight**| $\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)$ | $\epsilon_m$: weighted error rate $\in [0, 0.5)$ | High weight for low error; $\alpha_m \to \infty$ as $\epsilon_m \to 0$ |
| **M9: AdaBoost Weight Update** | $w_i^{(m+1)} = \frac{w_i^{(m)} \exp(-\alpha_m y_i h_m(\mathbf{x}_i))}{Z_m}$ | $Z_m$: normalization constant | Boosts weights of misclassified observations |
| **M9: XGBoost Split Gain** | $\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma$ | $G = \sum g_i$ (grad), $H = \sum h_i$ (Hessian) | 2nd-order Taylor expansion with tree regularization |
| **M10: $k$-Means Objective (WCSS)**| $J = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$ | $r_{ik} \in \{0, 1\}, \ \sum_k r_{ik} = 1$ | Alternating coordinate descent (Lloyd's algorithm) |
| **M10: $k$-Means++ Probability** | $P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum D(\mathbf{x})^2}$ | $D(\mathbf{x}_i)$: distance to nearest centroid | Provably guarantees $\mathcal{O}(\log K)$ competitive bound |
| **M10: GMM Responsibilities (E-step)**| $\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \mathbf{\Sigma}_j)}$ | $\sum_k \gamma_{ik} = 1, \gamma_{ik} \in [0, 1]$ | Posterior probability component $k$ generated point $i$ |
| **M10: Silhouette Coefficient** | $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ | $a$: intra-cluster dist, $b$: nearest cluster dist | Bound: $s(i) \in [-1, +1]$; $+1$ indicates optimal cluster |
| **M11: Precision & Recall** | $\text{Prec} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \ \text{Rec} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ | $\text{TP}, \text{FP}, \text{FN} \in \mathbb{N}$ | Precision penalizes false alarms; Recall penalizes misses |
| **M11: Harmonic $F_1$-Score** | $F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ | Range: $[0, 1]$ | Balanced single-number classification metric |
| **M11: McNemar's $\chi^2$ Statistic** | $\chi^2 = \frac{(\|e_{01} - e_{10}\| - 1)^2}{e_{01} + e_{10}}$ | $e_{01}, e_{10}$: discordant model error counts | Tests statistical significance of classifier difference |

---

## 2. High-Yield Conceptual Viva & Oral Examination Q&A

### Module 1 & 2: Foundations & Preprocessing
**Q1: Why does minimizing Mean Squared Error (MSE) correspond to finding the Maximum Likelihood Estimate under Gaussian noise?**  
*A*: **Because the negative log-likelihood of a univariate Gaussian distribution expands into a quadratic squared difference.** When assuming $y = f(\mathbf{x}) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the joint likelihood is $L \propto \prod \exp\left(-\frac{(y_i - f(\mathbf{x}_i))^2}{2\sigma^2}\right)$. Taking the negative natural logarithm yields $-\ln L = \text{const} + \frac{1}{2\sigma^2} \sum (y_i - f(\mathbf{x}_i))^2$. Maximizing likelihood is mathematically identical to minimizing $\sum (y_i - f(\mathbf{x}_i))^2$.

**Q2: What is Data Leakage in feature scaling, and how does it compromise test validation?**  
*A*: **Data leakage occurs when parameters from the test set ($\mu_{\text{test}}, \sigma_{\text{test}}, \min, \max$) are inadvertently used to normalize training data.** This leaks test distribution properties into training, causing the model to learn signals that do not exist in real-world deployment. To prevent this, transformers must execute `.fit()` exclusively on the training split, and apply `.transform()` blindly to validation and test splits.

---

### Module 3 & 4: Linear & Logistic Models
**Q3: Why is Ridge Regression $(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}$ guaranteed to be invertible even if $\mathbf{X}^T\mathbf{X}$ is singular?**  
*A*: **Because adding $\lambda \mathbf{I}$ shifts all eigenvalues strictly into the positive real domain.** The Gram matrix $\mathbf{X}^T\mathbf{X}$ is positive semi-definite with eigenvalues $\mu_i \ge 0$. Adding $\lambda \mathbf{I}$ ($\lambda > 0$) shifts every eigenvalue to $\lambda_i = \mu_i + \lambda \ge \lambda > 0$. Since all eigenvalues are strictly positive, the determinant $\det(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I}) = \prod \lambda_i > 0$, guaranteeing a non-singular, well-conditioned inverse.

**Q4: Why does Lasso produce exact feature sparsity ($w_j = 0$) while Ridge only shrinks weights towards zero?**  
*A*: **Due to the non-differentiable sharp corners (kinks) of the $L_1$ diamond constraint boundary at the coordinate axes.** In 2D, the level curves of the quadratic loss function first intersect the $L_1$ constraint region at a vertex lying directly on an axis (where $w_1 = 0$ or $w_2 = 0$). In contrast, the $L_2$ constraint is a smooth differentiable sphere, where tangency almost never occurs precisely on a coordinate axis.

---

### Module 5 & 6: Trees & Instance-Based Learning
**Q5: Why is Information Gain mathematically biased towards high-cardinality attributes, and how does C4.5 correct this?**  
*A*: **Information Gain rewards pure partitions regardless of whether the split generalizes.** An attribute like `Social_Security_Number` splits $N$ instances into $N$ singleton pure leaves ($H = 0$), producing maximum Information Gain. Quinlan's C4.5 introduces **Gain Ratio**, dividing Gain by **Split Information** ($-\sum \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$). High-cardinality attributes generate massive split entropy, heavily penalizing the Gain Ratio.

**Q6: What is the Cover-Hart Theorem, and why is it considered remarkable?**  
*A*: **Cover & Hart proved that the 1-Nearest Neighbor classifier has an asymptotic error rate bounded by at most twice the optimal Bayes error rate ($R_{1-\text{NN}} \le 2R^*$).** This is remarkable because the 1-NN rule requires zero knowledge of the underlying probability distributions, yet guarantees asymptotic performance within a factor of 2 of the theoretically perfect Bayes classifier.

---

### Module 7 & 8: Support Vector Machines & Bayesian Learning
**Q7: In Support Vector Machines, what is the geometric role of the Karush-Kuhn-Tucker (KKT) Complementary Slackness condition $\alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$?**  
*A*: **It proves that the decision boundary is determined exclusively by Support Vectors.** If a data point lies strictly outside the margin ($y_i(\mathbf{w}^T\mathbf{x}_i + b) > 1$), the bracket is positive, forcing $\alpha_i = 0$. Since $\mathbf{w} = \sum \alpha_i y_i \mathbf{x}_i$, points with $\alpha_i = 0$ exert zero influence on the model. Only points lying directly on the margin satisfy the bracket $= 0$ and have $\alpha_i > 0$.

**Q8: What is the Zero-Probability Problem in Naïve Bayes, and how does Laplace Smoothing fix it?**  
*A*: **Because Naïve Bayes computes class scores via a product chain $\prod P(x_k \mid C)$, a single unobserved feature value ($P(x_k \mid C) = 0$) zeroes out the entire class probability.** Laplace smoothing adds a uniform pseudo-count $\alpha = 1$ to the numerator and $\alpha K$ to the denominator: $\hat{P} = \frac{N_{jk} + 1}{N_j + K}$, guaranteeing all probabilities remain strictly positive.

---

### Module 9 & 10: Ensembles & Clustering
**Q9: How does Random Forest de-correlate individual trees compared to standard Bagging?**  
*A*: **By restricting the candidate split features at every node to a random subset of size $m \approx \sqrt{d}$.** In standard Bagging, if one feature has dominant predictive power, nearly every tree selects it as the root split, making all trees highly correlated ($\rho \to 1$). By forbidding that feature from being considered in many split subsets, Random Forests forces individual trees to discover alternative orthogonal predictive paths, reducing $\rho$ and slashing ensemble variance $\rho \sigma^2$.

**Q10: Why does $k$-Means clustering fail on concentric circles or non-convex manifolds?**  
*A*: **Because $k$-Means measures distance using the Euclidean metric, which partitions space into convex polyhedral Voronoi cells.** Concentric rings require non-convex, manifold-following decision boundaries. To cluster manifold data, one must use **DBSCAN** (density reachability) or **Spectral Clustering** (graph Laplacian eigenvectors).

---

## 3. Operational Rules & Exam Shortcuts

| Rule | Detail |
| :--- | :--- |
| **IQR outliers** | Always compute lower **and** upper Tukey bounds |
| **GD on logistic/regression** | Simultaneous weight update; scale features first |
| **SMOTE / scaling** | Fit only on training split — never before split |
| **Trees** | No feature scaling; continuous splits = binary thresholds at label-change midpoints |
| **Precision vs recall** | Screening / epidemics → recall; invasive follow-up / spam → precision |
| **Ensembles (webinar)** | Tune `n_estimators`, `max_depth` with CV; read Gini MDI for features |

---

### Module 11: Model Evaluation & Comparison
**Q11: When is the Precision-Recall (PR) Curve superior to the Receiver Operating Characteristic (ROC) Curve?**  
*A*: **On heavily imbalanced datasets where the negative class dominates the positive class.** In ROC curves, the False Positive Rate denominator is $\text{TN} + \text{FP}$. When $\text{TN} = 1,000,000$, adding 10,000 false positives barely changes FPR ($\approx 0.01$), making the ROC curve look deceptively flawless. In contrast, Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) includes FP directly against TP in the denominator, causing the PR curve to plunge and revealing severe model degradation.

