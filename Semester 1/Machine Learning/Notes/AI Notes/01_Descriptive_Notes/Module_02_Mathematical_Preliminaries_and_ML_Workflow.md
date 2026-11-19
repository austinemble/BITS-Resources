# Module 2: Mathematical Preliminaries & Machine Learning Workflow

**Course**: AIML ZG565 — Machine Learning  
**Prescribed References**: Christopher M. Bishop (R1, Chapters 1 & 2); Tan et al. (R2, Chapters 2 & 3); Lecture Notes  
**Syllabus module**: **M02** — Mathematical Preliminaries & ML Workflow (`module_map.yaml`)  
**Canonical video**: `Videos/Lecture_02_Mathematical_Preliminaries_and_ML_Workflow.mp4`  
**Contact sessions** (reference): Sessions 3–4; CS2 deck (Prof. Pritee).

---

## 1. Executive Summary & Foundational Definitions

Machine learning algorithms are fundamentally computational realizations of applied mathematics. Operating on real-world entities (images, acoustic waves, genomic sequences, tabular databases) requires encoding observations into abstract geometrical objects: high-dimensional real vector spaces $\mathbb{R}^d$.

To reason rigorously about optimization landscapes, loss minimization, geometric margins, and probabilistic certainty, machine learning rests upon four mathematical pillars:
1. **Linear Algebra**: Represents high-dimensional data, linear transformations, coordinate projections, and spectral decompositions.
2. **Multivariate Calculus & Matrix Analysis**: Supplies the mathematical engine for gradient descent, loss landscape curvature (Hessian), and parameter optimization.
3. **Probability & Decision Theory**: Quantifies epistemic and aleatoric uncertainty, formalizes risk minimization, and governs Bayesian inference.
4. **Information Theory**: Quantifies informational entropy, surprise, divergence between distributions, and coding complexity.

```
                    ┌─────────────────────────────────┐
                    │  Data Space: Features x ∈ ℝᵈ    │
                    └────────────────┬────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
┌─────────────────────────┐                     ┌─────────────────────────┐
│ Linear Transformations  │                     │ Optimization Analysis   │
│ A x = λ x (Spectral)    │                     │ ∇ f(x) = 0, H ⪰ 0       │
└────────────┬────────────┘                     └────────────┬────────────┘
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Machine Learning Workflow:      │
                    │ Ingestion ──> Preprocessing ──> │
                    │ Training ──> Validation         │
                    └─────────────────────────────────┘
```

---

## 2. Lecture-first: data types, quality & preprocessing (Sessions 3–4)

### 2.0 Five-stage ML pipeline (lecture)

1. Domain understanding & problem formulation  
2. Data collection, integration, cleansing  
3. Preprocessing & feature engineering  
4. Model training & optimization  
5. Evaluation, deployment, monitoring  

### 2.0.1 Stevens' measurement scales

| Scale | Operations | Central tendency | Examples |
| :--- | :--- | :--- | :--- |
| **Nominal** | $=, \neq$ | Mode | ID, eye colour, zip code |
| **Ordinal** | order comparisons | Median | service rating, grades |
| **Interval** | $+, -$ (no true zero) | Mean, SD | °C, calendar dates |
| **Ratio** | all arithmetic | geometric/harmonic mean | age, salary, weight, K |

**Bank customer attribute typing (exam)**: Name/Gender → nominal; Service rating → ordinal; Credit score → interval; Income band → ordinal; Region → nominal.

### 2.0.2 Dataset characteristics & data types

- **Dimensionality**, **sparsity**, **resolution**, **volume** (curse of dimensionality when $d$ grows).  
- **Structural types**: relational/record, transactional, document (bag-of-words), graph, spatial, temporal/sequence.

### 2.0.3 Outlier detection (exam numerics)

**Tukey IQR**: sort data; $Q_1$, $Q_3$, $\text{IQR} = Q_3 - Q_1$; bounds $[Q_1 - 1.5\,\text{IQR},\ Q_3 + 1.5\,\text{IQR}]$.

*Worked example*: $[10, 12, 11, 15, 11, 14, 13, 17, 12, 22, 14, 11]$ → sorted, $Q_1=11$, $Q_3=14.5$, IQR $=3.5$, upper bound $=19.75$ → **outlier: 22**.

**Three-sigma rule**: legitimate range $[\mu - 3\sigma,\ \mu + 3\sigma]$ (e.g. $\mu=50,\ \sigma=5$ → $[35, 65]$).

### 2.0.4 Sampling, scaling, encoding

- Simple random vs **stratified** sampling (preserve class proportions).  
- Min–max: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$; custom $[a,b]$: $x' = a + (b-a)\frac{x - x_{\min}}{x_{\max} - x_{\min}}$.  
- Z-score: $z = (x-\mu)/\sigma$.  
- **Equal-width** vs **equal-frequency** binning; **one-hot** (nominal) vs **label** encoding (ordinal).  

### 2.0.5 Inductive learning hypothesis (Mitchell)

> Any hypothesis that approximates the target well on a large enough training set will approximate it well on unobserved examples.

Hypothesis as attribute constraints, e.g. $\langle \text{Sunny}, \text{Warm}, ?, \text{Strong}, ?, ? \rangle$.

### 2.0.6 Vectorization & shape tracking (exam)

For $N$ examples, $d$ features: design matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$, weights $\mathbf{w} \in \mathbb{R}^{d}$, bias $b \in \mathbb{R}$.

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1} \in \mathbb{R}^{N}, \quad \mathbf{X}\mathbf{w} \in \mathbb{R}^{N}$$

### 2.0.7 Algorithms and feature scaling

| Scaling needed? | Examples |
| :--- | :--- |
| **No** (rank-invariant) | Decision trees, random forests, gradient boosting |
| **Yes** | $k$-NN, SVM, ridge/lasso, neural nets, batch gradient descent |

---

## 2. (Reference) Mathematical preliminaries

*Items below (SVD, full Hessian calculus, Box-Cox, MICE) are **external prerequisites / advanced reference** — not derived in live lectures.*

### 2.1 Linear Algebra & Vector Spaces

Let $\mathbf{x} \in \mathbb{R}^d$ denote a column vector of features:
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix}, \quad \mathbf{x}^T = [x_1, x_2, \dots, x_d]$$

- **Inner (Dot) Product**: The Euclidean inner product defines geometric projection and cosine similarity:
  $$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{j=1}^d u_j v_j = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos \theta$$
- **Vector $L_p$ Norms**: Quantify magnitude and enforce regularization penalties:
  $$\|\mathbf{x}\|_1 = \sum_{j=1}^d |x_j| \quad (\text{Lasso / Sparsity inducing})$$
  $$\|\mathbf{x}\|_2 = \sqrt{\sum_{j=1}^d x_j^2} = \sqrt{\mathbf{x}^T \mathbf{x}} \quad (\text{Ridge / Euclidean length})$$
  $$\|\mathbf{x}\|_\infty = \max_{1 \le j \le d} |x_j|$$
- **Matrix Decompositions**:
  - **Eigen-decomposition**: For a symmetric positive semi-definite matrix $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$:
    $$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i \implies \mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$$
    where $\mathbf{V}$ is an orthogonal matrix of eigenvectors ($\mathbf{V}^T\mathbf{V} = \mathbf{I}$) and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_d)$.
  - **Singular Value Decomposition (SVD)**: For an arbitrary matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$:
    $$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

---

### 2.2 Multivariate Calculus & Matrix Gradients

In optimizing scalar loss functions $\mathcal{L}(\boldsymbol{\theta})$ over parameter vectors $\boldsymbol{\theta} \in \mathbb{R}^d$:

- **The Gradient Vector**: Direction of steepest ascent:
  $$\nabla_{\boldsymbol{\theta}} \mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial \theta_1}, \frac{\partial \mathcal{L}}{\partial \theta_2}, \dots, \frac{\partial \mathcal{L}}{\partial \theta_d} \right]^T \in \mathbb{R}^d$$
- **The Hessian Matrix**: Second-order partial derivatives describing curvature:
  $$\mathbf{H} = \nabla_{\boldsymbol{\theta}}^2 \mathcal{L} \in \mathbb{R}^{d \times d}, \quad H_{jk} = \frac{\partial^2 \mathcal{L}}{\partial \theta_j \partial \theta_k}$$
  - If $\mathbf{H} \succ 0$ (positive definite: $\mathbf{z}^T \mathbf{H} \mathbf{z} > 0 \ \forall \mathbf{z} \neq \mathbf{0}$) at stationary point $\nabla \mathcal{L} = \mathbf{0}$, $\boldsymbol{\theta}^*$ is a **strict local minimum**.
  - If $\mathbf{H}$ has both positive and negative eigenvalues, $\boldsymbol{\theta}^*$ is a **saddle point**.

#### Canonical Matrix Derivatives:
$$\nabla_{\mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$$
$$\nabla_{\mathbf{x}} (\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{x} \quad (\text{or } 2\mathbf{A}\mathbf{x} \text{ if } \mathbf{A} \text{ is symmetric})$$

---

### 2.3 Probability Theory & Bayes Rule

Let $X$ and $Y$ be random variables.
- **Joint and Conditional Probability**:
  $$P(X=x, Y=y) = P(Y=y \mid X=x) P(X=x)$$
- **Bayes' Theorem**:
  $$P(Y=c \mid \mathbf{X}=\mathbf{x}) = \frac{P(\mathbf{X}=\mathbf{x} \mid Y=c) P(Y=c)}{P(\mathbf{X}=\mathbf{x})} = \frac{P(\mathbf{X}=\mathbf{x} \mid Y=c) P(Y=c)}{\sum_{k} P(\mathbf{X}=\mathbf{x} \mid Y=k) P(Y=k)}$$
  $$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$
- **Expectation, Covariance, and Correlation**:
  $$\mathbb{E}[X] = \mu_X = \int_{-\infty}^{\infty} x p(x) dx$$
  $$\text{Var}(X) = \sigma_X^2 = \mathbb{E}[(X - \mu_X)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
  $$\mathbf{\Sigma} = \text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T] \in \mathbb{R}^{d \times d}$$

---

### 2.4 Information Theory Foundations

- **Shannon Entropy**: Measures average uncertainty / unpredictability of a discrete random variable $X$:
  $$H(X) = - \sum_{k=1}^K P(x_k) \log_2 P(x_k)$$
- **Kullback-Leibler (KL) Divergence**: Measures information lost when approximating true distribution $P$ by $Q$:
  $$D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right) \ge 0$$
- **Cross-Entropy**: Standard loss function for multi-class classification:
  $$H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q) = - \sum_{x} P(x) \log Q(x)$$

---

## 3. The End-to-End Machine Learning Data Pipeline

![Discrete vs Continuous Attributes Lecture Slide](../assets/M02_data_types_attributes.jpg)
*Figure 2.1: Discrete vs Continuous Attributes slide — foundation of data preprocessing in ML (Aug 09 session).*

### 3.1 Data Preprocessing & Cleaning
1. **Handling Missing Values**:
   - Numerical: Median imputation (robust to outliers) or Mean imputation.
   - Categorical: Mode imputation or explicit `Missing` sentinel category.
   - Multivariate (reference only — not lectured): MICE or $k$-NN imputation.
2. **Feature Scaling**:
   - **Min-Max Normalization**: Scales values into a bounded interval $[0, 1]$:
     $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
     *Sensitive to extreme outliers which compress the non-outlier data.*
   - **Z-Score Standardization**: Centers data around $\mu = 0$ with unit variance $\sigma = 1$:
     $$x_{\text{standard}} = \frac{x - \mu}{\sigma}$$
     *Mandatory for distance-based algorithms (k-NN, SVM, K-Means) and gradient descent.*
3. **Data Skewness Removal** (lecture mentions log/sqrt; Box-Cox is **reference only**).
4. **Handling Class Imbalance**:
   - **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic instances along the line segments connecting $k$ minority nearest neighbors:
     $$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
   - **Random Under-Sampling / Tomek Links**: Removes majority class samples near decision boundaries.

---

## 4. Implementation note (exams are pencil-and-paper)

BITS mid-semester questions use **manual** preprocessing (IQR, min–max, z-score, mean imputation, stratified split counts). Fit every scaler or imputation statistic on **training data only**, then apply frozen parameters to validation and test.

*Optional Python reference (not tested):*

```python
# Illustrative only — CustomStandardScaler pattern
import numpy as np

class CustomStandardScaler:
    """Vectorized Z-Score Standardization transformer."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        # Prevent division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SyntheticSMOTE:
    """Simplified vectorized SMOTE for minority oversampling."""
    def __init__(self, k_neighbors: int = 3):
        self.k = k_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        minority_class = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
        X_min = X[y == minority_class]
        n_min = len(X_min)
        n_maj = len(X) - n_min
        n_synthetic_needed = n_maj - n_min

        if n_synthetic_needed <= 0 or n_min <= self.k:
            return X, y

        # Compute pairwise Euclidean distances among minority points
        dists = np.sqrt(np.sum((X_min[:, np.newaxis, :] - X_min[np.newaxis, :, :]) ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)

        # Get k nearest neighbor indices
        nn_indices = np.argsort(dists, axis=1)[:, :self.k]

        synthetic_samples = []
        for _ in range(n_synthetic_needed):
            idx = np.random.randint(0, n_min)
            neighbor_idx = np.random.choice(nn_indices[idx])
            diff = X_min[neighbor_idx] - X_min[idx]
            gap = np.random.uniform(0, 1)
            synthetic_samples.append(X_min[idx] + gap * diff)

        X_syn = np.array(synthetic_samples)
        y_syn = np.full(n_synthetic_needed, minority_class)

        X_resampled = np.vstack([X, X_syn])
        y_resampled = np.concatenate([y, y_syn])
        return X_resampled, y_resampled


if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic dataset with 2 features and severe class imbalance (90:10)
    X_maj = np.random.normal(loc=[5.0, 5.0], scale=1.0, size=(90, 2))
    X_min = np.random.normal(loc=[10.0, 10.0], scale=1.0, size=(10, 2))
    X = np.vstack([X_maj, X_min])
    y = np.array([0] * 90 + [1] * 10)

    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Scaled Feature Means:", np.mean(X_scaled, axis=0).round(4))
    print("Scaled Feature Stds:", np.std(X_scaled, axis=0).round(4))

    smote = SyntheticSMOTE(k_neighbors=3)
    X_bal, y_bal = smote.fit_resample(X_scaled, y)
    print(f"Original distribution: Class 0 = {np.sum(y == 0)}, Class 1 = {np.sum(y == 1)}")
    print(f"Resampled distribution: Class 0 = {np.sum(y_bal == 0)}, Class 1 = {np.sum(y_bal == 1)}")
```

---

## 5. Solved Numerical Examples (Exam Pattern)

### 5.0 Tukey IQR outlier detection (CS2 slide 32)

Data: $[10, 12, 11, 15, 11, 14, 13, 17, 12, 22, 14, 11]$.

Sorted ($N=12$): $Q_1=11$, $Q_3=14.5$, $\text{IQR}=3.5$. Bounds: $11-5.25=5.75$, $14.5+5.25=19.75$. **Outlier: 22**.

### 5.1 Mean imputation (practice numerics)

Column `[12, 15, \text{NA}, 18, 15]` → mean of observed $= (12+15+18+15)/4 = 15$ → imputed value **15** (NA excluded from denominator).

### 5.2 Scaling & covariance (5 students)

**Problem Statement (University Mid-Semester Exam Pattern, 6 Marks)**:  
Given the following raw observation table of 5 students with features:
- Feature 1 ($x_1$): Hours Studied ($[2, 4, 6, 8, 10]$)
- Feature 2 ($x_2$): Practice Tests Taken ($[1, 3, 2, 5, 4]$)

1. Compute the sample mean vector $\boldsymbol{\mu} = [\mu_1, \mu_2]^T$ and standard deviation vector $\boldsymbol{\sigma} = [\sigma_1, \sigma_2]^T$.
2. Compute the Z-score standardized matrix $\mathbf{Z}$.
3. Compute the Min-Max normalized matrix $\mathbf{X}_{\text{norm}}$ into range $[0, 1]$.
4. Calculate the empirical sample covariance $\text{Cov}(x_1, x_2)$ and correlation coefficient $r_{12}$.

---

### Step-by-Step Solution:

#### Step 1: Means and Standard Deviations
Sample size $N = 5$.

For $x_1 = [2, 4, 6, 8, 10]$:
$$\mu_1 = \frac{2 + 4 + 6 + 8 + 10}{5} = \frac{30}{5} = \mathbf{6.00}$$
$$\sigma_1^2 = \frac{\sum (x_{i1} - \mu_1)^2}{N} = \frac{(-4)^2 + (-2)^2 + 0^2 + 2^2 + 4^2}{5} = \frac{16 + 4 + 0 + 4 + 16}{5} = \frac{40}{5} = 8.00$$
$$\sigma_1 = \sqrt{8.00} \approx \mathbf{2.828}$$

For $x_2 = [1, 3, 2, 5, 4]$:
$$\mu_2 = \frac{1 + 3 + 2 + 5 + 4}{5} = \frac{15}{5} = \mathbf{3.00}$$
$$\sigma_2^2 = \frac{(-2)^2 + 0^2 + (-1)^2 + 2^2 + 1^2}{5} = \frac{4 + 0 + 1 + 4 + 1}{5} = \frac{10}{5} = 2.00$$
$$\sigma_2 = \sqrt{2.00} \approx \mathbf{1.414}$$

---

#### Step 2: Z-Score Standardized Matrix $\mathbf{Z}$
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

- Student 1:
  $$z_{11} = \frac{2 - 6}{2.828} = \frac{-4}{2.828} = -1.414, \quad z_{12} = \frac{1 - 3}{1.414} = \frac{-2}{1.414} = -1.414$$
- Student 2:
  $$z_{21} = \frac{4 - 6}{2.828} = -0.707, \quad z_{22} = \frac{3 - 3}{1.414} = 0.000$$
- Student 3:
  $$z_{31} = \frac{6 - 6}{2.828} = 0.000, \quad z_{32} = \frac{2 - 3}{1.414} = -0.707$$
- Student 4:
  $$z_{41} = \frac{8 - 6}{2.828} = +0.707, \quad z_{42} = \frac{5 - 3}{1.414} = +1.414$$
- Student 5:
  $$z_{51} = \frac{10 - 6}{2.828} = +1.414, \quad z_{52} = \frac{4 - 3}{1.414} = +0.707$$

Standardized matrix:
$$\mathbf{Z} = \begin{bmatrix} -1.414 & -1.414 \\ -0.707 & 0.000 \\ 0.000 & -0.707 \\ +0.707 & +1.414 \\ +1.414 & +0.707 \end{bmatrix}$$

---

#### Step 3: Min-Max Normalized Matrix $\mathbf{X}_{\text{norm}}$
- For $x_1$: $x_{\min} = 2, x_{\max} = 10 \implies \Delta_1 = 8$.
- For $x_2$: $x_{\min} = 1, x_{\max} = 5 \implies \Delta_2 = 4$.

$$x_{\text{norm}, 1} = \left[ \frac{2-2}{8}, \frac{4-2}{8}, \frac{6-2}{8}, \frac{8-2}{8}, \frac{10-2}{8} \right] = [\mathbf{0.00, 0.25, 0.50, 0.75, 1.00}]$$
$$x_{\text{norm}, 2} = \left[ \frac{1-1}{4}, \frac{3-1}{4}, \frac{2-1}{4}, \frac{5-1}{4}, \frac{4-1}{4} \right] = [\mathbf{0.00, 0.50, 0.25, 1.00, 0.75}]$$

---

#### Step 4: Sample Covariance and Correlation Coefficient
$$\text{Cov}(x_1, x_2) = \frac{1}{N} \sum_{i=1}^5 (x_{i1} - \mu_1)(x_{i2} - \mu_2)$$
$$\text{Cov}(x_1, x_2) = \frac{(-4)(-2) + (-2)(0) + (0)(-1) + (2)(2) + (4)(1)}{5} = \frac{8 + 0 + 0 + 4 + 4}{5} = \frac{16}{5} = \mathbf{3.20}$$

Pearson correlation coefficient:
$$r_{12} = \frac{\text{Cov}(x_1, x_2)}{\sigma_1 \sigma_2} = \frac{3.20}{2.828 \times 1.414} = \frac{3.20}{4.00} = \mathbf{+0.80}$$

*Strong positive linear correlation exists between hours studied and tests taken.*

---

## 6. Common Pitfalls, Trade-offs & Practical Guidelines

1. **Information Leakage via Unsupervised Preprocessing**:
   - Never compute min, max, mean, standard deviation, or imputation parameters on the entire dataset. Fit solely on `train_set` and apply the frozen statistics to `val_set` and `test_set`.
2. **MinMax vs. Standardization Decision Rule**:
   - Use **Min-Max** when features have strict bounded ranges (e.g., pixel intensities $[0, 255] \to [0, 1]$) or for neural network activation compatibility.
   - Use **Standardization** for algorithms assuming Gaussian error distributions, distance-based metrics (SVM, k-NN, PCA), or linear regression with $L_1/L_2$ regularization.
3. **Curse of Dimensionality in One-Hot Encoding**:
   - One-hot encoding high-cardinality categorical features (e.g., zip codes, product IDs with 10,000 levels) blows up feature dimensions, inducing extreme data sparsity. Use **Target Encoding**, **Weight of Evidence (WoE)**, or **Learned Embeddings**.

