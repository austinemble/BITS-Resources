# Module 4: Linear Models for Classification & Logistic Regression

**Course**: AIML ZG565 — Machine Learning  
**Prescribed References**: Christopher M. Bishop (R1, Chapter 4); Tan et al. (R2, Chapter 4)  
**Syllabus module**: **M04** — Linear Models for Classification & Logistic Regression (`module_map.yaml`)  
**Canonical videos**: `Lecture_04_Linear_Models_for_Classification.mp4`, `Lecture_05_Logistic_Regression_Logloss_Multiclass.mp4`  
**Contact sessions** (reference): Sessions 5–6.

---

## 1. Executive Summary & Foundational Definitions

Classification is the supervised learning task of mapping input vectors $\mathbf{x} \in \mathbb{R}^d$ to discrete categorical class labels $y \in \mathcal{C} = \{C_1, C_2, \dots, C_K\}$. Unlike regression where target outputs are continuous real numbers, classification decision boundaries partition the input space $\mathbb{R}^d$ into disjoint **decision regions** separated by $(d-1)$-dimensional manifolds known as **decision surfaces** or **hyperplanes**.

Linear classification models are characterized by decision boundaries that are linear functions of the input coordinates $\mathbf{x}$. When classes can be perfectly separated by a flat $(d-1)$-dimensional hyperplane, the data is termed **linearly separable**.

```
Input Vector x ∈ ℝᵈ ───> [ Linear Transformation z = wᵀx + b ] ───> [ Non-Linear Link σ(z) ] ───> Posterior P(y=1|x)
                                                                                  │
                                                                       Decision Threshold τ = 0.5
                                                                                  ▼
                                                                       Predicted Class ŷ ∈ {0, 1}
```

---

## 2. Theoretical Motivation & Core Principles

### 2.1 Discriminant Functions & Geometric Hyperplanes

A linear discriminant function computes a scalar activation score:

$$y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$$

Where $\mathbf{w}$ is the parameter weight vector and $w_0$ is the bias (or intercept).
- The decision boundary is the locus of all points where $y(\mathbf{x}) = 0$:
  $$\mathcal{S} = \{\mathbf{x} \in \mathbb{R}^d \mid \mathbf{w}^T \mathbf{x} + w_0 = 0\}$$
- **Orthogonality**: For any two points $\mathbf{x}_A, \mathbf{x}_B$ lying on the decision boundary $\mathcal{S}$:
  $$y(\mathbf{x}_A) = y(\mathbf{x}_B) = 0 \implies \mathbf{w}^T (\mathbf{x}_A - \mathbf{x}_B) = 0$$
  Therefore, the parameter vector $\mathbf{w}$ is strictly **orthogonal (normal)** to every vector lying within the decision boundary.
- **Signed Perpendicular Distance**: The normal distance $r$ from an arbitrary query point $\mathbf{x}$ to the decision hyperplane is given by:
  $$r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|_2} = \frac{\mathbf{w}^T \mathbf{x} + w_0}{\|\mathbf{w}\|_2}$$

---

### 2.2 Weight vector orthogonal to the decision boundary (exam proof)

Decision boundary: $\mathbf{w}^T \mathbf{x} + b = 0$. For any two points $\mathbf{x}_A, \mathbf{x}_B$ on the boundary, $\mathbf{w}^T \mathbf{x}_A + b = 0$ and $\mathbf{w}^T \mathbf{x}_B + b = 0$. Subtracting gives $\mathbf{w}^T(\mathbf{x}_A - \mathbf{x}_B) = 0$, so $\mathbf{w}$ is **perpendicular** to every vector parallel to the boundary plane. Signed distance from origin: $|b|/\|\mathbf{w}\|$.

> **Textbook supplement (not in BITS lectures 1–13)**: Fisher's LDA ($\mathbf{w} \propto \mathbf{S}_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$) — see Bishop R1; omitted from mid-sem focus.

---

### 2.3 Generative vs. Discriminative Paradigms

```
Generative Approach:
p(x|Cₖ), P(Cₖ) ───[ Bayes' Theorem ]───> P(Cₖ|x) = p(x|Cₖ)P(Cₖ) / ∑ p(x|Cⱼ)P(Cⱼ)
(Models full joint distribution P(x, y); e.g., Gaussian Discriminant Analysis, Naïve Bayes)

Discriminative Approach:
x ───[ Parametric Optimization ]───> P(Cₖ|x) directly
(Models conditional posterior P(y|x) directly; e.g., Logistic Regression, Neural Networks)
```

**Why Discriminative Modeling is Preferred for Tabular Classification**:  
As Vladimir Vapnik formulated: *"If you possess a restricted amount of data, solve the target problem directly and never solve a more general problem as an intermediate step."* Generative models waste modeling capacity estimating the input feature distribution $p(\mathbf{x})$, which can be extraordinarily complex, whereas the classification decision boundary itself may be simple.

---

### 2.4 Decision Theory: From Posterior to Decision

Classification has two stages (Bishop, *Pattern Recognition and Machine Learning*, §1.5):
1. **Inference**: estimate the posterior $P(C_k \mid \mathbf{x})$.
2. **Decision**: choose a class for each input $\mathbf{x}$.

The decision stage is where the cost of errors enters. In the 30 August 2026 lecture, Prof. Raole used disease screening to make the point: a false negative (a sick patient labelled healthy) and a false positive (a healthy patient flagged for follow-up) carry different consequences. The decision threshold should therefore depend on the application, not default to $0.5$.

**Loss matrix.** $L_{kj}$ is the cost of assigning an input whose true class is $C_k$ to class $C_j$. Correct decisions cost nothing, so $L_{kk} = 0$.

**Minimum expected loss rule.** Assign $\mathbf{x}$ to the class $j$ that minimizes the conditional risk
$$R_j(\mathbf{x}) = \sum_{k} L_{kj} \, P(C_k \mid \mathbf{x})$$
Averaging over the data gives $\mathbb{E}[L] = \int R_{j(\mathbf{x})}(\mathbf{x})\, p(\mathbf{x})\, d\mathbf{x}$, and choosing the minimum-risk class at each $\mathbf{x}$ minimizes this integral.

With 0/1 loss ($L_{kj} = 1$ for $k \neq j$), $R_j = 1 - P(C_j \mid \mathbf{x})$. The rule then reduces to picking the most probable class, which is the MAP decision from Module 8.

**Worked example (binary screening).** Classes are Sick ($S$) and Healthy ($H$). Costs: $L_{SH} = 10$ (missed disease), $L_{HS} = 1$ (unnecessary follow-up), and $L_{SS} = L_{HH} = 0$. For a patient with $p = P(S \mid \mathbf{x})$:
- Predict Healthy: $R_H = L_{SH}\,p = 10p$
- Predict Sick: $R_S = L_{HS}\,(1-p) = 1-p$

Predict Sick when $1 - p < 10p$, that is, when $p > \tfrac{1}{11} \approx 0.091$. Under these costs the threshold is about 9%, not 50%.

**Three routes to the same decision** (Bishop §1.5.4):
- **Generative**: model $p(\mathbf{x} \mid C_k)$ and the priors, then apply Bayes' theorem (Naïve Bayes, LDA).
- **Discriminative**: model $P(C_k \mid \mathbf{x})$ directly, then decide (logistic regression).
- **Discriminant function**: learn a function $f(\mathbf{x})$ that maps straight to a class label without estimating probabilities (perceptron, SVM).

Moving the threshold trades false negatives against false positives. That movement is the operating point on the ROC curve in Module 11.

**Regression.** Under squared-error loss, the optimal prediction is the conditional mean $\mathbb{E}[y \mid \mathbf{x}]$ (Bishop §1.5.5).

---

## 3. Logistic Regression Formulation

![Logistic Regression Introduction Whiteboard](../assets/M04_logistic_regression_whiteboard.jpg)
*Figure 4.1: Logistic Regression introduction — Y categorical values (Fraud/NotFraud, Pass/Fail), Binary, Multi-class and Ordinal classification types (Aug 23 session).*

### 3.1 The Log-Odds and the Sigmoid Function

In binary classification ($y \in \{0, 1\}$), we desire a probabilistic output $p(\mathbf{x}) = P(y = 1 \mid \mathbf{x}) \in [0, 1]$.  
The **odds ratio** is defined as $\frac{p(\mathbf{x})}{1 - p(\mathbf{x})} \in [0, \infty)$.  
Taking the natural logarithm yields the **logit** (log-odds), which maps to the entire real line $(-\infty, \infty)$:

$$\ln \left( \frac{P(y = 1 \mid \mathbf{x})}{1 - P(y = 1 \mid \mathbf{x})} \right) = \mathbf{w}^T \mathbf{x} + w_0$$

Solving for posterior probability $P(y = 1 \mid \mathbf{x})$:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + w_0) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + w_0)}}$$

Where $\sigma(z)$ is the **Standard Logistic Sigmoid Function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

#### Essential Mathematical Properties of $\sigma(z)$:
1. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$
2. **Derivative**:
   $$\frac{d\sigma}{dz} = \frac{d}{dz} (1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} (-e^{-z}) = \frac{1}{1 + e^{-z}} \left( \frac{e^{-z}}{1 + e^{-z}} \right) = \sigma(z)(1 - \sigma(z))$$

---

### 3.2 Maximum Likelihood Estimation & Binary Cross-Entropy Loss

Given independent training dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $y_i \in \{0, 1\}$, the target labels follow a Bernoulli distribution:
$$P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}, \quad \text{where } \hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$$

The Likelihood function is the joint product:
$$L(\mathbf{w}) = \prod_{i=1}^N P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \prod_{i=1}^N \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}$$

Taking the negative log-likelihood defines the **Binary Cross-Entropy Loss (Log-Loss)**:
$$\mathcal{L}(\mathbf{w}) = -\ln L(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

---

### 3.3 Complete Gradient Derivation via the Chain Rule

To minimize $\mathcal{L}(\mathbf{w})$ via gradient descent, compute $\frac{\partial \mathcal{L}}{\partial w_j}$ for parameter $w_j$:

Using the chain rule:
$$\frac{\partial \mathcal{L}_i}{\partial w_j} = \frac{\partial \mathcal{L}_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_j}$$

Where $z_i = \mathbf{w}^T \mathbf{x}_i = \sum_{k} w_k x_{ik}$.

1. **Loss derivative w.r.t prediction**:
   $$\frac{\partial \mathcal{L}_i}{\partial \hat{y}_i} = -\left[ \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right] = -\frac{y_i(1 - \hat{y}_i) - (1 - y_i)\hat{y}_i}{\hat{y}_i (1 - \hat{y}_i)} = \frac{\hat{y}_i - y_i}{\hat{y}_i (1 - \hat{y}_i)}$$
2. **Sigmoid derivative w.r.t linear logit $z_i$**:
   $$\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i (1 - \hat{y}_i)$$
3. **Linear logit derivative w.r.t weight $w_j$**:
   $$\frac{\partial z_i}{\partial w_j} = x_{ij}$$

Multiplying all three terms:
$$\frac{\partial \mathcal{L}_i}{\partial w_j} = \left[ \frac{\hat{y}_i - y_i}{\hat{y}_i (1 - \hat{y}_i)} \right] \cdot \left[ \hat{y}_i (1 - \hat{y}_i) \right] \cdot x_{ij} = (\hat{y}_i - y_i) x_{ij}$$

Summing over all $N$ training examples in vector matrix form:
$$\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) \in \mathbb{R}^M$$

> **Profound Architectural Insight**: The gradient of binary cross-entropy for logistic regression has the *identical mathematical structure* as the gradient of least-squares linear regression: $\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$! The only difference is that $\hat{\mathbf{y}} = \sigma(\mathbf{X}\mathbf{w})$ instead of $\mathbf{X}\mathbf{w}$.

---

### 3.4 Convexity and the Hessian Matrix

The second derivative matrix (Hessian) is:
$$\mathbf{H} = \nabla_{\mathbf{w}}^2 \mathcal{L}(\mathbf{w}) = \mathbf{X}^T \mathbf{R} \mathbf{X}$$

Where $\mathbf{R} \in \mathbb{R}^{N \times N}$ is a diagonal weighting matrix with elements:
$$R_{ii} = \hat{y}_i (1 - \hat{y}_i)$$

Since $0 < \hat{y}_i < 1$ for all real $z_i$, $R_{ii} > 0$ strictly. Therefore, for any non-zero vector $\mathbf{u} \in \mathbb{R}^M$:
$$\mathbf{u}^T \mathbf{H} \mathbf{u} = \mathbf{u}^T \mathbf{X}^T \mathbf{R} \mathbf{X} \mathbf{u} = (\mathbf{X}\mathbf{u})^T \mathbf{R} (\mathbf{X}\mathbf{u}) = \sum_{i=1}^N R_{ii} (\mathbf{x}_i^T \mathbf{u})^2 \ge 0$$

The Hessian $\mathbf{H}$ is strictly positive semi-definite everywhere. **Consequently, the cross-entropy loss landscape is strictly convex, guaranteeing that gradient descent cannot become trapped in non-global local minima.**

---

### 3.5 Multi-Class Generalization: Softmax Regression

For $K > 2$ mutually exclusive classes, the logit vector $\mathbf{z} = [\mathbf{w}_1^T \mathbf{x}, \dots, \mathbf{w}_K^T \mathbf{x}]^T$ is normalized via the **Softmax function**:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

The Categorical Cross-Entropy loss is:
$$\mathcal{L}(\mathbf{W}) = -\sum_{i=1}^N \sum_{k=1}^K y_{ik} \ln \hat{y}_{ik}$$

---

## 4. Implementation note

Mid-semester items: logit, sigmoid, one-step GD, confusion metrics — **no coding**. Optional reference:

```python
import numpy as np

class VectorizedLogisticRegression:
    """Vectorized Binary Logistic Regression with L2 Regularization."""
    def __init__(self, lr: float = 0.1, epochs: int = 1000, lambda_reg: float = 0.01):
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.w = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid to prevent overflow
        return np.where(z >= 0, 
                        1.0 / (1.0 + np.exp(-z)), 
                        np.exp(z) / (1.0 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        X_aug = np.hstack([np.ones((N, 1)), X])
        self.w = np.zeros(d + 1)

        loss_history = []
        for epoch in range(self.epochs):
            # Forward pass: logits & probabilities
            z = X_aug @ self.w
            y_hat = self._sigmoid(z)

            # Prevent log(0) numerical underflow
            eps = 1e-15
            y_hat_clipped = np.clip(y_hat, eps, 1.0 - eps)
            
            # Binary Cross Entropy Loss + L2 penalty (excluding bias w_0)
            bce = -np.mean(y * np.log(y_hat_clipped) + (1.0 - y) * np.log(1.0 - y_hat_clipped))
            reg_penalty = (self.lambda_reg / (2 * N)) * np.sum(self.w[1:] ** 2)
            loss_history.append(bce + reg_penalty)

            # Gradient: (1/N) * X^T (y_hat - y) + (lambda/N) * w
            grad = (X_aug.T @ (y_hat - y)) / N
            grad[1:] += (self.lambda_reg / N) * self.w[1:]

            # Parameter update
            self.w -= self.lr * grad

        return loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(X_aug @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    np.random.seed(42)
    # Generate 2D linearly separable clusters
    X_c0 = np.random.normal(loc=[-2, -2], scale=1.0, size=(100, 2))
    X_c1 = np.random.normal(loc=[+2, +2], scale=1.0, size=(100, 2))
    X = np.vstack([X_c0, X_c1])
    y = np.array([0] * 100 + [1] * 100)

    clf = VectorizedLogisticRegression(lr=0.5, epochs=500, lambda_reg=0.1)
    losses = clf.fit(X, y)
    print(f"Final Weights [w0, w1, w2]: {clf.w.round(4)}")
    acc = np.mean(clf.predict(X) == y) * 100.0
    print(f"Training Accuracy: {acc:.2f}%")
```

---

## 5. Solved Numerical Examples (Exam Pattern)

### 5.0 Sentiment classification (Session 6)

Weights $w_0=-0.2$, $w_1=1.2$ (positive word count), $w_2=-1.5$ (negative count). Review: $x_1=2$ (“great”, “good”), $x_2=1$ (“terrible”).

$z=-0.2+2.4-1.5=0.70$ → $\hat{y}=\sigma(0.70)\approx 0.668$ → **Positive** at threshold $0.5$.

### 5.1 One-step logistic GD — job offer (Session 6)

$\theta^{(0)}=[0.5,0.5,0.5]$, $\alpha=0.3$, sample $x_1=1.0$, $x_2=0.5$, $y=1$: $z=1.25$, $\hat{y}\approx 0.777$, error $\hat{y}-y=-0.223$. Updates: $\theta_0,\theta_1\to 0.5668$, $\theta_2\to 0.5334$.

### 5.2 Confusion matrix mnemonics & AUC benchmarks

- **First word**: True/False = prediction correct? **Second word**: Positive/Negative = predicted class.  
- **ROC/AUC**: sweep thresholds high→low; trapezoidal AUC. Guides: $0.5$ random; $0.7$–$0.8$ fair; $0.8$–$0.9$ excellent; $>0.9$ outstanding.

> **SMOTE**: apply only **after** train/test split — never on the full dataset before splitting (data leakage).

### 5.3 Batch gradient step on two samples

**Problem Statement (University Examination Pattern, 6 Marks)**:  
Given a binary logistic regression model with weight vector $\mathbf{w} = [w_0, w_1, w_2]^T = [-1.0, 0.5, 1.5]^T$ and learning rate $\eta = 0.2$.  
We observe two training samples:
- Sample 1: $\mathbf{x}_1 = [2, 1]^T$, true label $y_1 = 1$
- Sample 2: $\mathbf{x}_2 = [1, 0]^T$, true label $y_2 = 0$

1. Compute the linear score $z_i$ and predicted posterior probability $\hat{y}_i = \sigma(z_i)$ for both samples.
2. Compute the total Binary Cross-Entropy Loss across the two samples.
3. Compute the batch gradient $\nabla_{\mathbf{w}} \mathcal{L}$.
4. Update the weight vector $\mathbf{w}$ using 1 step of gradient descent.

---

### Step-by-Step Solution:

#### Step 1: Compute $z_i$ and $\hat{y}_i$
Augmented inputs: $\mathbf{x}_1 = [1, 2, 1]^T$ and $\mathbf{x}_2 = [1, 1, 0]^T$.

- **For Sample 1**:
  $$z_1 = \mathbf{w}^T \mathbf{x}_1 = w_0(1) + w_1(2) + w_2(1) = -1.0(1) + 0.5(2) + 1.5(1) = -1.0 + 1.0 + 1.5 = +1.50$$
  $$\hat{y}_1 = \sigma(1.50) = \frac{1}{1 + e^{-1.50}} = \frac{1}{1 + 0.2231} = \frac{1}{1.2231} \approx \mathbf{0.8176}$$

- **For Sample 2**:
  $$z_2 = \mathbf{w}^T \mathbf{x}_2 = w_0(1) + w_1(1) + w_2(0) = -1.0(1) + 0.5(1) + 1.5(0) = -1.0 + 0.5 + 0 = -0.50$$
  $$\hat{y}_2 = \sigma(-0.50) = \frac{1}{1 + e^{0.50}} = \frac{1}{1 + 1.6487} = \frac{1}{2.6487} \approx \mathbf{0.3775}$$

---

#### Step 2: Compute Total Binary Cross-Entropy Loss
$$\mathcal{L}_1 = -[1 \ln(0.8176) + 0] = -\ln(0.8176) = -(-0.2014) = \mathbf{0.2014}$$
$$\mathcal{L}_2 = -[0 + 1 \ln(1 - 0.3775)] = -\ln(0.6225) = -(-0.4740) = \mathbf{0.4740}$$

$$\mathcal{L}_{\text{total}} = \frac{\mathcal{L}_1 + \mathcal{L}_2}{2} = \frac{0.2014 + 0.4740}{2} = \frac{0.6754}{2} = \mathbf{0.3377}$$

---

#### Step 3: Compute Gradient Vector $\nabla_{\mathbf{w}} \mathcal{L}$
Prediction errors ($e_i = \hat{y}_i - y_i$):
- $e_1 = 0.8176 - 1.0 = \mathbf{-0.1824}$
- $e_2 = 0.3775 - 0.0 = \mathbf{+0.3775}$

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \sum_{i=1}^2 e_i \mathbf{x}_i = \frac{1}{2} \left[ (-0.1824)\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} + (0.3775)\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} \right]$$

Component-wise:
- Bias ($w_0$):
  $$\frac{\partial \mathcal{L}}{\partial w_0} = \frac{1}{2} [-0.1824(1) + 0.3775(1)] = \frac{1}{2} [+0.1951] = \mathbf{+0.09755}$$
- Weight $w_1$:
  $$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{1}{2} [-0.1824(2) + 0.3775(1)] = \frac{1}{2} [-0.3648 + 0.3775] = \frac{1}{2} [+0.0127] = \mathbf{+0.00635}$$
- Weight $w_2$:
  $$\frac{\partial \mathcal{L}}{\partial w_2} = \frac{1}{2} [-0.1824(1) + 0.3775(0)] = \frac{1}{2} [-0.1824] = \mathbf{-0.09120}$$

$$\nabla_{\mathbf{w}} \mathcal{L} = \begin{bmatrix} +0.09755 \\ +0.00635 \\ -0.09120 \end{bmatrix}$$

---

#### Step 4: Gradient Descent Weight Update
With $\eta = 0.2$:
$$\mathbf{w}^{(1)} = \mathbf{w}^{(0)} - \eta \nabla_{\mathbf{w}} \mathcal{L}$$

- $w_0^{(1)} = -1.0 - 0.2(+0.09755) = -1.0 - 0.01951 = \mathbf{-1.0195}$
- $w_1^{(1)} = 0.5 - 0.2(+0.00635) = 0.5 - 0.00127 = \mathbf{0.4987}$
- $w_2^{(1)} = 1.5 - 0.2(-0.09120) = 1.5 + 0.01824 = \mathbf{1.5182}$

$$\mathbf{w}^{(1)} = [-1.0195, 0.4987, 1.5182]^T$$

---

## 6. Common Pitfalls, Trade-offs & Practical Guidelines

1. **Numerical Instability in Cross-Entropy Evaluation**:
   - As $z \to \infty$, $\hat{y} \to 1.0 \implies \ln(1 - \hat{y}) = \ln(0) = -\infty$, triggering NaN errors.
   - *Fix*: Always use `np.logaddexp` or clip $\hat{y} \in [\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-15}$.
2. **Complete Separation & Exploding Weights**:
   - If training data is perfectly linearly separable, maximum likelihood drives $\|\mathbf{w}\| \to \infty$ because $\sigma(\infty) = 1.0$ yields zero loss. The decision boundary remains correct, but probability estimates become overconfident step functions.
   - *Remedy*: Always apply $L_2$ regularization ($\lambda > 0$) to keep weights bounded.
3. **Threshold Selection vs. Cost Asymmetry**:
   - Default threshold $\tau = 0.5$ assumes equal cost for False Positives (FP) and False Negatives (FN). In medical diagnostics or fraud detection where FN is catastrophic, lower the decision threshold ($\tau = 0.1 \sim 0.2$) to maximize recall.

