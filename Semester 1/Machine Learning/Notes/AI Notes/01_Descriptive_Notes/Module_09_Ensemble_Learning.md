# Module 9: Ensemble Learning

**Course**: AIML ZG565 — Machine Learning  
**Prescribed References**: Tan et al. (R2, Chapter 4); Christopher M. Bishop (R1, Chapter 14); Lecture Notes  
**Primary Teacher Followed**: Prof. Pritee Narendra Raole (Session 12: 2026-10-25)

---

## 1. Executive Summary & Foundational Definitions

Ensemble learning is a meta-learning methodology that constructs a diverse committee of individual base models (termed **weak learners**) and synergistically aggregates their individual outputs into a single composite predictor.

While single complex estimators suffer from an inherent tension between bias and variance, ensembles decouple this trade-off:
- **Bagging (Bootstrap Aggregation)** and **Random Forests** train deep, high-variance base trees in parallel on resampled datasets to dramatically reduce variance without increasing bias.
- **Boosting (AdaBoost, Gradient Boosting, XGBoost)** trains shallow, high-bias base learners sequentially, where each successor model specifically targets the residual mistakes of predecessor models, systematically driving systemic bias towards zero.

```
Bagging Paradigm (Parallel, Variance Reduction):
            ┌──> [ Bootstrap Sample 1 ] ──> [ Tree 1 (Deep) ] ──┐
            ├──> [ Bootstrap Sample 2 ] ──> [ Tree 2 (Deep) ] ──┼──> [ Aggregation (Average/Vote) ]
            └──> [ Bootstrap Sample M ] ──> [ Tree M (Deep) ] ──┘

Boosting Paradigm (Sequential, Bias Reduction):
[ Data D ] ──> [ Weak Tree 1 ] ──(Residuals)──> [ Weak Tree 2 ] ──(Residuals)──> [ Weak Tree M ] ──> [ Weighted Sum ]
```

---

## 2. Theoretical Foundations: Condorcet's Jury Theorem & Variance Shrinkage

### 2.1 Condorcet's Jury Theorem (1785)
Consider a committee of $M$ independent binary classifiers, each possessing individual accuracy $p > 0.5$. The probability that a simple majority vote makes the correct decision is given by the binomial cumulative distribution:

$$P_{\text{ensemble}} = \sum_{k=\lceil M/2 \rceil}^M \binom{M}{k} p^k (1-p)^{M-k}$$

As $M \to \infty$, $P_{\text{ensemble}} \to 1.0$ exponentially!  
*(Conversely, if base learners are worse than random guessing, $p < 0.5$, majority voting amplifies the error rate towards 100% failure).*

---

### 2.2 Theoretical Variance Reduction of Averaged Estimators
Let $M$ base models produce predictions $X_1, X_2, \dots, X_M$, each with variance $\sigma^2$ and pairwise correlation $\rho = \text{Corr}(X_i, X_j)$.  
The variance of the aggregated ensemble mean $\bar{X} = \frac{1}{M} \sum_{i=1}^M X_i$ is:

$$\text{Var}(\bar{X}) = \text{Var}\left( \frac{1}{M} \sum_{i=1}^M X_i \right) = \frac{1}{M^2} \left[ \sum_{i=1}^M \text{Var}(X_i) + \sum_{i \neq j} \text{Cov}(X_i, X_j) \right]$$
$$\text{Var}(\bar{X}) = \frac{1}{M^2} \left[ M \sigma^2 + M(M - 1) \rho \sigma^2 \right] = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$

#### Crucial Theoretical Insights:
1. If base models are **completely independent** ($\rho = 0$), the ensemble variance shrinks linearly by $\frac{1}{M}$:
   $$\text{Var}(\bar{X}) = \frac{\sigma^2}{M} \xrightarrow{M \to \infty} 0$$
2. In real data where models share training samples ($\rho > 0$), adding more trees cannot reduce variance below the asymptotic floor $\rho \sigma^2$.
3. **The Core Engineering Challenge of Bagging**: How can we minimize inter-tree correlation $\rho$?

---

## 3. Bagging & Random Forests


### 3.1 Bootstrap Aggregation (Bagging) & Out-of-Bag (OOB) Estimation
Given dataset $\mathcal{D}$ of size $N$, Bagging constructs $M$ bootstrap datasets $\mathcal{D}_1, \dots, \mathcal{D}_M$ by sampling $N$ observations **uniformly at random with replacement**.

#### The Out-of-Bag (OOB) Theorem:
For a dataset of size $N$, the probability that a specific training instance is *not* chosen in a single draw is $1 - \frac{1}{N}$.  
The probability that it is omitted across all $N$ independent draws is:

$$P(\text{OOB}) = \left( 1 - \frac{1}{N} \right)^N \xrightarrow{N \to \infty} \frac{1}{e} \approx 0.3679 \approx 36.8\%$$

Every bootstrap sample excludes roughly **$36.8\%$ of the dataset**. These omitted instances serve as a free, built-in validation set to compute the **Out-of-Bag (OOB) error**, eliminating the need for a separate validation split!

---

### 3.2 Random Forests: Eliminating Inter-Tree Correlation

Leo Breiman (2001) solved the correlation floor $\rho \sigma^2$ by introducing **Random Feature Subspace Projection (Feature Bagging)**:
1. At every internal node split of every tree, instead of searching across all $d$ features, randomly select a subset of $m$ features without replacement:
   - For classification: $m \approx \sqrt{d}$
   - For regression: $m \approx \frac{d}{3}$
2. Identify the best split strictly from this restricted candidate subset.
3. Grow each tree to maximum depth without pruning.

**Why this works**: If a single dominant feature exists (e.g., strong signal in $x_1$), standard Bagging selects $x_1$ as the root in virtually all trees, making trees highly correlated ($\rho \approx 1$). Random Forests forces other features to be chosen at the root in $(1 - \frac{m}{d})$ of the trees, de-correlating the ensemble and slashing $\rho$.

---

## 4. The Boosting Paradigm

### 4.1 AdaBoost (Adaptive Boosting) Algorithm

Yoav Freund and Robert Schapire (1997) formulated AdaBoost. Let training data be $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $y_i \in \{-1, +1\}$.

1. Initialize uniform sample weights: $w_i^{(1)} = \frac{1}{N}, \quad \forall i = 1, \dots, N$.
2. For rounds $m = 1, 2, \dots, M$:
   - Fit weak base classifier $h_m(\mathbf{x}) \in \{-1, +1\}$ minimizing weighted error:
     $$\epsilon_m = \sum_{i: y_i \neq h_m(\mathbf{x}_i)} w_i^{(m)}$$
   - Compute base classifier importance weight $\alpha_m$:
     $$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)$$
   - Update instance weights (increasing weights of misclassified points):
     $$w_i^{(m+1)} = \frac{w_i^{(m)} \exp\left( -\alpha_m y_i h_m(\mathbf{x}_i) \right)}{Z_m}$$
     where $Z_m = \sum_{i=1}^N w_i^{(m)} \exp\left( -\alpha_m y_i h_m(\mathbf{x}_i) \right)$ is the normalizing constant.
3. Final ensemble prediction:
   $$H(\mathbf{x}) = \text{sign}\left( \sum_{m=1}^M \alpha_m h_m(\mathbf{x}) \right)$$

---

### 4.2 Gradient Boosting Machines (GBM)

Jerome Friedman (1999) generalized boosting as numerical optimization in function space. Rather than reweighting samples, each successive base tree $h_m(\mathbf{x})$ directly fits the **pseudo-residuals** (negative gradients of the loss function):

$$r_{im} = - \left[ \frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F = F_{m-1}}$$

For Mean Squared Error loss $\mathcal{L} = \frac{1}{2}(y - F(\mathbf{x}))^2$:
$$r_{im} = y_i - F_{m-1}(\mathbf{x}_i) \quad (\text{Exact Residual!})$$

Model ensemble update with shrinkage (learning rate $\eta \in (0, 1]$):
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \gamma_m h_m(\mathbf{x})$$

---

### 4.3 Extreme Gradient Boosting (XGBoost)

Tianqi Chen and Carlos Guestrin (2016) developed XGBoost by incorporating two fundamental innovations:
1. **Second-Order Taylor Series Expansion of Loss**:
   $$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ \mathcal{L}(y_i, \hat{y}^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
   Where $g_i = \frac{\partial \mathcal{L}}{\partial \hat{y}^{(t-1)}}$ (first derivative / gradient) and $h_i = \frac{\partial^2 \mathcal{L}}{\partial (\hat{y}^{(t-1)})^2}$ (second derivative / Hessian).
2. **Explicit Regularization on Tree Complexity**:
   $$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
   Where $T$ is the number of terminal leaves, $w_j$ are leaf output weights, $\gamma$ is the leaf penalty, and $\lambda$ is $L_2$ leaf weight regularization.

#### Closed-Form Optimal Leaf Weights:
$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

#### Optimal Split Gain Criterion:
$$\text{Gain} = \frac{1}{2} \left[ \frac{\left( \sum_{i \in I_L} g_i \right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left( \sum_{i \in I_R} g_i \right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left( \sum_{i \in I} g_i \right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

---

## 5. Vectorized Implementation (From Scratch)

```python
import numpy as np

class DecisionStump:
    """Weak base classifier: 1-level decision tree splitting on a single feature."""
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class VectorizedAdaBoost:
    """Vectorized AdaBoost.M1 for Binary Classification."""
    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Convert y to {-1, +1}
        y = np.where(y <= 0, -1.0, 1.0)
        N, d = X.shape
        w = np.full(N, 1.0 / N)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            for feat in range(d):
                thresholds = np.unique(X[:, feat])
                for thresh in thresholds:
                    for polarity in [1, -1]:
                        preds = np.ones(N)
                        if polarity == 1:
                            preds[X[:, feat] < thresh] = -1
                        else:
                            preds[X[:, feat] > thresh] = -1

                        # Weighted error calculation
                        error = np.sum(w[y != preds])

                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.threshold = thresh
                            stump.feature_idx = feat

            # Clamp error to prevent div by zero
            min_error = np.clip(min_error, 1e-10, 1.0 - 1e-10)

            # Calculate alpha_m = 0.5 * ln((1 - eps) / eps)
            stump.alpha = 0.5 * np.log((1.0 - min_error) / min_error)

            # Update sample weights
            preds = stump.predict(X)
            w *= np.exp(-stump.alpha * y * preds)
            w /= np.sum(w)  # Normalize

            self.estimators.append(stump)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        clf_preds = [stump.alpha * stump.predict(X) for stump in self.estimators]
        return np.where(np.sum(clf_preds, axis=0) >= 0, 1, 0)


if __name__ == "__main__":
    np.random.seed(42)
    # Non-linear XOR classification
    X = np.random.randn(200, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

    adaboost = VectorizedAdaBoost(n_estimators=30).fit(X, y)
    acc = np.mean(adaboost.predict(X) == y) * 100.0
    print(f"AdaBoost (30 Stumps) Training Accuracy on XOR: {acc:.1f}%")
```

---

## 6. Solved Numerical Example (Exam Pattern)

**Problem Statement (University Examination Pattern, 6 Marks)**:  
Consider a dataset of 5 points with a 1D feature $x$ and binary target $y \in \{-1, +1\}$:

| Instance ($i$) | Feature $x$ | True Label $y$ | Initial Weight $w_i^{(1)}$ |
| :---: | :---: | :---: | :---: |
| 1 | 1.0 | +1 | 0.20 |
| 2 | 2.0 | +1 | 0.20 |
| 3 | 3.0 | -1 | 0.20 |
| 4 | 4.0 | -1 | 0.20 |
| 5 | 5.0 | +1 | 0.20 |

In Round 1, weak classifier $h_1(x)$ applies threshold split:
$$h_1(x) = \begin{cases} +1 & \text{if } x < 2.5 \\ -1 & \text{if } x \ge 2.5 \end{cases}$$

1. Identify which instances are misclassified by $h_1(x)$.
2. Calculate the weighted training error $\epsilon_1$.
3. Compute the classifier importance coefficient $\alpha_1$.
4. Calculate the unnormalized updated weights, the normalization factor $Z_1$, and the final normalized weights $\mathbf{w}^{(2)}$ for Round 2.

---

### Step-by-Step Solution:

#### Step 1: Evaluate $h_1(x)$ Predictions
- Point 1 ($x=1.0$): $h_1(1.0) = +1$, True $y_1 = +1 \implies$ **Correct**
- Point 2 ($x=2.0$): $h_1(2.0) = +1$, True $y_2 = +1 \implies$ **Correct**
- Point 3 ($x=3.0$): $h_1(3.0) = -1$, True $y_3 = -1 \implies$ **Correct**
- Point 4 ($x=4.0$): $h_1(4.0) = -1$, True $y_4 = -1 \implies$ **Correct**
- Point 5 ($x=5.0$): $h_1(5.0) = -1$, True $y_5 = +1 \implies$ **MISCLASSIFIED!**

Only **Instance 5** is misclassified.

---

#### Step 2: Compute Weighted Error $\epsilon_1$
$$\epsilon_1 = \sum_{i: y_i \neq h_1(x_i)} w_i^{(1)} = w_5^{(1)} = \mathbf{0.20}$$

---

#### Step 3: Compute Classifier Importance $\alpha_1$
$$\alpha_1 = \frac{1}{2} \ln \left( \frac{1 - \epsilon_1}{\epsilon_1} \right) = \frac{1}{2} \ln \left( \frac{1 - 0.20}{0.20} \right) = \frac{1}{2} \ln(4.0) = \frac{1}{2} (1.3863) = \mathbf{0.6931}$$

Factors:
- For correctly classified points ($y_i h_1(x_i) = +1$):
  $$e^{-\alpha_1} = e^{-0.6931} = \frac{1}{2} = \mathbf{0.50}$$
- For misclassified points ($y_i h_1(x_i) = -1$):
  $$e^{+\alpha_1} = e^{+0.6931} = 2.0 = \mathbf{2.00}$$

---

#### Step 4: Weight Update & Normalization
Compute unnormalized weights $u_i = w_i^{(1)} \exp(-\alpha_1 y_i h_1(x_i))$:
- Points 1, 2, 3, 4 (Correct):
  $$u_1 = u_2 = u_3 = u_4 = 0.20 \times 0.50 = \mathbf{0.10}$$
- Point 5 (Misclassified):
  $$u_5 = 0.20 \times 2.00 = \mathbf{0.40}$$

Compute normalization factor $Z_1$:
$$Z_1 = \sum_{i=1}^5 u_i = 0.10 + 0.10 + 0.10 + 0.10 + 0.40 = \mathbf{0.80}$$

Compute final normalized weights $w_i^{(2)} = \frac{u_i}{Z_1}$:
- For Points 1, 2, 3, 4:
  $$w_1^{(2)} = w_2^{(2)} = w_3^{(2)} = w_4^{(2)} = \frac{0.10}{0.80} = \frac{1}{8} = \mathbf{0.125}$$
- For Point 5:
  $$w_5^{(2)} = \frac{0.40}{0.80} = \frac{1}{2} = \mathbf{0.500}$$

Verification: $\sum w_i^{(2)} = 4(0.125) + 0.500 = 0.500 + 0.500 = 1.000$.

*Conclusion: The weight of the solitary misclassified instance escalated from $20\% \to 50\%$ of the total sample weight distribution, forcing Classifier 2 to prioritize it.*

---

## 7. Common Pitfalls, Trade-offs & Practical Guidelines

1. **Random Forest vs. Gradient Boosting Guidelines**:
   - Use **Random Forests** when out-of-the-box robustness without extensive hyperparameter tuning is required; highly resistant to overfitting and trivially parallelizable across CPU cores.
   - Use **Gradient Boosting (XGBoost / LightGBM / CatBoost)** when squeezing maximum predictive accuracy on competitive tabular benchmarks; requires careful tuning of learning rate $\eta$, max depth, and regularization $\lambda$.
2. **Subsampling & Learning Rate Coupling**:
   - In gradient boosting, reducing the learning rate $\eta$ requires a proportional increase in the number of trees ($M$).
   - Rule of thumb: Set $\eta \le 0.05$ and use **Early Stopping** based on validation loss rather than fixing $M$ arbitrarily.

