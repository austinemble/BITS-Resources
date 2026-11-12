# AIML ZG565: Machine Learning — Comprehensive Examination Question-Answer Bank

**Course**: AIML ZG565 — Machine Learning  
**Standard**: 50 High-Yield University Examination Questions (15 Easy, 25 Medium, 10 Tough)  
**Marking Scheme**: 6 Marks Each (Point-wise model solutions: 2 Marks Concept, 2 Marks Derivation/Math, 2 Marks Numerical/Insight)  
**Primary Teacher Followed**: Prof. Pritee Narendra Raole (BITS Pilani WILP)

---

# Part I: Foundations & Preprocessing (Modules 1 & 2)

### Q1. [Easy] State Tom Mitchell's formal definition of machine learning. Formulate the $(T, P, E)$ triplet for an autonomous robotic package delivery vehicle. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - A computer program is said to **learn** from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$:
     $$\text{Learning} \iff P(T, E_{\text{new}}) > P(T, E_{\text{old}})$$
2. **Mathematical / System Formulation (2 Marks)**:
   - **Task ($T$)**: Navigate public urban sidewalks and roads to transport packages from logistics hubs to customer doorsteps while avoiding pedestrians, traffic, and physical obstacles.
   - **Performance Measure ($P$)**: Percentage of successful on-time deliveries without human intervention, collision rate per 1,000 km, battery energy efficiency, and passenger/pedestrian safety compliance score.
   - **Experience ($E$)**: Millions of historical telemetry frames comprising LiDAR 3D point clouds, stereoscopic RGB video streams, IMU accelerometer feeds, GPS coordinates, and actuator wheel odometry.
3. **Diagram / Practical Insight (2 Marks)**:
   - Classical expert systems fail here because urban driving edge cases cannot be enumerated in deterministic if-else rules. The autonomous vehicle learns non-linear control mappings $\mathbf{a}_t = \pi(\mathbf{s}_t)$ through supervised imitation learning and reinforcement feedback.

---

### Q2. [Easy] Explain the Least Mean Squares (LMS) weight update rule in the context of Mitchell's checkers learning system. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The LMS algorithm updates the weight vector $\mathbf{w}$ of a linear evaluation function $\hat{V}(b) = \sum_{j=0}^d w_j x_j$ incrementally to minimize squared error between the model's estimate and the training value $V_{\text{train}}(b)$ provided by the Critic.
2. **Mathematical Derivation (2 Marks)**:
   - Let loss on board state $b$ be $\mathcal{L}(b) = \frac{1}{2} (V_{\text{train}}(b) - \hat{V}(b))^2$.
   - Differentiating with respect to parameter $w_j$:
     $$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial \hat{V}(b)} \cdot \frac{\partial \hat{V}(b)}{\partial w_j} = -(V_{\text{train}}(b) - \hat{V}(b)) \cdot x_j$$
   - Applying gradient descent ($w_j \leftarrow w_j - \eta \frac{\partial \mathcal{L}}{\partial w_j}$):
     $$w_j \leftarrow w_j + \eta \left( V_{\text{train}}(b) - \hat{V}(b) \right) x_j$$
3. **Numerical Walkthrough / Insight (2 Marks)**:
   - If current board prediction is $\hat{V}(b) = 15$, target is $V_{\text{train}}(b) = 25$, learning rate $\eta = 0.1$, and feature $x_1 = 4$:
     $$\Delta w_1 = 0.1 \times (25 - 15) \times 4 = 0.1 \times 10 \times 4 = +4.0$$
   - Weight $w_1$ increases by $+4.0$, adjusting the model in the precise direction of the Critic's target.

---

### Q3. [Medium] Compare Min-Max Normalization and Z-Score Standardization mathematically. Under what algorithmic conditions is each strictly required? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - **Min-Max Normalization** scales raw feature values into a rigid bounded interval $[0, 1]$.
   - **Z-Score Standardization** transforms features to have zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
2. **Mathematical Formulations (2 Marks)**:
   $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
   $$z = \frac{x - \mu}{\sigma}, \quad \text{where } \mu = \frac{1}{N}\sum_{i=1}^N x_i, \ \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$$
3. **Comparative Insight & Operational Rules (2 Marks)**:
   - **Use Min-Max**: When downstream algorithms expect bounded inputs (e.g., image pixels $[0, 255] \to [0, 1]$ for neural activation functions like Sigmoid).
   - **Use Standardization**: For distance-based algorithms (SVM, $k$-NN, K-Means) and regularized models (Ridge/Lasso). Min-Max collapses when extreme outliers are present, compressing legitimate data into an infinitesimally narrow band, whereas Z-score retains outlier magnitude.

---

### Q4. [Medium] Define the empirical sample covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$. Prove that $\mathbf{\Sigma}$ is always positive semi-definite. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The covariance matrix summarizes pairwise linear dependencies across $d$ feature dimensions. Diagonal entries $\Sigma_{jj} = \text{Var}(X_j)$, while off-diagonal entries $\Sigma_{jk} = \text{Cov}(X_j, X_k)$.
2. **Mathematical Formulation & Proof (2 Marks)**:
   - Let $\tilde{\mathbf{X}} \in \mathbb{R}^{N \times d}$ denote the mean-centered design matrix ($\tilde{x}_{ij} = x_{ij} - \mu_j$).
   - The covariance matrix is:
     $$\mathbf{\Sigma} = \frac{1}{N} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$$
   - To prove positive semi-definiteness, we must show that for any non-zero vector $\mathbf{v} \in \mathbb{R}^d$: $\mathbf{v}^T \mathbf{\Sigma} \mathbf{v} \ge 0$.
     $$\mathbf{v}^T \mathbf{\Sigma} \mathbf{v} = \mathbf{v}^T \left( \frac{1}{N} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} \right) \mathbf{v} = \frac{1}{N} (\tilde{\mathbf{X}}\mathbf{v})^T (\tilde{\mathbf{X}}\mathbf{v})$$
   - Let $\mathbf{u} = \tilde{\mathbf{X}}\mathbf{v} \in \mathbb{R}^N$. Then:
     $$\mathbf{v}^T \mathbf{\Sigma} \mathbf{v} = \frac{1}{N} \mathbf{u}^T \mathbf{u} = \frac{1}{N} \|\mathbf{u}\|_2^2 = \frac{1}{N} \sum_{i=1}^N u_i^2 \ge 0 \quad \text{(Q.E.D.)}$$
3. **Geometric Insight (2 Marks)**:
   - Since $\mathbf{\Sigma} \succeq 0$, all eigenvalues are real and non-negative ($\lambda_i \ge 0$). In PCA, these eigenvectors define the orthogonal axes of maximum variance, and eigenvalues represent the variance explained along those axes.

---

### Q5. [Medium] Explain how SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic observations. Why is it superior to naive random over-sampling? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - SMOTE generates synthetic feature vectors along the line segments connecting minority class instances to their $k$-nearest minority neighbors, rather than duplicating existing records.
2. **Mathematical Algorithmic Formulation (2 Marks)**:
   - For each minority instance $\mathbf{x}_i$:
     1. Find its $k$-nearest neighbors within the minority class: $\mathcal{N}_k(\mathbf{x}_i)$.
     2. Randomly pick one neighbor $\mathbf{x}_{zi} \in \mathcal{N}_k(\mathbf{x}_i)$.
     3. Compute the synthetic vector:
        $$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
3. **Diagram & Practical Insight (2 Marks)**:
   ```
   Minority Points: ●-------x_new-------●
                    x_i                 x_zi
   ```
   - **Naive Over-Sampling**: Duplicates identical data points, causing models (especially decision trees) to memorize single points and severely overfit.
   - **SMOTE Advantage**: Synthesizes convex interpolations that expand the decision region of the minority class in feature space, improving out-of-sample generalization.

---

# Part II: Linear Models for Regression & Classification (Modules 3 & 4)

### Q6. [Easy] State the Ordinary Least Squares (OLS) objective function and write the Normal Equations in matrix notation. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - OLS finds the parameter vector $\mathbf{w}$ that minimizes the sum of squared vertical differences between observed targets $\mathbf{y}$ and linear predictions $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$.
2. **Mathematical Formulation (2 Marks)**:
   $$\mathcal{L}(\mathbf{w}) = \frac{1}{2} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 = \frac{1}{2} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})$$
   - Taking the gradient $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$ yields the **Normal Equations**:
     $$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{y}$$
   - When $\mathbf{X}^T\mathbf{X}$ is invertible:
     $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
3. **Geometric Insight (2 Marks)**:
   - The predicted vector $\hat{\mathbf{y}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ is the orthogonal projection of $\mathbf{y}$ onto the subspace spanned by the column vectors of $\mathbf{X}$. The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is perpendicular to the column space: $\mathbf{X}^T \mathbf{e} = \mathbf{0}$.

---

### Q7. [Medium] Derive the closed-form parameter solution for Ridge Regression ($L_2$ regularization). Why is $(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})$ guaranteed to be invertible for $\lambda > 0$? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Ridge regression augments the OLS loss with a quadratic penalty on parameter magnitudes $\frac{\lambda}{2} \|\mathbf{w}\|_2^2$, shrinking weights to prevent overfitting and resolve collinearity.
2. **Mathematical Derivation (2 Marks)**:
   $$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w}) + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w}$$
   - Expanding and taking the vector gradient:
     $$\nabla_{\mathbf{w}} \mathcal{L}_{\text{Ridge}} = -\mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w}) + \lambda \mathbf{w} = -\mathbf{X}^T \mathbf{y} + (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{0}$$
     $$(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{X}^T \mathbf{y} \implies \mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
3. **Proof of Invertibility (2 Marks)**:
   - $\mathbf{X}^T\mathbf{X}$ is symmetric positive semi-definite with eigenvalues $\mu_i \ge 0$.
   - The eigenvalues of $(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})$ are $\lambda_i = \mu_i + \lambda$.
   - Since $\lambda > 0$ and $\mu_i \ge 0$, every eigenvalue $\lambda_i \ge \lambda > 0$ is strictly positive.
   - Determinant $\det(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I}) = \prod_{i=1}^d \lambda_i > 0$, guaranteeing strict invertibility.

---

### Q8. [Tough] Derive the Bias-Variance decomposition of the expected squared prediction error for a regression estimator. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The expected generalization error of a learning algorithm on unseen test points decomposes into three fundamentally independent terms: **Bias squared** (underfitting error), **Variance** (sensitivity to training set fluctuations), and **Irreducible Noise** $\sigma^2$.
2. **Mathematical Derivation (2 Marks)**:
   - Let $y = f(\mathbf{x}) + \epsilon$ where $\mathbb{E}[\epsilon] = 0, \text{Var}(\epsilon) = \sigma^2$.
   - Let $\hat{f}(\mathbf{x}) = \hat{f}(\mathbf{x}; \mathcal{D})$ be the estimator trained on dataset $\mathcal{D}$, and $\bar{f}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x})]$.
   - The prediction error at point $\mathbf{x}$ is:
     $$y - \hat{f}(\mathbf{x}) = (f(\mathbf{x}) - \bar{f}(\mathbf{x})) + (\bar{f}(\mathbf{x}) - \hat{f}(\mathbf{x})) + \epsilon$$
   - Squaring both sides:
     $$(y - \hat{f})^2 = (f - \bar{f})^2 + (\bar{f} - \hat{f})^2 + \epsilon^2 + 2(f - \bar{f})(\bar{f} - \hat{f}) + 2(f - \bar{f})\epsilon + 2(\bar{f} - \hat{f})\epsilon$$
   - Taking expectation $\mathbb{E}_{\mathcal{D}, \epsilon}[\cdot]$:
     - Cross-term $\mathbb{E}[2(f - \bar{f})(\bar{f} - \hat{f})] = 2(f - \bar{f})\mathbb{E}[\bar{f} - \hat{f}] = 0$ (since $\mathbb{E}[\hat{f}] = \bar{f}$).
     - Cross-terms involving $\epsilon$ evaluate to $0$ because noise is independent with zero mean.
     $$\mathbb{E}[(y - \hat{f})^2] = \underbrace{(f(\mathbf{x}) - \bar{f}(\mathbf{x}))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(\mathbf{x}) - \bar{f}(\mathbf{x}))^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$
3. **Practical Insight (2 Marks)**:
   - As model complexity increases, Bias monotonically decreases while Variance escalates. Optimal generalization requires tuning regularization to hit the minimum of the U-shaped total error curve.

---

### Q9. [Medium] Prove that the derivative of the Standard Logistic Sigmoid function is $\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ squashes real-valued logits $z \in (-\infty, \infty)$ into probabilistic outputs $\sigma(z) \in (0, 1)$. Its derivative expresses the rate of change of posterior probability.
2. **Mathematical Derivation (2 Marks)**:
   $$\sigma(z) = (1 + e^{-z})^{-1}$$
   - Applying the power and chain rules:
     $$\frac{d\sigma(z)}{dz} = -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z}) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$
   - Factoring the fraction:
     $$\frac{d\sigma(z)}{dz} = \left( \frac{1}{1 + e^{-z}} \right) \cdot \left( \frac{e^{-z}}{1 + e^{-z}} \right)$$
   - Noting that $\frac{e^{-z}}{1 + e^{-z}} = \frac{1 + e^{-z} - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$:
     $$\frac{d\sigma(z)}{dz} = \sigma(z) (1 - \sigma(z)) \quad \text{(Q.E.D.)}$$
3. **Practical Insight (2 Marks)**:
   - Maximum gradient occurs at $z = 0$, where $\sigma(0) = 0.5 \implies \sigma'(0) = 0.5(1 - 0.5) = 0.25$.
   - As $|z| \to \infty$, $\sigma(z) \to 0$ or $1$, causing $\sigma'(z) \to 0$. This property causes **vanishing gradients** when saturated sigmoids are stacked in deep networks.

---

### Q10. [Tough] Derive the gradient of the Binary Cross-Entropy (Log-Loss) function with respect to weights $\mathbf{w}$ in Logistic Regression. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Binary cross-entropy evaluates the negative log-likelihood of Bernoulli-distributed labels $y_i \in \{0, 1\}$ given model probabilities $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$.
2. **Mathematical Derivation (2 Marks)**:
   $$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
   - Let $z_i = \mathbf{w}^T \mathbf{x}_i \implies \hat{y}_i = \sigma(z_i)$. By the multivariate chain rule:
     $$\frac{\partial \mathcal{L}_i}{\partial w_j} = \frac{\partial \mathcal{L}_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_j}$$
   - Compute each term:
     $$\frac{\partial \mathcal{L}_i}{\partial \hat{y}_i} = -\left[ \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right] = \frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)}$$
     $$\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i (1 - \hat{y}_i)$$
     $$\frac{\partial z_i}{\partial w_j} = x_{ij}$$
   - Multiplying terms:
     $$\frac{\partial \mathcal{L}_i}{\partial w_j} = \left[ \frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)} \right] \cdot [\hat{y}_i (1 - \hat{y}_i)] \cdot x_{ij} = (\hat{y}_i - y_i) x_{ij}$$
   - In vector matrix form across all $N$ instances:
     $$\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
3. **Diagram / Comparative Insight (2 Marks)**:
   - The gradient has the exact same structural form as linear regression, $\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$, but with non-linear probabilistic predictions $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$.

---

### Q11. [Medium] Formulate Fisher's Linear Discriminant for binary classification. How does it optimize class separability? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Fisher's Linear Discriminant seeks a projection vector $\mathbf{w}$ that projects data points $y = \mathbf{w}^T \mathbf{x}$ such that the distance between projected class means is maximized while the variance within each projected class is minimized.
2. **Mathematical Derivation (2 Marks)**:
   - Let class means be $\mathbf{m}_1, \mathbf{m}_2$. Projected means are $\tilde{m}_1 = \mathbf{w}^T \mathbf{m}_1, \tilde{m}_2 = \mathbf{w}^T \mathbf{m}_2$.
   - **Between-Class Scatter**: $\mathbf{S}_B = (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^T$.
   - **Within-Class Scatter**: $\mathbf{S}_W = \sum_{i \in C_1} (\mathbf{x}_i - \mathbf{m}_1)(\mathbf{x}_i - \mathbf{m}_1)^T + \sum_{i \in C_2} (\mathbf{x}_i - \mathbf{m}_2)(\mathbf{x}_i - \mathbf{m}_2)^T$.
   - Fisher's criterion to maximize:
     $$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$
   - Setting $\nabla_{\mathbf{w}} J = \mathbf{0}$ yields $\mathbf{S}_B \mathbf{w} \propto \mathbf{S}_W \mathbf{w}$. Since $\mathbf{S}_B \mathbf{w}$ lies in the direction of $(\mathbf{m}_1 - \mathbf{m}_2)$:
     $$\mathbf{w} \propto \mathbf{S}_W^{-1} (\mathbf{m}_1 - \mathbf{m}_2)$$
3. **Practical Insight (2 Marks)**:
   - Simple projection along $(\mathbf{m}_1 - \mathbf{m}_2)$ ignores covariance, causing overlapping projections if classes have high variance along that axis. Multiplying by $\mathbf{S}_W^{-1}$ rotates the projection vector away from directions of high within-class spread.

---

### Q12. [Medium] Explain the Softmax regression function for multi-class classification ($K > 2$). Why is it invariant to constant additions to logits? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Softmax generalizes the binary logistic sigmoid to $K$ mutually exclusive categories by mapping a vector of $K$ real-valued affine scores $\mathbf{z} = [z_1, \dots, z_K]^T$ into a normalized probability distribution.
2. **Mathematical Formulation (2 Marks)**:
   $$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$
   - Clearly, $P(y = k \mid \mathbf{x}) \in (0, 1)$ and $\sum_{k=1}^K P(y = k \mid \mathbf{x}) = 1.0$.
3. **Invariance Proof & Practical Implementation (2 Marks)**:
   - Let a constant $c \in \mathbb{R}$ be added to all logits: $z_k' = z_k + c$.
     $$P(y = k \mid \mathbf{x}) = \frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^c \cdot e^{z_k}}{e^c \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
   - **Numerical Stability Trick**: Setting $c = -\max(z_1, \dots, z_K)$ ensures the largest exponent is $e^0 = 1$, preventing floating-point overflow during software execution.

---

# Part III: Decision Trees & Instance-Based Learning (Modules 5 & 6)

### Q13. [Easy] Define Shannon Entropy and Gini Impurity. Calculate both for a binary node with 30 positive and 10 negative samples. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - **Entropy** $H(S) = -\sum p_i \log_2 p_i$ measures expected informational uncertainty in bits.
   - **Gini Impurity** $I_G(S) = 1 - \sum p_i^2$ measures the probability that a randomly chosen element is incorrectly labeled.
2. **Step-by-Step Numerical Calculations (2 Marks)**:
   - Total samples $N = 30 + 10 = 40$.
   - $p_+ = \frac{30}{40} = 0.75$, $p_- = \frac{10}{40} = 0.25$.
   - **Entropy**:
     $$H(S) = - [0.75 \log_2(0.75) + 0.25 \log_2(0.25)]$$
     $$H(S) = - [0.75(-0.4150) + 0.25(-2.0000)] = - [-0.3113 - 0.5000] = \mathbf{0.8113 \text{ bits}}$$
   - **Gini Impurity**:
     $$I_G(S) = 1 - [(0.75)^2 + (0.25)^2] = 1 - [0.5625 + 0.0625] = 1 - 0.6250 = \mathbf{0.3750}$$
3. **Comparative Insight (2 Marks)**:
   - Both metrics reach maximum at equal balance ($p=0.5 \implies H=1.0, I_G=0.5$) and minimum at pure state ($p=1 \implies H=0, I_G=0$). Gini is preferred in CART because it avoids slow logarithmic evaluations.

---

### Q14. [Medium] Why is Information Gain biased toward high-cardinality attributes? How does C4.5's Gain Ratio penalize this? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Information Gain measures entropy reduction: $\text{Gain}(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$. It favors attributes with many values (e.g., `Customer_ID`) because partitioning into $|S|$ singleton subsets creates zero-entropy subsets ($H(S_v) = 0$), yielding maximum gain despite zero predictive generalization.
2. **Mathematical Formulation (2 Marks)**:
   - C4.5 defines **Split Information** (the entropy of the attribute's values):
     $$\text{SplitInfo}(S, A) = - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2 \left( \frac{|S_v|}{|S|} \right)$$
   - The **Gain Ratio** normalizes Information Gain by Split Information:
     $$\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}$$
3. **Numerical Illustration (2 Marks)**:
   - If an attribute splits $N = 100$ instances into 100 unique values:
     $$\text{SplitInfo} = - \sum_{i=1}^{100} \frac{1}{100} \log_2\left(\frac{1}{100}\right) = \log_2(100) \approx 6.64 \text{ bits}$$
   - This large denominator drastically suppresses the Gain Ratio, preventing the tree from choosing uninformative high-cardinality attributes.

---

### Q15. [Medium] Explain how Decision Trees handle continuous-valued attributes during split evaluation. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - A continuous attribute $A$ is dynamically discretized into a binary test predicate $A \le c$ versus $A > c$ by identifying candidate split thresholds $c$ that maximize information gain.
2. **Mathematical Algorithmic Formulation (2 Marks)**:
   1. Sort all unique values of attribute $A$ observed in dataset $S$ in ascending order: $\{a_1, a_2, \dots, a_m\}$.
   2. Identify candidate split thresholds $c_j$ as midpoints between adjacent distinct values:
      $$c_j = \frac{a_j + a_{j+1}}{2}$$
   3. **Fayyad & Irani Optimization**: Only consider midpoints where the target class label changes between $a_j$ and $a_{j+1}$. (Proven that the threshold maximizing Information Gain must lie between adjacent samples of differing classes).
3. **Walkthrough / Insight (2 Marks)**:
   - For attribute values $[10, 15, 20]$ with labels $[\text{No}, \text{Yes}, \text{Yes}]$:
     - Midpoint between 10 (No) and 15 (Yes): $c_1 = \frac{10 + 15}{2} = 12.5$.
     - Midpoint between 15 (Yes) and 20 (Yes): No class change; skip!
   - Evaluate $\text{Gain}(S, A \le 12.5)$ to determine if this split outperforms other features.

---

### Q16. [Medium] Differentiate between Eager Learning and Lazy Learning. Provide two algorithmic examples of each. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - **Eager Learners**: Analyze the entire training dataset prior to receiving queries, constructing an explicit, compact global hypothesis $f(\mathbf{x}; \boldsymbol{\theta})$ and discarding raw training data.
   - **Lazy Learners**: Defer all computation until a specific query point $\mathbf{x}_q$ arrives, retaining all training instances in memory and forming local hypotheses on demand.
2. **Structural & Complexity Comparison (2 Marks)**:
   - **Training Time**: Eager is computationally demanding ($\mathcal{O}(Nd^2)$ or more); Lazy is trivial ($\mathcal{O}(1)$ storage).
   - **Inference Time**: Eager is fast ($\mathcal{O}(d)$ evaluation); Lazy is slow ($\mathcal{O}(Nd)$ distance scans).
   - **Decision Surface**: Eager models globally constrained surfaces (planes, axis-aligned boxes); Lazy models arbitrary non-linear local boundaries.
3. **Algorithmic Categorization (2 Marks)**:
   - **Eager Examples**: Logistic Regression, Decision Trees (CART/C4.5), Support Vector Machines, Multilayer Perceptrons.
   - **Lazy Examples**: $k$-Nearest Neighbors ($k$-NN), Locally Weighted Regression (LWR), Case-Based Reasoning (CBR).

---

### Q17. [Tough] State the Cover-Hart Theorem for 1-Nearest Neighbor classification. Explain its asymptotic bounds and significance. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The Cover-Hart Theorem (1967) establishes the mathematical relationship between the asymptotic error probability of the non-parametric 1-Nearest Neighbor classifier and the theoretically optimal Bayes error rate.
2. **Mathematical Formulation (2 Marks)**:
   - Let $R^*$ denote the Bayes optimal minimum risk:
     $$R^* = \mathbb{E}_{\mathbf{x}} [1 - \max_k P(C_k \mid \mathbf{x})]$$
   - In the infinite sample limit ($N \to \infty$), the risk of the 1-NN rule $R_{1-\text{NN}}$ is bounded by:
     $$R^* \le R_{1-\text{NN}} \le 2 R^* (1 - R^*) \le 2 R^*$$
   - For binary classification, substituting worst-case $1 - R^* \le 1$ yields:
     $$R_{1-\text{NN}} \le 2 R^*$$
3. **Theoretical & Practical Significance (2 Marks)**:
   - The theorem proves that even without knowing the true underlying probability density functions, an extremely simple memory-based rule (1-NN) guarantees an error rate never worse than twice the optimal error achievable by an omniscient Bayes classifier.

---

### Q18. [Medium] Describe the Curse of Dimensionality and its impact on distance-based algorithms like $k$-NN. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The Curse of Dimensionality refers to the geometric phenomena that arise when data dimensions $d \to \infty$, where volume grows exponentially and data points become exponentially sparse.
2. **Mathematical Formulation (2 Marks)**:
   - As $d \to \infty$, the Euclidean distance between a query point and its nearest neighbor converges to the distance to its furthest neighbor:
     $$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$
   - In high dimensions, every point becomes nearly equidistant from every other point.
3. **Diagram / Practical Impact on $k$-NN (2 Marks)**:
   - Because all distances converge to roughly the same scalar value, the concept of "nearest" neighbor becomes statistically meaningless.
   - Additionally, finding $k$ neighbors requires searching an exponentially expanding hypercube volume, causing $k$-NN to lose its local approximation capability unless dimensionality reduction (PCA, t-SNE) is applied first.

---

# Part IV: Support Vector Machines & Bayesian Learning (Modules 7 & 8)

### Q19. [Easy] Define the geometric margin of a linear Support Vector Machine. Why do canonical hyperplanes satisfy $\gamma = \frac{2}{\|\mathbf{w}\|}$? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The geometric margin $\gamma$ is the perpendicular distance between the parallel bounding hyperplanes supporting the positive and negative classes that separate the data with zero training violations.
2. **Mathematical Derivation (2 Marks)**:
   - Let decision boundary be $\mathbf{w}^T \mathbf{x} + b = 0$.
   - Scale $(\mathbf{w}, b)$ such that the closest points (support vectors) satisfy:
     $$\mathbf{w}^T \mathbf{x}^+ + b = +1 \quad \text{and} \quad \mathbf{w}^T \mathbf{x}^- + b = -1$$
   - Subtracting the two equations:
     $$\mathbf{w}^T (\mathbf{x}^+ - \mathbf{x}^-) = 2$$
   - Projecting the vector $(\mathbf{x}^+ - \mathbf{x}^-)$ onto the unit normal vector $\frac{\mathbf{w}}{\|\mathbf{w}\|_2}$:
     $$\text{Margin } \gamma = \frac{\mathbf{w}^T (\mathbf{x}^+ - \mathbf{x}^-)}{\|\mathbf{w}\|_2} = \frac{2}{\|\mathbf{w}\|_2}$$
3. **Optimization Link (2 Marks)**:
   - Maximizing margin $\gamma = \frac{2}{\|\mathbf{w}\|_2}$ is mathematically equivalent to minimizing $\frac{1}{2} \|\mathbf{w}\|_2^2$, transforming the hard margin problem into a convex quadratic program.

---

### Q20. [Tough] Formulate the Soft-Margin SVM primal optimization problem. Explain the role of slack variables $\xi_i$ and the penalty parameter $C$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - For non-linearly separable data with label noise, hard margins are infeasible. Soft-margin SVM introduces non-negative slack variables $\xi_i \ge 0$ allowing margin violations while penalizing their total magnitude.
2. **Mathematical Formulation (2 Marks)**:
   $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \xi_i$$
   $$\text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \ \forall i=1,\dots,N$$
   - Interpretation of slack $\xi_i$:
     - $\xi_i = 0$: Point lies outside or on the correct margin.
     - $0 < \xi_i \le 1$: Point is correctly classified but violates the margin.
     - $\xi_i > 1$: Point is misclassified.
3. **Trade-off Analysis of Hyperparameter $C$ (2 Marks)**:
   - $C$ controls the trade-off between maximizing margin width ($\frac{1}{2}\|\mathbf{w}\|^2$) and minimizing classification errors ($\sum \xi_i$).
   - **Large $C$**: Heavy penalty on slack $\implies$ Narrower margin, fits training noise (high variance, overfitting).
   - **Small $C$**: Tolerates margin violations $\implies$ Wider margin, smoother boundary (high bias, robust to outliers).

---

### Q21. [Medium] State Mercer's Theorem. Explain how the Kernel Trick bypasses explicit feature mapping $\boldsymbol{\phi}(\mathbf{x})$ into high-dimensional space. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The Kernel Trick computes the inner product between mapped vectors $\langle \boldsymbol{\phi}(\mathbf{x}), \boldsymbol{\phi}(\mathbf{z}) \rangle$ directly in the original low-dimensional input coordinates via a kernel function $K(\mathbf{x}, \mathbf{z})$, without ever explicitly constructing the high-dimensional feature space $\mathcal{H}$.
2. **Mathematical Formulation of Mercer's Theorem (2 Marks)**:
   - A symmetric continuous function $K(\mathbf{x}, \mathbf{z})$ is a valid inner-product kernel in a Hilbert space if and only if its Gram matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$ ($K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$) is **positive semi-definite** for all finite point configurations:
     $$\sum_{i=1}^N \sum_{j=1}^N c_i c_j K(\mathbf{x}_i, \mathbf{x}_j) \ge 0, \quad \forall \mathbf{c} \in \mathbb{R}^N$$
3. **Computational Insight (2 Marks)**:
   - For an RBF Gaussian kernel $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$, the mapped space $\mathcal{H}$ has **infinite dimensions** (via Taylor series expansion of $\exp$). Explicitly computing $\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^\infty$ is physically impossible, yet evaluating $K(\mathbf{x}, \mathbf{z})$ takes only $\mathcal{O}(d)$ time.

---

### Q22. [Easy] Distinguish between Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) hypothesis selection. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - **MLE** selects the hypothesis that maximizes the probability of observing the data $P(D \mid h)$, treating all hypotheses as equally likely *a priori*.
   - **MAP** incorporates a prior probability distribution $P(h)$, selecting the hypothesis that maximizes the posterior distribution $P(h \mid D)$.
2. **Mathematical Formulations (2 Marks)**:
   $$h_{\text{ML}} = \arg\max_{h \in \mathcal{H}} P(D \mid h)$$
   $$h_{\text{MAP}} = \arg\max_{h \in \mathcal{H}} P(h \mid D) = \arg\max_{h \in \mathcal{H}} \frac{P(D \mid h) P(h)}{P(D)} = \arg\max_{h \in \mathcal{H}} P(D \mid h) P(h)$$
3. **Comparative Insight (2 Marks)**:
   - In log-space: $\ln P(h \mid D) = \ln P(D \mid h) + \ln P(h)$.
   - The prior term $\ln P(h)$ acts as a **regularizer**. For instance, assuming a zero-mean Gaussian prior on weights $P(\mathbf{w}) \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ transforms MLE linear regression directly into **Ridge Regression** ($L_2$ penalty).

---

### Q23. [Medium] Formulate the Naïve Bayes classifier. Why does the conditional independence assumption make high-dimensional classification computationally tractable? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Naïve Bayes predicts class $v_j$ by computing posterior probabilities via Bayes' theorem under the simplifying assumption that all feature attributes are mutually conditionally independent given the class.
2. **Mathematical Formulation (2 Marks)**:
   $$P(x_1, \dots, x_d \mid v_j) = \prod_{k=1}^d P(x_k \mid v_j)$$
   - The classification rule is:
     $$v_{\text{NB}} = \arg\max_{v_j \in V} P(v_j) \prod_{k=1}^d P(x_k \mid v_j)$$
3. **Complexity Reduction Insight (2 Marks)**:
   - Without conditional independence, modeling $d$ binary features requires estimating $2^d - 1$ joint conditional probabilities per class, which scales exponentially and causes severe data sparsity.
   - Under Naïve Bayes conditional independence, only $2d$ parameters per class must be estimated, reducing parameter complexity from $\mathcal{O}(2^d)$ to $\mathcal{O}(d)$.

---

### Q24. [Easy] Explain the Zero-Probability Problem in Naïve Bayes. Show how Laplace Smoothing resolves it. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - If a feature value $x_k$ never co-occurs with class $v_j$ in training data, empirical frequency yields $P(x_k \mid v_j) = 0$. In Naïve Bayes' product chain, this zero annihilates all other conditional probabilities, producing posterior score $= 0$.
2. **Mathematical Formulation (2 Marks)**:
   - **Laplace (Add-1) Smoothing** adds pseudo-count $\alpha = 1$ to each count:
     $$\hat{P}(x_k = c \mid v_j) = \frac{N_{jk} + 1}{N_j + K_k}$$
     Where $N_{jk}$ is the count of feature $k$ having value $c$ in class $v_j$, $N_j$ is the total count of class $v_j$, and $K_k$ is the number of distinct categories of feature $k$.
3. **Numerical Illustration (2 Marks)**:
   - If feature `Wind = Strong` appears 0 times in 5 `Play = No` samples with $K=2$ values:
     $$\hat{P}(\text{Strong} \mid \text{No}) = \frac{0 + 1}{5 + 2} = \frac{1}{7} \approx 0.1429 > 0$$
   - The probability remains strictly positive, preserving evidence from other features.

---

# Part V: Ensembles & Unsupervised Learning (Modules 9 & 10)

### Q25. [Medium] Explain Condorcet's Jury Theorem and prove why ensemble averaging reduces predictive variance. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Condorcet's Jury Theorem proves that if $M$ independent voters/models each have accuracy $p > 0.5$, the probability that the majority vote is correct converges exponentially to $1.0$ as $M \to \infty$.
2. **Mathematical Variance Proof (2 Marks)**:
   - Let $M$ base models produce predictions $X_i$ with variance $\sigma^2$ and pairwise correlation $\rho$.
   - The ensemble average is $\bar{X} = \frac{1}{M} \sum_{i=1}^M X_i$.
   $$\text{Var}(\bar{X}) = \frac{1}{M^2} \left[ \sum_{i=1}^M \text{Var}(X_i) + \sum_{i \neq j} \text{Cov}(X_i, X_j) \right] = \frac{1}{M^2} [M\sigma^2 + M(M-1)\rho \sigma^2]$$
   $$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
3. **Practical Insight (2 Marks)**:
   - If models are uncorrelated ($\rho = 0$), variance shrinks to $\frac{\sigma^2}{M} \to 0$.
   - In correlated ensembles ($\rho > 0$), variance cannot drop below $\rho \sigma^2$. This mathematical floor is why Random Forests uses feature bagging to force $\rho$ lower.

---

### Q26. [Easy] Prove that roughly 36.8% of training data is omitted in each bootstrap sample of Bagging (Out-of-Bag instances). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - In Bootstrap Aggregation, a sample of size $N$ is drawn uniformly with replacement from a dataset of size $N$. Instances not selected in a given bootstrap draw are designated Out-Of-Bag (OOB).
2. **Mathematical Derivation (2 Marks)**:
   - The probability that a specific instance $i$ is *not* picked in 1 single draw is $1 - \frac{1}{N}$.
   - The probability that it is omitted across all $N$ independent draws is:
     $$P(\text{OOB}) = \left( 1 - \frac{1}{N} \right)^N$$
   - Taking the limit as dataset size $N \to \infty$:
     $$\lim_{N \to \infty} \left( 1 - \frac{1}{N} \right)^N = \lim_{N \to \infty} \left[ \left( 1 + \frac{-1}{N} \right)^{-N} \right]^{-1} = e^{-1} = \frac{1}{e} \approx 0.367879 \approx \mathbf{36.8\%}$$
3. **Practical Utility (2 Marks)**:
   - Because each tree omits $\approx 36.8\%$ of data, these OOB samples act as a built-in cross-validation set. The OOB error provides an unbiased estimate of generalization error without holding out a separate validation set.

---

### Q27. [Tough] Derive the AdaBoost classifier weight update formula $\alpha_m = \frac{1}{2} \ln \left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$ by minimizing Exponential Loss. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - AdaBoost fits base classifiers sequentially to minimize the Exponential Loss objective function $\mathcal{L} = \sum_{i=1}^N \exp(-y_i F_m(\mathbf{x}_i))$ where $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \alpha_m h_m(\mathbf{x})$.
2. **Mathematical Derivation (2 Marks)**:
   - At round $m$, with fixed $F_{m-1}$ and weights $w_i^{(m)} = \exp(-y_i F_{m-1}(\mathbf{x}_i))$:
     $$\mathcal{L}(\alpha_m) = \sum_{i=1}^N w_i^{(m)} \exp(-\alpha_m y_i h_m(\mathbf{x}_i))$$
   - Splitting instances into correctly classified ($y_i h_m = +1$) and misclassified ($y_i h_m = -1$):
     $$\mathcal{L}(\alpha_m) = e^{-\alpha_m} \sum_{y_i = h_m} w_i^{(m)} + e^{\alpha_m} \sum_{y_i \neq h_m} w_i^{(m)} = e^{-\alpha_m}(1 - \epsilon_m) + e^{\alpha_m} \epsilon_m$$
   - Differentiating with respect to $\alpha_m$ and setting to zero:
     $$\frac{d\mathcal{L}}{d\alpha_m} = -e^{-\alpha_m}(1 - \epsilon_m) + e^{\alpha_m}\epsilon_m = 0 \implies e^{2\alpha_m} \epsilon_m = 1 - \epsilon_m$$
     $$e^{2\alpha_m} = \frac{1 - \epsilon_m}{\epsilon_m} \implies 2\alpha_m = \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right) \implies \alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right) \quad \text{(Q.E.D.)}$$
3. **Behavioral Analysis (2 Marks)**:
   - If $\epsilon_m = 0.5$ (random guess), $\alpha_m = \frac{1}{2}\ln(1) = 0$ (zero voting power).
   - As $\epsilon_m \to 0$ (perfect weak learner), $\alpha_m \to \infty$ (dominates ensemble).

---

### Q28. [Medium] Formulate the $k$-Means objective function (Distortion Measure). Prove that Lloyd's algorithm converges in a finite number of iterations. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - $k$-Means partitions $N$ observations into $K$ clusters to minimize the Within-Cluster Sum of Squares (WCSS), also called the Distortion Measure $J$.
2. **Mathematical Objective & Coordinate Descent (2 Marks)**:
   $$J(\mathbf{R}, \boldsymbol{\mu}) = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2, \quad r_{ik} \in \{0, 1\}, \ \sum_{k=1}^K r_{ik} = 1$$
   - **Assignment Step**: Minimize $J$ w.r.t $r_{ik}$ holding $\boldsymbol{\mu}_k$ fixed $\implies$ assign each point to nearest centroid. $J$ strictly decreases or stays same.
   - **Update Step**: Minimize $J$ w.r.t $\boldsymbol{\mu}_k$ holding $r_{ik}$ fixed $\implies \boldsymbol{\mu}_k = \frac{\sum r_{ik} \mathbf{x}_i}{\sum r_{ik}}$ (centroid mean). $J$ strictly decreases or stays same.
3. **Proof of Finite Convergence (2 Marks)**:
   - The number of possible distinct partitions of $N$ points into $K$ clusters is finite (bounded by $K^N$).
   - Since $J \ge 0$ is bounded below, and each alternating step strictly reduces $J$ unless cluster assignments do not change, the algorithm cannot cycle infinitely through configurations. Therefore, it **must terminate at a local minimum in a finite number of iterations**.

---

### Q29. [Medium] Describe the $k$-Means++ initialization algorithm. Why is it mathematically superior to random initialization? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - $k$-Means++ seeds initial centroids sequentially using a non-uniform probability distribution proportional to squared Euclidean distance to the nearest existing centroid.
2. **Mathematical Algorithm (2 Marks)**:
   1. Choose first centroid $\boldsymbol{\mu}_1$ uniformly at random from $\mathcal{D}$.
   2. For subsequent centroids $k = 2, \dots, K$:
      - For each point $\mathbf{x}_i$, compute squared distance to closest chosen centroid: $D(\mathbf{x}_i)^2 = \min_{j < k} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$.
      - Select next centroid $\boldsymbol{\mu}_k$ with probability:
        $$P(\mathbf{x}_i = \boldsymbol{\mu}_k) = \frac{D(\mathbf{x}_i)^2}{\sum_{m=1}^N D(\mathbf{x}_m)^2}$$
3. **Optimality Bound (2 Marks)**:
   - Standard random seeding often places multiple centroids in the same natural cluster, leading to poor local minima.
   - Arthur & Vassilvitskii proved that $k$-Means++ guarantees an expected distortion bound: $\mathbb{E}[J] \le 8(\ln K + 2) J_{\text{opt}}$, provably within $\mathcal{O}(\log K)$ of the global optimum.

---

### Q30. [Tough] Derive the Expectation-Maximization (EM) update equations for a Gaussian Mixture Model (GMM). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - GMM models data density as a superposition of $K$ Gaussian densities: $p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)$. Because direct log-likelihood maximization is intractable, EM introduces latent cluster indicators $\mathbf{z}_i \in \{0, 1\}^K$.
2. **Mathematical Derivation of E-Step and M-Step (2 Marks)**:
   - **E-Step**: Compute posterior responsibilities $\gamma_{ik} = P(z_{ik} = 1 \mid \mathbf{x}_i)$ via Bayes' Rule:
     $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \mathbf{\Sigma}_j)}$$
   - **M-Step**: Maximize expected complete-data log-likelihood w.r.t parameters:
     - Effective cluster count: $N_k = \sum_{i=1}^N \gamma_{ik}$
     - Mean: $\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} \mathbf{x}_i$
     - Covariance: $\mathbf{\Sigma}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^T$
     - Prior weight: $\pi_k^{\text{new}} = \frac{N_k}{N}$
3. **Equivalence Insight (2 Marks)**:
   - When covariances are constrained to $\mathbf{\Sigma}_k = \sigma^2 \mathbf{I}$ and $\sigma^2 \to 0$, responsibilities collapse to hard binary indicators $\gamma_{ik} \in \{0, 1\}$, and EM simplifies exactly into the Lloyd $k$-Means algorithm.

---

# Part VI: Model Evaluation, Comparison & Emerging Requirements (Module 11)

### Q31. [Easy] Define Precision, Recall, Specificity, and $F_1$-score using a Confusion Matrix. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Confusion matrix counts binary prediction outcomes: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
2. **Mathematical Formulations (2 Marks)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}, \qquad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
3. **Practical Insight (2 Marks)**:
   - **Precision** penalizes False Alarms (Type I error).
   - **Recall** penalizes Missed Detections (Type II error).
   - $F_1$ is the harmonic mean; it approaches 0 if either Precision or Recall drops to 0, providing a reliable single metric on imbalanced datasets.

---

### Q32. [Medium] Explain the ROC Curve and Area Under the ROC Curve (ROC-AUC). What is the exact probabilistic meaning of ROC-AUC? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - The Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Sensitivity) against False Positive Rate ($1 - \text{Specificity}$) across all discrimination thresholds $\tau \in [0, 1]$.
2. **Mathematical Formulation (2 Marks)**:
   $$\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \qquad \text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{TN}(\tau) + \text{FP}(\tau)}$$
   $$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) d(\text{FPR})$$
3. **Probabilistic Interpretation (2 Marks)**:
   $$\text{ROC-AUC} = P(\text{Score}(\mathbf{x}^+) > \text{Score}(\mathbf{x}^-))$$
   - ROC-AUC equals the exact probability that a randomly chosen positive instance is scored higher by the model than a randomly chosen negative instance. $\text{AUC} = 0.5$ represents random guessing; $\text{AUC} = 1.0$ represents perfect ranking.

---

### Q33. [Medium] Explain McNemar's Test for comparing two machine learning classifiers. State the test statistic and degree of freedom. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - McNemar's test is a paired non-parametric statistical hypothesis test evaluated on a $2 \times 2$ contingency table of discordant prediction errors between two classifiers evaluated on the exact same test dataset.
2. **Contingency Table & Mathematical Statistic (2 Marks)**:
   | | Classifier B Correct | Classifier B Incorrect |
   | :--- | :---: | :---: |
   | **Classifier A Correct** | $e_{00}$ | $e_{01}$ (A right, B wrong) |
   | **Classifier A Incorrect** | $e_{10}$ (A wrong, B right) | $e_{11}$ |

   - Under null hypothesis $H_0$: Classifiers have equal error rates ($P(e_{01}) = P(e_{10})$).
   - For $e_{01} + e_{10} \ge 25$, the continuity-corrected Chi-Square statistic is:
     $$\chi^2 = \frac{(|e_{01} - e_{10}| - 1)^2}{e_{01} + e_{10}} \sim \chi^2_1$$
3. **Decision Rule (2 Marks)**:
   - Degrees of freedom $= 1$.
   - For significance level $\alpha = 0.05$, critical value is $\chi^2_{\text{critical}} = 3.841$. If $\chi^2 > 3.841$, reject $H_0$ and conclude that the difference between the two models is statistically significant.

---

### Q34. [Medium] Differentiate between Demographic Parity and Equalized Odds in algorithmic fairness. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - Fairness metrics ensure that predictive machine learning models do not systematically discriminate against individuals based on sensitive attributes $A$ (e.g., race, gender, age).
2. **Mathematical Formulations (2 Marks)**:
   - **Demographic Parity (Statistical Parity)**: Requires equal positive acceptance rates across protected groups:
     $$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$
   - **Equalized Odds**: Requires both True Positive Rate and False Positive Rate to be equal across protected groups:
     $$P(\hat{Y} = 1 \mid A = 0, Y = y) = P(\hat{Y} = 1 \mid A = 1, Y = y), \quad \forall y \in \{0, 1\}$$
3. **Trade-off Insight (2 Marks)**:
   - Demographic Parity forces equal positive predictions even if base ground-truth rates differ between groups, potentially degrading predictive accuracy.
   - Equalized Odds conditions on true outcome $Y$, penalizing models that make disparate error rates on qualified versus unqualified candidates across groups.

---

### Q35. [Easy] Explain SHAP (Shapley Additive Explanations) for model interpretability. What mathematical axioms does SHAP uniquely satisfy? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Formal Definition (2 Marks)**:
   - SHAP is a game-theoretic feature attribution framework that explains individual model predictions by calculating the marginal contribution of each feature across all possible feature coalitions.
2. **Mathematical Formulation (2 Marks)**:
   $$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
   - Explanation model: $g(z') = \phi_0 + \sum_{i=1}^d \phi_i z_i'$.
3. **Axiomatic Properties (2 Marks)**:
   - SHAP is the only feature attribution method that provably satisfies:
     1. **Efficiency**: Feature contributions sum exactly to the difference between model output and expected baseline: $\sum \phi_i = f(\mathbf{x}) - \mathbb{E}[f(\mathbf{x})]$.
     2. **Symmetry**: Identical features receive identical attributions.
     3. **Additivity**: Attributions across ensemble trees can be summed linearly.

---

# Part VII: Step-by-Step Solved Exam Numericals (Questions 36 to 50)

### Q36. [Medium] Numerical: Given points $(1, 2), (2, 3), (3, 5)$, compute the OLS parameter vector $\mathbf{w} = [w_0, w_1]^T$ using the Normal Equations. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Matrix Setup (2 Marks)**:
   $$\mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}$$
2. **Matrix Multiplications (2 Marks)**:
   $$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 1+1+1 & 1+2+3 \\ 1+2+3 & 1+4+9 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$$
   $$\mathbf{X}^T \mathbf{y} = \begin{bmatrix} 2+3+5 \\ 2(1)+3(2)+5(3) \end{bmatrix} = \begin{bmatrix} 10 \\ 23 \end{bmatrix}$$
3. **Inversion & Weight Vector (2 Marks)**:
   $$\det = (3)(14) - (6)(6) = 42 - 36 = 6$$
   $$(\mathbf{X}^T \mathbf{X})^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}$$
   $$\mathbf{w} = \frac{1}{6} \begin{bmatrix} 14(10) - 6(23) \\ -6(10) + 3(23) \end{bmatrix} = \frac{1}{6} \begin{bmatrix} 140 - 138 \\ -60 + 69 \end{bmatrix} = \frac{1}{6} \begin{bmatrix} 2 \\ 9 \end{bmatrix} = \begin{bmatrix} \mathbf{0.333} \\ \mathbf{1.500} \end{bmatrix}$$
   Fitted line: $\hat{y} = 0.333 + 1.500 x$.

---

### Q37. [Medium] Numerical: For the dataset in Q36, compute Ridge regression weights with $\lambda = 2.0$ (penalizing only slope $w_1$). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Formulate Regularized Matrix (2 Marks)**:
   $$\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_{\text{reg}} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix} + \begin{bmatrix} 0 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 16 \end{bmatrix}$$
2. **Matrix Inversion (2 Marks)**:
   $$\det = (3)(16) - (6)(6) = 48 - 36 = 12$$
   $$(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_{\text{reg}})^{-1} = \frac{1}{12} \begin{bmatrix} 16 & -6 \\ -6 & 3 \end{bmatrix}$$
3. **Weight Vector & Shrinkage Comparison (2 Marks)**:
   $$\mathbf{w}_{\text{Ridge}} = \frac{1}{12} \begin{bmatrix} 16(10) - 6(23) \\ -6(10) + 3(23) \end{bmatrix} = \frac{1}{12} \begin{bmatrix} 160 - 138 \\ -60 + 69 \end{bmatrix} = \frac{1}{12} \begin{bmatrix} 22 \\ 9 \end{bmatrix} = \begin{bmatrix} \mathbf{1.833} \\ \mathbf{0.750} \end{bmatrix}$$
   Slope $w_1$ shrank from $1.500 \to 0.750$.

---

### Q38. [Medium] Numerical: In binary logistic regression, $\mathbf{w} = [-1.0, 0.5]^T$, learning rate $\eta = 0.1$. For point $x = 4.0$ with true label $y = 1$, compute prediction, loss, and updated weight vector after 1 step. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Forward Pass Logit & Probability (2 Marks)**:
   Augmented feature: $\mathbf{x} = [1.0, 4.0]^T$.
   $$z = w_0(1) + w_1(4) = -1.0 + 0.5(4.0) = -1.0 + 2.0 = +1.0$$
   $$\hat{y} = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} = \frac{1}{1 + 0.3679} \approx \mathbf{0.7311}$$
2. **Binary Cross-Entropy Loss (2 Marks)**:
   $$\mathcal{L} = - [1 \cdot \ln(0.7311) + 0] = -\ln(0.7311) \approx \mathbf{0.3132}$$
3. **Gradient & Parameter Update (2 Marks)**:
   Error: $e = \hat{y} - y = 0.7311 - 1.0 = -0.2689$.
   $$\nabla_{\mathbf{w}} \mathcal{L} = e \cdot \mathbf{x} = -0.2689 \begin{bmatrix} 1.0 \\ 4.0 \end{bmatrix} = \begin{bmatrix} -0.2689 \\ -1.0756 \end{bmatrix}$$
   $$\mathbf{w}^{(1)} = \mathbf{w}^{(0)} - \eta \nabla \mathcal{L} = \begin{bmatrix} -1.0 \\ 0.5 \end{bmatrix} - 0.1 \begin{bmatrix} -0.2689 \\ -1.0756 \end{bmatrix} = \begin{bmatrix} -1.0 + 0.0269 \\ 0.5 + 0.1076 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.9731} \\ \mathbf{0.6076} \end{bmatrix}$$

---

### Q39. [Medium] Numerical: Given a set $S$ of 14 examples: 9 Yes, 5 No. An attribute $A$ splits $S$ into $S_1$ (6 Yes, 2 No) and $S_2$ (3 Yes, 3 No). Calculate $\text{Gain}(S, A)$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Base Entropy $H(S)$ (2 Marks)**:
   $$H(S) = - \left[ \frac{9}{14} \log_2\left(\frac{9}{14}\right) + \frac{5}{14} \log_2\left(\frac{5}{14}\right) \right] = - [0.6429(-0.6374) + 0.3571(-1.4854)] = \mathbf{0.9403 \text{ bits}}$$
2. **Subset Entropies $H(S_1)$ and $H(S_2)$ (2 Marks)**:
   - $S_1$ ($|S_1| = 8$): $p_+ = \frac{6}{8} = 0.75, p_- = 0.25$.
     $$H(S_1) = - [0.75 \log_2(0.75) + 0.25 \log_2(0.25)] = \mathbf{0.8113 \text{ bits}}$$
   - $S_2$ ($|S_2| = 6$): $p_+ = \frac{3}{6} = 0.50, p_- = 0.50$.
     $$H(S_2) = \mathbf{1.0000 \text{ bits}}$$
3. **Information Gain (2 Marks)**:
   $$H(S \mid A) = \frac{8}{14}(0.8113) + \frac{6}{14}(1.0000) = 0.4636 + 0.4286 = 0.8922 \text{ bits}$$
   $$\text{Gain}(S, A) = H(S) - H(S \mid A) = 0.9403 - 0.8922 = \mathbf{0.0481 \text{ bits}}$$

---

### Q40. [Medium] Numerical: For query point $\mathbf{x}_q = [2, 3]^T$, compute Euclidean distances to: $\mathbf{x}_1=[1, 1]^T (y=0)$, $\mathbf{x}_2=[2, 5]^T (y=1)$, $\mathbf{x}_3=[4, 3]^T (y=1)$. Predict class using 3-NN distance-weighted voting ($w_i = 1/d_i^2$). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Distance Calculations (2 Marks)**:
   - $d_1 = \sqrt{(2-1)^2 + (3-1)^2} = \sqrt{1 + 4} = \sqrt{5} \approx 2.236$
   - $d_2 = \sqrt{(2-2)^2 + (3-5)^2} = \sqrt{0 + 4} = \sqrt{4} = 2.000$
   - $d_3 = \sqrt{(2-4)^2 + (3-3)^2} = \sqrt{4 + 0} = \sqrt{4} = 2.000$
2. **Inverse Squared Distance Weights (2 Marks)**:
   - $w_1 = \frac{1}{d_1^2} = \frac{1}{5} = \mathbf{0.200}$ (Class 0)
   - $w_2 = \frac{1}{d_2^2} = \frac{1}{4} = \mathbf{0.250}$ (Class 1)
   - $w_3 = \frac{1}{d_3^2} = \frac{1}{4} = \mathbf{0.250}$ (Class 1)
3. **Class Aggregation & Decision (2 Marks)**:
   - Total Score for Class 0: $w_1 = \mathbf{0.200}$
   - Total Score for Class 1: $w_2 + w_3 = 0.250 + 0.250 = \mathbf{0.500}$
   - Decision: Since $0.500 > 0.200$, predicted class is $\hat{y} = \mathbf{1}$.

---

### Q41. [Tough] Numerical: Given support vectors $\mathbf{x}_1 = [2, 0]^T (y_1=+1)$ and $\mathbf{x}_2 = [0, 0]^T (y_2=-1)$. Find the maximum margin hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$, margin width, and classify $\mathbf{x}_{\text{test}} = [3, 1]^T$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Hyperplane Equations for Support Vectors (2 Marks)**:
   Support vectors satisfy $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$:
   - For $\mathbf{x}_1$: $+1(w_1(2) + w_2(0) + b) = 1 \implies 2w_1 + b = 1$
   - For $\mathbf{x}_2$: $-1(w_1(0) + w_2(0) + b) = 1 \implies -b = 1 \implies \mathbf{b = -1.0}$
   - Substitute $b = -1$: $2w_1 - 1 = 1 \implies 2w_1 = 2 \implies \mathbf{w_1 = 1.0}$.
   - Since $x_2$ has no variance between support vectors, $w_2 = 0$.
   $$\mathbf{w} = [1.0, 0.0]^T, \quad b = -1.0$$
   Boundary equation: $x_1 - 1 = 0 \implies x_1 = 1.0$.
2. **Margin Width (2 Marks)**:
   $$\|\mathbf{w}\|_2 = \sqrt{1.0^2 + 0.0^2} = 1.0$$
   $$\text{Margin } \gamma = \frac{2}{\|\mathbf{w}\|_2} = \frac{2}{1.0} = \mathbf{2.00}$$
3. **Classification of $\mathbf{x}_{\text{test}} = [3, 1]^T$ (2 Marks)**:
   $$f(\mathbf{x}_{\text{test}}) = w_1(3) + w_2(1) + b = 1.0(3) + 0 - 1.0 = +2.0 > 0$$
   $$\hat{y}_{\text{test}} = \mathbf{+1}$$

---

### Q42. [Medium] Numerical: In Naïve Bayes, class priors are $P(C_1) = 0.6, P(C_2) = 0.4$. For binary features $\mathbf{x} = [1, 0]^T$:  
$P(x_1=1 \mid C_1) = 0.7, P(x_2=0 \mid C_1) = 0.8$.  
$P(x_1=1 \mid C_2) = 0.3, P(x_2=0 \mid C_2) = 0.2$.  
Compute posterior probabilities $P(C_1 \mid \mathbf{x})$ and $P(C_2 \mid \mathbf{x})$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Unnormalized Posterior Scores (2 Marks)**:
   $$\text{Score}(C_1) = P(C_1) \cdot P(x_1=1 \mid C_1) \cdot P(x_2=0 \mid C_1) = 0.6 \times 0.7 \times 0.8 = \mathbf{0.336}$$
   $$\text{Score}(C_2) = P(C_2) \cdot P(x_1=1 \mid C_2) \cdot P(x_2=0 \mid C_2) = 0.4 \times 0.3 \times 0.2 = \mathbf{0.024}$$
2. **Total Evidence (2 Marks)**:
   $$P(\mathbf{x}) = \text{Score}(C_1) + \text{Score}(C_2) = 0.336 + 0.024 = \mathbf{0.360}$$
3. **Normalized Probabilities & Prediction (2 Marks)**:
   $$P(C_1 \mid \mathbf{x}) = \frac{0.336}{0.360} = \frac{14}{15} \approx \mathbf{93.33\%}$$
   $$P(C_2 \mid \mathbf{x}) = \frac{0.024}{0.360} = \frac{1}{15} \approx \mathbf{6.67\%}$$
   Predicted class: $\mathbf{C_1}$.

---

### Q43. [Medium] Numerical: In AdaBoost round 1 with 4 samples ($w_i = 0.25$), a weak classifier misclassifies sample 3. Compute error $\epsilon_1$, classifier weight $\alpha_1$, and updated normalized weights $\mathbf{w}^{(2)}$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Error $\epsilon_1$ and Alpha $\alpha_1$ (2 Marks)**:
   $$\epsilon_1 = w_3 = \mathbf{0.25}$$
   $$\alpha_1 = \frac{1}{2} \ln \left( \frac{1 - 0.25}{0.25} \right) = \frac{1}{2} \ln(3.0) = \frac{1}{2}(1.0986) \approx \mathbf{0.5493}$$
2. **Unnormalized Weight Multipliers (2 Marks)**:
   - For correct samples (1, 2, 4): $e^{-\alpha_1} = e^{-0.5493} = \frac{1}{\sqrt{3}} \approx 0.5774$.
     $$u_1 = u_2 = u_4 = 0.25 \times 0.5774 = \mathbf{0.1443}$$
   - For misclassified sample 3: $e^{+\alpha_1} = e^{+0.5493} = \sqrt{3} \approx 1.7321$.
     $$u_3 = 0.25 \times 1.7321 = \mathbf{0.4330}$$
3. **Normalization Factor $Z_1$ and Weights (2 Marks)**:
   $$Z_1 = 3(0.1443) + 0.4330 = 0.4330 + 0.4330 = \mathbf{0.8660}$$
   $$w_1^{(2)} = w_2^{(2)} = w_4^{(2)} = \frac{0.1443}{0.8660} = \frac{1}{6} \approx \mathbf{0.1667}$$
   $$w_3^{(2)} = \frac{0.4330}{0.8660} = \frac{1}{2} = \mathbf{0.5000}$$

---

### Q44. [Medium] Numerical: In $k$-Means with $K=2$, points are $\{1, 2, 5, 6, 9\}$. Initial centroids: $\mu_1 = 1.0, \mu_2 = 5.0$. Perform 1 iteration and compute updated centroids. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Distance Comparison & Cluster Assignment (2 Marks)**:
   - Point 1: $|1-1|=0$ vs $|1-5|=4 \implies$ Cluster 1
   - Point 2: $|2-1|=1$ vs $|2-5|=3 \implies$ Cluster 1
   - Point 5: $|5-1|=4$ vs $|5-5|=0 \implies$ Cluster 2
   - Point 6: $|6-1|=5$ vs $|6-5|=1 \implies$ Cluster 2
   - Point 9: $|9-1|=8$ vs $|9-5|=4 \implies$ Cluster 2
   Partitions: $C_1 = \{1, 2\}$, $C_2 = \{5, 6, 9\}$.
2. **Centroid Re-computation (2 Marks)**:
   $$\mu_1^{(1)} = \frac{1 + 2}{2} = \mathbf{1.50}$$
   $$\mu_2^{(1)} = \frac{5 + 6 + 9}{3} = \frac{20}{3} \approx \mathbf{6.67}$$
3. **Distortion Measure (Inertia) (2 Marks)**:
   $$J_1 = (1 - 1.5)^2 + (2 - 1.5)^2 = 0.25 + 0.25 = 0.50$$
   $$J_2 = (5 - 6.67)^2 + (6 - 6.67)^2 + (9 - 6.67)^2 = 2.7889 + 0.4489 + 5.4289 = 8.6667$$
   $$J_{\text{total}} = 0.50 + 8.6667 = \mathbf{9.1667}$$

---

### Q45. [Medium] Numerical: Given confusion matrix: $\text{TP} = 90, \text{FP} = 30, \text{FN} = 10, \text{TN} = 870$. Calculate Accuracy, Precision, Recall, Specificity, and $F_1$-score. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Total Sample Count & Accuracy (2 Marks)**:
   $$N = 90 + 30 + 10 + 870 = 1,000$$
   $$\text{Accuracy} = \frac{90 + 870}{1000} = \frac{960}{1000} = \mathbf{96.00\%}$$
2. **Precision, Recall, Specificity (2 Marks)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{90}{90 + 30} = \frac{90}{120} = \mathbf{75.00\%}$$
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{90}{90 + 10} = \frac{90}{100} = \mathbf{90.00\%}$$
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{870}{870 + 30} = \frac{870}{900} \approx \mathbf{96.67\%}$$
3. **$F_1$-Score (2 Marks)**:
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.75 \times 0.90}{0.75 + 0.90} = 2 \cdot \frac{0.675}{1.65} = \frac{1.35}{1.65} \approx \mathbf{81.82\%}$$

---

### Q46. [Tough] Numerical: Two classifiers evaluated on 500 test samples yield discordant error counts $e_{01} = 38$ (A correct, B wrong) and $e_{10} = 12$ (A wrong, B correct). Perform McNemar's test ($\alpha = 0.05, \chi^2_{\text{critical}} = 3.841$). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Hypothesis Formulation & Sample Size Check (2 Marks)**:
   - Null Hypothesis $H_0$: Classifier A and Classifier B have equal marginal error rates ($P(e_{01}) = P(e_{10})$).
   - Discordant count: $e_{01} + e_{10} = 38 + 12 = 50 \ge 25$ (Gaussian continuity-corrected test is valid).
2. **Compute McNemar's Chi-Square Statistic (2 Marks)**:
   $$\chi^2 = \frac{(|e_{01} - e_{10}| - 1)^2}{e_{01} + e_{10}} = \frac{(|38 - 12| - 1)^2}{38 + 12} = \frac{(26 - 1)^2}{50} = \frac{25^2}{50} = \frac{625}{50} = \mathbf{12.50}$$
3. **Statistical Decision & Conclusion (2 Marks)**:
   $$\chi^2_{\text{calculated}} = \mathbf{12.50} > \chi^2_{\text{critical}} = \mathbf{3.841}$$
   - **Conclusion**: Decisively reject $H_0$. Classifier A is statistically significantly superior to Classifier B at the $95\%$ confidence level ($p < 0.001$).

---

### Q47. [Medium] Numerical: Given target vector $\mathbf{y} = [1, 0, 1, 1, 0]^T$ and predicted continuous probabilities $\mathbf{\hat{y}} = [0.9, 0.2, 0.8, 0.4, 0.1]^T$. Compute the ROC-AUC score using pairwise rank comparisons. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Partition Scores by Class (2 Marks)**:
   - Positives ($y=1$): $\{0.9, 0.8, 0.4\} \implies n_+ = 3$.
   - Negatives ($y=0$): $\{0.2, 0.1\} \implies n_- = 2$.
   - Total positive-negative pairs: $n_+ \times n_- = 3 \times 2 = 6$ pairs.
2. **Pairwise Ranking Evaluations (2 Marks)**:
   - Positive $0.9$: $0.9 > 0.2$ (Win), $0.9 > 0.1$ (Win) $\implies 2$ points.
   - Positive $0.8$: $0.8 > 0.2$ (Win), $0.8 > 0.1$ (Win) $\implies 2$ points.
   - Positive $0.4$: $0.4 > 0.2$ (Win), $0.4 > 0.1$ (Win) $\implies 2$ points.
3. **ROC-AUC Calculation (2 Marks)**:
   - Total Wins $= 2 + 2 + 2 = 6$.
   $$\text{AUC} = \frac{\text{Total Wins}}{n_+ \times n_-} = \frac{6}{6} = \mathbf{1.00}$$
   The classifier achieves a perfect ROC-AUC score of 1.0 (flawless rank ordering).

---

### Q48. [Medium] Numerical: In an automated loan approval model, Group $A=0$ has 100 applicants with 40 approvals. Group $A=1$ has 200 applicants with 120 approvals. Check whether Demographic Parity is satisfied. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Mathematical Definition of Demographic Parity (2 Marks)**:
   - Demographic Parity requires identical positive prediction rates across protected demographic attributes:
     $$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$
2. **Calculate Group Acceptance Rates (2 Marks)**:
   - For Group $A = 0$:
     $$P(\hat{Y} = 1 \mid A = 0) = \frac{40}{100} = \mathbf{0.40 \ (40\%)}$$
   - For Group $A = 1$:
     $$P(\hat{Y} = 1 \mid A = 1) = \frac{120}{200} = \mathbf{0.60 \ (60\%)}$$
3. **Disparate Impact Ratio & Verdict (2 Marks)**:
   $$\text{Disparate Impact Ratio} = \frac{P(\hat{Y} = 1 \mid A = 0)}{P(\hat{Y} = 1 \mid A = 1)} = \frac{0.40}{0.60} = \frac{2}{3} \approx \mathbf{0.667}$$
   - **Verdict**: Demographic Parity is **violated**. Under the legal four-fifths rule ($\text{ratio} < 0.80$), a ratio of $0.667$ indicates actionable disparate impact against Group $0$.

---

### Q49. [Tough] Numerical: In XGBoost, a leaf node contains 3 samples with first gradients $g = [-0.5, -0.4, 0.2]$ and second Hessians $h = [0.25, 0.20, 0.15]$. With $L_2$ leaf regularization $\lambda = 1.0$, compute the optimal leaf weight $w^*$ and the leaf quality score. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Sum of Gradients and Hessians (2 Marks)**:
   $$G = \sum g_i = -0.5 - 0.4 + 0.2 = \mathbf{-0.70}$$
   $$H = \sum h_i = 0.25 + 0.20 + 0.15 = \mathbf{0.60}$$
2. **Compute Optimal Leaf Weight $w^*$ (2 Marks)**:
   $$w^* = - \frac{G}{H + \lambda} = - \frac{-0.70}{0.60 + 1.00} = - \frac{-0.70}{1.60} = \frac{0.70}{1.60} = \mathbf{+0.4375}$$
3. **Compute Leaf Quality Score (2 Marks)**:
   $$\text{Score} = \frac{1}{2} \frac{G^2}{H + \lambda} = \frac{1}{2} \frac{(-0.70)^2}{1.60} = \frac{1}{2} \frac{0.49}{1.60} = \frac{0.49}{3.20} \approx \mathbf{0.1531}$$
   This quality score is utilized by XGBoost to determine split gain when considering candidate partitions.

---

### Q50. [Medium] Numerical: Given raw vector $\mathbf{x} = [2, 4, 6, 8, 10]^T$, compute its Min-Max normalized vector in range $[0, 1]$ and its Z-score standardized vector. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Summary Statistics (2 Marks)**:
   - $x_{\min} = 2.0, x_{\max} = 10.0 \implies \Delta = 10.0 - 2.0 = 8.0$.
   - Mean $\mu = \frac{2+4+6+8+10}{5} = \mathbf{6.0}$.
   - Variance $\sigma^2 = \frac{(-4)^2 + (-2)^2 + 0^2 + 2^2 + 4^2}{5} = \frac{16+4+0+4+16}{5} = \frac{40}{5} = 8.0$.
   - Standard deviation $\sigma = \sqrt{8.0} \approx \mathbf{2.828}$.
2. **Min-Max Normalization (2 Marks)**:
   $$x_{\text{norm}, i} = \frac{x_i - 2.0}{8.0}$$
   $$\mathbf{x}_{\text{norm}} = [0.00, 0.25, 0.50, 0.75, 1.00]^T$$
3. **Z-Score Standardization (2 Marks)**:
   $$z_i = \frac{x_i - 6.0}{2.828}$$
   - $z_1 = \frac{2-6}{2.828} = \mathbf{-1.414}$
   - $z_2 = \frac{4-6}{2.828} = \mathbf{-0.707}$
   - $z_3 = \frac{6-6}{2.828} = \mathbf{0.000}$
   - $z_4 = \frac{8-6}{2.828} = \mathbf{+0.707}$
   - $z_5 = \frac{10-6}{2.828} = \mathbf{+1.414}$
   $$\mathbf{z} = [-1.414, -0.707, 0.000, +0.707, +1.414]^T$$

