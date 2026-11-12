# Module 12: Optimization & Regularization for Deep Models

**Course**: AIML ZG511 — Deep Neural Networks  
**Prescribed References**: Zhang et al. (T1, Ch 3.6-3.7, 4.6-4.7, Ch 12); Goodfellow et al. (R1, Ch 7 & Ch 8)  
**Primary Teacher Followed**: Prof. Shakshi Sharma (Session 13: 2026-10-31)

---

## 1. Challenges in Deep Model Optimization

Unlike linear models where loss surfaces are strictly convex bowls, deep neural network loss surfaces are non-convex, non-linear high-dimensional landscapes:

```
Loss Surface Challenges:
1. Ill-conditioned Curvature (Hessian condition number λ_max / λ_min >> 1)
2. Saddle Points (Gradients = 0, but Hessian has mixed positive/negative eigenvalues)
3. Local Minima & Plateaus
4. Vanishing / Exploding Gradients
```

---

## 2. Gradient Descent Variants & Optimizers

![Optimizers and Regularization](../assets/M12_optimizers_and_regularization.jpg)
*Figure 12.1: Convergence trajectories of SGD, Momentum, RMSProp, and Adam on non-convex surfaces (Prof. Shakshi Sharma Lecture 13).*

### 2.1 SGD with Momentum
Standard SGD oscillates violently across narrow ravines. Momentum adds an exponential moving average of past gradients, dampening oscillations and accelerating through ravines:
$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t) \quad (\text{typically } \beta = 0.9)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_t$$

### 2.2 RMSProp (Root Mean Square Propagation — Hinton, 2012)
Maintains an exponential moving average of squared gradients to adaptively scale learning rates per parameter:
$$\mathbf{s}_t = \gamma \mathbf{s}_{t-1} + (1 - \gamma) (\nabla \mathcal{L}_t)^{\odot 2} \quad (\text{typically } \gamma = 0.99)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \nabla \mathcal{L}_t$$

### 2.3 Adam (Adaptive Moment Estimation — Kingma & Ba, 2014)
Combines the benefits of Momentum (1st moment) and RMSProp (2nd moment) with bias correction:
1. **1st Moment (Mean)**: $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$ ($\beta_1 = 0.9$)
2. **2nd Moment (Uncentered Variance)**: $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$ ($\beta_2 = 0.999$)
3. **Bias Corrections** (corrects initialization bias towards zero):
   $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
4. **Parameter Update**:
   $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

---

## 3. Regularization Techniques for Deep Models

Regularization prevents overfitting, improving generalization to unseen data.

### 3.1 Weight Decay ($L_2$ Regularization)
Augments the objective function with a penalty on the squared Euclidean norm of parameters:
$$\tilde{\mathcal{L}}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
Gradient update step:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \left( \nabla \mathcal{L}(\mathbf{w}_t) + \lambda \mathbf{w}_t \right) = (1 - \eta \lambda) \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$$
The factor $(1 - \eta \lambda) < 1$ shrinks the weight vector by a constant fraction at every step, driving uninformative weights toward zero.

### 3.2 Dropout (Srivastava et al., 2014)
- **During Training**: Randomly zeros out activations of neurons with probability $p$ independently at each step.
  $$\tilde{\mathbf{a}} = \frac{1}{1 - p} \left( \mathbf{m} \odot \mathbf{a} \right), \quad m_i \sim \operatorname{Bernoulli}(1 - p)$$
  *(Inverted dropout divides by $1-p$ during training so that test-time code requires no modification)*.
- **Why It Works**: Prevents co-adaptation of neurons; can be mathematically interpreted as training an exponential ensemble of $2^M$ thinned sub-networks.

### 3.3 Batch Normalization vs Layer Normalization

| Normalization Technique | Normalized Dimension | Batch Size Dependency | Inference Mechanics | Primary Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **Batch Normalization (BN)** | Across the mini-batch dimension $B$ for each feature channel independently. | Dependent on batch size ($B \ge 16$ required for stable statistics). | Uses frozen running mean and variance computed during training. | CNNs (ResNet, VGG, ConvNets) |
| **Layer Normalization (LN)** | Across all feature channels $C$ for each individual instance independently. | **Zero batch dependency** (operates identically for $B=1$). | Computes exact mean and variance directly on the single test instance. | Transformers (BERT, GPT, ViT), RNNs |

#### Mathematical Formulation of LayerNorm:
For an activation vector $\mathbf{x} \in \mathbb{R}^D$:
$$\mu = \frac{1}{D} \sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$$
$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$\mathbf{y} = \boldsymbol{\gamma} \odot \hat{\mathbf{x}} + \boldsymbol{\beta}$$
where $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^D$ are learnable scale and shift parameters.
