# Module 5: Deep Feedforward Neural Networks (DFNN) & Backpropagation

**Course**: AIML ZG511 — Deep Neural Networks  
**Prescribed References**: Zhang et al. (T1, Ch 5); Goodfellow et al. (R1, Ch 6); Companion: `M5_DFNN_Handwritten..pdf`, `BPNNDerivation.pdf`  
**Primary Teacher Followed**: Prof. Shakshi Sharma (Session 6: 2026-08-29, Recording `Shakshi -- 7.mp4`)  
**Exam Reference**: Prof. Mithun Ghosh Question Paper (Q27, Q28, Q29, Q30, Q31, Q32, Q33, Q34)

---

## 1. Multi-Layer Perceptrons (MLP) as Solution to XOR

A single linear hyperplane cannot partition the non-linearly separable XOR truth table. By chaining hidden layers parameterized with non-linear activation functions, an MLP bends and folds the input coordinate space, projecting non-separable points into a latent manifold where they become linearly separable.

```
Input Layer           Hidden Layer           Output Layer
   (x1) ---------\ /----> [ h1 ] ---------\
                  X                        +-----> ( y_hat )
   (x2) ---------/ \----> [ h2 ] ---------/
```

### Why Non-Linear Activation Functions Are Indispensable
If all activation functions were linear ($g(z) = c z$), a multi-layer network with $L$ layers would collapse mathematically into a single linear transformation:
$$\mathbf{h}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$$
$$\mathbf{h}^{[2]} = \mathbf{W}^{[2]}\mathbf{h}^{[1]} + \mathbf{b}^{[2]} = \mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = (\mathbf{W}^{[2]}\mathbf{W}^{[1]})\mathbf{x} + (\mathbf{W}^{[2]}\mathbf{b}^{[1]} + \mathbf{b}^{[2]}) = \mathbf{W}'\mathbf{x} + \mathbf{b}'$$
*Conclusion*: No matter how many hidden layers are stacked, a deep linear network cannot compute anything beyond linear regression or classification. Non-linear activations introduce the curvature necessary for non-linear decision boundaries.

---

## 2. Activation Functions in Deep Networks

| Activation Function | Formula $g(z)$ | Derivative $g'(z)$ | Output Range | Key Pros & Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $g(z)(1 - g(z))$ | $(0, 1)$ | **Pros**: Smooth probability output.<br>**Cons**: Vanishing gradient for $|z|>4$ ($g' \approx 0$); not zero-centered. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - g(z)^2$ | $(-1, 1)$ | **Pros**: Zero-centered, stronger gradients than sigmoid.<br>**Cons**: Still saturates and causes vanishing gradient. |
| **ReLU** | $\max(0, z)$ | $\begin{cases}1 & z > 0 \\ 0 & z < 0\end{cases}$ | $[0, +\infty)$ | **Pros**: Does not saturate for $z>0$; very fast computation.<br>**Cons**: "Dying ReLU" problem when $z \le 0$ ($g' = 0$). |
| **Leaky ReLU** | $\max(\alpha z, z)$, $\alpha \approx 0.01$ | $\begin{cases}1 & z > 0 \\ \alpha & z \le 0\end{cases}$ | $(-\infty, +\infty)$ | **Pros**: Eliminates dying ReLU by maintaining small non-zero gradient $\alpha$ for $z < 0$. |
| **GELU** | $z \cdot \Phi(z) = z \cdot P(X \le z)$ | Complex analytical | $(-\approx 0.17, +\infty)$ | **Pros**: Probabilistic gating; state-of-the-art in Transformers & modern MLPs. |

---

## 3. Vectorized Forward Propagation Algorithm

![DFNN MLP Architecture](../assets/M05_dfnn_mlp_architecture.jpg)
*Figure 5.1: Multi-layer feedforward architecture with layer indexing and weight tensor dimensions (Slide 18, DNN_M5_DFNN (1).pdf, Prof. Shakshi Sharma Lecture 6).*

Consider an $L$-layer network where layer $l \in \{1, 2, \dots, L\}$ has $n^{[l]}$ neurons. Layer $0$ is the input layer with $n^{[0]} = d$ features.
- Weight matrix for layer $l$: $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$
- Bias vector for layer $l$: $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$
- Input activation: $\mathbf{a}^{[0]} = \mathbf{x} \in \mathbb{R}^{n^{[0]}}$

### Forward Propagation Equations (Layer by Layer):
$$\text{For } l = 1, 2, \dots, L:$$
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$
$$\text{Prediction: } \hat{\mathbf{y}} = \mathbf{a}^{[L]}$$

For a mini-batch of $B$ training instances $\mathbf{A}^{[0]} = \mathbf{X}^T \in \mathbb{R}^{n^{[0]} \times B}$:
$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \mathbf{1}_B^T \in \mathbb{R}^{n^{[l]} \times B}$$
$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

---

## 4. Rigorous Vectorized Backpropagation (The Chain Rule)

![Backpropagation Derivation](../assets/M05_backpropagation_derivation.jpg)
*Figure 5.2: Computation graph and vectorized chain rule derivations for error propagation $\boldsymbol{\delta}^{[l]}$ (Slide 34, DNN_M5_DFNN (1).pdf, Prof. Shakshi Sharma Lecture 6).*

Backpropagation computes the gradient of scalar loss $\mathcal{L}$ with respect to every weight $W_{jk}^{[l]}$ and bias $b_j^{[l]}$ via the chain rule of multivariable calculus.

### 4.1 Definition of Layer Error Vector $\boldsymbol{\delta}^{[l]}$
Define the sensitivity / error vector of layer $l$ as the partial derivative of loss $\mathcal{L}$ with respect to the pre-activation vector $\mathbf{z}^{[l]}$:
$$\boldsymbol{\delta}^{[l]} \triangleq \nabla_{\mathbf{z}^{[l]}} \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial z_1^{[l]}} \\ \frac{\partial \mathcal{L}}{\partial z_2^{[l]}} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial z_{n^{[l]}}^{[l]}} \end{bmatrix} \in \mathbb{R}^{n^{[l]}}$$

### 4.2 Output Layer Sensitivity $\boldsymbol{\delta}^{[L]}$
By the chain rule:
$$\delta_i^{[L]} = \frac{\partial \mathcal{L}}{\partial z_i^{[L]}} = \sum_{k} \frac{\partial \mathcal{L}}{\partial a_k^{[L]}} \frac{\partial a_k^{[L]}}{\partial z_i^{[L]}}$$
- **For element-wise activations** (e.g., Sigmoid with Squared Error Loss):
  $$\boldsymbol{\delta}^{[L]} = (\mathbf{a}^{[L]} - \mathbf{y}) \odot g'^{[L]}(\mathbf{z}^{[L]})$$
- **For Softmax with Categorical Cross-Entropy**:
  $$\boldsymbol{\delta}^{[L]} = \hat{\mathbf{y}} - \mathbf{y}$$

### 4.3 Backward Error Propagation from Layer $l+1$ to Layer $l$
Applying the chain rule through the matrix transformation $\mathbf{z}^{[l+1]} = \mathbf{W}^{[l+1]} \mathbf{a}^{[l]} + \mathbf{b}^{[l+1]}$:
$$\frac{\partial \mathcal{L}}{\partial a_j^{[l]}} = \sum_{k=1}^{n^{[l+1]}} \frac{\partial \mathcal{L}}{\partial z_k^{[l+1]}} \frac{\partial z_k^{[l+1]}}{\partial a_j^{[l]}} = \sum_{k=1}^{n^{[l+1]}} \delta_k^{[l+1]} W_{kj}^{[l+1]} = \left( (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right)_j$$
Now connecting $a_j^{[l]} = g^{[l]}(z_j^{[l]})$:
$$\delta_j^{[l]} = \frac{\partial \mathcal{L}}{\partial z_j^{[l]}} = \frac{\partial \mathcal{L}}{\partial a_j^{[l]}} \cdot g'^{[l]}(z_j^{[l]})$$

In compact vectorized form:
$$\boldsymbol{\delta}^{[l]} = \left( (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right) \odot g'^{[l]}(\mathbf{z}^{[l]})$$
where $\odot$ denotes the Hadamard (element-wise) product.

### 4.4 Parameter Gradients & Updates
1. **Weight Matrix Gradient**:
   $$\frac{\partial \mathcal{L}}{\partial W_{jk}^{[l]}} = \delta_j^{[l]} a_k^{[l-1]} \implies \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$$
2. **Bias Vector Gradient**:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]} \in \mathbb{R}^{n^{[l]}}$$

### 4.5 Gradient Descent Parameter Update
$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$
$$\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$$

---

## 5. Complete Step-by-Step Numerical Walkthrough (Exam Q33)

*(Directly following Prof. Shakshi Sharma's solved exam numerical in `M5_DFNN_Handwritten..pdf` and `DNN_M5_DFNN_Numerical.pdf`)*

### Network Specifications:
- **Architecture**: 2 input neurons ($x_1, x_2$), 2 hidden neurons ($h_1, h_2$), 1 output neuron ($\hat{y}$).
- **Activation Functions**: Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ for both hidden and output layers.
- **Loss Function**: Squared Error $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.
- **Training Sample**: $\mathbf{x} = [0.05, 0.10]^T$, Target $y = 0.01$. Learning rate $\eta = 0.5$.
- **Initial Weights & Biases**:
  $$\mathbf{W}^{[1]} = \begin{bmatrix} 0.15 & 0.20 \\ 0.25 & 0.30 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.35 \\ 0.35 \end{bmatrix}$$
  $$\mathbf{W}^{[2]} = \begin{bmatrix} 0.40 & 0.45 \end{bmatrix}, \quad b^{[2]} = 0.60$$

### Step 1: Forward Pass
1. **Hidden Layer Net Inputs**:
   $$z_1^{[1]} = 0.15(0.05) + 0.20(0.10) + 0.35 = 0.0075 + 0.0200 + 0.35 = 0.3775$$
   $$z_2^{[1]} = 0.25(0.05) + 0.30(0.10) + 0.35 = 0.0125 + 0.0300 + 0.35 = 0.3925$$
2. **Hidden Layer Activations**:
   $$a_1^{[1]} = \sigma(0.3775) = \frac{1}{1 + e^{-0.3775}} \approx 0.59327$$
   $$a_2^{[1]} = \sigma(0.3925) = \frac{1}{1 + e^{-0.3925}} \approx 0.59688$$
3. **Output Layer Net Input**:
   $$z_1^{[2]} = 0.40(0.59327) + 0.45(0.59688) + 0.60 = 0.23731 + 0.26860 + 0.60 = 1.10591$$
4. **Output Activation**:
   $$\hat{y} = a_1^{[2]} = \sigma(1.10591) = \frac{1}{1 + e^{-1.10591}} \approx 0.75137$$
5. **Initial Loss**:
   $$\mathcal{L} = \frac{1}{2}(0.75137 - 0.01)^2 = \frac{1}{2}(0.74137)^2 \approx 0.27481$$

### Step 2: Backward Pass (Error Sensitivities)
1. **Output Layer Error $\delta^{[2]}$**:
   $$\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial z_1^{[2]}} = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y}) = (0.75137 - 0.01) \cdot 0.75137(1 - 0.75137)$$
   $$\delta^{[2]} = 0.74137 \cdot (0.75137 \cdot 0.24863) = 0.74137 \cdot 0.18681 \approx 0.13849$$
2. **Hidden Layer Errors $\boldsymbol{\delta}^{[1]}$**:
   $$\delta_1^{[1]} = (W_{11}^{[2]} \delta^{[2]}) \cdot a_1^{[1]}(1 - a_1^{[1]}) = (0.40 \cdot 0.13849) \cdot 0.59327(1 - 0.59327)$$
   $$\delta_1^{[1]} = 0.055396 \cdot 0.24130 \approx 0.01337$$
   $$\delta_2^{[1]} = (W_{12}^{[2]} \delta^{[2]}) \cdot a_2^{[1]}(1 - a_2^{[1]}) = (0.45 \cdot 0.13849) \cdot 0.59688(1 - 0.59688)$$
   $$\delta_2^{[1]} = 0.062321 \cdot 0.24062 \approx 0.01500$$

### Step 3: Complete Parameter Updates ($\eta = 0.5$)
1. **Output Layer Weights ($W^{[2]}$) and Bias ($b^{[2]}$)**:
   $$\frac{\partial \mathcal{L}}{\partial W_{11}^{[2]}} = \delta^{[2]} a_1^{[1]} = 0.13849 \cdot 0.59327 \approx 0.08216 \implies W_{11}^{[2](\text{new})} = 0.40 - 0.5(0.08216) = \mathbf{0.35892}$$
   $$\frac{\partial \mathcal{L}}{\partial W_{12}^{[2]}} = \delta^{[2]} a_2^{[1]} = 0.13849 \cdot 0.59688 \approx 0.08266 \implies W_{12}^{[2](\text{new})} = 0.45 - 0.5(0.08266) = \mathbf{0.40867}$$
   $$\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \delta^{[2]} = 0.13849 \implies b^{[2](\text{new})} = 0.60 - 0.5(0.13849) = \mathbf{0.53076}$$
2. **Hidden Layer Weights ($W^{[1]}$) and Biases ($b^{[1]}$)**:
   $$\frac{\partial \mathcal{L}}{\partial W_{11}^{[1]}} = \delta_1^{[1]} x_1 = 0.01337 \cdot 0.05 \approx 0.000669 \implies W_{11}^{[1](\text{new})} = 0.15 - 0.5(0.000669) = \mathbf{0.14967}$$
   $$\frac{\partial \mathcal{L}}{\partial W_{12}^{[1]}} = \delta_1^{[1]} x_2 = 0.01337 \cdot 0.10 \approx 0.001337 \implies W_{12}^{[1](\text{new})} = 0.20 - 0.5(0.001337) = \mathbf{0.19933}$$
   $$\frac{\partial \mathcal{L}}{\partial W_{21}^{[1]}} = \delta_2^{[1]} x_1 = 0.01500 \cdot 0.05 \approx 0.000750 \implies W_{21}^{[1](\text{new})} = 0.25 - 0.5(0.000750) = \mathbf{0.24963}$$
   $$\frac{\partial \mathcal{L}}{\partial W_{22}^{[1]}} = \delta_2^{[1]} x_2 = 0.01500 \cdot 0.10 \approx 0.001500 \implies W_{22}^{[1](\text{new})} = 0.30 - 0.5(0.001500) = \mathbf{0.29925}$$
   $$\frac{\partial \mathcal{L}}{\partial b_1^{[1]}} = \delta_1^{[1]} = 0.01337 \implies b_1^{[1](\text{new})} = 0.35 - 0.5(0.01337) = \mathbf{0.34332}$$
   $$\frac{\partial \mathcal{L}}{\partial b_2^{[1]}} = \delta_2^{[1]} = 0.01500 \implies b_2^{[1](\text{new})} = 0.35 - 0.5(0.01500) = \mathbf{0.34250}$$

---

## 6. Summary Checklist for Examination

- [x] Multi-layer perceptron structural necessity for non-linearly separable problems (XOR).
- [x] Proof that deep linear networks collapse to single affine transformations: $\mathbf{W}' = \mathbf{W}^{[L]} \dots \mathbf{W}^{[1]}$.
- [x] Five primary activation functions (Sigmoid, Tanh, ReLU, LeakyReLU, GELU) with formulas and saturation properties.
- [x] Vectorized forward propagation equations for layer $l$: $\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$, $\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$.
- [x] Rigorous chain rule derivation of error sensitivity: $\boldsymbol{\delta}^{[l]} = ((\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]}) \odot g'^{[l]}(\mathbf{z}^{[l]})$.
- [x] Complete end-to-end forward/backward propagation solved numerical walkthrough (Exam Q33).
