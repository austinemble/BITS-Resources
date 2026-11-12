# Deep Neural Networks — Module-Wise Q&A Study Guide & Formula Sheet

**Course**: AIML ZG511 — Deep Neural Networks  
**Target**: Quick Revision, Viva/Oral Exam Preparation, and Formula Verification  
**Primary Teacher Followed**: Prof. Shakshi Sharma

---

## 1. Master Formula Sheet

| Concept | Canonical Formula | Key Remarks |
| :--- | :--- | :--- |
| **Perceptron Decision Rule** | $\hat{y} = \mathbb{I}(\mathbf{w}^T \mathbf{x} + b \ge 0)$ | Output $\in \{0, 1\}$; linear hyperplane boundary. |
| **Perceptron Update** | $\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}$, $b \leftarrow b + \eta (y - \hat{y})$ | Updates only on misclassification ($y \ne \hat{y}$). |
| **Linear Regression MSE** | $J(\mathbf{w}, b) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2$ | Strictly convex; global minimum guaranteed. |
| **Normal Equations** | $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ | Exact closed-form solution; $\mathcal{O}(d^3)$ cost. |
| **Sigmoid Activation** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$; saturates for $|z|>4$. |
| **Binary Cross-Entropy** | $\mathcal{L} = -[y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$ | Cancels sigmoid derivative: $\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$. |
| **Softmax Function** | $\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}$ | Invariant to translation: $\text{softmax}(\mathbf{z}) = \text{softmax}(\mathbf{z} - c\mathbf{1})$. |
| **Categorical Cross-Entropy**| $\mathcal{L} = -\sum_{c=1}^C y_c \ln \hat{y}_c$ | Gradient w.r.t logits: $\nabla_{\mathbf{z}} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$. |
| **Backprop Error Propagation**| $\boldsymbol{\delta}^{[l]} = ((\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]}) \odot g'^{[l]}(\mathbf{z}^{[l]})$ | Vectorized sensitivity vector at hidden layer $l$. |
| **Backprop Weight Gradient** | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$ | Outer product of error and previous activation. |
| **CNN Output Spatial Size** | $O = \lfloor \frac{I - K + 2P}{S} \rfloor + 1$ | $I$: input, $K$: kernel, $P$: pad, $S$: stride. |
| **CNN Parameter Count** | $C_{\text{out}} \times (C_{\text{in}} \cdot K^2 + 1)$ | Independent of input spatial height and width! |
| **LSTM Cell State Update** | $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$ | Linear additive highway avoiding vanishing gradient. |
| **GRU State Update** | $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$ | Coupled forget and input gating. |
| **Scaled Dot-Product Attn** | $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$ | $\sqrt{d_k}$ prevents variance explosion and saturation. |
| **ViT Patch Count** | $N = \frac{H \cdot W}{P^2}$ | $P \times P$: patch resolution (e.g. $16 \times 16$). |
| **Weight Decay ($L_2$)** | $\mathbf{w}_{t+1} = (1 - \eta \lambda)\mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$ | Multiplicative shrinkage factor $(1 - \eta \lambda) < 1$. |
| **Layer Normalization** | $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$ | Statistics computed across channels per instance. |

---

## 2. High-Yield Conceptual Q&A by Module

### Module 1 & 2: Foundations & Perceptrons
**Q: Can a perceptron with a step function learn if initialized with weights all zero?**  
*A*: **Yes.** For the Perceptron Learning Algorithm, initial weights $\mathbf{w}=\mathbf{0}, b=0$ are valid. On the first misclassified instance, the weights immediately update to $\mathbf{w} \leftarrow \pm \eta \mathbf{x}$. *(Contrast this with multi-layer neural networks using backpropagation, where zero weight initialization causes symmetric hidden neurons and identical gradient updates, preventing feature diversity).*

**Q: Why does the XOR problem require a non-linear activation in the hidden layer?**  
*A*: If the hidden layer uses a linear activation ($g(z) = z$), the two-layer network collapses into a single affine transformation $\mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = \mathbf{W}'\mathbf{x} + \mathbf{b}'$, which remains strictly linear and incapable of separating XOR. Non-linear activation breaks linearity.

---

### Module 3 & 4: Linear Neural Networks
**Q: Why is Mean Squared Error (MSE) poor for classification with Sigmoid activations?**  
*A*: With MSE $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$ and $\hat{y} = \sigma(z)$, the gradient is $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = (\hat{y} - y)\sigma(z)(1 - \sigma(z))\mathbf{x}$. When a prediction is completely wrong (e.g., $y=1$ but $z = -10 \implies \hat{y} \approx 0$), the term $\sigma(z)(1 - \sigma(z)) \approx 0$ causes the gradient to vanish! The network stops learning precisely when it is most wrong. Cross-Entropy eliminates $\sigma'(z)$, yielding gradient $(\hat{y} - y)\mathbf{x}$.

**Q: Explain the numeric overflow issue in Softmax and how to solve it.**  
*A*: For large logit values (e.g., $z_i = 1000$), $e^{1000}$ overflows standard floating-point representation (`inf`), resulting in `NaN`. This is solved by using the translation invariance property:
$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$
Subtracting the maximum logit ensures the largest exponent is $e^0 = 1$, completely preventing overflow.

---

### Module 5: Deep Feedforward Networks & Backprop
**Q: What is the "Dying ReLU" problem, and how do variants solve it?**  
*A*: If a neuron's pre-activation $z \le 0$ across all training data, $\text{ReLU}(z) = 0$ and its derivative is $0$. Gradient updates cease, and the neuron remains permanently inactive ("dead"). **Leaky ReLU** ($\max(\alpha z, z)$ with $\alpha = 0.01$) and **PReLU** (learnable $\alpha$) ensure a small non-zero slope for $z < 0$, allowing dead neurons to recover.

**Q: Why do we transpose the weight matrix in backpropagation?**  
*A*: In the forward pass, layer $l+1$ pre-activation is $\mathbf{z}^{[l+1]} = \mathbf{W}^{[l+1]}\mathbf{a}^{[l]}$. By multivariable calculus, the sensitivity w.r.t the input activation is $\nabla_{\mathbf{a}^{[l]}} \mathcal{L} = (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]}$. Transposing $\mathbf{W}^{[l+1]} \in \mathbb{R}^{n^{[l+1]} \times n^{[l]}}$ yields an $n^{[l]} \times n^{[l+1]}$ operator that correctly maps error from the $(l+1)$-th dimension back down to the $l$-th dimension.

---

### Module 6 & 7: CNNs & Deep Architectures
**Q: Why is a stack of two $3 \times 3$ convolutions better than one $5 \times 5$ convolution?**  
*A*:
1. **Receptive Field**: Two $3 \times 3$ convolutions have an identical receptive field ($3 + (3-1) = 5$).
2. **Parameters**: For $C$ channels, one $5 \times 5$ layer has $25C^2$ parameters; two $3 \times 3$ layers have $2 \times 9C^2 = 18C^2$ parameters (**28% savings**).
3. **Non-linearity**: Two layers incorporate two non-linear activations instead of one, increasing model expressiveness.

**Q: How do Residual Connections solve the vanishing gradient problem in deep networks?**  
*A*: The residual formulation $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ has gradient:
$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$$
Even if the learned transformation gradient $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ vanishes (approaches $\mathbf{0}$), the identity matrix $\mathbf{I}$ guarantees an unimpeded gradient signal of magnitude $1.0$ flowing directly back to early layers.

---

### Module 8 & 9: RNNs, LSTM, GRU
**Q: Why does standard RNN suffer from vanishing gradients while LSTM does not?**  
*A*: In a vanilla RNN, the temporal gradient involves the product $\prod_{j=k+1}^t \operatorname{diag}(1 - \mathbf{h}_j^2) \mathbf{W}_{hh}^T$. Because $|\tanh'| \le 1$ and $\|\mathbf{W}_{hh}\|$ is often $< 1$, the gradient decays exponentially. In LSTM, the cell state update is **additive**: $C_t = f_t C_{t-1} + i_t \tilde{C}_t$. The derivative is $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. When the forget gate is saturated at $1$, the error gradient flows across hundreds of time steps with zero attenuation (constant error carousel).

---

### Module 10 & 11: Attention & Transformers
**Q: Why are Transformers faster to train than RNNs despite having more parameters?**  
*A*: RNNs are inherently sequential: step $t$ cannot be computed until step $t-1$ finishes ($\mathcal{O}(T)$ sequential operations), prohibiting parallelization. Transformers compute self-attention across all tokens simultaneously using parallel matrix multiplication ($\mathcal{O}(1)$ sequential operations), fully saturating GPU tensor cores.

**Q: What is the computational complexity of Self-Attention w.r.t sequence length $N$?**  
*A*: Computing $\mathbf{Q}\mathbf{K}^T$ for an $N \times d$ sequence requires $(N \times d) \times (d \times N) \implies \mathcal{O}(N^2 d)$ time and memory. This quadratic $\mathcal{O}(N^2)$ scaling limits standard Transformers on very long documents.

---

### Module 12: Optimization & Regularization
**Q: Why does Adam outperform SGD with Momentum on sparse gradient problems?**  
*A*: Adam maintains both 1st moments (momentum) and 2nd moments (RMSProp). The second moment $\mathbf{v}_t$ adaptively scales the step size inversely with the gradient magnitude $\sqrt{\mathbf{v}_t}$. Features that receive infrequent, sparse updates get larger effective learning steps, whereas dense, frequent features get dampened steps.

**Q: Why is Batch Normalization problematic for small batch sizes ($B < 8$)?**  
*A*: Batch Normalization estimates the population mean and variance from the mini-batch sample: $\sigma_B^2 = \frac{1}{B}\sum (x_i - \mu_B)^2$. For tiny batch sizes, the empirical sample variance has high statistical noise, introducing wild fluctuations in normalized activations and degrading performance.
