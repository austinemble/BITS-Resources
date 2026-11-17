# Deep Neural Networks — Module-Wise Q&A Study Guide & Master Formula Sheet

**Course**: AIML ZG511 — Deep Neural Networks  
**Target**: Quick Revision, Viva/Oral Exam Preparation, and Master Formula Verification  
**Primary Faculty Benchmark**: Prof. Shakshi Sharma & Prof. Mithun Ghosh (BITS Pilani WILP)

---

## 1. Master Formula Sheet (All 12 Modules)

| Category / Topic | Mathematical Formulation | Key Parameters & Exam Remarks |
| :--- | :--- | :--- |
| **Perceptron Decision Rule** | $\hat{y} = \mathbb{I}(\mathbf{w}^T \mathbf{x} + b \ge 0)$ | Output $\in \{0, 1\}$; linear hyperplane boundary $\mathbf{w}^T \mathbf{x} + b = 0$. |
| **PLA Parameter Updates** | $\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}, \quad b \leftarrow b + \eta (y - \hat{y})$ | $\eta \in (0, 1]$; updates occur only on misclassification ($y \ne \hat{y}$). |
| **Novikoff's Bound** | $k \le \left(\frac{R}{\gamma}\right)^2$ | $R = \max \|\mathbf{x}\|$, $\gamma$ is geometric margin; finite convergence iff separable. |
| **Linear Regression MSE** | $J(\mathbf{w}, b) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2$ | Strictly convex quadratic bowl; global minimum guaranteed. |
| **Normal Equations** | $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ | Exact closed-form solution; computational complexity $\mathcal{O}(d^3)$. |
| **Sigmoid (Logistic)** | $\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$ | Range $(0, 1)$; maximum derivative $\sigma'(0) = 0.25$; saturates for $|z| > 4$. |
| **Binary Cross-Entropy (BCE)**| $\mathcal{L} = -[y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$ | Cancels sigmoid derivative: $\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$. |
| **Softmax Normalization** | $\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}$ | Shift invariance: $\text{softmax}(\mathbf{z} - c\mathbf{1}) = \text{softmax}(\mathbf{z})$, used for numerical stability. |
| **Categorical Cross-Entropy**| $\mathcal{L} = -\sum_{c=1}^C y_c \ln \hat{y}_c = -\ln \hat{y}_{\text{true}}$ | Gradient w.r.t logits: $\nabla_{\mathbf{z}} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$. |
| **DFNN Forward Propagation**| $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \quad \mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$ | $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$, $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$. |
| **Backprop Output Error** | $\boldsymbol{\delta}^{[L]} \triangleq \nabla_{\mathbf{z}^{[L]}} \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$ | For Softmax + CCE, Sigmoid + BCE, or Identity + MSE. |
| **Backprop Hidden Error** | $\boldsymbol{\delta}^{[l]} = \left( (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right) \odot g'^{[l]}(\mathbf{z}^{[l]})$ | Vectorized sensitivity vector at hidden layer $l$; $\odot$ is Hadamard product. |
| **Backprop Weight Gradients**| $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$ | Outer product of layer error and previous layer activation. |
| **CNN Output Spatial Size** | $O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$ | $I$: input size, $K$: kernel size, $P$: padding, $S$: stride. |
| **CNN "Same" Padding** | $P = \frac{K - 1}{2}$ | Guarantees $O = I$ when stride $S = 1$ (requires odd kernel size $K$). |
| **CNN Parameter Count** | $C_{\text{out}} \times (C_{\text{in}} \cdot K^2 + 1)$ | Independent of input spatial height and width; $+1$ accounts for bias. |
| **CNN Inference FLOPs** | $2 \times H_{\text{out}} W_{\text{out}} \times (C_{\text{in}} \cdot K^2) \times C_{\text{out}}$ | Accounts for multiply-accumulate operations in feature map generation. |
| **Receptive Field Growth** | $RF_l = RF_{l-1} + (K_l - 1) \prod_{i=1}^{l-1} S_i$ | Base case $RF_0 = 1$; stacking $3\times 3$ convs gives $RF = 3, 5, 7, 9$. |
| **Dilated Conv Effective $K$**| $k_{\text{eff}} = k + (k - 1)(r - 1)$ | For $k=3, r=2 \implies k_{\text{eff}} = 5$; expands RF with 0 parameter increase. |
| **ResNet Residual Mapping** | $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x} \implies \frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}}\left(\frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}\right)$ | Identity term $\mathbf{I}$ guarantees uninhibited backward gradient flow. |
| **Vanilla RNN Hidden State** | $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$ | Recurrent weights $\mathbf{W}_{hh}$ shared across all time steps. |
| **Vanilla RNN Parameters** | $h(d + h + V + 1) + V$ | $d$: input dim, $h$: hidden dim, $V$: vocabulary output dim. |
| **LSTM Cell State Update** | $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$ | Linear additive conveyor belt solving vanishing gradient: $\frac{\partial C_t}{\partial C_{t-1}} \approx \mathbf{f}_t$. |
| **LSTM Hidden Output** | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$ | Filtered through output gate $\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$. |
| **LSTM Parameters** | $4 \times [h(d + h) + h] = 4(dh + h^2 + h)$ | 4 parameter matrices for Forget, Input, Candidate, Output gates. |
| **GRU State Interpolation**| $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$ | Single state vector; coupled reset ($\mathbf{r}_t$) and update ($\mathbf{z}_t$) gates. |
| **GRU Parameters** | $3 \times [h(d + h) + h] = 3(dh + h^2 + h)$ | Exactly 25% fewer parameters than LSTM ($\frac{3}{4} = 0.75$). |
| **Scaled Dot-Product Attn** | $\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$ | $\sqrt{d_k}$ normalizes variance to $1.0$, preventing softmax gradient saturation. |
| **Multi-Head Attention** | $\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$ | $d_k = d_v = d_{\text{model}} / h$; projects into $h$ distinct representation subspaces. |
| **Sinusoidal Positional Enc** | $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d})$ | Linear shift invariance allows attention to learn relative positions. |
| **ViT Patch Extraction** | $N = \frac{H \cdot W}{P^2}$ patches of size $P \times P$ | For $224\times 224$ and $P=16 \implies N = 196$ patches; sequence length $197$ with `[CLS]`. |
| **Momentum Optimizer** | $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \mathbf{g}_t, \quad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_t$ | $\beta \approx 0.9$; dampens oscillations across high-curvature ravines. |
| **RMSprop Optimizer** | $\mathbf{s}_t = \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^{\odot 2}, \quad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$ | Adaptive per-parameter learning rate scaling inversely with gradient energy. |
| **Adam Bias Corrections** | $\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$ | Proof: $\mathbb{E}[\mathbf{m}_t] = \mathbf{g}(1 - \beta_1^t)$; division guarantees unbiased estimators. |
| **Adam Parameter Update** | $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$ | Combines 1st moment (momentum) and 2nd moment (RMSprop). |
| **AdamW Update** | $\boldsymbol{\theta}_{t+1} = (1 - \eta \lambda)\boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$ | Decouples weight decay from adaptive moment gradients. |
| **Inverted Dropout** | $\tilde{\mathbf{a}} = \frac{1}{1 - p} (\mathbf{m} \odot \mathbf{a}), \quad m_i \sim \text{Bernoulli}(1 - p)$ | Scaling by $\frac{1}{1-p}$ during training leaves inference code untouched. |
| **Layer Normalization** | $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i$ | Normalized across feature channels $D$; 100% independent of batch size $B$. |

---

## 2. High-Yield Oral Exam & Viva Checklist

### Modules 1–3: Foundations & Linear Models
- [x] **AI vs ML vs DL**: AI is intelligent machines; ML is statistical pattern learning; DL is multi-layer representation learning without manual feature engineering.
- [x] **Why Andrew Ng curve plateaus for ML**: Hypothesis space is constrained; model capacity saturates regardless of data size. Deep networks have power-law scaling ($E \propto N^{-\alpha}$).
- [x] **Four Pillars of DL**: Data ($\mathcal{D}$), Model Architecture ($f(\mathbf{x}; \boldsymbol{\theta})$), Objective Loss Function ($\mathcal{L}$), Optimization Algorithm (SGD/Adam).
- [x] **Biological vs Artificial Neuron**: Dendrites $\to$ inputs $\mathbf{x}$; Synapse $\to$ weights $\mathbf{w}$; Soma $\to$ linear sum $\sum w_i x_i + b$; Axon hillock $\to$ activation $g(z)$.
- [x] **Why zero weights work for PLA but fail for backpropagation**: In PLA, first mistake breaks symmetry immediately ($\mathbf{w} \leftarrow \pm \eta \mathbf{x}$). In backprop, all hidden neurons receive identical gradients, preventing feature specialization (symmetric trap).
- [x] **Why Normal Equations fail for big data**: Inverting $(\mathbf{X}^T \mathbf{X})$ requires $\mathcal{O}(d^3)$ time and $\mathcal{O}(d^2)$ memory, intractable when $d > 10^5$.

### Modules 4–5: Classification, DFNN & Backprop
- [x] **Why Linear Regression fails for Classification**: Produces unbounded outputs outside $[0, 1]$; extreme outlier points pull the boundary away from true separations.
- [x] **Why BCE cancels the Sigmoid derivative**: $\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}$ and $\sigma'(z) = \hat{y}(1-\hat{y})$. Product yields $\hat{y}-y$, preventing vanishing gradient on wrong predictions.
- [x] **Softmax Numeric Overflow Fix**: Subtract maximum logit: $\text{softmax}(\mathbf{z}) = \text{softmax}(\mathbf{z} - \max(\mathbf{z}))$. Largest exponent is $e^0 = 1.0$.
- [x] **Why Deep Linear Networks collapse**: $\mathbf{W}^{[L]}\dots\mathbf{W}^{[1]}\mathbf{x} = \mathbf{W}'\mathbf{x}$. No non-linear decision boundaries possible without non-linear activations.
- [x] **Dying ReLU problem**: If pre-activation $z \le 0$ across all training points, $\text{ReLU}'(z) = 0$ permanently. Solved by Leaky ReLU ($\alpha = 0.01$) or GELU.
- [x] **Why weight matrix is transposed in backprop**: Forward pass maps $\mathbb{R}^{n^{[l-1]}} \to \mathbb{R}^{n^{[l]}}$ via $\mathbf{W}^{[l]}$. Backward error propagation maps sensitivities from layer $l+1$ back to layer $l$, requiring operator of shape $n^{[l]} \times n^{[l+1]}$, which is $(\mathbf{W}^{[l+1]})^T$.

### Modules 6–7: CNNs & Deep Architectures
- [x] **Three CNN Principles**: Local connectivity (locality), parameter sharing (weight sharing), translation equivariance.
- [x] **Why $1\times 1$ conv is a bottleneck**: Shrinks channel depth before expensive $3\times 3$ or $5\times 5$ convolutions, reducing parameters and FLOPs by $10\times$ to $12\times$.
- [x] **Why VGG stacks $3\times 3$ convs**: Two $3\times 3$ convs have effective RF $5\times 5$ with $18C^2$ params vs $25C^2$ (28% savings) and two non-linear activations.
- [x] **ResNet Degradation vs Overfitting**: Deeper networks exhibited higher *training* error, proving optimization failure rather than overfitting.
- [x] **Residual Gradient Highway**: $\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathcal{H}}(\frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I})$. The identity matrix $\mathbf{I}$ guarantees non-vanishing gradient flow to early layers.

### Modules 8–10: Sequence Models, RNN, LSTM, Attention
- [x] **Why feedforward fails on sequences**: Cannot handle variable length $T$; no internal state memory; no parameter sharing across time steps.
- [x] **Why vanilla RNN suffers from vanishing gradient**: Temporal Jacobian involves product $\prod_{j=k+1}^t \operatorname{diag}(1-\mathbf{h}_j^2)\mathbf{W}_{hh}^T$. With $\|\mathbf{W}_{hh}\| < 1$ and $|\tanh'| \le 1$, gradient decays exponentially ($0.4^t \to 10^{-8}$ after 20 steps).
- [x] **Why LSTM solves vanishing gradient**: Additive cell state highway $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$. $\frac{\partial C_t}{\partial C_{t-1}} \approx \mathbf{f}_t$. When $\mathbf{f}_t \approx 1$, error flows with zero attenuation.
- [x] **LSTM vs GRU**: LSTM has 3 gates, 2 states, $4(dh+h^2+h)$ params. GRU has 2 gates, 1 state, $3(dh+h^2+h)$ params (25% fewer parameters, faster training).
- [x] **Seq2Seq Information Bottleneck**: Compressing an entire sentence into a single static vector $\mathbf{c} = \mathbf{h}_{T_x}$ loses early token context. Attention provides direct access to all encoder states.
- [x] **Why divide by $\sqrt{d_k}$ in Attention**: Dot product of two $d_k$-dim unit variance vectors has variance $d_k$. Dividing by $\sqrt{d_k}$ scales variance back to $1.0$, preventing softmax saturation and gradient death.

### Modules 11–12: Transformers, ViT & Optimization
- [x] **Why Transformers train faster than RNNs**: RNNs require $\mathcal{O}(T)$ sequential steps. Transformers compute self-attention across all tokens in parallel ($\mathcal{O}(1)$ sequential steps), fully utilizing GPU tensor cores.
- [x] **Pre-LN vs Post-LN**: Post-LN has unnormalized gradients at the output, requiring delicate learning rate warmup. Pre-LN places LayerNorm inside the residual branch, maintaining an uninterrupted identity gradient highway.
- [x] **How ViT processes images**: Splits $H \times W \times C$ into $N = HW/P^2$ patches ($16\times 16$), linearly projects to dimension $D$, prepends learnable `[CLS]` token, adds 1D learnable position embeddings, and feeds into a standard Transformer Encoder.
- [x] **CNN vs ViT Inductive Bias**: CNN has hardcoded locality and translation equivariance; trains well on small data. ViT has zero spatial inductive bias; requires large-scale pre-training but achieves higher scaling capacity.
- [x] **Adam Bias Correction Proof**: Uncorrected $\mathbb{E}[\mathbf{m}_t] = \mathbf{g}(1 - \beta_1^t)$. Dividing by $(1 - \beta_1^t)$ provides an exact unbiased estimator of the first moment.
- [x] **Adam vs AdamW**: In Adam, $L_2$ regularization is scaled inversely by gradient variance, reducing weight decay for active parameters. AdamW decouples weight decay directly from gradient updates: $\boldsymbol{\theta} \leftarrow (1 - \eta \lambda)\boldsymbol{\theta} - \dots$.
- [x] **Batch Normalization vs Layer Normalization**: BN normalizes across batch dimension $B$ (fails when $B < 8$; struggles with variable length sequences). LN normalizes across channel dimension $D$ per sample (works for $B=1$; ideal for Transformers and RNNs).
