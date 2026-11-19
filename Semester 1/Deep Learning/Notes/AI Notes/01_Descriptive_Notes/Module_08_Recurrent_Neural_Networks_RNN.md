# Module 8: Recurrent Neural Networks (RNN)

**Course**: AIML ZG511 — Deep Neural Networks  
**Prescribed References**: Zhang et al. (T1, Ch 9); Goodfellow et al. (R1, Ch 10); Jurafsky & Martin (R3, Ch 9)  
**Syllabus module**: **M08** — Recurrent Neural Networks (`module_map.yaml`)  
**Canonical video**: `Videos/Lecture_09_Recurrent_Neural_Networks_RNN_BPTT.mp4`  
**Exam Reference**: Prof. Mithun Ghosh Question Paper (Q44, Q45, Q46, Q47, Q48)

---

## 1. Sequence Modelling & Failure of Feedforward NNs

Feedforward neural networks (MLPs, standard CNNs) assume that input vectors $\mathbf{x}^{(i)}$ are independent and identically distributed (i.i.d.). They fail on sequential and temporal data (natural language, speech, time series) for three fundamental reasons:
1. **Inability to Handle Variable Sequence Lengths**: Natural language sentences and time series vary dynamically in length $T$. Feedforward networks require fixed-sized input dimensions $d$.
2. **Lack of Temporal Memory (State Persistence)**: Feedforward networks process each token in total isolation without retaining information about previously observed tokens.
3. **Absence of Temporal Parameter Sharing**: A phrase or statistical pattern learned at position $t_1$ is not recognized at position $t_2$ without duplicating parameters for every possible temporal offset.

---

## 2. Standard RNN Architecture & Hidden States

Recurrent Neural Networks introduce a **cyclical recurrent feedback loop**, maintaining a persistent hidden state vector $\mathbf{h}_t \in \mathbb{R}^H$ that acts as a working memory of all historical tokens up to time step $t$.

![Recurrent Neural Network BPTT](../assets/M08_recurrent_neural_network_bptt.jpg)
*Figure 8.1: Unfolding a Recurrent Neural Network across sequential time steps and Backpropagation Through Time (BPTT) (Slide 19, DNN_M7_RNN_annotated_1.pdf, Prof. Shakshi Sharma Lecture 9).*

### 2.1 State Transition Equations
At each discrete time step $t \in \{1, 2, \dots, T\}$:
1. **Pre-activation Net Input**:
   $$\mathbf{z}_t = \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h$$
2. **Hidden State Activation**:
   $$\mathbf{h}_t = \tanh(\mathbf{z}_t) = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
3. **Output Emission (Logits & Softmax)**:
   $$\mathbf{o}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$
   $$\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{o}_t)$$
where:
- $\mathbf{x}_t \in \mathbb{R}^d$: Input feature vector at time $t$.
- $\mathbf{h}_t \in \mathbb{R}^h$: Hidden state vector at time $t$ (initialized to $\mathbf{h}_0 = \mathbf{0}$).
- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$: Input-to-hidden transition weight matrix.
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$: Hidden-to-hidden recurrent transition matrix (**shared across all time steps**).
- $\mathbf{b}_h \in \mathbb{R}^h$: Hidden layer bias vector.
- $\mathbf{W}_{hy} \in \mathbb{R}^{V \times h}$: Hidden-to-output emission matrix.
- $\mathbf{b}_y \in \mathbb{R}^V$: Output bias vector.

---

## 3. Parameter Counting Formulation & Numerical Verification (Exam Q45)

### 3.1 General Parameter Count Formula
$$\text{Parameters} = \underbrace{(d \cdot h)}_{\mathbf{W}_{xh}} + \underbrace{(h \cdot h)}_{\mathbf{W}_{hh}} + \underbrace{h}_{\mathbf{b}_h} + \underbrace{(h \cdot V)}_{\mathbf{W}_{hy}} + \underbrace{V}_{\mathbf{b}_y} = h(d + h + V + 1) + V$$

### 3.2 Evaluation for BITS Examination Dimensions:
Given input embedding dimension $d = 100$, hidden state dimension $h = 256$, and output vocabulary size $V = 10,000$:
1. $\mathbf{W}_{xh}: 256 \times 100 = \mathbf{25,600}$ weights.
2. $\mathbf{W}_{hh}: 256 \times 256 = \mathbf{65,536}$ recurrent weights.
3. $\mathbf{b}_h: \mathbf{256}$ hidden biases.
4. $\mathbf{W}_{hy}: 10,000 \times 256 = \mathbf{2,560,000}$ output projection weights.
5. $\mathbf{b}_y: \mathbf{10,000}$ output biases.
$$\text{Total Learnable Parameters} = 25,600 + 65,536 + 256 + 2,560,000 + 10,000 = \mathbf{2,661,392} \approx 2.66 \text{ Million}$$

---

## 4. Solved Forward Pass Numerical (Exam Q46)

**Problem Statement (6 Marks)**:  
A vanilla RNN has $d = 2$ inputs, $h = 3$ hidden units, and $V = 2$ outputs with $\phi = \tanh$ and initial state $\mathbf{h}_0 = [0, 0, 0]^T$. The parameters are:
$$\mathbf{W}_{xh} = \begin{bmatrix} 0.5 & 0.3 \\ 0.2 & 0.4 \\ 0.1 & 0.2 \end{bmatrix}, \quad \mathbf{W}_{hh} = \begin{bmatrix} 0.3 & 0.2 & 0.1 \\ 0.1 & 0.4 & 0.2 \\ 0.2 & 0.1 & 0.3 \end{bmatrix}, \quad \mathbf{b}_h = \begin{bmatrix} 0.1 \\ 0.1 \\ 0.1 \end{bmatrix}$$
$$\mathbf{W}_{ho} = \begin{bmatrix} 0.4 & 0.3 & 0.5 \\ 0.2 & 0.6 & 0.1 \end{bmatrix}, \quad \mathbf{b}_o = \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}$$
For input $\mathbf{x}_1 = [1.0, 0.5]^T$, compute $\mathbf{z}_1$, $\mathbf{h}_1$, and output $\mathbf{o}_1$ (show all intermediate steps to 4 decimal places).

### Step-by-Step Solution:
1. **Compute Input Projection $\mathbf{W}_{xh} \mathbf{x}_1$**:
   $$\mathbf{W}_{xh} \mathbf{x}_1 = \begin{bmatrix} 0.5(1.0) + 0.3(0.5) \\ 0.2(1.0) + 0.4(0.5) \\ 0.1(1.0) + 0.2(0.5) \end{bmatrix} = \begin{bmatrix} 0.50 + 0.15 \\ 0.20 + 0.20 \\ 0.10 + 0.10 \end{bmatrix} = \begin{bmatrix} 0.65 \\ 0.40 \\ 0.20 \end{bmatrix}$$
2. **Compute Recurrent Transition $\mathbf{W}_{hh} \mathbf{h}_0$**:
   $$\text{Since } \mathbf{h}_0 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} \implies \mathbf{W}_{hh} \mathbf{h}_0 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$
3. **Compute Pre-activation Net Input $\mathbf{z}_1$**:
   $$\mathbf{z}_1 = \mathbf{W}_{xh} \mathbf{x}_1 + \mathbf{W}_{hh} \mathbf{h}_0 + \mathbf{b}_h = \begin{bmatrix} 0.65 \\ 0.40 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.1 \\ 0.1 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.7500 \\ 0.5000 \\ 0.3000 \end{bmatrix}}$$
4. **Compute Hidden State Activation $\mathbf{h}_1 = \tanh(\mathbf{z}_1)$**:
   $$\tanh(0.7500) = \frac{e^{0.75} - e^{-0.75}}{e^{0.75} + e^{-0.75}} = \frac{2.11700 - 0.47237}{2.11700 + 0.47237} \approx \mathbf{0.6351}$$
   $$\tanh(0.5000) = \frac{e^{0.50} - e^{-0.50}}{e^{0.50} + e^{-0.50}} = \frac{1.64872 - 0.60653}{1.64872 + 0.60653} \approx \mathbf{0.4621}$$
   $$\tanh(0.3000) = \frac{e^{0.30} - e^{-0.30}}{e^{0.30} + e^{-0.30}} = \frac{1.34986 - 0.74082}{1.34986 + 0.74082} \approx \mathbf{0.2913}$$
   $$\mathbf{h}_1 = \mathbf{\begin{bmatrix} 0.6351 \\ 0.4621 \\ 0.2913 \end{bmatrix}}$$
5. **Compute Output Vector $\mathbf{o}_1 = \mathbf{W}_{ho} \mathbf{h}_1 + \mathbf{b}_o$**:
   $$o_{1,1} = 0.4(0.6351) + 0.3(0.4621) + 0.5(0.2913) + 0.1 = 0.25404 + 0.13863 + 0.14565 + 0.1 = \mathbf{0.6383}$$
   $$o_{1,2} = 0.2(0.6351) + 0.6(0.4621) + 0.1(0.2913) + 0.1 = 0.12702 + 0.27726 + 0.02913 + 0.1 = \mathbf{0.5334}$$
   $$\mathbf{o}_1 = \mathbf{\begin{bmatrix} 0.6383 \\ 0.5334 \end{bmatrix}}$$

---

## 5. Backpropagation Through Time (BPTT) & Vanishing Gradients (Exam Q47, Q48)

### 5.1 BPTT Multivariable Chain Rule Derivation
For sequence loss $L = \sum_{t=1}^T \ell_t$, the gradient with respect to recurrent weight $\mathbf{W}_{hh}$ sums contributions across all time steps:
$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \frac{\partial \ell_t}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \sum_{k=1}^t \frac{\partial \ell_t}{\partial \hat{\mathbf{y}}_t} \frac{\partial \hat{\mathbf{y}}_t}{\partial \mathbf{h}_t} \left( \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \right) \frac{\partial^+ \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$
where the temporal Jacobian chain between time step $t$ and an earlier time step $k$ is:
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \prod_{j=k+1}^t \operatorname{diag}\left(1 - \mathbf{h}_j^2\right) \mathbf{W}_{hh}^T$$

### 5.2 Mathematical Proof of Gradient Vanishing
Because $\mathbf{h}_j = \tanh(\mathbf{z}_j)$, the derivative is bounded:
$$|\tanh'(z)| = |1 - \tanh^2(z)| \le 1.0 \quad (\text{and } \le 0.5 \text{ for typical activations } |z| > 0.6)$$
Let $\|\mathbf{W}_{hh}\|_2 \le \lambda$. The upper bound on the temporal Jacobian norm is:
$$\left\| \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \right\| \le \prod_{j=k+1}^t \|\operatorname{diag}(1 - \mathbf{h}_j^2)\| \cdot \|\mathbf{W}_{hh}\| \le (\gamma \lambda)^{t - k}$$
where $\gamma < 1$ is the average activation derivative.

### 5.3 Concrete Numerical Attenuation Trace:
Given typical values $\|\mathbf{W}_{hh}\| \approx 0.8$ and $|\phi'| \approx 0.5$:
$$\text{Contraction factor per step: } \rho = 0.8 \times 0.5 = 0.40$$
- **After 5 time steps**:
  $$\rho^5 = (0.40)^5 = \mathbf{0.01024} \quad (\approx 99.0\% \text{ signal attenuation!})$$
- **After 10 time steps**:
  $$\rho^{10} = (0.40)^{10} = \mathbf{1.048 \times 10^{-4}} \quad (0.01\% \text{ gradient signal remains!})$$
- **After 20 time steps**:
  $$\rho^{20} = (0.40)^{20} = \mathbf{1.099 \times 10^{-8}} \quad (\text{Essentially ZERO; complete gradient extinction!})$$

### 5.4 Consequences & Practical Remedies:
- **Consequences**:
  1. *Loss of Long-Term Dependencies*: Early tokens cannot influence parameters; the network acts as a short-memory model (Markov model of order $k \le 5$).
  2. *Catastrophic Drift*: In Seq2Seq models, sentences longer than 15 words suffer grammatical decay.
- **Remedies**:
  1. *Gradient Clipping*: Fixes exploding gradients by scaling $\mathbf{g} \leftarrow \theta \frac{\mathbf{g}}{\|\mathbf{g}\|}$ when $\|\mathbf{g}\| > \theta$.
  2. *Gated Architectures*: Deploy LSTM or GRU with additive linear state paths ($\frac{\partial C_t}{\partial C_{t-1}} = f_t$).

---

## 6. Summary Checklist for Examination

- [x] Three reasons why feedforward MLPs fail on sequential data.
- [x] Vanilla RNN equations for hidden state update $\mathbf{h}_t$ and output emission $\mathbf{o}_t$.
- [x] Derivation of parameter count formula and evaluation for $d=100, h=256, V=10000$ (2,661,392 params).
- [x] Complete step-by-step solved forward pass numerical (Exam Q46) with $\mathbf{z}_1, \mathbf{h}_1, \mathbf{o}_1$.
- [x] BPTT unrolled chain rule derivation: $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod \operatorname{diag}(1-\mathbf{h}_j^2)\mathbf{W}_{hh}^T$.
- [x] Concrete numerical trace of gradient vanishing over 5, 10, and 20 time steps ($0.40^t$).
- [x] Gradient clipping formulation for exploding gradients.
