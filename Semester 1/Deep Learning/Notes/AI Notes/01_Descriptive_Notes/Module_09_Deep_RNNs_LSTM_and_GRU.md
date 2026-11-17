# Module 9: Deep Recurrent Neural Networks (LSTM & GRU)

**Course**: AIML ZG511 — Deep Neural Networks  
**Prescribed References**: Zhang et al. (T1, Ch 10); Goodfellow et al. (R1, Ch 10); Jurafsky & Martin (R3, Ch 9)  
**Primary Teacher Followed**: Prof. Shakshi Sharma (Session 10: 2026-10-10, Recording `Shakshi -- 10.mp4`)  
**Exam Reference**: Prof. Mithun Ghosh Question Paper (Q49, Q50)

---

## 1. Gated Memory Mechanisms: Why LSTM & GRU?

Standard vanilla RNNs fail on long sequences because their recurrent state $\mathbf{h}_t$ undergoes continuous matrix multiplications and saturating non-linear transformations ($\tanh$), leading to exponential gradient vanishing.

To solve this, **Long Short-Term Memory (LSTM)** networks (Hochreiter & Schmidhuber, 1997) and **Gated Recurrent Units (GRU)** (Cho et al., 2014) introduce **multiplicative gating mechanisms** and an **additive linear state conveyor belt** that protects gradient flow over long temporal separations.

---

## 2. Long Short-Term Memory (LSTM) Architecture (Exam Q50)

![LSTM Gated Architecture](../assets/M09_lstm_gated_architecture.jpg)
*Figure 9.1: The internal gating structure of an LSTM memory cell showing forget, input, and output gates controlling the cell state highway (Slide 77, DNN_M7_RNN_annotated_1.pdf, Prof. Shakshi Sharma Lecture 10).*

### 2.1 The Two States and Three Gates
An LSTM cell maintains two distinct internal vectors at each time step $t$:
1. **Cell State $\mathbf{C}_t \in \mathbb{R}^h$**: The long-term internal memory conveyor belt.
2. **Hidden State $\mathbf{h}_t \in \mathbb{R}^h$**: The short-term external working memory output.

The information flow is regulated by three sigmoid-activated gates ($\sigma(z) \in [0, 1]$):
1. **Forget Gate ($\mathbf{f}_t$)**: Decides what information to discard from the previous cell state $\mathbf{C}_{t-1}$:
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) = \sigma(\mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{W}_{xf} \mathbf{x}_t + \mathbf{b}_f)$$
   *(If $f_{t,j} \approx 0$, the $j$-th memory element is erased; if $f_{t,j} \approx 1$, it is completely retained).*

2. **Input Gate ($\mathbf{i}_t$) & Candidate Cell State ($\tilde{\mathbf{C}}_t$)**: Decides what new information to write into memory:
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) = \sigma(\mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{W}_{xi} \mathbf{x}_t + \mathbf{b}_i)$$
   $$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) = \tanh(\mathbf{W}_{hc} \mathbf{h}_{t-1} + \mathbf{W}_{xc} \mathbf{x}_t + \mathbf{b}_c)$$

3. **Cell State Update ($\mathbf{C}_t$)**: The linear additive highway:
   $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
   where $\odot$ denotes element-wise (Hadamard) product.

4. **Output Gate ($\mathbf{o}_t$) & Visible Hidden State ($\mathbf{h}_t$)**: Decides what parts of the internal cell state are emitted to the network:
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) = \sigma(\mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{W}_{xo} \mathbf{x}_t + \mathbf{b}_o)$$
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

### 2.2 Mathematical Proof: Why LSTM Solves Vanishing Gradients
In a vanilla RNN, the temporal gradient $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \operatorname{diag}(1 - \mathbf{h}_t^2) \mathbf{W}_{hh}^T$ constantly attenuates because the derivative of $\tanh$ is strictly $\le 1$ and $\|\mathbf{W}_{hh}\|$ causes geometric decay.

In an LSTM, differentiate the cell state update equation:
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
with respect to the prior cell state $\mathbf{C}_{t-1}$:
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t + \mathbf{C}_{t-1} \frac{\partial \mathbf{f}_t}{\partial \mathbf{C}_{t-1}} + \tilde{\mathbf{C}}_t \frac{\partial \mathbf{i}_t}{\partial \mathbf{C}_{t-1}} + \mathbf{i}_t \frac{\partial \tilde{\mathbf{C}}_t}{\partial \mathbf{C}_{t-1}}$$
The leading term is simply the forget gate activation $\mathbf{f}_t$!
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} \approx \operatorname{diag}(\mathbf{f}_t)$$
Across a temporal sequence of $T$ steps:
$$\frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_1} = \prod_{t=2}^T \operatorname{diag}(\mathbf{f}_t)$$
**The Key Insight**: If the network learns that an input feature is important, it sets the forget gate $\mathbf{f}_t \approx \mathbf{1}$. Then:
$$\prod_{t=2}^T \mathbf{f}_t \approx \mathbf{1}$$
The error gradient propagates backward across hundreds of time steps with **zero mathematical attenuation**, completely bypassing the vanishing gradient trap!

---

## 3. Gated Recurrent Unit (GRU — Cho et al., 2014) (Exam Q49)

![GRU Architecture](../assets/M09_gru_reset_update_gates.jpg)
*Figure 9.2: Gated Recurrent Unit (GRU) unrolled architecture with coupled reset and update gates (Slide 62, DNN_M7_RNN_annotated_1.pdf, Prof. Shakshi Sharma Lecture 10).*

The GRU streamlines the LSTM architecture by:
1. Merging the separate cell state and hidden state into a single recurrent hidden state vector $\mathbf{h}_t$.
2. Reducing the number of gates from 3 to 2: **Reset Gate ($\mathbf{r}_t$)** and **Update Gate ($\mathbf{z}_t$)**.
3. Coupling the forget and input gates into a single convex combination: $(1 - \mathbf{z}_t)$ and $\mathbf{z}_t$.

### 3.1 GRU Equations
1. **Reset Gate ($\mathbf{r}_t$)**: Determines how much of the past hidden state to forget when computing new candidate activations:
   $$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r) = \sigma(\mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{W}_{xr} \mathbf{x}_t + \mathbf{b}_r)$$
2. **Update Gate ($\mathbf{z}_t$)**: Controls how much of the previous state is copied vs updated:
   $$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z) = \sigma(\mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{W}_{xz} \mathbf{x}_t + \mathbf{b}_z)$$
3. **Candidate Hidden State ($\tilde{\mathbf{h}}_t$)**:
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h) = \tanh(\mathbf{W}_{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
   *(If $r_{t,j} \approx 0$, the unit acts as if it is reading the first symbol of an entirely new sequence).*
4. **Final Hidden State Interpolation**:
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$
   *(If $z_{t,j} \approx 0$, the previous state is simply copied forward: $h_{t,j} = h_{t-1,j}$; if $z_{t,j} \approx 1$, the state is completely replaced by $\tilde{h}_{t,j}$).*

---

## 4. Parameter Count Derivations: LSTM vs GRU

### 4.1 LSTM Parameter Formulation
An LSTM has 4 affine transformation blocks ($f, i, c, o$), each with input weights ($h \times d$), recurrent weights ($h \times h$), and biases ($h$):
$$\text{Parameters}_{\text{LSTM}} = 4 \times [h(d + h) + h] = 4(dh + h^2 + h)$$

### 4.2 GRU Parameter Formulation
A GRU has 3 affine transformation blocks ($r, z, h$), each with input weights ($h \times d$), recurrent weights ($h \times h$), and biases ($h$):
$$\text{Parameters}_{\text{GRU}} = 3 \times [h(d + h) + h] = 3(dh + h^2 + h)$$

### 4.3 Concrete Numerical Comparison:
Given input feature dimension $d = 100$ and hidden state dimension $h = 256$:
- **Per Block Parameters**:
  $$h(d + h) + h = 256(100 + 256) + 256 = 256(356) + 256 = 91,136 + 256 = \mathbf{91,392}$$
- **LSTM Total**:
  $$\text{Params}_{\text{LSTM}} = 4 \times 91,392 = \mathbf{365,568} \text{ parameters}$$
- **GRU Total**:
  $$\text{Params}_{\text{GRU}} = 3 \times 91,392 = \mathbf{274,176} \text{ parameters}$$
**Savings**: GRU uses exactly $25\%$ fewer parameters than LSTM ($\frac{3}{4} = 0.75$), executing faster per epoch and exhibiting lower risk of overfitting on modest training corpora.

---

## 5. Comparative Evaluation: LSTM vs GRU (Exam Q50)

| Evaluation Criterion | Long Short-Term Memory (LSTM) | Gated Recurrent Unit (GRU) |
| :--- | :--- | :--- |
| **Internal State Structure** | Two separate states: Cell State $\mathbf{C}_t$ (long-term) and Hidden State $\mathbf{h}_t$ (short-term). | Single unified Hidden State $\mathbf{h}_t$. |
| **Number of Gating Units** | **Three gates**: Forget ($\mathbf{f}_t$), Input ($\mathbf{i}_t$), and Output ($\mathbf{o}_t$). | **Two gates**: Reset ($\mathbf{r}_t$) and Update ($\mathbf{z}_t$). |
| **Total Parameter Count** | $4 \times (dh + h^2 + h)$ (100% parameter baseline). | $3 \times (dh + h^2 + h)$ (25% fewer parameters). |
| **Vanishing Gradient Protection** | Protected via pure additive linear cell state highway ($\frac{\partial C_t}{\partial C_{t-1}} \approx \mathbf{f}_t$). | Protected via adaptive linear interpolation highway: $(1 - \mathbf{z}_t)\mathbf{h}_{t-1}$. |
| **Computational Efficiency** | Higher memory footprint; slightly slower forward and backward passes. | Lower memory footprint; significantly faster training iterations. |
| **Empirical Performance** | Preferred on large datasets requiring fine-grained control of long-range dependencies (e.g., long document modeling, speech). | Preferred on small-to-medium datasets where data is scarce and compute is constrained (e.g., sensor telemetry, edge devices). |

---

## 6. Summary Checklist for Examination

- [x] LSTM core architecture: 2 states ($\mathbf{C}_t, \mathbf{h}_t$) and 3 gates ($\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t$).
- [x] Complete mathematical equations for all 6 LSTM update steps.
- [x] Mathematical proof showing $\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$ and how $\mathbf{f}_t \approx 1$ preserves gradient flow across time steps.
- [x] GRU core architecture: 1 state ($\mathbf{h}_t$) and 2 gates ($\mathbf{r}_t, \mathbf{z}_t$).
- [x] Parameter count formulas: $4(dh + h^2 + h)$ for LSTM vs $3(dh + h^2 + h)$ for GRU, with concrete evaluation for $d=100, h=256$.
- [x] Four-point comparative matrix between LSTM and GRU.
