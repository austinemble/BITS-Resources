# Module 10: Attention Mechanism

**Course**: AIML ZG511 — Deep Neural Networks  
**Prescribed References**: Zhang et al. (T1, Ch 11); Jurafsky & Martin (R3, Ch 10)  
**Primary Teacher Followed**: Prof. Shakshi Sharma (Session 11: 2026-10-17)

---

## 1. Motivation: The Information Bottleneck in Seq2Seq

In standard Seq2Seq models, the entire input sentence (regardless of whether it contains 5 words or 100 words) is compressed into a single fixed-length context vector $\mathbf{c} = \mathbf{h}_T$. For long sentences, information regarding early tokens decays, severely degrading translation performance.

The **Attention Mechanism** (Bahdanau et al., 2014; Luong et al., 2015) eliminates this bottleneck: instead of forcing the encoder to compress everything into one vector, the decoder is granted dynamic, direct access to **all encoder hidden states** $(\mathbf{h}_1, \dots, \mathbf{h}_{T_x})$.

---

## 2. The Query, Key, Value (QKV) Conceptual Framework

![Attention QKV Mechanism](../assets/M10_attention_qkv_mechanism.jpg)
*Figure 10.1: Database retrieval analogy for Query, Key, and Value representations (Prof. Shakshi Sharma Lecture 11).*

The attention mechanism operates as a **differentiable, soft memory retrieval**:
- **Query ($\mathbf{q}$)**: The current search target or internal state seeking relevant information.
- **Key ($\mathbf{k}$)**: The label or identifier associated with each stored memory entry.
- **Value ($\mathbf{v}$)**: The actual informational payload returned based on matching relevance.

$$\text{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \sum_{j} \alpha(\mathbf{q}, \mathbf{k}_j) \mathbf{v}_j$$
where $\alpha(\mathbf{q}, \mathbf{k}_j)$ is a normalized scalar probability reflecting compatibility.

---

## 3. Attention Scoring Functions

| Scoring Name | Mathematical Function $a(\mathbf{q}, \mathbf{k})$ | Characteristics |
| :--- | :--- | :--- |
| **Dot-Product Attention** | $\mathbf{q}^T \mathbf{k}$ | Fast, zero extra parameters; requires $d_q = d_k$. |
| **Scaled Dot-Product** | $\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}$ | **Industry standard**; scale factor $\sqrt{d_k}$ prevents softmax saturation. |
| **Bahdanau (Additive)** | $\mathbf{v}_a^T \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})$ | Handles different query/key dimensions; slightly higher compute cost. |
| **Bilinear / General** | $\mathbf{q}^T \mathbf{W} \mathbf{k}$ | Parameterized projection matrix $\mathbf{W} \in \mathbb{R}^{d_q \times d_k}$. |

### Why Scale by $\frac{1}{\sqrt{d_k}}$?
Assume components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and unit variance ($\mathbb{E}[q_i] = 0, \operatorname{Var}(q_i) = 1$).
The dot product is $z = \mathbf{q}^T \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$.
$$\operatorname{Var}(z) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k \cdot (1 \cdot 1) = d_k \implies \operatorname{StdDev}(z) = \sqrt{d_k}$$
For large embedding dimensions (e.g., $d_k = 512$ or $1024$), dot-product scores grow large in magnitude ($|z| \gg 1$), pushing the softmax into regions with vanishingly small gradients. Dividing by $\sqrt{d_k}$ rescales the variance back to $1.0$, preserving healthy gradient flow!

---

## 4. Scaled Dot-Product Attention in Matrix Form

![Scaled Dot Product](../assets/M10_scaled_dot_product_attention.jpg)
*Figure 10.2: Compact matrix multiplication and softmax normalization for parallel attention computation (Prof. Shakshi Sharma Lecture 11).*

Given query matrix $\mathbf{Q} \in \mathbb{R}^{N \times d_k}$, key matrix $\mathbf{K} \in \mathbb{R}^{M \times d_k}$, and value matrix $\mathbf{V} \in \mathbb{R}^{M \times d_v}$:
$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$
- $\mathbf{Q} \mathbf{K}^T \in \mathbb{R}^{N \times M}$: Raw compatibility matrix between all $N$ queries and $M$ keys.
- $\mathbf{A} = \operatorname{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{N \times M}$: Attention weight matrix (rows sum to 1).
- Output $\in \mathbb{R}^{N \times d_v}$: Weighted combination of value vectors.

---

## 5. Multi-Head Attention (MHA)

Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, **Multi-Head Attention** projects queries, keys, and values $h$ times with learned linear projections:

$$\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$
$$\text{where } \text{head}_i = \operatorname{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V)$$
- Projections: $\mathbf{W}_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.
- Typically $d_k = d_v = d_{\text{model}} / h$ (e.g., $d_{\text{model}} = 512, h = 8 \implies d_k = 64$). Total compute cost is identical to single-head attention.
- **Intuitive Benefit**: Allows the model to jointly attend to information from different representation subspaces at different positions (e.g., one head tracks syntactic subject-verb agreement, another head tracks pronoun coreference).
