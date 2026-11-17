# Module 11: Transformers & Vision Transformers (ViT)

**Course**: AIML ZG511 — Deep Neural Networks  
**Prescribed References**: Zhang et al. (T1, Ch 11); Jurafsky & Martin (R3, Ch 10); Vaswani et al. (2017); Dosovitskiy et al. (2020)  
**Primary Teacher Followed**: Prof. Shakshi Sharma (Session 12: 2026-10-24 & Session 13: 2026-10-31, Recordings `Shakshi -- 12.mp4`, `Shakshi -- 13.mp4`)  
**Exam Reference**: Prof. Mithun Ghosh Question Paper (Q52) & Companion HTML Modules

---

## 1. The Transformer Paradigm ("Attention Is All You Need", 2017)

Before 2017, state-of-the-art sequence models relied on recurrent (LSTM, GRU) or convolutional backbones. The Transformer architecture discarded recurrence entirely, relying exclusively on multi-head self-attention.
- **Advantage over RNNs**: Total elimination of sequential dependency. All tokens in a sequence are processed in parallel ($\mathcal{O}(1)$ sequential operations vs $\mathcal{O}(T)$ for RNNs), maximizing GPU parallelization.
- **Direct Context Modeling**: Path length between any two tokens across the sequence is $\mathcal{O}(1)$, eliminating long-range decay.

---

## 2. Transformer Architecture: Encoder & Decoder Blocks

![Transformer Architecture](../assets/M11_transformer_encoder_decoder_block.jpg)
*Figure 11.1: Detailed schema of the canonical Transformer Encoder and Decoder blocks with residual connections and Layer Normalization (Slide 12, DNN_M9_Transformer (1)_annotated_annotated.pdf, Prof. Shakshi Sharma Lecture 12).*

### 2.1 The Transformer Encoder Block
Each encoder layer comprises two primary sub-layers wrapped in residual connections:
1. **Multi-Head Self-Attention (MHA)**:
   $$\mathbf{x}_{\text{attn}} = \operatorname{LayerNorm}(\mathbf{x} + \operatorname{MultiHead}(\mathbf{x}, \mathbf{x}, \mathbf{x}))$$
2. **Position-wise Feed-Forward Network (FFN)**: Applied to each position independently:
   $$\operatorname{FFN}(\mathbf{z}) = \max(0, \mathbf{z} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2 = \operatorname{GELU}(\mathbf{z} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$
   $$\mathbf{x}_{\text{out}} = \operatorname{LayerNorm}(\mathbf{x}_{\text{attn}} + \operatorname{FFN}(\mathbf{x}_{\text{attn}}))$$
   *(Typically the inner projection dimension expands by $4\times$: $d_{\text{ff}} = 4 \times d_{\text{model}}$)*.

### 2.2 Pre-LN vs Post-LN Normalization
- **Post-LN (Original Vaswani 2017)**: LayerNorm placed *after* the residual addition:
  $$\mathbf{x}_{l} = \operatorname{LayerNorm}(\mathbf{x}_{l-1} + \operatorname{SubLayer}(\mathbf{x}_{l-1}))$$
  *Drawback*: Gradients at the output layer are unnormalized, requiring strict warm-up learning schedules to prevent training divergence.
- **Pre-LN (Modern Standard: ViT, GPT, LLaMA)**: LayerNorm placed *inside* the residual branch before the transformation:
  $$\mathbf{x}_{l} = \mathbf{x}_{l-1} + \operatorname{SubLayer}(\operatorname{LayerNorm}(\mathbf{x}_{l-1}))$$
  *Benefit*: Maintains an uninhibited identity gradient highway throughout the entire depth of the network, enabling stable training of 100+ layer transformers without delicate warmup.

### 2.3 The Transformer Decoder Block
Contains three sub-layers:
1. **Masked Multi-Head Self-Attention**: Prevents positions from attending to subsequent future tokens by adding an upper-triangular mask matrix $\mathbf{M}$:
   $$\operatorname{MaskedAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}$$
   where $M_{ij} = 0$ for $i \ge j$, and $M_{ij} = -\infty$ for $i < j$. Since $e^{-\infty} = 0$, attention weights to future tokens are strictly zero, ensuring causal autoregression.
2. **Cross-Attention (Encoder-Decoder Attention)**: Queries come from the previous decoder layer; Keys and Values come from the final encoder output.
3. **Position-wise Feed-Forward Network (FFN)**.

---

## 3. Positional Encoding: Preserving Token Order

Because self-attention is permutation-equivariant (treating inputs as an unordered bag of tokens), explicit order information must be added to token embeddings.

### Sinusoidal Positional Encoding (Vaswani et al., 2017):
For token position $pos \in [0, T-1]$ and feature dimension index $i \in [0, d_{\text{model}}/2 - 1]$:
$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$
$$PE_{(pos, 2i + 1)} = \cos\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$

### Key Properties:
1. **Bounded Range**: Values are strictly bounded in $[-1, 1]$.
2. **Deterministic & Unique**: Each position receives a unique multi-frequency signature (lower dimensions oscillate rapidly; higher dimensions oscillate slowly).
3. **Linear Shift Invariance**: For any fixed temporal offset $k$, $PE_{pos + k}$ is a linear transformation of $PE_{pos}$:
   $$\begin{bmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(k \cdot \omega_i) & \sin(k \cdot \omega_i) \\ -\sin(k \cdot \omega_i) & \cos(k \cdot \omega_i) \end{bmatrix} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$
   This allows the attention mechanism to easily attend based on relative token offsets.

---

## 4. Vision Transformers (ViT — Dosovitskiy et al., 2020) (Exam Q52)

![Vision Transformer Pipeline](../assets/M11_vision_transformer_vit_pipeline.jpg)
*Figure 11.2: End-to-end Vision Transformer (ViT) pipeline: 2D patch extraction, linear embedding projection, learnable [CLS] token, and Transformer Encoder stack (Slide 53, DNN_M9_Transformer (1)_annotated_annotated.pdf, Prof. Shakshi Sharma Lecture 13).*

Vision Transformers demonstrate that pure standard Transformer encoders can be applied directly to 2D image data without convolutional inductive bias, achieving state-of-the-art accuracy when trained on large-scale datasets.

### 4.1 The Patch Extraction & Embedding Pipeline
1. **Input Image**: Given RGB image $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ (e.g., $224 \times 224 \times 3$).
2. **Patch Partitioning**: Divide image into a grid of non-overlapping square patches of size $P \times P$ (e.g., $P = 16$).
3. **Patch Count ($N$)**:
   $$N = \frac{H \cdot W}{P^2} = \frac{224 \times 224}{16 \times 16} = 14 \times 14 = \mathbf{196 \text{ patches}}$$
4. **Patch Flattening**: Flatten each patch into a 1D vector $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2 C}$:
   $$\text{Flattened dimension} = 16 \times 16 \times 3 = \mathbf{768}$$
5. **Linear Patch Projection**: Project each flattened patch to the Transformer latent dimension $D = 768$ using learnable matrix $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$:
   $$\mathbf{e}_i = \mathbf{x}_p^{(i)} \mathbf{E} \in \mathbb{R}^D$$
6. **Prepend Learnable Class Token (`[CLS]`)**: Following BERT, prepend a learnable token $\mathbf{x}_{\text{class}} \in \mathbb{R}^D$ at position 0 to aggregate sequence-wide representations for classification. Sequence length becomes $N + 1 = 197$.
7. **Add 1D Learnable Positional Embeddings**:
   $$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^{(1)}\mathbf{E}; \dots; \mathbf{x}_p^{(N)}\mathbf{E}] + \mathbf{E}_{\text{pos}}, \quad \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$$
8. **Transformer Encoder Processing**: Pass sequence $\mathbf{z}_0$ through $L$ identical Transformer Encoder blocks.
9. **MLP Classification Head**: Extract the output representation of the `[CLS]` token at layer $L$ ($\mathbf{z}_L^0$), pass through LayerNorm and a final linear layer to predict target class probabilities:
   $$\hat{\mathbf{y}} = \operatorname{softmax}(\operatorname{Linear}(\mathbf{z}_L^0))$$

---

## 5. ViT-Base Parameter Count Derivation

For the canonical **ViT-Base** configuration:
- Image shape: $224 \times 224 \times 3$
- Patch size: $P = 16 \implies N = 196$ patches
- Hidden embedding dimension: $D = 768$
- Number of Encoder layers: $L = 12$
- Number of Attention Heads: $H = 12 \implies d_k = D / H = 64$
- MLP hidden expansion: $d_{\text{ff}} = 4 \times D = 3072$

### Layer-by-Layer Parameter Inventory:
1. **Patch Embedding Projection**:
   $$\mathbf{W}_E: (16 \times 16 \times 3) \times 768 + 768 = 768 \times 768 + 768 = \mathbf{590,592}$$
2. **Positional Embeddings & Class Token**:
   $$\mathbf{x}_{\text{class}}: 768, \quad \mathbf{E}_{\text{pos}}: (196 + 1) \times 768 = 151,296 \implies \mathbf{152,064}$$
3. **One Transformer Encoder Layer**:
   - *Multi-Head Attention*:
     - $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$: $3 \times (768 \times 768 + 768) = 1,771,776$
     - $\mathbf{W}^O$: $768 \times 768 + 768 = 590,592$
     - MHA Subtotal = $2,362,368$
   - *LayerNorm 1*: $2 \times 768 = 1,536$
   - *MLP Feed-Forward Block*:
     - Linear 1 ($768 \to 3072$): $768 \times 3072 + 3072 = 2,362,368$
     - Linear 2 ($3072 \to 768$): $3072 \times 768 + 768 = 2,360,064$
     - MLP Subtotal = $4,722,432$
   - *LayerNorm 2*: $2 \times 768 = 1,536$
   - **Total per Encoder Layer**: $2,362,368 + 1,536 + 4,722,432 + 1,536 = \mathbf{7,087,872}$
4. **All 12 Encoder Layers**:
   $$12 \times 7,087,872 = \mathbf{85,054,464}$$
5. **Final LayerNorm & Classification Head (ImageNet 1000 classes)**:
   - Final LayerNorm: $2 \times 768 = 1,536$
   - Linear Head: $768 \times 1000 + 1000 = 769,000$
   - Head Subtotal = $\mathbf{770,536}$

$$\text{Total ViT-Base Parameters} = 590,592 + 152,064 + 85,054,464 + 770,536 \approx \mathbf{86.57 \text{ Million}}$$

---

## 6. Summary Checklist for Examination

- [x] Elimination of recurrence in Transformers and $\mathcal{O}(1)$ sequential path length.
- [x] Complete schematic of Transformer Encoder (MHA + FFN) and Decoder (Masked MHA + Cross-Attn + FFN).
- [x] Post-LN vs Pre-LN architectural trade-offs and gradient stability.
- [x] Sinusoidal positional encoding formulas and linear shift property.
- [x] Vision Transformer (ViT) patch extraction pipeline ($N = HW/P^2$), linear projection, and `[CLS]` token.
- [x] Inductive bias comparison: CNN local translation equivariance vs ViT global self-attention.
- [x] Complete parameter count derivation for ViT-Base ($\approx 86.5\text{M}$ parameters).
