# Module 6: Convolutional Neural Networks (CNN)

**Course**: AIML ZG511 — Deep Neural Networks  
**Prescribed References**: Zhang et al. (T1, Ch 7); Goodfellow et al. (R1, Ch 9)  
**Primary Teacher Followed**: Prof. Shakshi Sharma (Session 7: 2026-09-05 & Session 8: 2026-09-12)

---

## 1. Why CNNs for Visual Data?

When a multi-layer perceptron (MLP) is applied to high-resolution image data, three fundamental pathologies emerge:
1. **Parameter Explosion**: Flattening a $1000 \times 1000$ RGB image yields $3 \times 10^6$ inputs. Connecting this to a hidden layer of 1,000 neurons requires $3 \times 10^9$ weights (3 billion parameters), causing catastrophic overfitting and memory exhaustion.
2. **Destruction of 2D Spatial Geometry**: Flattening an image tensor destroys the neighborhood relationships between adjacent pixels.
3. **Lack of Translation Invariance**: In an MLP, an object appearing in the top-left corner activates entirely different weights than if it appeared in the bottom-right corner.

### The Two Guiding Principles of CNNs:
- **Locality (Local Receptive Fields)**: Neurons in early layers process only a small, localized patch of the input space.
- **Stationarity / Weight Sharing (Translation Equivariance)**: The same learned convolutional kernel slides across the entire image, detecting features invariant to their spatial position.

---

## 2. The 2D Discrete Convolution Operation

![CNN Kernel Operation](../assets/M06_cnn_convolution_kernel_operation.jpg)
*Figure 6.1: Discrete cross-correlation sliding window, element-wise dot product, and feature map generation (Prof. Shakshi Sharma Lecture 7).*

Given a 2D input feature map $\mathbf{X} \in \mathbb{R}^{H \times W}$ and a 2D kernel/filter $\mathbf{K} \in \mathbb{R}^{K_h \times K_w}$:
$$S(i, j) = (\mathbf{X} * \mathbf{K})(i, j) = \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} X(i + m, j + n) \cdot K(m, n)$$
*(Technically in deep learning libraries, this operation is cross-correlation without kernel flipping, but conventionally referred to as convolution).*

---

## 3. Padding, Stride, and Output Dimension Formulas

![Pooling and Feature Maps](../assets/M06_pooling_and_feature_maps.jpg)
*Figure 6.2: Impact of padding $P$, stride $S$, and pooling operations on spatial dimensions (Prof. Shakshi Sharma Lecture 7).*

### 3.1 Spatial Dimension Formula
For an input of height $H_{in}$ and width $W_{in}$, kernel size $K_h \times K_w$, total padding $P_h, P_w$ (split evenly across borders), and stride $S_h, S_w$:
$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} - K_w + 2P_w}{S_w} \right\rfloor + 1$$

### 3.2 Types of Padding:
- **Valid Padding ($P = 0$)**: No zero-padding added. Output spatial dimension shrinks: $H_{out} = H_{in} - K + 1$.
- **Same Padding**: Zero-padding added such that $H_{out} = H_{in}$ when stride $S = 1$:
  $$2P = K - 1 \implies P = \frac{K - 1}{2} \quad (\text{requires odd kernel size } K)$$

---

## 4. Multi-Channel Convolutions & $1 \times 1$ Convolutions

### 4.1 Input & Output Channels
In real computer vision systems, inputs and feature maps have depth (channels):
- Input tensor: $\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W}$
- To produce $C_{out}$ output feature maps, we deploy $C_{out}$ distinct 3D filters, each of shape $C_{in} \times K_h \times K_w$:
  $$\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}, \quad \mathbf{b} \in \mathbb{R}^{C_{out}}$$
- **Total Trainable Parameters**:
  $$\text{Parameters} = C_{out} \times (C_{in} \times K_h \times K_w + 1)$$

### 4.2 The Role of $1 \times 1$ Convolutions (Network in Network / Inception)
A $1 \times 1$ convolution has kernel size $K_h = K_w = 1$. It operates across the channel dimension at each pixel independently:
- **Dimensionality Reduction**: Shrinks channels from $C_{in}$ to $C_{out} < C_{in}$ without changing spatial resolution.
- **Cross-Channel Feature Pooling**: Computes linear combinations of feature maps followed by a non-linear activation.

---

## 5. Pooling Layers

Pooling introduces spatial downsampling and local translation invariance.
- **Max Pooling**: Selects the maximum value within a $K \times K$ pooling window. Captures the presence of dominant features.
- **Average Pooling**: Computes the arithmetic mean. Smooths feature activations.
- **Global Average Pooling (GAP)**: Computes the average over the entire spatial dimensions $H \times W$ of each channel, collapsing $\mathbb{R}^{C \times H \times W} \to \mathbb{R}^C$. Replaces parameter-heavy fully connected layers in modern CNNs.

---

## 6. Architecture Breakdown: LeNet-5 (LeCun et al., 1998)

The foundation of modern CNN architectures, designed for handwritten digit recognition (MNIST):

| Layer | Type | Input Shape | Kernel / Stride / Pad | Output Shape | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Image | - | - | $1 \times 32 \times 32$ | 0 |
| **C1** | Conv2D | $1 \times 32 \times 32$ | 6 kernels of $5 \times 5$, $S=1, P=0$ | $6 \times 28 \times 28$ | $6 \times (1 \times 5 \times 5 + 1) = \mathbf{156}$ |
| **S2** | AvgPool | $6 \times 28 \times 28$ | $2 \times 2$, $S=2$ | $6 \times 14 \times 14$ | $6 \times 2 = \mathbf{12}$ (scale+bias) |
| **C3** | Conv2D | $6 \times 14 \times 14$ | 16 kernels of $5 \times 5$, $S=1, P=0$ | $16 \times 10 \times 10$ | $\mathbf{1,516}$ (custom sparse) |
| **S4** | AvgPool | $16 \times 10 \times 10$| $2 \times 2$, $S=2$ | $16 \times 5 \times 5$ | $16 \times 2 = \mathbf{32}$ |
| **C5** | Conv2D | $16 \times 5 \times 5$ | 120 kernels of $5 \times 5$, $S=1, P=0$| $120 \times 1 \times 1$ | $120 \times (16 \times 5 \times 5 + 1) = \mathbf{48,120}$ |
| **F6** | Dense | $120$ | Fully connected | $84$ | $120 \times 84 + 84 = \mathbf{10,164}$ |
| **Output**| RBF / Softmax | $84$ | Classification (10 digits) | $10$ | $84 \times 10 = \mathbf{840}$ |
