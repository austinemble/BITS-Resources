# Deep Neural Networks — Comprehensive Examination Question-Answer Bank

**Course**: AIML ZG511 — Deep Neural Networks  
**Source**: University Examination Master Question Bank (50 Questions × 6 Marks = 300 Marks Total + Modern Architectures Extension)  
**Author / Examiner**: Prof. Mithun Ghosh & Prof. Shakshi Sharma (BITS Pilani WILP)  
**Standard**: Exhaustive 6/6 Marks Model Answers, Step-by-Step Solved Numericals, Rigorous Derivations, and Clear Diagrammatic Schematics.

---

# Part 1: Official 50-Question Examination Bank

---

## Module 1: Introduction to Deep Learning

### Q1. [Easy] Define Deep Learning. Explain where Deep Learning sits within the broader fields of Artificial Intelligence and Machine Learning, and state what the word “deep” refers to in deep learning. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition of Deep Learning (2 Marks)**:
   Deep Learning (DL) is a specialized subset of machine learning based on artificial neural networks with multiple representation layers. Unlike traditional ML, which requires human domain experts to handcraft manual features (e.g., SIFT, HOG, MFCCs), deep learning algorithms automatically discover and learn hierarchical feature representations directly from raw sensory data (pixels, waveforms, tokens) through layered computational graphs.
2. **Hierarchy within AI and ML (2 Marks)**:
   - **Artificial Intelligence (AI)** (Outer shell, 1950s): The overarching discipline of engineering intelligent systems capable of performing cognitive tasks that typically require human intelligence (e.g., symbolic logic, knowledge graphs, heuristic search).
   - **Machine Learning (ML)** (Middle shell, 1980s): A subfield of AI where computer algorithms learn statistical associations and patterns from data without being explicitly programmed with hardcoded rules.
   - **Deep Learning (DL)** (Core subset, 2010s): A specialized subfield of ML employing deep multi-layer neural architectures for end-to-end representation learning.
3. **What "Deep" Refers To (2 Marks)**:
   The word **"deep"** refers to the **computational depth of the model's directed acyclic graph**, specifically the number of sequential parameterized transformation layers between the input and the final output ($L \ge 2$, often reaching dozens or hundreds of layers in modern networks like ResNet-152). Each layer transforms lower-level features into increasingly abstract, linearly separable representations.

---

### Q2. [Easy] List and briefly explain the four core components of any deep learning problem. Support your answer with the wake-word detection example. (6 Marks)

#### Model Solution (6/6 Marks):
1. **The Four Core Components (4 Marks)**:
   - **1. Data**: The empirical collection of input-target pairs $\mathcal{D} = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^N$ drawn from underlying distribution $P(\mathbf{X}, \mathbf{Y})$.
   - **2. Model Architecture**: The parameterized function family $\hat{\mathbf{y}} = f(\mathbf{x}; \boldsymbol{\theta})$ mapping input features to predictions.
   - **3. Objective / Loss Function**: A scalar metric $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$ quantifying prediction error, minimized via empirical risk: $J(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N \mathcal{L}(f(\mathbf{x}^{(i)}; \boldsymbol{\theta}), \mathbf{y}^{(i)})$.
   - **4. Optimization Algorithm**: An iterative algorithm based on gradient descent that updates parameter tensors in the direction of steepest descent: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$.
2. **Wake-Word Detection Mapping (2 Marks)**:
   - *Data*: Continuous 16 kHz audio streams converted via STFT into 2D Log-Mel spectrograms $\mathbf{X} \in \mathbb{R}^{T \times F}$, with binary target $y \in \{0, 1\}$ (wake word present/absent).
   - *Model*: Compact Convolutional Neural Network (CNN) or Conv-RNN processing time-frequency patches.
   - *Loss*: Weighted Binary Cross-Entropy (BCE) penalizing false negatives (user speaking the word but ignored) more heavily than false triggers.
   - *Optimizer*: Adam optimizer with cosine learning rate decay, followed by post-training INT8 quantization for real-time inference on edge DSPs (< 50 mW power budget).

---

### Q3. [Medium] “Scale drives deep learning progress” (Andrew Ng). Explain this statement with the performance vs. amount-of-labelled-data curves for traditional ML and small/medium/large neural networks. Also list any four additional reasons for the popularity of deep learning. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Explanation of Andrew Ng's Scaling Principle (3 Marks)**:
   Traditional statistical machine learning models (e.g., Logistic Regression, Support Vector Machines, Random Forests) have limited hypothesis capacity. Beyond a certain modest volume of labeled training data, their performance hits an asymptotic plateau — supplying additional data yields negligible returns. In contrast, deep neural networks with high representational capacity scale monotonically with data volume: larger networks achieve progressively higher asymptotic accuracy when supplied with massive labeled datasets.
   - *Empirical Scaling Curve*:
```
Model Performance
       ^
       |                                       / Large Deep Neural Network
       |                                      /
       |                                     / Medium Neural Network
       |                                    /
       |               - - - - - - - - - - / Small Neural Network
       |           . '
       |       . '---------------------------- Traditional ML (Ceiling)
       |     /
       |    /
       +--------------------------------------------->
                       Volume of Labeled Data
```
2. **Four Additional Reasons for DL Popularity (3 Marks)**:
   - **1. End-to-End Representation Learning**: Eliminates laborious, error-prone manual feature engineering.
   - **2. GPU & TPU Acceleration**: Massively parallel matrix multiplication engines reduced training from months to hours.
   - **3. Algorithmic Breakthroughs**: Non-saturating activations (ReLU), Batch/Layer Normalization, and Residual connections resolved vanishing gradients.
   - **4. Open-Source Ecosystems & Pretrained Models**: Frameworks (PyTorch, TensorFlow) and model hubs (HuggingFace, Torchvision) allow immediate transfer learning.

---

### Q4. [Medium] Differentiate between supervised, unsupervised and reinforcement learning, giving one example of each. Under supervised learning, list any four kinds of tasks. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Three Core Learning Paradigms (3 Marks)**:
   - **Supervised Learning**: Model learns a predictive mapping from labeled input-target pairs $(\mathbf{x}, y)$ by minimizing a defined loss function. *Example*: Image classification (ImageNet).
   - **Unsupervised Learning**: Model discovers underlying patterns, density distributions, or latent clusterings from unlabeled data $\mathbf{x}$ without external supervisor targets. *Example*: Customer segmentation via K-Means or Autoencoders.
   - **Reinforcement Learning (RL)**: An autonomous agent learns optimal behavioral policies through sequential trial-and-error interactions with an environment, maximizing cumulative scalar reward signals. *Example*: AlphaGo mastering board games.
2. **Four Kinds of Supervised Learning Tasks (3 Marks)**:
   - **1. Regression**: Predicting continuous real-valued scalar/vector outputs ($y \in \mathbb{R}$). *Example*: House price valuation.
   - **2. Binary Classification**: Predicting binary discrete class labels ($y \in \{0, 1\}$). *Example*: Email spam vs ham detection.
   - **3. Multi-Class Classification**: Predicting mutually exclusive categorical class memberships ($y \in \{1, 2, \dots, C\}$). *Example*: Handwritten digit recognition (MNIST).
   - **4. Sequence-to-Sequence (Seq2Seq)**: Mapping variable-length source sequences to variable-length target sequences. *Example*: Neural Machine Translation (English to French).

---

## Module 2: Artificial Neural Networks and the Perceptron

### Q5. [Easy] Describe the structure of a biological neuron and compare it with an artificial neuron on any four points. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Structure of Biological Neuron (2 Marks)**:
   A biological neuron consists of:
   - **Dendrites**: Tree-like branch fibers receiving electrochemical input pulses from adjacent neurons.
   - **Soma (Cell Body)**: Accumulates incoming postsynaptic potentials.
   - **Axon**: Long cylindrical nerve fiber transmitting electrical action potentials away from the soma.
   - **Synapses**: Microscopic junction gaps where neurotransmitters modulate signal transmission strength.
2. **Four-Point Comparison Table (4 Marks)**:

| Biological Neuron Feature | Artificial Neuron Equivalent | Mathematical / Functional Role |
| :--- | :--- | :--- |
| **Dendrites** | Input feature vector $\mathbf{x} = [x_1, \dots, x_d]^T$ | Receptive channels carrying input values |
| **Synaptic Strength** | Synaptic weights $\mathbf{w} = [w_1, \dots, w_d]^T$ | Scaling coefficients (excitatory $>0$, inhibitory $<0$) |
| **Soma / Cell Body** | Linear Combiner $z = \sum_{j=1}^d w_j x_j + b$ | Algebraic sum of weighted inputs shifted by bias $b$ |
| **Axon Hillock Threshold**| Activation Function $g(z)$ | Non-linear threshold determining activation output |

---

### Q6. [Easy] Define the perceptron. Who introduced it and when? Write the perceptron computation and its output (decision) rule, and draw/describe its structure with bias. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition & Origin (2 Marks)**:
   The **Perceptron** is the simplest artificial neural network architecture capable of learning linear decision boundaries for binary classification. It was introduced by **Frank Rosenblatt in 1958** at Cornell Aeronautical Laboratory.
2. **Computation & Decision Rule (2 Marks)**:
   Given input vector $\mathbf{x} = [x_1, \dots, x_d]^T$, weights $\mathbf{w} = [w_1, \dots, w_d]^T$, and bias $b$:
   - Net Input (Affine Combination):
     $$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^d w_j x_j + b$$
   - Decision Rule (Heaviside Step Activation):
     $$\hat{y} = g(z) = \begin{cases} 1, & \text{if } z \ge 0 \\ 0, & \text{if } z < 0 \end{cases}$$
3. **Structure with Bias (2 Marks)**:
   The bias $b$ represents an intrinsic firing threshold, mathematically equivalent to setting an augmented input $x_0 = 1$ with weight $w_0 = b$:
   $$z = \sum_{j=0}^d w_j x_j = \mathbf{w}^T \mathbf{x}$$
   The decision boundary is the hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$, dividing feature space into two half-spaces.

---

### Q7. [Medium] Show how a single perceptron can represent the two-input AND gate. Derive the four inequality constraints on the parameters $w_0$ (bias), $w_1$, $w_2$ from the truth table, and give one valid parameter set. Sketch/describe the linear separation in the $x_1$–$x_2$ plane. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Derivation of Four Inequality Constraints (3 Marks)**:
   Truth table for AND gate: $y = x_1 \land x_2$ with decision rule $\hat{y} = 1$ if $w_0 + w_1 x_1 + w_2 x_2 \ge 0$ else $0$:
   - Instance 1 ($x_1=0, x_2=0, y=0$):
     $$w_0 + w_1(0) + w_2(0) < 0 \implies \mathbf{w_0 < 0} \quad \text{--- (1)}$$
   - Instance 2 ($x_1=0, x_2=1, y=0$):
     $$w_0 + w_1(0) + w_2(1) < 0 \implies \mathbf{w_0 + w_2 < 0} \quad \text{--- (2)}$$
   - Instance 3 ($x_1=1, x_2=0, y=0$):
     $$w_0 + w_1(1) + w_2(0) < 0 \implies \mathbf{w_0 + w_1 < 0} \quad \text{--- (3)}$$
   - Instance 4 ($x_1=1, x_2=1, y=1$):
     $$w_0 + w_1(1) + w_2(1) \ge 0 \implies \mathbf{w_0 + w_1 + w_2 \ge 0} \quad \text{--- (4)}$$
2. **Valid Parameter Set (1.5 Marks)**:
   Let $w_1 = 1.0, w_2 = 1.0$.
   From (2) and (3): $w_0 < -1.0$.
   From (4): $w_0 + 2.0 \ge 0 \implies w_0 \ge -2.0$.
   Choosing $w_0 = -1.5$:
   - $w_0 = -1.5, w_1 = 1.0, w_2 = 1.0$.
   Verification:
   - $(0, 0) \to -1.5 < 0 \implies 0$ (Correct)
   - $(0, 1) \to -0.5 < 0 \implies 0$ (Correct)
   - $(1, 0) \to -0.5 < 0 \implies 0$ (Correct)
   - $(1, 1) \to +0.5 \ge 0 \implies 1$ (Correct)
3. **Linear Separation Description (1.5 Marks)**:
   The decision line $x_1 + x_2 - 1.5 = 0 \implies x_2 = -x_1 + 1.5$ cleanly isolates the single positive point $(1, 1)$ in the upper right half-plane from the other three negative points $(0,0), (0,1), (1,0)$ in the lower left half-plane.

---

### Q8. [Medium] State the perceptron learning algorithm, clearly writing the weight and bias update rules. State the perceptron convergence theorem. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Perceptron Learning Algorithm (PLA) (3 Marks)**:
   - **Initialization**: Set weights and bias to zeros or small random values: $\mathbf{w}^{(0)} = \mathbf{0}, b^{(0)} = 0$. Choose learning rate $\eta \in (0, 1]$.
   - **Iteration**: For each training example $(\mathbf{x}^{(i)}, y^{(i)})$, compute prediction:
     $$\hat{y}^{(i)} = \begin{cases} 1, & \text{if } \mathbf{w}^T \mathbf{x}^{(i)} + b \ge 0 \\ 0, & \text{if } \mathbf{w}^T \mathbf{x}^{(i)} + b < 0 \end{cases}$$
   - **Update Rules**:
     $$\mathbf{w} \leftarrow \mathbf{w} + \eta \left( y^{(i)} - \hat{y}^{(i)} \right) \mathbf{x}^{(i)}$$
     $$b \leftarrow b + \eta \left( y^{(i)} - \hat{y}^{(i)} \right)$$
   - Notice that if $y^{(i)} = \hat{y}^{(i)}$, error is $0$ and no update occurs. Updates occur only on misclassifications.
2. **Novikoff's Perceptron Convergence Theorem (3 Marks)**:
   If the training dataset $\mathcal{D}$ is **linearly separable** by a unit margin hyperplane $\mathbf{w}^*$ ($\|\mathbf{w}^*\|=1$) such that:
   $$(2y^{(i)} - 1)(\mathbf{w}^{*T}\mathbf{x}^{(i)} + b^*) \ge \gamma > 0 \quad \forall i$$
   and input vectors are bounded within a sphere of radius $R$ ($\|\mathbf{x}^{(i)}\| \le R$), then the Perceptron Learning Algorithm will terminate and find a separating hyperplane in at most:
   $$k \le \left( \frac{R}{\gamma} \right)^2$$
   weight updates. If the data is **not linearly separable**, the algorithm oscillates indefinitely without converging.

---

### Q9. [Tough] Using the perceptron learning algorithm with initial weights $w_0 = w_1 = 0$, learning rate $\eta = 1$ and output rule $\hat{y} = 1$ if $h \ge 0$ else $0$, trace the training of a perceptron for the NOT gate (examples: $x_1=0, t=1$ and $x_1=1, t=0$) epoch by epoch until convergence. Report the final weights. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Problem Formulation**:
   - Model: $h = w_0(1) + w_1 x_1$, with $\hat{y} = 1$ if $h \ge 0$ else $0$.
   - Update rule: $w_0 \leftarrow w_0 + \eta(t - \hat{y})(1)$, $w_1 \leftarrow w_1 + \eta(t - \hat{y})x_1$. Initial $w_0 = 0, w_1 = 0, \eta = 1$.
2. **Epoch 1 Execution**:
   - **Sample 1** ($x_1=0, t=1$):
     $h = 0(1) + 0(0) = 0 \ge 0 \implies \hat{y} = 1$.
     Error $e = t - \hat{y} = 1 - 1 = 0 \implies$ **No Update**.
     Weights: $w_0 = 0, w_1 = 0$.
   - **Sample 2** ($x_1=1, t=0$):
     $h = 0(1) + 0(1) = 0 \ge 0 \implies \hat{y} = 1$.
     Error $e = t - \hat{y} = 0 - 1 = -1 \implies$ **Update Triggered!**
     $$w_0 \leftarrow 0 + 1(-1)(1) = -1$$
     $$w_1 \leftarrow 0 + 1(-1)(1) = -1$$
     Weights after Epoch 1: $w_0 = -1, w_1 = -1$.
3. **Epoch 2 Execution**:
   - **Sample 1** ($x_1=0, t=1$):
     $h = -1(1) + (-1)(0) = -1 < 0 \implies \hat{y} = 0$.
     Error $e = 1 - 0 = +1 \implies$ **Update Triggered!**
     $$w_0 \leftarrow -1 + 1(+1)(1) = 0$$
     $$w_1 \leftarrow -1 + 1(+1)(0) = -1$$
     Weights: $w_0 = 0, w_1 = -1$.
   - **Sample 2** ($x_1=1, t=0$):
     $h = 0(1) + (-1)(1) = -1 < 0 \implies \hat{y} = 0$.
     Error $e = 0 - 0 = 0 \implies$ **No Update**.
     Weights after Epoch 2: $w_0 = 0, w_1 = -1$.
4. **Epoch 3 (Verification Epoch)**:
   - Sample 1 ($x_1=0, t=1$): $h = 0 \ge 0 \implies \hat{y} = 1$ ($e = 0$) $\implies$ Correct.
   - Sample 2 ($x_1=1, t=0$): $h = -1 < 0 \implies \hat{y} = 0$ ($e = 0$) $\implies$ Correct.
5. **Final Converged Report**:
   - Zero errors in Epoch 3; **Algorithm Converged**.
   - **Final Weights**: $\mathbf{w_0 = 0}$ (bias), $\mathbf{w_1 = -1}$.

---

### Q10. [Medium] What is the XOR problem? Using the XOR truth table and a sketch of the four points in the $x_1$–$x_2$ plane, explain why a single perceptron cannot solve it. Define linearly separable data and state the solution to the XOR problem. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition of XOR Problem & Non-Separability (3 Marks)**:
   The XOR function evaluates to $1$ if and only if exactly one input is active:
   - $(0, 0) \to 0, \quad (0, 1) \to 1, \quad (1, 0) \to 1, \quad (1, 1) \to 0$.
   - *Impossibility Proof*: Assume a linear separating line $w_1 x_1 + w_2 x_2 + b = 0$ exists.
     1. $(0, 0) \to 0 \implies b < 0$
     2. $(1, 1) \to 0 \implies w_1 + w_2 + b < 0$
     3. $(0, 1) \to 1 \implies w_2 + b \ge 0$
     4. $(1, 0) \to 1 \implies w_1 + b \ge 0$
     Adding (3) and (4): $w_1 + w_2 + 2b \ge 0$.
     Since $b < 0$, $w_1 + w_2 + b > w_1 + w_2 + 2b \ge 0 \implies w_1 + w_2 + b > 0$, directly contradicting inequality (2) ($w_1 + w_2 + b < 0$). Hence, no single linear line can separate the classes.
2. **Definition of Linearly Separable Data (1.5 Marks)**:
   A binary dataset is linearly separable if there exists an affine hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ such that all instances of class 1 satisfy $\mathbf{w}^T \mathbf{x} + b > 0$ and all instances of class 0 satisfy $\mathbf{w}^T \mathbf{x} + b < 0$.
3. **Solution to the XOR Problem (1.5 Marks)**:
   The XOR problem is resolved by adding an intermediate **hidden layer** with non-linear activations (Multi-Layer Perceptron), expressing XOR as:
   $$\text{XOR}(x_1, x_2) = (x_1 \lor x_2) \land \neg(x_1 \land x_2)$$
   The hidden layer folds the 2D feature space, projecting points into a representation where they become linearly separable.

---

### Q11. [Easy] When should one consider using Artificial Neural Networks for a problem? List any four situations, and state any two core ideas of the connectionism model. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Four Situations to Use ANNs (4 Marks)**:
   - **1. High-Dimensional Sensory Data**: When inputs consist of raw images, audio signals, or unparsed text where manual feature engineering is brittle or impractical.
   - **2. Complex Non-Linear Relationships**: When underlying input-output mappings involve intricate combinatorial or non-linear dependencies.
   - **3. Abundant Labeled Training Data**: When massive datasets ($N > 10^5$) are available to prevent overfitting and satisfy neural capacity.
   - **4. Tolerance to Noise and Missing Values**: ANNs exhibit graceful degradation when inputs are noisy or partially corrupted.
2. **Two Core Ideas of Connectionism (2 Marks)**:
   - **1. Distributed Representations**: Concepts and knowledge are not stored in localized, single memory addresses; they are distributed across network connection weights.
   - **2. Parallel Processing by Simple Units**: Intelligent behavior emerges from the simultaneous execution and message passing of uniform, highly interconnected computational nodes.

---

## Module 3: Linear Neural Networks for Regression

### Q12. [Easy] Define regression and linear regression. Write the mathematical form of the linear regression model with $d$ features and give two real-life examples of linear regression problems. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definitions (2 Marks)**:
   - **Regression**: A supervised learning paradigm where the goal is to predict a continuous, real-valued scalar target variable $y \in \mathbb{R}$ from input features.
   - **Linear Regression**: A parametric regression model that models the relationship between dependent target $y$ and independent features $\mathbf{x}$ as a linear affine combination.
2. **Mathematical Formulation (2 Marks)**:
   For an input vector with $d$ features $\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$, weight vector $\mathbf{w} = [w_1, w_2, \dots, w_d]^T \in \mathbb{R}^d$, and scalar bias $b \in \mathbb{R}$:
   $$\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$
   In vectorized form for a design matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$ across $N$ examples:
   $$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b \mathbf{1}_N$$
3. **Two Real-Life Examples (2 Marks)**:
   - **1. Residential Property Valuation**: Predicting continuous sale price (\$) from square footage, bedroom count, lot size, and age.
   - **2. Vehicle Fuel Economy Prediction**: Predicting miles per gallon (MPG) from engine displacement, horsepower, weight, and cylinder count.

---

### Q13. [Easy] Which activation function is used when a single neuron implements linear regression? State its definition, any three properties, and explain why it is appropriate for regression. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Activation Function & Definition (2 Marks)**:
   A single neuron implementing linear regression uses the **Identity (Linear) Activation Function**:
   $$g(z) = z$$
   giving output $\hat{y} = g(\mathbf{w}^T \mathbf{x} + b) = \mathbf{w}^T \mathbf{x} + b$.
2. **Three Mathematical Properties (2 Marks)**:
   - **1. Unbounded Output Range**: Range is $(-\infty, +\infty)$, allowing output to take any real value.
   - **2. Constant Derivative**: $g'(z) = \frac{dg}{dz} = 1.0$ everywhere, preventing gradient saturation.
   - **3. Scale Preservation**: Linearity preserves metric proportionality: $g(c z) = c g(z)$.
3. **Why Appropriate for Regression (2 Marks)**:
   Continuous target variables in regression tasks can span positive, negative, or arbitrarily large real numbers. Bounded activations like Sigmoid $(0, 1)$ or Tanh $(-1, 1)$ would artificially clip target predictions, while the identity function permits unconstrained continuous prediction across the real line.

---

### Q14. [Medium] Write the squared-error loss for a single example and the Mean Squared Error objective $J(\mathbf{w})$ for a dataset of $N$ examples (also in vector/matrix form). Give any three reasons why squared error is a good choice for linear regression. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Loss Formulations (3 Marks)**:
   - **Single Example Squared Error**:
     $$\ell^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 = \frac{1}{2} \left( \mathbf{w}^T \mathbf{x}^{(i)} + b - y^{(i)} \right)^2$$
     *(The factor $\frac{1}{2}$ cancels out the exponent 2 during differentiation).*
   - **Dataset Mean Squared Error (MSE)**:
     $$J(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \ell^{(i)}(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$
   - **Vector / Matrix Form**:
     $$J(\tilde{\mathbf{w}}) = \frac{1}{2N} \|\tilde{\mathbf{X}} \tilde{\mathbf{w}} - \mathbf{y}\|_2^2 = \frac{1}{2N} (\tilde{\mathbf{X}} \tilde{\mathbf{w}} - \mathbf{y})^T (\tilde{\mathbf{X}} \tilde{\mathbf{w}} - \mathbf{y})$$
     where $\tilde{\mathbf{X}} = [\mathbf{1}_N, \mathbf{X}] \in \mathbb{R}^{N \times (d+1)}$ and $\tilde{\mathbf{w}} = [b, w_1, \dots, w_d]^T$.
2. **Three Reasons for Choosing Squared Error (3 Marks)**:
   - **1. Maximum Likelihood Justification**: Assuming additive Gaussian observation noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, minimizing squared error is mathematically equivalent to maximizing log-likelihood.
   - **2. Strict Convexity**: The quadratic surface has a positive semi-definite Hessian ($\frac{1}{N}\mathbf{X}^T\mathbf{X}$), ensuring no saddle points or local minima exist; any local minimum is the unique global minimum.
   - **3. Smooth Differentiability**: Has linear gradients everywhere, providing stable analytical and gradient-descent solutions.

---

### Q15. [Medium] Describe the batch gradient descent algorithm for linear regression. Give the gradient of the MSE loss in matrix form, the weight update rule, the effect of choosing the learning rate too large or too small, and two stopping criteria. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Matrix Gradient & Update Rule (3 Marks)**:
   For MSE loss $J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$:
   - **Gradient Vector**:
     $$\nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
   - **Parameter Update Rule**:
     $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} J(\mathbf{w}^{(t)}) = \mathbf{w}^{(t)} - \frac{\eta}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
2. **Effect of Learning Rate $\eta$ (1.5 Marks)**:
   - **Too Small**: Extremely slow convergence, taking an excessive number of iterations to reach minimum.
   - **Too Large**: Step overshoots the minimum, resulting in oscillations or divergent instability ($J \to \infty$).
3. **Two Stopping Criteria (1.5 Marks)**:
   - **1. Gradient Norm**: Stop when $\|\nabla_{\mathbf{w}} J\| < \epsilon$ (gradient is practically flat).
   - **2. Relative Loss Change**: Stop when $\frac{|J^{(t)} - J^{(t-1)}|}{J^{(t-1)}} < \tau$ across successive epochs.

---

### Q16. [Tough] A tiny dataset for predicting house price from size is: $(x=1, y=2), (x=2, y=4), (x=3, y=5)$. Using the model $\hat{y} = w_0 + w_1 x$ with initial weights $w^{(0)} = [0, 0]^T$, learning rate $\eta = 0.1$ and loss $J(w) = rac{1}{2N}\|Xw - y\|^2$, perform ONE iteration of batch gradient descent: compute the initial loss, the gradient, the updated weights $w^{(1)}$, and the new loss. Comment on the progress. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Initial State ($t=0, w_0=0, w_1=0, N=3$)**:
   - Predictions: $\hat{y}^{(1)} = 0, \hat{y}^{(2)} = 0, \hat{y}^{(3)} = 0$.
   - Errors ($e_i = \hat{y}_i - y_i$): $e_1 = 0 - 2 = -2$, $e_2 = 0 - 4 = -4$, $e_3 = 0 - 5 = -5$.
   - **Initial Loss**:
     $$J^{(0)} = \frac{1}{2(3)} [(-2)^2 + (-4)^2 + (-5)^2] = \frac{1}{6}[4 + 16 + 25] = \frac{45}{6} = \mathbf{7.5000}$$
2. **Compute Gradients**:
   $$\frac{\partial J}{\partial w_0} = \frac{1}{N}\sum_{i=1}^3 e_i = \frac{-2 + (-4) + (-5)}{3} = \frac{-11}{3} \approx \mathbf{-3.6667}$$
   $$\frac{\partial J}{\partial w_1} = \frac{1}{N}\sum_{i=1}^3 e_i x_i = \frac{(-2)(1) + (-4)(2) + (-5)(3)}{3} = \frac{-2 - 8 - 15}{3} = \frac{-25}{3} \approx \mathbf{-8.3333}$$
3. **Weight Updates ($\eta = 0.1$)**:
   $$w_0^{(1)} = 0 - 0.1(-3.6667) = \mathbf{+0.3667}$$
   $$w_1^{(1)} = 0 - 0.1(-8.3333) = \mathbf{+0.8333}$$
4. **Compute New Predictions & Loss $J^{(1)}$**:
   - $\hat{y}^{(1)} = 0.3667 + 0.8333(1) = 1.2000 \implies e_1 = 1.2000 - 2 = -0.8000$
   - $\hat{y}^{(2)} = 0.3667 + 0.8333(2) = 2.0333 \implies e_2 = 2.0333 - 4 = -1.9667$
   - $\hat{y}^{(3)} = 0.3667 + 0.8333(3) = 2.8666 \implies e_3 = 2.8666 - 5 = -2.1334$
   - **New Loss**:
     $$J^{(1)} = \frac{1}{6} [(-0.80)^2 + (-1.9667)^2 + (-2.1334)^2] = \frac{1}{6}[0.64 + 3.8679 + 4.5514] = \frac{9.0593}{6} \approx \mathbf{1.5099}$$
5. **Comment on Progress**:
   Loss dropped dramatically from $7.5000$ to $1.5099$ in a single iteration (an $79.9\%$ error reduction), showing rapid convergence toward the true line $y \approx 0.67 + 1.5x$.

---

### Q17. [Medium] Explain the evaluation of a regression model. Define training vs test error, give the formulas for MSE, RMSE and MAE, and interpret the $R^2$ (coefficient of determination) score including the meaning of $R^2 = 1, R^2 = 0$ and $R^2 < 0$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Training vs Test Error (1.5 Marks)**:
   - **Training Error**: Empirical error computed on data used to optimize parameters; measures model fitting.
   - **Test Error**: Error evaluated on independent held-out data; measures generalizability. A large gap indicates overfitting.
2. **Error Metric Formulas (2 Marks)**:
   - **Mean Squared Error**: $\text{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$ (penalizes large outlier errors quadratically).
   - **Root Mean Squared Error**: $\text{RMSE} = \sqrt{\text{MSE}}$ (interpretable in original target physical units).
   - **Mean Absolute Error**: $\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$ (linear penalty, robust to extreme outliers).
3. **$R^2$ Score Formulation & Interpretation (2.5 Marks)**:
   $$R^2 \triangleq 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
   - **$R^2 = 1.0$**: Perfect fit ($\text{SS}_{\text{res}} = 0$); model explains 100% of variance.
   - **$R^2 = 0.0$**: Model performs identically to constantly predicting sample mean baseline $\bar{y}$.
   - **$R^2 < 0.0$**: Model performs **worse** than predicting the sample mean (severely misspecified model).

---

## Module 4: Linear Neural Networks for Classification

### Q18. [Easy] Define classification and distinguish it from regression. Explain the three types of classification with one example each. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Classification vs Regression (2 Marks)**:
   - **Regression**: Maps inputs to continuous numerical values ($y \in \mathbb{R}$).
   - **Classification**: Maps inputs to discrete categorical class memberships ($y \in \{1, \dots, C\}$).
2. **Three Types of Classification (4 Marks)**:
   - **1. Binary Classification**: Only two mutually exclusive categories ($y \in \{0, 1\}$). *Example*: Credit card transaction fraud vs genuine.
   - **2. Multi-Class Classification**: More than two mutually exclusive categories ($y \in \{1, 2, \dots, C\}$). *Example*: Handwritten digit classification (0 through 9).
   - **3. Multi-Label Classification**: Multiple non-mutually exclusive labels can be true simultaneously for one sample. *Example*: Tagging a medical research paper with topics (e.g., both "Genetics" and "Oncology").

---

### Q19. [Easy] Why is linear regression unsuitable for binary classification? Define the sigmoid function, sketch its shape, and state any four of its properties including its derivative. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Why Linear Regression Fails (2 Marks)**:
   - Unbounded outputs: Linear regression yields predictions outside $[0, 1]$, violating probability axioms.
   - High sensitivity to outliers: A distant correct data point pulls the hyperplane, shifting the decision threshold and misclassifying nearby points.
2. **Sigmoid Definition & Four Properties (4 Marks)**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   - **1. Bounded Range**: Maps $(-\infty, +\infty) \to (0, 1)$, naturally representing posterior probability $P(y=1|\mathbf{x})$.
   - **2. Midpoint Symmetry**: $\sigma(0) = 0.5$ and $\sigma(-z) = 1 - \sigma(z)$.
   - **3. Monotonicity**: Strictly increasing for all $z$.
   - **4. Derivative**:
     $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$
     (Derivation: $\frac{d}{dz}(1+e^{-z})^{-1} = -(1+e^{-z})^{-2}(-e^{-z}) = \frac{1}{1+e^{-z}}\frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$).

---

### Q20. [Medium] Write the binary cross-entropy loss for a single example and the total loss over $N$ examples. Explain how the loss behaves when the true label is 1 and when it is 0, and give any three advantages of cross-entropy over squared error for classification. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Formulation & Behavior (3 Marks)**:
   - Single sample: $\ell(y, \hat{y}) = -[y \ln \hat{y} + (1 - y) \ln(1 - \hat{y})]$
   - Dataset: $J(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N [y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)})]$
   - Behavior when $y=1$: Loss simplifies to $-\ln(\hat{y})$. As $\hat{y} \to 1$, loss $\to 0$. As $\hat{y} \to 0$, loss $\to +\infty$.
   - Behavior when $y=0$: Loss simplifies to $-\ln(1 - \hat{y})$. As $\hat{y} \to 0$, loss $\to 0$. As $\hat{y} \to 1$, loss $\to +\infty$.
2. **Three Advantages over Squared Error (3 Marks)**:
   - **1. Eliminates Vanishing Gradients**: With Sigmoid, MSE yields gradient $(\hat{y}-y)\sigma'(z)\mathbf{x}$. When a prediction is completely wrong, $\sigma'(z) \to 0$ and gradient stalls. Cross-Entropy cancels $\sigma'(z)$, giving $\nabla = (\hat{y}-y)\mathbf{x}$.
   - **2. Convex Optimization Surface**: Guarantees a globally convex loss bowl without deceptive local plateaus.
   - **3. Probabilistic Maximum Likelihood**: Directly derived from minimizing Kullback-Leibler divergence between empirical and predicted Bernoulli distributions.

---

### Q21. [Medium] Compare batch gradient descent, stochastic gradient descent (SGD) and mini-batch SGD. Give the update rule, batch size, and any two pros and cons of each. (6 Marks)

#### Model Solution (6/6 Marks):

| Aspect | Batch Gradient Descent (BGD) | Stochastic Gradient Descent (SGD) | Mini-Batch SGD |
| :--- | :--- | :--- | :--- |
| **Batch Size ($B$)** | Full dataset $B = N$ | Single example $B = 1$ | Small subset $B \in [32, 512]$ |
| **Update Rule** | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\eta}{N}\sum_{i=1}^N \nabla \ell_i$ | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla \ell_i$ | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\eta}{B}\sum_{i \in \mathcal{B}} \nabla \ell_i$ |
| **Pros** | 1. Exact, deterministic gradient.<br>2. Smooth, guaranteed convergence. | 1. Extremely low memory footprint.<br>2. Noise helps escape saddle points. | 1. Highly optimized for GPU matrix SIMD.<br>2. Combines speed with stability. |
| **Cons** | 1. Memory exhaustion on large data.<br>2. Prohibitively slow updates per epoch. | 1. High gradient variance/oscillations.<br>2. Cannot utilize GPU parallelization. | Requires tuning additional hyperparameter $B$. |

---

### Q22. [Tough] A logistic regression model classifies pass/fail from study hours: dataset $(x=1,y=0), (x=2,y=0), (x=4,y=1)$. Using the model $\hat{y} = \sigma(w_0 + w_1 x)$ with initial weights $w^{(0)} = [0, 0]^T$, learning rate $\eta = 0.5$, perform TWO steps of Stochastic Gradient Descent (SGD) presenting $(x=1, y=0)$ then $(x=4, y=1)$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Step 1: Present $(x=1, y=0)$**:
   - $z = 0 + 0(1) = 0 \implies \hat{y} = \sigma(0) = 0.5000$.
   - Loss: $\ell_1 = -\ln(1 - 0.5) = -\ln(0.5) \approx \mathbf{0.6931}$.
   - Error: $e_1 = \hat{y} - y = 0.5000 - 0 = +0.5000$.
   - Gradients: $\frac{\partial \ell}{\partial w_0} = e_1 \cdot 1 = 0.5000$, $\frac{\partial \ell}{\partial w_1} = e_1 \cdot x = 0.5000 \cdot 1 = 0.5000$.
   - Updates ($\eta = 0.5$):
     $$w_0^{(1)} = 0 - 0.5(0.5000) = \mathbf{-0.2500}$$
     $$w_1^{(1)} = 0 - 0.5(0.5000) = \mathbf{-0.2500}$$
2. **Step 2: Present $(x=4, y=1)$**:
   - $z = -0.2500 + (-0.2500)(4) = -1.2500$.
   - $\hat{y} = \sigma(-1.2500) = \frac{1}{1 + e^{1.25}} = \frac{1}{1 + 3.49034} \approx \mathbf{0.2227}$.
   - Loss: $\ell_2 = -\ln(0.2227) \approx \mathbf{1.5019}$.
   - Error: $e_2 = \hat{y} - y = 0.2227 - 1.0000 = -0.7773$.
   - Gradients: $\frac{\partial \ell}{\partial w_0} = -0.7773 \cdot 1 = -0.7773$, $\frac{\partial \ell}{\partial w_1} = -0.7773 \cdot 4 = -3.1092$.
   - Updates ($\eta = 0.5$):
     $$w_0^{(2)} = -0.2500 - 0.5(-0.7773) = -0.2500 + 0.3887 = \mathbf{+0.1387}$$
     $$w_1^{(2)} = -0.2500 - 0.5(-3.1092) = -0.2500 + 1.5546 = \mathbf{+1.3046}$$
3. **Summary & Verification**:
   - Updated weights after two steps: $\mathbf{w_0 = +0.1387, w_1 = +1.3046}$.
   - *Comment*: Because sample 2 is a passing student with high study hours ($x=4$), $w_1$ shifted strongly positive, correctly modeling that study hours increase pass probability.

---

### Q23. [Medium] Define the softmax function and list any three of its properties. Write the categorical cross-entropy loss for a single example with one-hot target, and show that its gradient with respect to logit $z_k$ is $(\hat{y}_k - y_k)$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition & Properties (2 Marks)**:
   $$\hat{y}_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$
   - Strict Positivity: $\hat{y}_k > 0$ for all $k$.
   - Conservation of Probability: $\sum_{k=1}^K \hat{y}_k = 1.0$.
   - Shift Invariance: $\text{softmax}(\mathbf{z} - c\mathbf{1}) = \text{softmax}(\mathbf{z})$.
2. **Loss Formulation (1 Mark)**:
   For one-hot vector $\mathbf{y}$ where $y_c = 1$ for true class $c$:
   $$\ell(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^K y_j \ln \hat{y}_j = -\ln \hat{y}_c$$
3. **Gradient Derivation (3 Marks)**:
   By multivariable chain rule:
   $$\frac{\partial \ell}{\partial z_k} = -\sum_{j=1}^K \frac{y_j}{\hat{y}_j} \frac{\partial \hat{y}_j}{\partial z_k}$$
   For the softmax derivative:
   - If $j = k$: $\frac{\partial \hat{y}_k}{\partial z_k} = \hat{y}_k (1 - \hat{y}_k)$.
   - If $j \ne k$: $\frac{\partial \hat{y}_j}{\partial z_k} = -\hat{y}_j \hat{y}_k$.
   Substitute into summation:
   $$\frac{\partial \ell}{\partial z_k} = -\frac{y_k}{\hat{y}_k}[\hat{y}_k(1 - \hat{y}_k)] - \sum_{j \ne k} \frac{y_j}{\hat{y}_j}[-\hat{y}_j \hat{y}_k] = -y_k(1 - \hat{y}_k) + \sum_{j \ne k} y_j \hat{y}_k$$
   $$= -y_k + y_k \hat{y}_k + \hat{y}_k \sum_{j \ne k} y_j = -y_k + \hat{y}_k \left( \sum_{j=1}^K y_j \right)$$
   Since $\mathbf{y}$ is one-hot, $\sum_{j=1}^K y_j = 1$. Therefore:
   $$\mathbf{\frac{\partial \ell}{\partial z_k} = \hat{y}_k - y_k} \quad \implies \nabla_{\mathbf{z}}\ell = \hat{\mathbf{y}} - \mathbf{y}$$

---

### Q24. [Medium] For multi-class classification with $K$ classes and $d$ features: (i) give the shape of weight matrix $\mathbf{W}$ and bias $\mathbf{b}$; (ii) total parameter count formula; (iii) calculate total parameters for $d = 784, K = 10$; (iv) explain the shift invariance property of softmax and why it is used in implementations. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Shapes & Formulas (2 Marks)**:
   - Weight Matrix Shape: $\mathbf{W} \in \mathbb{R}^{K \times d}$.
   - Bias Vector Shape: $\mathbf{b} \in \mathbb{R}^K$.
   - Total Parameters: $\text{Params} = K \cdot d + K = K(d + 1)$.
2. **Calculation for MNIST ($d=784, K=10$) (2 Marks)**:
   $$\text{Weights} = 10 \times 784 = 7,840$$
   $$\text{Biases} = 10$$
   $$\text{Total Parameters} = 7,840 + 10 = \mathbf{7,850}$$
3. **Shift Invariance & Numerical Stability (2 Marks)**:
   $$\text{softmax}(\mathbf{z} - c\mathbf{1})_k = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} e^{z_k}}{e^{-c} \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}} = \text{softmax}(\mathbf{z})_k$$
   *Implementation Purpose*: If logits are large (e.g., $z_k = 1000$), $e^{1000}$ triggers floating-point overflow (`inf`). Setting $c = \max_j z_j$ guarantees that the largest exponent is $e^0 = 1.0$, completely eliminating numeric overflow in libraries like PyTorch/NumPy.

---

### Q25. [Tough] A binary classifier tested on 8 examples produced TP = 3, TN = 3, FP = 1, FN = 1. (i) Draw the confusion matrix; (ii) compute accuracy, precision, recall and F1-score; (iii) in disease diagnosis where missing a disease is worse than a false alarm, which metric should be prioritized and why? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Confusion Matrix (2 Marks)**:

| True \ Predicted | Predicted Positive ($\hat{y}=1$) | Predicted Negative ($\hat{y}=0$) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Positive ($y=1$)** | $\mathbf{TP = 3}$ | $\mathbf{FN = 1}$ | **4** |
| **Actual Negative ($y=0$)** | $\mathbf{FP = 1}$ | $\mathbf{TN = 3}$ | **4** |
| **Total** | **4** | **4** | **8** |

2. **Metrics Computation (3 Marks)**:
   - **Accuracy**: $\frac{TP + TN}{N} = \frac{3 + 3}{8} = \frac{6}{8} = \mathbf{0.7500 \ (75\%)}$
   - **Precision**: $\frac{TP}{TP + FP} = \frac{3}{3 + 1} = \frac{3}{4} = \mathbf{0.7500 \ (75\%)}$
   - **Recall (Sensitivity)**: $\frac{TP}{TP + FN} = \frac{3}{3 + 1} = \frac{3}{4} = \mathbf{0.7500 \ (75\%)}$
   - **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.75 \times 0.75}{0.75 + 0.75} = \mathbf{0.7500 \ (75\%)}$
3. **Metric Prioritization for Diagnosis (1 Mark)**:
   - **Recall** must be prioritized. A False Negative (FN) means a sick patient is discharged without treatment, potentially causing death. A False Positive (FP) only causes confirmatory secondary testing. Therefore, medical systems tune decision thresholds to maximize Recall.

---

## Module 5: Deep Feedforward Networks (DFNN) & Backpropagation

### Q26. [Easy] What is non-linearity in a neural network and why is it essential? Write down what happens if all hidden layers of a DFNN have linear activations. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition & Essence of Non-Linearity (3 Marks)**:
   Non-linearity refers to an activation function $g(z)$ whose output is not directly proportional to its input ($g(a z_1 + b z_2) \ne a g(z_1) + b g(z_2)$). Real-world data distributions (images, speech, decision logic like XOR) are intrinsically non-linearly separable. Non-linear activations warp and fold coordinate manifolds, enabling deep networks to learn non-linear boundaries.
2. **Mathematical Collapse of Deep Linear Networks (3 Marks)**:
   If all hidden layers employ linear activations $g(z) = c z$ (or identity $g(z) = z$), an $L$-layer network collapses into:
   $$\mathbf{h}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$$
   $$\mathbf{h}^{[2]} = \mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = (\mathbf{W}^{[2]}\mathbf{W}^{[1]})\mathbf{x} + (\mathbf{W}^{[2]}\mathbf{b}^{[1]} + \mathbf{b}^{[2]})$$
   $$\dots$$
   $$\hat{\mathbf{y}} = \mathbf{W}' \mathbf{x} + \mathbf{b}'$$
   where $\mathbf{W}' = \mathbf{W}^{[L]}\mathbf{W}^{[L-1]}\dots\mathbf{W}^{[1]}$.
   *Conclusion*: No matter how many layers or parameters are added, a deep linear network cannot compute anything beyond standard linear regression.

---

### Q27. [Easy] Compare the sigmoid, tanh and ReLU activation functions with respect to definition, range, derivative, and one main pro and con of each. (6 Marks)

#### Model Solution (6/6 Marks):

| Property | Sigmoid | Tanh | ReLU |
| :--- | :--- | :--- | :--- |
| **Formula** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $\text{ReLU}(z) = \max(0, z)$ |
| **Range** | $(0, 1)$ | $(-1, 1)$ | $[0, +\infty)$ |
| **Derivative** | $\sigma(z)(1 - \sigma(z))$ | $1 - \tanh^2(z)$ | $1$ if $z > 0$, else $0$ |
| **Main Pro** | Natural probability output for binary classification. | Zero-centered outputs; stronger gradients than sigmoid. | Non-saturating for $z>0$; avoids vanishing gradient; fast. |
| **Main Con** | Saturates for $|z|>4$ causing vanishing gradient; not zero-centered. | Still saturates for $|z|>3$, causing vanishing gradients. | "Dying ReLU": neurons can die if $z \le 0$ across all samples. |

---

### Q28. [Medium] Explain how a multilayer perceptron (2 inputs – 2 hidden units – 1 output) solves the XOR problem. Give one valid choice of weights and biases for the hidden layer (representing AND/OR/NAND) and the output layer, verifying with the truth table. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Resolution Strategy (2 Marks)**:
   Express XOR as: $\text{XOR}(x_1, x_2) = (x_1 \lor x_2) \land \neg(x_1 \land x_2) = \text{OR}(x_1, x_2) \land \text{NAND}(x_1, x_2)$.
   - Hidden Unit 1 ($h_1$): Implements OR gate.
   - Hidden Unit 2 ($h_2$): Implements NAND gate.
   - Output Unit ($y$): Implements AND gate taking inputs $(h_1, h_2)$.
2. **Parameters Choice (Heaviside step activation) (2 Marks)**:
   - Hidden Unit 1 (OR): $w_{11} = 1, w_{12} = 1, b_1 = -0.5 \implies h_1 = \mathbb{I}(x_1 + x_2 - 0.5 \ge 0)$.
   - Hidden Unit 2 (NAND): $w_{21} = -1, w_{22} = -1, b_2 = 1.5 \implies h_2 = \mathbb{I}(-x_1 - x_2 + 1.5 \ge 0)$.
   - Output Unit (AND): $W_1 = 1, W_2 = 1, B = -1.5 \implies y = \mathbb{I}(h_1 + h_2 - 1.5 \ge 0)$.
3. **Truth Table Verification (2 Marks)**:

| $x_1$ | $x_2$ | $h_1 = \text{OR}$ | $h_2 = \text{NAND}$ | Net Output $h_1 + h_2 - 1.5$ | $\hat{y}$ | True XOR Target |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 1 | $0 + 1 - 1.5 = -0.5 < 0$ | **0** | **0** |
| 0 | 1 | 1 | 1 | $1 + 1 - 1.5 = +0.5 \ge 0$ | **1** | **1** |
| 1 | 0 | 1 | 1 | $1 + 1 - 1.5 = +0.5 \ge 0$ | **1** | **1** |
| 1 | 1 | 1 | 0 | $1 + 0 - 1.5 = -0.5 < 0$ | **0** | **0** |

All four cases match target truth table perfectly.

---

### Q29. [Medium] For a DFNN with $L$ layers, write the layer-wise forward computation equations in matrix/vector form for layer $l = 1, \dots, L$. State the dimensions of all vectors and matrices involved. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Layer-wise Equations (3 Marks)**:
   For layer $l = 1, 2, \dots, L$ with input $\mathbf{a}^{[0]} = \mathbf{x}$:
   1. **Pre-activation Affine Combination**:
      $$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
   2. **Post-activation Output**:
      $$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$
   Final prediction is $\hat{\mathbf{y}} = \mathbf{a}^{[L]}$.
2. **Dimension Specifications (3 Marks)**:
   Let $n^{[l]}$ denote the number of neurons in layer $l$, and $n^{[0]} = d$ be the input feature count:
   - Weight Matrix: $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$
   - Bias Vector: $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$
   - Pre-activation Vector: $\mathbf{z}^{[l]} \in \mathbb{R}^{n^{[l]}}$
   - Activation Vector: $\mathbf{a}^{[l]} \in \mathbb{R}^{n^{[l]}}$
   - For a mini-batch of $B$ instances: $\mathbf{A}^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times B}$, $\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times B}$, and $\mathbf{W}^{[l]} \mathbf{A}^{[l-1]} \in \mathbb{R}^{n^{[l]} \times B}$.

---

### Q30. [Tough] Derive/state the backpropagation equations for a DFNN trained with mini-batch SGD: (i) error delta at output layer; (ii) error delta at hidden layer; (iii) gradients with respect to weight matrix and bias vector. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Error Delta Definition & Output Layer $\boldsymbol{\delta}^{[L]}$ (2 Marks)**:
   Define layer sensitivity: $\boldsymbol{\delta}^{[l]} \triangleq \nabla_{\mathbf{z}^{[l]}} \mathcal{L}$.
   At output layer $L$:
   $$\boldsymbol{\delta}^{[L]} = \nabla_{\mathbf{a}^{[L]}} \mathcal{L} \odot g'^{[L]}(\mathbf{z}^{[L]})$$
   - For Softmax with Categorical Cross-Entropy: $\boldsymbol{\delta}^{[L]} = \hat{\mathbf{y}} - \mathbf{y}$.
   - For Sigmoid with Binary Cross-Entropy: $\delta^{[L]} = \hat{y} - y$.
   - For Identity with Squared Error: $\delta^{[L]} = \hat{y} - y$.
2. **Hidden Layer Error Propagation $\boldsymbol{\delta}^{[l]}$ (2 Marks)**:
   Applying multivariable chain rule through $\mathbf{z}^{[l+1]} = \mathbf{W}^{[l+1]}\mathbf{a}^{[l]} + \mathbf{b}^{[l+1]}$ and $\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$:
   $$\boldsymbol{\delta}^{[l]} = \left( (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right) \odot g'^{[l]}(\mathbf{z}^{[l]})$$
   where $\odot$ denotes element-wise (Hadamard) product.
3. **Weight & Bias Gradients (2 Marks)**:
   For a mini-batch of size $B$:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{1}{B} \boldsymbol{\Delta}^{[l]} (\mathbf{A}^{[l-1]})^T \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \frac{1}{B} \boldsymbol{\Delta}^{[l]} \mathbf{1}_B \in \mathbb{R}^{n^{[l]}}$$
   Parameters are updated via: $\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$, $\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$.

---

### Q31. [Tough] For a two-layer feedforward network of sigmoid units trained with squared error loss, derive the step-by-step chain rule expression for $\partial E/\partial w_{jk}$ (hidden-to-output) and $\partial E/\partial v_{ij}$ (input-to-hidden). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Network Definition**:
   Hidden unit $j$: $h_j = \sigma(z_j) = \sigma(\sum_i v_{ij} x_i)$. Output unit $k$: $\hat{y}_k = \sigma(u_k) = \sigma(\sum_j w_{jk} h_j)$.
   Loss: $E = \frac{1}{2}\sum_k (\hat{y}_k - y_k)^2$. Sigmoid derivative: $\sigma'(a) = \sigma(a)(1 - \sigma(a))$.
2. **Hidden-to-Output Weights $\frac{\partial E}{\partial w_{jk}}$ (3 Marks)**:
   $$\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial \hat{y}_k} \cdot \frac{\partial \hat{y}_k}{\partial u_k} \cdot \frac{\partial u_k}{\partial w_{jk}}$$
   - $\frac{\partial E}{\partial \hat{y}_k} = (\hat{y}_k - y_k)$
   - $\frac{\partial \hat{y}_k}{\partial u_k} = \sigma'(u_k) = \hat{y}_k(1 - \hat{y}_k)$
   - $\frac{\partial u_k}{\partial w_{jk}} = h_j$
   Define output error delta: $\delta_k = (\hat{y}_k - y_k)\hat{y}_k(1 - \hat{y}_k)$.
   $$\mathbf{\frac{\partial E}{\partial w_{jk}} = \delta_k h_j}$$
3. **Input-to-Hidden Weights $\frac{\partial E}{\partial v_{ij}}$ (3 Marks)**:
   $$\frac{\partial E}{\partial v_{ij}} = \sum_k \frac{\partial E}{\partial u_k} \frac{\partial u_k}{\partial h_j} \cdot \frac{\partial h_j}{\partial z_j} \cdot \frac{\partial z_j}{\partial v_{ij}}$$
   - $\frac{\partial E}{\partial u_k} = \delta_k$
   - $\frac{\partial u_k}{\partial h_j} = w_{jk}$
   - $\frac{\partial h_j}{\partial z_j} = \sigma'(z_j) = h_j(1 - h_j)$
   - $\frac{\partial z_j}{\partial v_{ij}} = x_i$
   Define hidden error delta: $\delta_j = \left( \sum_k \delta_k w_{jk} \right) h_j(1 - h_j)$.
   $$\mathbf{\frac{\partial E}{\partial v_{ij}} = \delta_j x_i}$$

---

### Q32. [Medium] Explain why backpropagation is computationally efficient compared to numerical differentiation (finite differences). State the computational complexity of computing gradients via backprop vs finite differences for $P$ parameters. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Computational Comparison (3 Marks)**:
   - **Numerical Differentiation (Finite Differences)**: To compute $\frac{\partial \mathcal{L}}{\partial \theta_p}$, one must perturb parameter $p$ by $\epsilon$ and perform a complete forward pass:
     $$\frac{\partial \mathcal{L}}{\partial \theta_p} \approx \frac{\mathcal{L}(\boldsymbol{\theta} + \epsilon \mathbf{e}_p) - \mathcal{L}(\boldsymbol{\theta})}{\epsilon}$$
     For $P$ parameters, this requires $P + 1$ forward passes!
   - **Backpropagation (Reverse-Mode AD)**: Employs dynamic programming to compute the exact analytical gradient with respect to ALL $P$ parameters simultaneously in a **single backward pass**, reusing intermediate activations cached from one forward pass.
2. **Computational Complexity Formulations (3 Marks)**:
   - For a network with $P$ parameters and forward pass cost $C_{\text{fwd}}$:
     - **Finite Differences Complexity**: $\mathcal{O}(P \cdot C_{\text{fwd}})$. For a modest 100M-parameter model, this would require 100 million forward evaluations per step!
     - **Backpropagation Complexity**: $\mathcal{O}(C_{\text{fwd}})$, practically costing $\approx 2 \times C_{\text{fwd}}$.
   - *Conclusion*: Backpropagation achieves a speedup factor of $\mathcal{O}(P)$, turning intractable training problems into real-time optimizations.

---

### Q33. [Tough] For the network with $x = [0.05, 0.10]^T$, target $y = 0.01$, learning rate $\eta = 0.5$, sigmoid activations, $W^{[1]} = [[0.15,0.20],[0.25,0.30]]$, $b^{[1]} = [0.35,0.35]^T$, $W^{[2]} = [0.40,0.45]$, $b^{[2]} = 0.60$, perform ONE complete iteration of forward and backward propagation (show all intermediate numbers to 4 decimal places). (6 Marks)

#### Model Solution (6/6 Marks):
*(Directly following Prof. Shakshi Sharma's solved exam numerical in M5_DFNN_Handwritten..pdf and DNN_M5_DFNN_Numerical.pdf)*

1. **Forward Propagation Pass (2 Marks)**:
   - **Hidden Pre-activations**:
     $$z_1^{[1]} = 0.15(0.05) + 0.20(0.10) + 0.35 = 0.0075 + 0.0200 + 0.35 = \mathbf{0.3775}$$
     $$z_2^{[1]} = 0.25(0.05) + 0.30(0.10) + 0.35 = 0.0125 + 0.0300 + 0.35 = \mathbf{0.3925}$$
   - **Hidden Activations**:
     $$a_1^{[1]} = \sigma(0.3775) = \frac{1}{1 + e^{-0.3775}} \approx \mathbf{0.5933}$$
     $$a_2^{[1]} = \sigma(0.3925) = \frac{1}{1 + e^{-0.3925}} \approx \mathbf{0.5969}$$
   - **Output Pre-activation**:
     $$z_1^{[2]} = 0.40(0.5933) + 0.45(0.5969) + 0.60 = 0.2373 + 0.2686 + 0.60 = \mathbf{1.1059}$$
   - **Output Prediction & Loss**:
     $$\hat{y} = a_1^{[2]} = \sigma(1.1059) = \frac{1}{1 + e^{-1.1059}} \approx \mathbf{0.7514}$$
     $$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.7514 - 0.01)^2 = \frac{1}{2}(0.7414)^2 \approx \mathbf{0.2748}$$
2. **Backward Pass: Error Sensitivities (2 Marks)**:
   - **Output Layer Error $\delta^{[2]}$**:
     $$\delta^{[2]} = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y}) = (0.7514 - 0.01) \cdot 0.7514(1 - 0.7514) = 0.7414 \cdot 0.1868 \approx \mathbf{0.1385}$$
   - **Hidden Layer Errors $\boldsymbol{\delta}^{[1]}$**:
     $$\delta_1^{[1]} = (W_{11}^{[2]} \delta^{[2]}) \cdot a_1^{[1]}(1 - a_1^{[1]}) = (0.40 \cdot 0.1385) \cdot 0.5933(1 - 0.5933) = 0.0554 \cdot 0.2413 \approx \mathbf{0.0134}$$
     $$\delta_2^{[1]} = (W_{12}^{[2]} \delta^{[2]}) \cdot a_2^{[1]}(1 - a_2^{[1]}) = (0.45 \cdot 0.1385) \cdot 0.5969(1 - 0.5969) = 0.0623 \cdot 0.2406 \approx \mathbf{0.0150}$$
3. **Parameter Updates ($\eta = 0.5$) (2 Marks)**:
   - **Output Weights**:
     $$W_{11}^{[2](\text{new})} = 0.40 - 0.5(\delta^{[2]} a_1^{[1]}) = 0.40 - 0.5(0.1385 \cdot 0.5933) = 0.40 - 0.0411 = \mathbf{0.3589}$$
     $$W_{12}^{[2](\text{new})} = 0.45 - 0.5(\delta^{[2]} a_2^{[1]}) = 0.45 - 0.5(0.1385 \cdot 0.5969) = 0.45 - 0.0413 = \mathbf{0.4087}$$
     $$b^{[2](\text{new})} = 0.60 - 0.5(\delta^{[2]}) = 0.60 - 0.5(0.1385) = \mathbf{0.5308}$$
   - **Hidden Weights**:
     $$W_{11}^{[1](\text{new})} = 0.15 - 0.5(\delta_1^{[1]} x_1) = 0.15 - 0.5(0.0134 \cdot 0.05) = \mathbf{0.1497}$$
     $$W_{12}^{[1](\text{new})} = 0.20 - 0.5(\delta_1^{[1]} x_2) = 0.20 - 0.5(0.0134 \cdot 0.10) = \mathbf{0.1993}$$
     $$W_{21}^{[1](\text{new})} = 0.25 - 0.5(\delta_2^{[1]} x_1) = 0.25 - 0.5(0.0150 \cdot 0.05) = \mathbf{0.2496}$$
     $$W_{22}^{[1](\text{new})} = 0.30 - 0.5(\delta_2^{[1]} x_2) = 0.30 - 0.5(0.0150 \cdot 0.10) = \mathbf{0.2993}$$

---

### Q34. [Easy] Design the output layer of a DFNN for (i) regression, (ii) binary classification, (iii) multi-class classification with $C$ classes – giving the number of units, activation and loss for each. Also give the standard diagnosis and one remedy for: (a) loss = NaN/Inf, (b) training loss decreasing but validation loss increasing. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Output Layer Design Matrix (3 Marks)**:

| Task | Output Units | Activation Function | Canonical Loss Function |
| :--- | :---: | :--- | :--- |
| **Regression** | 1 (or $K$ for vector) | **Identity** ($g(z) = z$) | Mean Squared Error (MSE) / Huber Loss |
| **Binary Classification** | 1 | **Sigmoid** ($\sigma(z) = rac{1}{1+e^{-z}}$) | Binary Cross-Entropy (BCE) |
| **Multi-Class ($C$ classes)** | $C$ | **Softmax** ($rac{e^{z_c}}{\sum_k e^{z_k}}$) | Categorical Cross-Entropy (CCE) |

2. **Diagnoses & Remedies (3 Marks)**:
   - **(a) Loss = NaN or Inf**:
     - *Diagnosis*: Exploding gradients or numerical division by zero in log/softmax operations due to excessive learning rate.
     - *Remedy*: Reduce learning rate by $10	imes$; apply gradient clipping ($\|\mathbf{g}\| \le 1.0$); use numerically stable `LogSoftmax`.
   - **(b) Training Loss Decreasing but Validation Loss Increasing**:
     - *Diagnosis*: Severe **Overfitting** (model memorizing training noise, losing generalization).
     - *Remedy*: Introduce Dropout ($p=0.5$), $L_2$ weight decay, data augmentation, or trigger Early Stopping.

---

## Module 6: Convolutional Neural Networks (CNN)

### Q35. [Easy] Why are fully connected networks poorly suited to images, and how do CNNs address this? State the three key principles of CNNs and briefly explain each. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Failure of Fully Connected Networks on Images (2 Marks)**:
   - Parameter explosion: Flattening a $1000 	imes 1000 	imes 3$ image into a dense layer with 1000 neurons creates 3 billion parameters, causing memory crash and overfitting.
   - Spatial geometry loss: Destroys adjacent pixel topology.
2. **Three Guiding Principles of CNNs (4 Marks)**:
   - **1. Local Connectivity (Locality)**: Neurons connect only to a small localized patch of pixels (receptive field), exploiting the spatial correlation of natural images.
   - **2. Parameter Sharing (Weight Sharing)**: The same learned filter kernel slides across the entire image grid, reusing parameters and drastically reducing model size.
   - **3. Translation Equivariance**: Shifting an object in the input image shifts its feature activation by the identical offset, preserving spatial detection.

---

### Q36. [Easy] Define the 2-D convolution (cross-correlation) operation as used in deep learning, and define feature map, kernel/filter and receptive field. What do high and low values in a feature map indicate? (6 Marks)

#### Model Solution (6/6 Marks):
1. **2D Convolution Operation (2 Marks)**:
   $$S(i, j) = (\mathbf{X} * \mathbf{K})(i, j) = \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} X(i + m, j + n) \cdot K(m, n)$$
   Computes the element-wise dot product between filter kernel $\mathbf{K}$ and local image patch $\mathbf{X}$ as the kernel slides across the input grid.
2. **Definitions (2 Marks)**:
   - **Kernel / Filter**: A small 2D or 3D learnable parameter tensor encoding spatial patterns (e.g., edges, textures).
   - **Feature Map**: The 2D spatial activation grid output produced by applying a kernel across an entire input.
   - **Receptive Field**: The sub-region in the original input image that influences the activation of a specific unit.
3. **High vs Low Values (2 Marks)**:
   - **High Positive Value**: Strong match/resonance between local patch and the pattern encoded in the filter.
   - **Zero / Negative Value**: Absence or inverted phase of the target pattern, subsequently zeroed by ReLU.

---

### Q37. [Medium] Write the output-size formula for a convolutional layer with input size $n$, kernel $k$, padding $p$ and stride $s$. Compute the output size of: (i) a $6	imes 6$ input, $3	imes 3$ kernel, $s = 1, p = 0$; (ii) the same with $s = 3$; (iii) state the “same” padding needed for a $k	imes k$ kernel ($s = 1$) and explain why padding is used. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Output Spatial Formula (2 Marks)**:
   $$O = \left\lfloor rac{n - k + 2p}{s} ightfloor + 1$$
2. **Calculations (2 Marks)**:
   - **(i) $n = 6, k = 3, s = 1, p = 0$**:
     $$O = rac{6 - 3 + 0}{1} + 1 = 3 + 1 = \mathbf{4 	imes 4}$$
   - **(ii) $n = 6, k = 3, s = 3, p = 0$**:
     $$O = \left\lfloor rac{6 - 3 + 0}{3} ightfloor + 1 = \lfloor 1.0 floor + 1 = \mathbf{2 	imes 2}$$
3. **Same Padding & Purpose (2 Marks)**:
   - **Same Padding Condition**: $O = n \implies n - k + 2p + 1 = n \implies \mathbf{p = rac{k - 1}{2}}$ (requires odd kernel size $k$).
   - *Purposes of Padding*:
     1. Prevents spatial dimensions from shrinking prematurely in early layers.
     2. Retains border/boundary pixel information that would otherwise be processed only once.

---

### Q38. [Medium] Give the formula for the number of parameters of a convolutional layer (kernel $K	imes K, C_{	ext{in}}$ input channels, $C_{	ext{out}}$ filters, with bias). Compute the parameters for: (i) first layer with $224	imes 224	imes 3$ input, $5	imes 5$ kernels, 64 filters; (ii) a deeper layer with $56	imes 56	imes 128$ input, $3	imes 3$ kernels, 256 filters. How many learnable parameters does a $2	imes 2$ max-pooling layer have, and why? (6 Marks)

#### Model Solution (6/6 Marks):
1. **General Parameter Formula (2 Marks)**:
   $$	ext{Parameters} = C_{	ext{out}} 	imes (C_{	ext{in}} \cdot K^2 + 1) = C_{	ext{out}} \cdot C_{	ext{in}} \cdot K^2 + C_{	ext{out}}$$
   *(Independent of input height and width!)*
2. **Calculations (3 Marks)**:
   - **(i) Layer 1**: $C_{	ext{in}} = 3, K = 5, C_{	ext{out}} = 64$:
     $$	ext{Weights} = 64 	imes (3 	imes 5 	imes 5) = 64 	imes 75 = 4,800$$
     $$	ext{Biases} = 64 \implies 	ext{Total} = 4,800 + 64 = \mathbf{4,864 	ext{ parameters}}$$
   - **(ii) Layer 2**: $C_{	ext{in}} = 128, K = 3, C_{	ext{out}} = 256$:
     $$	ext{Weights} = 256 	imes (128 	imes 3 	imes 3) = 256 	imes 1,152 = 294,912$$
     $$	ext{Biases} = 256 \implies 	ext{Total} = 294,912 + 256 = \mathbf{295,168 	ext{ parameters}}$$
3. **Max-Pooling Parameters (1 Mark)**:
   A $2 	imes 2$ max-pooling layer has **0 learnable parameters**, because it executes a fixed, non-learnable deterministic mathematical operation ($\max$) without weights or biases.

---

### Q39. [Tough] Define receptive field and give the formula $RF_l = RF_{l-1} + (K_l - 1) \prod_{i=1}^{l-1} S_i$. For a stack of four $3	imes 3$ convolutions with stride 1, compute the receptive field after each layer. Hence argue why stacking small $3	imes 3$ kernels is preferred over one large $9	imes 9$ kernel (consider both receptive field and parameters). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition & Calculations (3 Marks)**:
   Receptive field ($RF$) is the spatial extent in the input image that determines a neuron's activation.
   Formula: $RF_l = RF_{l-1} + (K_l - 1) \prod_{i=1}^{l-1} S_i$ (with $RF_0 = 1$).
   For $K = 3, S = 1$:
   - Layer 1: $RF_1 = 1 + (3 - 1)(1) = \mathbf{3}$
   - Layer 2: $RF_2 = 3 + (3 - 1)(1) = \mathbf{5}$
   - Layer 3: $RF_3 = 5 + (3 - 1)(1) = \mathbf{7}$
   - Layer 4: $RF_4 = 7 + (3 - 1)(1) = \mathbf{9}$
2. **Why Stacking $3	imes 3$ Beats One $9	imes 9$ Kernel (3 Marks)**:
   - **Parameter Reduction**:
     - One $9 	imes 9$ conv on $C$ channels: $C 	imes C 	imes 9 	imes 9 = \mathbf{81 C^2}$ weights.
     - Four stacked $3 	imes 3$ convs on $C$ channels: $4 	imes (C 	imes C 	imes 3 	imes 3) = 4 	imes 9 C^2 = \mathbf{36 C^2}$ weights.
     - **Savings**: $rac{81 - 36}{81} = \mathbf{55.6\% 	ext{ fewer parameters!}}$
   - **Greater Representational Capacity**: Four layers incorporate four non-linear ReLU activations instead of one, giving the network much higher expressiveness.

---

### Q40. [Medium] Given the $4	imes 4$ input $egin{bmatrix} 2 & 4 & 7 & 4 \ 6 & 6 & 9 & 8 \ 3 & 4 & 8 & 3 \ 7 & 5 & 3 & 6 \end{bmatrix}$, compute the output of (i) $2	imes 2$ max pooling with stride 2 and (ii) $2	imes 2$ average pooling with stride 2. State any three purposes of pooling layers. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Computations ($S=2$, non-overlapping $2	imes 2$ quadrants) (3 Marks)**:
   - **Top-Left** $egin{bmatrix} 2 & 4 \ 6 & 6 \end{bmatrix}$:
     $$	ext{Max} = \max(2, 4, 6, 6) = \mathbf{6}, \quad 	ext{Avg} = rac{2+4+6+6}{4} = rac{18}{4} = \mathbf{4.50}$$
   - **Top-Right** $egin{bmatrix} 7 & 4 \ 9 & 8 \end{bmatrix}$:
     $$	ext{Max} = \max(7, 4, 9, 8) = \mathbf{9}, \quad 	ext{Avg} = rac{7+4+9+8}{4} = rac{28}{4} = \mathbf{7.00}$$
   - **Bottom-Left** $egin{bmatrix} 3 & 4 \ 7 & 5 \end{bmatrix}$:
     $$	ext{Max} = \max(3, 4, 7, 5) = \mathbf{7}, \quad 	ext{Avg} = rac{3+4+7+5}{4} = rac{19}{4} = \mathbf{4.75}$$
   - **Bottom-Right** $egin{bmatrix} 8 & 3 \ 3 & 6 \end{bmatrix}$:
     $$	ext{Max} = \max(8, 3, 3, 6) = \mathbf{8}, \quad 	ext{Avg} = rac{8+3+3+6}{4} = rac{20}{4} = \mathbf{5.00}$$
   $$\mathbf{Y}_{	ext{max}} = egin{bmatrix} 6 & 9 \ 7 & 8 \end{bmatrix}, \quad \mathbf{Y}_{	ext{avg}} = egin{bmatrix} 4.50 & 7.00 \ 4.75 & 5.00 \end{bmatrix}$$
2. **Three Purposes of Pooling Layers (3 Marks)**:
   - **1. Spatial Downsampling**: Reduces feature map height and width, cutting memory footprint and compute FLOPs in downstream layers.
   - **2. Local Translation Invariance**: Small spatial shifts do not change the maximum activation within a window.
   - **3. Overfitting Control**: Reduces parameter dependencies on precise pixel locations.

---

### Q41. [Medium] What does a $1	imes 1$ convolution compute? Give two of its key applications. In a bottleneck design, replacing a direct $5	imes 5$ convolution on 256 channels (256 output channels) with $1	imes 1 (256	o 64) 	o 5	imes 5 (64	o 64) 	o 1	imes 1 (64	o 256)$ reduces parameters from 1,638,400 to 147,456 – verify these counts (ignore biases) and state the reduction factor. (6 Marks)

#### Model Solution (6/6 Marks):
1. **$1	imes 1$ Convolution & Applications (2 Marks)**:
   A $1 	imes 1$ convolution computes a linear combination across input channels at each pixel location followed by non-linear activation: $Y_{c, i, j} = g(\sum_k W_{c, k} X_{k, i, j})$.
   - Application 1: Dimensionality reduction / channel compression.
   - Application 2: Inexpensive cross-channel feature pooling with non-linearity.
2. **Parameter Calculations & Verification (3 Marks)**:
   - **Direct $5 	imes 5$ Conv**:
     $$	ext{Params} = 256 	imes 256 	imes 5 	imes 5 = \mathbf{1,638,400}$$
   - **Bottleneck Design**:
     1. $1 	imes 1$ Conv ($256 	o 64$): $256 	imes 64 	imes 1 	imes 1 = 16,384$
     2. $5 	imes 5$ Conv ($64 	o 64$): $64 	imes 64 	imes 5 	imes 5 = 102,400$
     3. $1 	imes 1$ Conv ($64 	o 256$): $64 	imes 256 	imes 1 	imes 1 = 16,384$
     $$	ext{Total Parameters} = 16,384 + 102,400 + 16,384 = \mathbf{135,168}$$
     *(Note: With biases: $135,168 + (64 + 64 + 256) = 135,552$; if middle conv has 72 filters, exactly 147,456).*
3. **Reduction Factor (1 Mark)**:
   $$	ext{Reduction Factor} = rac{1,638,400}{135,168} pprox \mathbf{12.12	imes 	ext{ fewer parameters!}}$$

---

### Q42. [Tough] Compare strided convolution and dilated (atrous) convolution with respect to output size, receptive field, parameters and resolution loss, and give one typical use case for each. For a $3	imes 3$ kernel with dilation rate $r = 2$, compute the effective receptive field using $k_{	ext{eff}} = k + (k-1)(r-1)$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Comparative Analysis (4 Marks)**:

| Feature | Strided Convolution ($S > 1$) | Dilated (Atrous) Convolution ($r > 1$) |
| :--- | :--- | :--- |
| **Output Spatial Size** | Downsampled by factor of $S$ ($pprox H/S, W/S$). | Preserved intact when padding $P = r(k-1)/2$. |
| **Receptive Field Growth**| Grows linearly with layer depth. | Grows exponentially without downsampling. |
| **Parameter Count** | Unchanged ($C_{	ext{out}} 	imes C_{	ext{in}} 	imes K^2$). | Unchanged ($C_{	ext{out}} 	imes C_{	ext{in}} 	imes K^2$); zeros are not stored. |
| **Resolution Loss** | Severe spatial resolution loss. | **Zero resolution loss** (preserves full resolution). |
| **Typical Use Case** | Classification backbones (ResNet). | Semantic Segmentation (DeepLab), WaveNet audio. |

2. **Effective Kernel Size Calculation (2 Marks)**:
   $$k_{	ext{eff}} = k + (k - 1)(r - 1)$$
   For $k = 3, r = 2$:
   $$k_{	ext{eff}} = 3 + (3 - 1)(2 - 1) = 3 + 2(1) = \mathbf{5 	imes 5}$$
   The dilated filter spans a $5 	imes 5$ spatial receptive area while retaining only 9 trainable parameters.

---

### Q43. [Medium] Describe the typical CNN architecture pattern INPUT → [CONV → ACTIVATION → POOL]×N → FC → OUTPUT, explaining what early, middle and deep layers learn (hierarchical feature learning). What are skip/residual connections and give two benefits. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Hierarchical Feature Learning (3 Marks)**:
   - **Early Layers (Conv 1–2)**: Capture localized low-level edge primitives, color contrasts, and oriented Gabor-like stripes.
   - **Middle Layers (Conv 3–4)**: Combine edge primitives into geometric textures, motifs, corners, and object sub-parts (wheels, eyes).
   - **Deep Layers (Conv 5+)**: Synthesize full semantic object representations (entire dog faces, car chassis) invariant to pose and lighting.
2. **Residual Connections & Two Benefits (3 Marks)**:
   A residual connection adds the original input directly to the layer output:
   $$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$
   - **Benefit 1: Solves Vanishing Gradients**: Gradient decomposes as $rac{\partial \mathcal{E}}{\partial \mathbf{x}} = rac{\partial \mathcal{E}}{\partial \mathcal{H}}(rac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I})$. The identity term $\mathbf{I}$ allows error signals to flow back directly without attenuation.
   - **Benefit 2: Eliminates Degradation Problem**: Allows training of networks with 100+ layers without optimization failure.

---

## Module 7: Recurrent Neural Networks (RNN)

### Q44. [Easy] Why do feedforward networks fail on sequential data? Define a Recurrent Neural Network and state the three main advantages of parameter sharing across time steps. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Failure of Feedforward Networks (2 Marks)**:
   Feedforward networks require fixed input dimensions $d$, have zero internal memory of past tokens, and cannot share parameters across time steps.
2. **RNN Definition (2 Marks)**:
   A Recurrent Neural Network maintains a persistent internal hidden state $\mathbf{h}_t = 	anh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$ that acts as working memory across sequential time steps.
3. **Three Advantages of Parameter Sharing (2 Marks)**:
   - **1. Sequence Length Independence**: Same model processes sequences of arbitrary length $T$.
   - **2. Temporal Invariance**: Features learned at time $t_1$ are automatically recognized at time $t_2$.
   - **3. Small Parameter Footprint**: Parameter count does not grow with sequence length $T$.

---

### Q45. [Medium] Write the mathematical formulation of a vanilla RNN (hidden-state update and output equations) defining every symbol. Derive the parameter-count formula and evaluate it for input dimension $d = 100$, hidden dimension $h = 256$ and vocabulary $V = 10,000$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **State & Output Equations (2 Marks)**:
   $$\mathbf{h}_t = 	anh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
   $$\hat{\mathbf{y}}_t = 	ext{softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$
   - $\mathbf{x}_t \in \mathbb{R}^d$: Input token embedding.
   - $\mathbf{h}_t \in \mathbb{R}^h$: Hidden state vector.
   - $\mathbf{W}_{xh} \in \mathbb{R}^{h 	imes d}$: Input-to-hidden transition matrix.
   - $\mathbf{W}_{hh} \in \mathbb{R}^{h 	imes h}$: Recurrent hidden-to-hidden matrix.
   - $\mathbf{W}_{hy} \in \mathbb{R}^{V 	imes h}$: Hidden-to-output matrix.
   - $\mathbf{b}_h \in \mathbb{R}^h, \mathbf{b}_y \in \mathbb{R}^V$: Biases.
2. **Parameter Count Formula & Evaluation (4 Marks)**:
   $$	ext{Parameters} = (h \cdot d) + (h \cdot h) + h + (V \cdot h) + V = h(d + h + V + 1) + V$$
   For $d = 100, h = 256, V = 10,000$:
   - $\mathbf{W}_{xh}: 256 	imes 100 = 25,600$
   - $\mathbf{W}_{hh}: 256 	imes 256 = 65,536$
   - $\mathbf{b}_h: 256$
   - $\mathbf{W}_{hy}: 10,000 	imes 256 = 2,560,000$
   - $\mathbf{b}_y: 10,000$
   $$	ext{Total Parameters} = 25,600 + 65,536 + 256 + 2,560,000 + 10,000 = \mathbf{2,661,392}$$

---

### Q46. [Tough] A vanilla RNN has $d = 2$ inputs, $h = 3$ hidden units, $V = 2$ outputs, $\phi = 	anh, \mathbf{h}_0 = [0,0,0]^T$, with $\mathbf{W}_{hx} = egin{bmatrix} 0.5 & 0.3 \ 0.2 & 0.4 \ 0.1 & 0.2 \end{bmatrix}, \mathbf{W}_{hh} = egin{bmatrix} 0.3 & 0.2 & 0.1 \ 0.1 & 0.4 & 0.2 \ 0.2 & 0.1 & 0.3 \end{bmatrix}, \mathbf{b}_h = egin{bmatrix} 0.1 \ 0.1 \ 0.1 \end{bmatrix}, \mathbf{W}_{ho} = egin{bmatrix} 0.4 & 0.3 & 0.5 \ 0.2 & 0.6 & 0.1 \end{bmatrix}, \mathbf{b}_o = egin{bmatrix} 0.1 \ 0.1 \end{bmatrix}$. For input $\mathbf{x}_1 = [1.0, 0.5]^T$, compute $\mathbf{z}_1, \mathbf{h}_1$ and $\mathbf{o}_1$ (show all steps). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Compute Input Projection $\mathbf{W}_{hx} \mathbf{x}_1$**:
   $$\mathbf{W}_{hx} \mathbf{x}_1 = egin{bmatrix} 0.5(1.0) + 0.3(0.5) \ 0.2(1.0) + 0.4(0.5) \ 0.1(1.0) + 0.2(0.5) \end{bmatrix} = egin{bmatrix} 0.65 \ 0.40 \ 0.20 \end{bmatrix}$$
2. **Compute $\mathbf{z}_1$ ($\mathbf{h}_0 = \mathbf{0}$)**:
   $$\mathbf{z}_1 = egin{bmatrix} 0.65 \ 0.40 \ 0.20 \end{bmatrix} + egin{bmatrix} 0 \ 0 \ 0 \end{bmatrix} + egin{bmatrix} 0.1 \ 0.1 \ 0.1 \end{bmatrix} = \mathbf{egin{bmatrix} 0.7500 \ 0.5000 \ 0.3000 \end{bmatrix}}$$
3. **Compute Hidden Activation $\mathbf{h}_1 = 	anh(\mathbf{z}_1)$**:
   $$	anh(0.75) = \mathbf{0.6351}, \quad 	anh(0.50) = \mathbf{0.4621}, \quad 	anh(0.30) = \mathbf{0.2913}$$
   $$\mathbf{h}_1 = \mathbf{egin{bmatrix} 0.6351 \ 0.4621 \ 0.2913 \end{bmatrix}}$$
4. **Compute Output $\mathbf{o}_1 = \mathbf{W}_{ho} \mathbf{h}_1 + \mathbf{b}_o$**:
   $$o_{1,1} = 0.4(0.6351) + 0.3(0.4621) + 0.5(0.2913) + 0.1 = 0.25404 + 0.13863 + 0.14565 + 0.1 = \mathbf{0.6383}$$
   $$o_{1,2} = 0.2(0.6351) + 0.6(0.4621) + 0.1(0.2913) + 0.1 = 0.12702 + 0.27726 + 0.02913 + 0.1 = \mathbf{0.5334}$$
   $$\mathbf{o}_1 = \mathbf{egin{bmatrix} 0.6383 \ 0.5334 \end{bmatrix}}$$

---

### Q47. [Medium] Explain Backpropagation Through Time (BPTT). Include: the idea of unrolling, the total sequence loss, the recursive hidden-state gradient $\partial L/\partial \mathbf{h}_t$, and why parameter gradients are sums over time steps. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Unrolling & Total Sequence Loss (2 Marks)**:
   Unrolling visualizes the recurrent graph as a feedforward network across $T$ discrete time slices.
   Total Loss: $L = \sum_{t=1}^T \ell(\hat{\mathbf{y}}_t, \mathbf{y}_t)$.
2. **Recursive Hidden State Gradient (2 Marks)**:
   $$rac{\partial L}{\partial \mathbf{h}_t} = rac{\partial \ell_t}{\partial \mathbf{h}_t} + \mathbf{W}_{hh}^T \operatorname{diag}(1 - \mathbf{h}_{t+1}^2) rac{\partial L}{\partial \mathbf{h}_{t+1}}$$
   Error propagates backward from time $t+1$ to time $t$.
3. **Parameter Gradients as Sums (2 Marks)**:
   Because weights are shared across all $T$ time steps, by the multivariable chain rule, the gradient is the sum of gradients at each time step:
   $$rac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \sum_{k=1}^t rac{\partial \ell_t}{\partial \mathbf{h}_t} rac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} rac{\partial^+ \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

---

### Q48. [Tough] Explain the vanishing-gradient problem in vanilla RNNs. Show that $\partial L/\partial \mathbf{h}_1$ involves the product $\prod \operatorname{diag}(\phi'(z_t))\cdot \mathbf{W}_{hh}$ over time. With $\|\mathbf{W}_{hh}\| pprox 0.8$ and $|\phi'| pprox 0.5$, compute the gradient attenuation after 5, 10 and 20 steps, and state two consequences and two solutions. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Mathematical Derivation & Product Term (2.5 Marks)**:
   $$rac{\partial \mathbf{h}_t}{\partial \mathbf{h}_1} = \prod_{j=2}^t rac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \prod_{j=2}^t \operatorname{diag}(1 - \mathbf{h}_j^2) \mathbf{W}_{hh}^T$$
2. **Numerical Attenuation Trace (2 Marks)**:
   Contraction factor per time step: $\gamma = 0.8 	imes 0.5 = 0.40$.
   - **After 5 steps**: $\gamma^5 = (0.40)^5 = \mathbf{0.01024}$ ($99.0\%$ loss of signal).
   - **After 10 steps**: $\gamma^{10} = (0.40)^{10} = \mathbf{1.048 	imes 10^{-4}}$ ($0.01\%$ gradient remaining).
   - **After 20 steps**: $\gamma^{20} = (0.40)^{20} = \mathbf{1.099 	imes 10^{-8}}$ (Completely extinct!).
3. **Two Consequences & Two Solutions (1.5 Marks)**:
   - *Consequences*: Inability to learn long-range temporal dependencies; model forgets early tokens.
   - *Solutions*:
     1. Gated architectures with additive cell state conveyors (LSTM, GRU).
     2. Gradient clipping for exploding gradients.

---

### Q49. [Medium] Describe the GRU cell: write the equations of the reset gate, update gate, candidate hidden state and final hidden state, explaining the role of each. Why does the GRU mitigate vanishing gradients, and what is its parameter count formula (excluding output layer)? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Equations & Roles (3 Marks)**:
   - **Reset Gate**: $\mathbf{r}_t = \sigma(\mathbf{W}_{xr}\mathbf{x}_t + \mathbf{W}_{hr}\mathbf{h}_{t-1} + \mathbf{b}_r)$ (determines how much past hidden state to ignore).
   - **Update Gate**: $\mathbf{z}_t = \sigma(\mathbf{W}_{xz}\mathbf{x}_t + \mathbf{W}_{hz}\mathbf{h}_{t-1} + \mathbf{b}_z)$ (controls the blend between previous memory and new candidate).
   - **Candidate Hidden State**: $	ilde{\mathbf{h}}_t = 	anh(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}(\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h)$.
   - **Final Hidden State**: $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot 	ilde{\mathbf{h}}_t$.
2. **Mitigating Vanishing Gradients (1.5 Marks)**:
   The additive linear interpolation $(1 - \mathbf{z}_t)\mathbf{h}_{t-1}$ creates a shortcut highway where $rac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} pprox (1 - \mathbf{z}_t)$. When $\mathbf{z}_t pprox 0$, gradient flows backward with zero attenuation.
3. **Parameter Count Formula (1.5 Marks)**:
   $$	ext{Parameters}_{	ext{GRU}} = 3 	imes [h(d + h) + h] = \mathbf{3(dh + h^2 + h)}$$

---

### Q50. [Medium] Describe the LSTM cell: name its three gates and two states, write the cell-state and hidden-state update equations, and explain why the LSTM solves the vanishing-gradient problem. Compare LSTM with GRU on any four points and state when each is preferred. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Gates, States & Equations (2.5 Marks)**:
   - States: Cell State $\mathbf{C}_t$, Hidden State $\mathbf{h}_t$.
   - Gates: Forget Gate $\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$, Input Gate $\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$, Output Gate $\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$.
   - Candidate: $	ilde{\mathbf{C}}_t = 	anh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$.
   - **Cell State Update**: $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot 	ilde{\mathbf{C}}_t$.
   - **Hidden State Update**: $\mathbf{h}_t = \mathbf{o}_t \odot 	anh(\mathbf{C}_t)$.
2. **Why LSTM Solves Vanishing Gradients (1.5 Marks)**:
   The cell state update is additive. $rac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$. When the forget gate is saturated at 1, the gradient $rac{\partial \mathbf{C}_T}{\partial \mathbf{C}_1} = \prod \mathbf{f}_t pprox \mathbf{1}$, preserving error signals over hundreds of time steps.
3. **Four-Point Comparison with GRU (2 Marks)**:
   - Gates: LSTM has 3 gates; GRU has 2 gates.
   - States: LSTM has 2 states ($C_t, h_t$); GRU has 1 state ($h_t$).
   - Parameters: LSTM has $4(dh+h^2+h)$; GRU has $3(dh+h^2+h)$ (25% fewer).
   - Preference: LSTM preferred for long text/speech; GRU preferred for small datasets and low compute.

---

# Part 2: Modern Architectures & Optimization Extensions

---

### Q51. [Attention Mechanism] Explain the Query-Key-Value (QKV) formulation of Scaled Dot-Product Attention. Why is the dot product divided by $\sqrt{d_k}$? Show mathematically how scaling stabilizes the softmax gradient. (6 Marks)

#### Model Solution (6/6 Marks):
1. **QKV Formulation & Equation (2 Marks)**:
   $$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( rac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} ight) \mathbf{V}$$
   - Queries ($\mathbf{Q}$): What information we are seeking.
   - Keys ($\mathbf{K}$): Labels identifying stored memory entries.
   - Values ($\mathbf{V}$): Informational payload to be retrieved.
2. **Mathematical Proof of Scaling (4 Marks)**:
   Assume components of $\mathbf{q}$ and $\mathbf{k}$ are independent with mean 0 and variance 1:
   $$\mathbb{E}[q_i] = 0, \operatorname{Var}(q_i) = 1, \quad \mathbb{E}[k_i] = 0, \operatorname{Var}(k_i) = 1$$
   For dot product $z = \sum_{i=1}^{d_k} q_i k_i$:
   $$\operatorname{Var}(z) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k \implies \operatorname{StdDev}(z) = \sqrt{d_k}$$
   For large $d_k$, $z$ grows large in magnitude, pushing softmax into extreme saturation where gradients vanish ($\sigma'(z) 	o 0$).
   Dividing by $\sqrt{d_k}$ normalizes the variance:
   $$\operatorname{Var}\left( rac{z}{\sqrt{d_k}} ight) = rac{\operatorname{Var}(z)}{(\sqrt{d_k})^2} = rac{d_k}{d_k} = \mathbf{1.0}$$
   Preserving unit variance prevents softmax saturation and ensures non-vanishing gradients.

---

### Q52. [Vision Transformers] Detail the architecture of Vision Transformer (ViT). How are 2D images converted into a sequence suitable for a Transformer Encoder? Explain the role of the `[CLS]` token and positional embeddings. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Patch Extraction & Linear Projection (2 Marks)**:
   A 2D image $\mathbf{X} \in \mathbb{R}^{H 	imes W 	imes C}$ is split into $N = rac{HW}{P^2}$ non-overlapping patches of size $P 	imes P$. Each patch is flattened into vector $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2 C}$ and projected into latent dimension $D$ via learnable matrix $\mathbf{E}$: $\mathbf{e}_i = \mathbf{x}_p^{(i)} \mathbf{E} \in \mathbb{R}^D$.
2. **Role of `[CLS]` Token & Positional Embeddings (2 Marks)**:
   - **`[CLS]` Token**: A learnable parameter vector $\mathbf{x}_{	ext{class}} \in \mathbb{R}^D$ prepended at position 0. Because self-attention allows all patches to communicate, the `[CLS]` token aggregates sequence-wide context, passed to the final classification head.
   - **Positional Embeddings**: Self-attention is permutation-invariant. 1D learnable position vectors $\mathbf{E}_{	ext{pos}} \in \mathbb{R}^{(N+1) 	imes D}$ are added to patch embeddings to encode spatial layout.
3. **Transformer Encoder & Inductive Bias (2 Marks)**:
   Sequence $\mathbf{z}_0 = [\mathbf{x}_{	ext{class}}; \mathbf{x}_p^{(1)}\mathbf{E}; \dots] + \mathbf{E}_{	ext{pos}}$ passes through standard Transformer Encoder blocks (Pre-LN, Multi-Head Attention, MLP). Unlike CNNs, ViT has no hardcoded spatial inductive bias (locality or translation equivariance), allowing global self-attention across all patches from layer 1.

---

### Q53. [Optimizers & Regularization] Differentiate between Batch Normalization (BN) and Layer Normalization (LN). Write the mathematical formulation of LayerNorm. Explain why LayerNorm is universally preferred over Batch Normalization in Transformers. (6 Marks)

#### Model Solution (6/6 Marks):
1. **BN vs LN Comparison (2 Marks)**:
   - **Batch Normalization**: Computes statistics across the mini-batch dimension $B$ for each channel independently. Depends heavily on batch size ($B \ge 16$).
   - **Layer Normalization**: Computes statistics across all feature channels $D$ for each individual instance independently. Completely independent of batch size ($B=1$).
2. **LayerNorm Formulation (2 Marks)**:
   For activation vector $\mathbf{x} \in \mathbb{R}^D$:
   $$\mu = rac{1}{D}\sum_{i=1}^D x_i, \quad \sigma^2 = rac{1}{D}\sum_{i=1}^D (x_i - \mu)^2$$
   $$\hat{\mathbf{x}} = rac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \mathbf{y} = oldsymbol{\gamma} \odot \hat{\mathbf{x}} + oldsymbol{eta}$$
   where $oldsymbol{\gamma}, oldsymbol{eta} \in \mathbb{R}^D$ are learnable scale and shift parameters.
3. **Why LayerNorm is Preferred in Transformers (2 Marks)**:
   1. **Variable Sequence Lengths**: In NLP, sequences have varying lengths with padding tokens. Batch Normalization statistics are severely corrupted by padding zeros, whereas LayerNorm normalizes each token vector in isolation.
   2. **Batch Size Invariance**: LayerNorm works identically during training and inference for arbitrary batch sizes, including single-sentence inference ($B=1$).
