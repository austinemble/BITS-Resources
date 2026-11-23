# Course Syllabus: Mathematical Foundations for Machine Learning (AIML ZC416 / ZG514)

**Institution**: Birla Institute of Technology & Science (BITS), Pilani  
**Division**: Work Integrated Learning Programmes (WILP)  
**Degree**: M.Tech in AI & ML / Data Science  
**Credits**: 4 (Credit Model: 3 – 1 – 0)  
**Lead Faculty**: Prof. Mandar Vijay Datar  

---

## 1. Course Modules Overview

```
[Module 01: Linear Systems & Gauss Elimination]
        │
        ▼
[Module 02: Vector Spaces, Subspaces & Bases]
        │
        ▼
[Module 03: Inner Products, Norms & Analytic Geometry]
        │
        ▼
[Module 04: Linear Mappings, Eigenvalues & Diagonalization]
        │
        ▼
[Module 05: Singular Value Decomposition (SVD)]
        │
        ▼
[Module 06: Vector Calculus, Gradients & Jacobians]
        │
        ▼
[Module 07: Automatic Differentiation & Backpropagation]
        │
        ▼
[Module 08: Multivariate Taylor Series & Extrema]
        │
        ▼
[Module 09: Unconstrained Optimization & Gradient Descent]
        │
        ▼
[Module 10: Constrained Optimization, Lagrange & KKT]
        │
        ▼
[Module 11: Advanced Optimizers: Momentum, RMSProp, Adam]
        │
        ▼
[Module 12: Principal Component Analysis (PCA)]
        │
        ▼
[Module 13: Low-Rank Matrix Approximations]
        │
        ▼
[Module 14: Support Vector Machines (SVM) & Kernel Methods]
```

---

## 2. Detailed Modular Breakdown

### Module 01: Systems of Linear Equations & Gaussian Elimination
- Systems of linear equations $Ax = b$ in matrix notation.
- Augmented matrix $[A \mid b]$ and elementary row operations (swapping, scaling, row replacement).
- Row Echelon Form (REF) and Reduced Row Echelon Form (RREF).
- Pivot columns, rank of coefficient matrix $\text{rank}(A)$ vs augmented matrix $\text{rank}([A \mid b])$.
- Rouché-Capelli Theorem: unique solution, infinitely many solutions (expressed as particular solution + span of direction vectors), and inconsistent/no solution.
- Matrix inversion via Gauss-Jordan elimination $[A \mid I] \to [I \mid A^{-1}]$.

### Module 02: Vector Spaces, Subspaces, Linear Independence & Bases
- Definition of algebraic fields, groups, and vector space axioms.
- Vector subspaces: closure under vector addition and scalar multiplication.
- Linear combinations, span, and minimal spanning sets.
- Linear independence and dependence tests via determinant and row reduction.
- Basis of a vector space, non-uniqueness of bases, and dimension.
- Fundamental subspaces of a matrix: Null space $\mathcal{N}(A)$, Column space $\mathcal{C}(A)$, Row space $\mathcal{R}(A)$, Left null space $\mathcal{N}(A^\top)$.
- Rank-Nullity Theorem: $\text{rank}(A) + \text{nullity}(A) = n$.

### Module 03: Inner Products, Norms & Analytic Geometry
- Inner products on real vector spaces: positive definiteness, symmetry, and linearity in the first argument.
- Matrix-induced inner products $\langle x, y \rangle_A = x^\top A y$ where $A \succ 0$ (symmetric positive definite).
- Induced norms $\|x\| = \sqrt{\langle x, x \rangle}$ and $L_p$ norms ($L_1, L_2, L_\infty$).
- Distance metric $d(x, y) = \|x - y\|$ and angle between vectors $\cos \theta = \frac{\langle x, y \rangle}{\|x\| \|y\|}$.
- Cauchy-Schwarz inequality $|\langle x, y \rangle| \le \|x\| \|y\|$ and Triangle inequality $\|x + y\| \le \|x\| + \|y\|$.
- Orthogonal vectors, orthogonal complements $W^\perp$, and orthonormal bases.
- Gram-Schmidt orthogonalization process and orthogonal projection onto subspaces.

### Module 04: Linear Mappings, Matrix Representations, Eigenvalues & Eigenvectors
- Linear mappings $T: V \to W$ and coordinate transformation matrices.
- Change of basis and similarity transformations $B = P^{-1} A P$.
- Determinants and their geometric interpretation as signed volume scaling factors.
- Eigenvalues $\lambda$ and eigenvectors $v$: $(A - \lambda I)v = 0$.
- Characteristic polynomial $p(\lambda) = \det(A - \lambda I) = 0$.
- Algebraic multiplicity vs geometric multiplicity; defectiveness.
- Matrix diagonalization $A = P D P^{-1}$ and powers of matrices $A^k = P D^k P^{-1}$.
- Spectral Theorem for real symmetric matrices: existence of orthonormal basis of eigenvectors $A = Q \Lambda Q^\top$.

### Module 05: Singular Value Decomposition (SVD) & Spectral Theory
- Motivation: decomposing arbitrary non-square matrices $A \in \mathbb{R}^{m \times n}$.
- Gram matrices $A^\top A$ and $A A^\top$, their eigenvalues $\lambda_i \ge 0$, and singular values $\sigma_i = \sqrt{\lambda_i}$.
- Full SVD formulation: $A = U \Sigma V^\top$ where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal, and $\Sigma \in \mathbb{R}^{m \times n}$ is rectangular diagonal.
- Geometric interpretation: action of a matrix as rotation/reflection ($V^\top$), scaling along orthogonal axes ($\Sigma$), and rotation/reflection ($U$).
- Moore-Penrose pseudo-inverse $A^+ = V \Sigma^+ U^\top$ and minimal-norm least squares solutions.

### Module 06: Vector Calculus, Gradients, Jacobians & Hessians
- Multivariable functions $f: \mathbb{R}^n \to \mathbb{R}$ and vector fields $f: \mathbb{R}^n \to \mathbb{R}^m$.
- Partial derivatives and the gradient operator $\nabla f(x) \in \mathbb{R}^n$.
- Jacobian matrix $J_f(x) \in \mathbb{R}^{m \times n}$ of vector-valued functions.
- Hessian matrix $H(x) \in \mathbb{R}^{n \times n}$ of second-order partial derivatives and symmetry (Schwarz's theorem).
- Multivariable chain rule: composition of vector-valued functions and matrix product of Jacobians.
- Matrix calculus identities: derivatives of linear forms $\nabla_x (a^\top x) = a$, quadratic forms $\nabla_x (x^\top A x) = (A + A^\top)x = 2Ax$ (for symmetric $A$).

### Module 07: Automatic Differentiation & Backpropagation
- Computational graphs: nodes as operations/variables, directed edges as data flows.
- Forward accumulation mode: propagating directional derivatives alongside forward evaluation.
- Reverse accumulation mode (Backpropagation): propagating adjoint sensitivities from output loss backward to inputs.
- Matrix derivatives in neural network layers: $\frac{\partial L}{\partial W} = \delta x^\top$.
- Computational complexity: forward mode $O(n)$ vs reverse mode $O(m)$ for $f: \mathbb{R}^n \to \mathbb{R}^m$.

### Module 08: Multivariate Taylor Series & Extrema
- Linearization (first-order Taylor approximation) around operating point $x_0$: $f(x) \approx f(x_0) + \nabla f(x_0)^\top (x - x_0)$.
- Quadratic approximation (second-order Taylor expansion):
  $$f(x) \approx f(x_0) + \nabla f(x_0)^\top (x - x_0) + \frac{1}{2} (x - x_0)^\top H(x_0) (x - x_0)$$
- Stationary / critical points: $\nabla f(x^*) = 0$.
- Second derivative test via eigenvalues of Hessian $H(x^*)$:
  - $H(x^*) \succ 0$ (all eigenvalues $> 0$): strict local minimum.
  - $H(x^*) \prec 0$ (all eigenvalues $< 0$): strict local maximum.
  - Indefinite Hessian (positive and negative eigenvalues): saddle point.

### Module 09: Unconstrained Optimization & Gradient Descent
- General unconstrained optimization problem: $\min_{x} f(x)$.
- Gradient Descent update rule: $x^{(t+1)} = x^{(t)} - \eta \nabla f(x^{(t)})$.
- Step size / learning rate selection: exact line search $\phi(\alpha) = f(x - \alpha \nabla f(x))$, backtracking Armijo condition, binary search.
- Convergence properties: convex functions, condition number $\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$, oscillations in ill-conditioned ravines.

### Module 10: Constrained Optimization, Lagrange Multipliers & KKT Conditions
- Equality constrained optimization: $\min f(x)$ s.t. $g_i(x) = 0$.
- Geometric condition: gradient of objective must be collinear with gradient of constraints $\nabla f(x^*) = -\sum \lambda_i \nabla g_i(x^*)$.
- Lagrangian function: $L(x, \lambda) = f(x) + \lambda^\top g(x)$.
- Inequality constrained optimization: $\min f(x)$ s.t. $g_i(x) \le 0, h_j(x) = 0$.
- Karush-Kuhn-Tucker (KKT) conditions:
  1. Stationarity: $\nabla_x L(x^*, \lambda^*, \mu^*) = 0$.
  2. Primal feasibility: $g_i(x^*) \le 0, h_j(x^*) = 0$.
  3. Dual feasibility: $\mu_i^* \ge 0$.
  4. Complementary slackness: $\mu_i^* g_i(x^*) = 0$ for all $i$.

### Module 11: Advanced First- & Second-Order Optimizers
- Classical momentum: velocity update $v^{(t+1)} = \beta v^{(t)} + \eta \nabla f(x^{(t)})$, parameter update $x^{(t+1)} = x^{(t)} - v^{(t+1)}$.
- Nesterov Accelerated Gradient (NAG): looking ahead before computing gradient.
- Adaptive learning rate methods:
  - AdaGrad: accumulating squared gradients $G_t = G_{t-1} + g_t \odot g_t$.
  - RMSProp: exponentially decaying squared gradient average $v_t = \beta v_{t-1} + (1-\beta) g_t^2$.
  - Adam (Adaptive Moment Estimation): combining first moment $m_t$ and second raw moment $v_t$ with bias correction terms $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$.

### Module 12: Principal Component Analysis (PCA) & Dimensionality Reduction
- Unsupervised learning and feature space reduction.
- Centering data: empirical mean vector $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$.
- Sample covariance matrix: $S = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^\top = \frac{1}{n-1} \tilde{X}^\top \tilde{X}$.
- Maximum variance formulation: finding unit vector $u_1$ maximizing $u_1^\top S u_1$ subject to $\|u_1\|_2 = 1 \implies S u_1 = \lambda_1 u_1$.
- Minimum reconstruction error formulation: minimizing $\|x - \tilde{x}\|^2$.
- Principal components as eigenvectors of $S$ sorted in descending order of eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d$.
- Proportion of variance explained: $\frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$.

### Module 13: Low-Rank Matrix Approximations
- Best rank-$k$ approximation of a matrix $A$ in Frobenius and spectral norms.
- Eckart-Young-Mirsky Theorem: $A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$ achieves minimal error $\min_{\text{rank}(B) \le k} \|A - B\|_F = \sqrt{\sum_{j=k+1}^r \sigma_j^2}$.
- Spectral norm approximation error: $\|A - A_k\|_2 = \sigma_{k+1}$.
- Connection between SVD and PCA: PCA on centered matrix $\tilde{X}$ is equivalent to right singular vectors $V$ of $\tilde{X}$.

### Module 14: Classification with Support Vector Machines (SVM) & Kernels
- Linear classifiers: decision rule $y = \text{sign}(w^\top x + b)$.
- Geometric margin vs functional margin.
- Hard-margin SVM primal optimization: $\min_{w, b} \frac{1}{2} \|w\|^2$ s.t. $y_i(w^\top x_i + b) \ge 1$.
- Soft-margin SVM with slack variables $\xi_i \ge 0$: $\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i$ s.t. $y_i(w^\top x_i + b) \ge 1 - \xi_i$.
- Equivalent unconstrained formulation with Hinge Loss: $\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w^\top x_i + b))$.
- Wolfe Dual formulation: $\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum \sum \alpha_i \alpha_j y_i y_j x_i^\top x_j$ s.t. $0 \le \alpha_i \le C$ and $\sum \alpha_i y_i = 0$.
- Support vectors: points with $\alpha_i > 0$ lying on or violating the margin.
- The Kernel Trick: replacing inner products $x_i^\top x_j$ with kernel function $k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$ (Linear, Polynomial, Radial Basis Function RBF / Gaussian).
