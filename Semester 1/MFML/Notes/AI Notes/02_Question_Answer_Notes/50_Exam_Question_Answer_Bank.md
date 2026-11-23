# Comprehensive 50-Question Examination Bank & Model Solutions

**Course**: AIML ZC416 — Mathematical Foundations for Machine Learning  
**Prescribed Textbooks**: T1 Deisenroth, Faisal & Ong (*MML*); R1 Boyd & Vandenberghe (*VMLS*); R2 Boyd & Vandenberghe (*Convex Optimization*)  
**Primary Faculty**: Prof. Mandar Vijay Datar (Lead Instructor) | **Contributors**: Prof. Saurabh, Prof. Michael Alphonse, Prof. Upendar Mendu  
**Target Coverage**: BITS Pilani Mid-Semester Examination (Closed Book, 30%) & Comprehensive Examination (Open Book, 40%)

---

## Structure & Marking Scheme
Each model answer is structured according to the official BITS Pilani university marking rubric (6 Marks total):
- **Part I: Definition & Theoretical Basis** (2 Marks)
- **Part II: Mathematical Derivation & Step-by-Step Calculation** (2 Marks)
- **Part III: Geometric Insight, Diagram & Dimensional Verification** (2 Marks)

Questions are stratified into three difficulty tiers:
- **Tier 1: Foundational / Easy** (Questions 1–15)
- **Tier 2: Core Analytical / Medium** (Questions 16–40)
- **Tier 3: Advanced Comprehensive / Tough** (Questions 41–50)

---

## Tier 1: Foundational Questions (Easy — 15 Questions)

### Q1. Elementary Row Operations and Invariance of Solution Sets
**Question (6 Marks)**: State the three elementary row operations. Prove that applying any elementary row operation to an augmented matrix $[A \mid b]$ preserves the exact solution set of the linear system $Ax = b$.

**Model Answer**:
1. **Definition & Operations (2 Marks)**:
   The three Elementary Row Operations (EROs) are:
   - $R_i \leftrightarrow R_j$: Swap rows $i$ and $j$.
   - $R_i \leftarrow c R_i$ ($c \ne 0$): Scale row $i$ by a non-zero scalar $c$.
   - $R_i \leftarrow R_i + c R_j$ ($i \ne j$): Add a scalar multiple of row $j$ to row $i$.
2. **Mathematical Invariance Proof (2 Marks)**:
   Each ERO can be represented by premultiplication by an invertible **elementary matrix** $E \in \mathbb{R}^{m \times m}$.
   - For any vector $x$, if $A x = b$, then multiplying both sides by $E$ yields $(E A) x = (E b)$. Thus, every solution of $A x = b$ is a solution of $(E A) x = E b$.
   - Conversely, because every elementary matrix is invertible ($E^{-1}$ exists and is itself an elementary matrix of the same type), $(E A) x = E b \implies E^{-1} (E A) x = E^{-1} (E b) \implies A x = b$.
   Therefore, the solution sets are identical: $\mathcal{S}_{A, b} = \mathcal{S}_{EA, Eb}$.
3. **Insight & Conclusion (2 Marks)**:
   Geometrically, EROs do not move the hyperplanes defined by the system; they form linear combinations of the existing constraint equations, preserving their unique intersection manifold in $\mathbb{R}^n$.

---

### Q2. Linear Independence Definition and Testing
**Question (6 Marks)**: Define linear independence for a set of vectors $\{v_1, \dots, v_k\}$ in $\mathbb{R}^n$. Determine whether the vectors $v_1 = (1, 2, 3)^\top, v_2 = (4, 5, 6)^\top, v_3 = (7, 8, 9)^\top$ are linearly independent.

**Model Answer**:
1. **Definition (2 Marks)**:
   A set of vectors $\{v_1, \dots, v_k\}$ in $\mathbb{R}^n$ is **linearly independent** if the vector equation $c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = \mathbf{0}$ has only the trivial solution $c_1 = c_2 = \cdots = c_k = 0$.
2. **Step-by-Step Calculation (2 Marks)**:
   Form the matrix $V = [v_1 \mid v_2 \mid v_3]$:
   $$V = \begin{bmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{bmatrix}$$
   Compute the determinant along Row 1:
   $$\det(V) = 1(45 - 48) - 4(18 - 24) + 7(12 - 15) = 1(-3) - 4(-6) + 7(-3) = -3 + 24 - 21 = \mathbf{0}$$
3. **Conclusion & Relation (2 Marks)**:
   Because $\det(V) = 0$, the matrix is singular, meaning there exist non-trivial solutions.
   Indeed, notice $v_1 + v_3 = (8, 10, 12)^\top = 2 v_2 \implies v_1 - 2 v_2 + v_3 = \mathbf{0}$.
   Thus, $\{v_1, v_2, v_3\}$ is **linearly dependent**.

---

### Q3. Subspace Verification
**Question (6 Marks)**: State the subspace criterion. Prove whether the subset $W = \{(x_1, x_2, x_3) \in \mathbb{R}^3 : 2x_1 - x_2 + 3x_3 = 0\}$ is a vector subspace of $\mathbb{R}^3$.

**Model Answer**:
1. **Subspace Criterion (2 Marks)**:
   A subset $W \subseteq V$ is a subspace if: (i) $\mathbf{0} \in W$, (ii) $\forall u, v \in W, u + v \in W$, (iii) $\forall u \in W, c \in \mathbb{R}, c u \in W$.
2. **Verification (2 Marks)**:
   - **Zero vector**: $2(0) - (0) + 3(0) = 0 \implies \mathbf{0} \in W$.
   - **Linear combinations**: Let $u = (u_1, u_2, u_3) \in W$ and $v = (v_1, v_2, v_3) \in W$, so $2u_1 - u_2 + 3u_3 = 0$ and $2v_1 - v_2 + 3v_3 = 0$.
     For any $\alpha, \beta \in \mathbb{R}$, consider $w = \alpha u + \beta v$:
     $$2(\alpha u_1 + \beta v_1) - (\alpha u_2 + \beta v_2) + 3(\alpha u_3 + \beta v_3) = \alpha(2u_1 - u_2 + 3u_3) + \beta(2v_1 - v_2 + 3v_3) = \alpha(0) + \beta(0) = 0$$
3. **Geometric Insight (2 Marks)**:
   Since $w \in W$, $W$ is closed under linear combinations and is a **subspace of $\mathbb{R}^3$**. Geometrically, $W$ represents a 2-dimensional plane passing through the origin with normal vector $n = (2, -1, 3)^\top$.

---

### Q4. Determinant Properties and Geometric Interpretation
**Question (6 Marks)**: Explain the geometric interpretation of the determinant $\det(A)$ for $A \in \mathbb{R}^{2 \times 2}$ and $A \in \mathbb{R}^{3 \times 3}$. State four algebraic properties of determinants.

**Model Answer**:
1. **Geometric Interpretation (2 Marks)**:
   - In 2D, $|\det(A)|$ represents the **area** of the parallelogram spanned by the column vectors of $A$.
   - In 3D, $|\det(A)|$ represents the **volume** of the parallelepiped spanned by the column vectors of $A$.
   - The sign of $\det(A)$ indicates whether the transformation preserves ($> 0$) or reverses ($< 0$) orientation.
2. **Four Core Properties (2 Marks)**:
   - $\det(A B) = \det(A) \det(B)$
   - $\det(A^\top) = \det(A)$
   - $\det(A^{-1}) = \frac{1}{\det(A)}$ (if $A$ is invertible)
   - $\det(c A) = c^n \det(A)$ for $A \in \mathbb{R}^{n \times n}$ and scalar $c$.
3. **Singularity Test (2 Marks)**:
   $\det(A) = 0$ if and only if $A$ is singular (non-invertible), which occurs when the volume collapses to zero due to linearly dependent column vectors.

---

### Q5. Dot Product vs General Inner Product
**Question (6 Marks)**: Define a general inner product $\langle x, y \rangle$ on $\mathbb{R}^n$. Explain why every inner product on $\mathbb{R}^n$ can be written as $\langle x, y \rangle_A = x^\top A y$ where $A$ is symmetric positive definite.

**Model Answer**:
1. **Axioms (2 Marks)**:
   An inner product satisfies: (i) Positive definiteness $\langle x, x \rangle > 0$ for $x \ne \mathbf{0}$, (ii) Symmetry $\langle x, y \rangle = \langle y, x \rangle$, (iii) Linearity in first argument $\langle \alpha x + z, y \rangle = \alpha \langle x, y \rangle + \langle z, y \rangle$.
2. **Matrix Representation Derivation (2 Marks)**:
   Let $\{e_1, \dots, e_n\}$ be the standard basis. Any vectors $x, y \in \mathbb{R}^n$ can be expanded as $x = \sum x_i e_i$ and $y = \sum y_j e_j$.
   $$\langle x, y \rangle = \left\langle \sum_{i=1}^n x_i e_i, \sum_{j=1}^n y_j e_j \right\rangle = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \langle e_i, e_j \rangle = x^\top A y$$
   where $A_{ij} = \langle e_i, e_j \rangle$.
3. **SPD Requirements (2 Marks)**:
   - By symmetry, $A_{ij} = \langle e_i, e_j \rangle = \langle e_j, e_i \rangle = A_{ji} \implies A = A^\top$.
   - By positive definiteness, for all $x \ne \mathbf{0}$, $x^\top A x = \langle x, x \rangle > 0 \implies A \succ 0$.
   Hence, $A$ must be Symmetric Positive Definite.

---

### Q6. Cauchy-Schwarz Inequality Proof
**Question (6 Marks)**: State and prove the Cauchy-Schwarz inequality for an arbitrary inner product space $(V, \langle \cdot, \cdot \rangle)$. Under what condition does equality hold?

**Model Answer**:
1. **Statement (2 Marks)**:
   For all $x, y \in V$, $|\langle x, y \rangle| \le \|x\| \|y\|$, where $\|x\| = \sqrt{\langle x, x \rangle}$.
2. **Analytical Proof (2 Marks)**:
   If $y = \mathbf{0}$, both sides are 0, so equality holds trivially. Assume $y \ne \mathbf{0}$.
   For any real scalar $t \in \mathbb{R}$, consider the vector $x - t y$. By positive definiteness:
   $$\|x - t y\|^2 = \langle x - t y, x - t y \rangle = \langle x, x \rangle - 2t \langle x, y \rangle + t^2 \langle y, y \rangle \ge 0$$
   This is a quadratic polynomial $a t^2 + b t + c \ge 0$ in $t$, where $a = \|y\|^2 > 0, b = -2\langle x, y \rangle, c = \|x\|^2$.
   For this quadratic to be non-negative for all $t$, its discriminant must be non-positive:
   $$\Delta = b^2 - 4ac \le 0 \implies (-2\langle x, y \rangle)^2 - 4(\|y\|^2)(\|x\|^2) \le 0$$
   $$4 \langle x, y \rangle^2 \le 4 \|x\|^2 \|y\|^2 \implies |\langle x, y \rangle| \le \|x\| \|y\|$$
3. **Equality Condition (2 Marks)**:
   Equality holds ($\Delta = 0$) if and only if there exists $t$ such that $\|x - ty\|^2 = 0 \implies x = t y$, meaning $x$ and $y$ are **linearly dependent**.

---

### Q7. Matrix Inversion via Gauss-Jordan Elimination
**Question (6 Marks)**: Using the Gauss-Jordan elimination method, find the inverse of $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$.

**Model Answer**:
1. **Setup Augmented Matrix (2 Marks)**:
   $$[A \mid I] = \left[\begin{array}{cc|cc} 1 & 2 & 1 & 0 \\ 3 & 4 & 0 & 1 \end{array}\right]$$
2. **Row Operations (2 Marks)**:
   - $R_2 \leftarrow R_2 - 3 R_1$:
     $$\left[\begin{array}{cc|cc} 1 & 2 & 1 & 0 \\ 0 & -2 & -3 & 1 \end{array}\right]$$
   - $R_2 \leftarrow -\frac{1}{2} R_2$:
     $$\left[\begin{array}{cc|cc} 1 & 2 & 1 & 0 \\ 0 & 1 & 3/2 & -1/2 \end{array}\right]$$
   - $R_1 \leftarrow R_1 - 2 R_2$:
     $$\left[\begin{array}{cc|cc} 1 & 0 & 1 - 2(3/2) & 0 - 2(-1/2) \\ 0 & 1 & 3/2 & -1/2 \end{array}\right] = \left[\begin{array}{cc|cc} 1 & 0 & -2 & 1 \\ 0 & 1 & 3/2 & -1/2 \end{array}\right]$$
3. **Verification (2 Marks)**:
   $$A^{-1} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}$$
   Check: $A A^{-1} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix} = \begin{bmatrix} -2 + 3 & 1 - 1 \\ -6 + 6 & 3 - 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I$. Correct!

---

### Q8. Sylvester's Criterion for Positive Definiteness
**Question (6 Marks)**: State Sylvester's criterion. Determine whether the matrix $A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$ is positive definite.

**Model Answer**:
1. **Sylvester's Criterion (2 Marks)**:
   A real symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive definite ($A \succ 0$) if and only if all of its leading principal minors $\Delta_k = \det(A_{1..k, 1..k})$ are strictly positive for $k = 1, \dots, n$.
2. **Calculation (2 Marks)**:
   - Check symmetry: $A^\top = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix} = A$ (Symmetric).
   - Minor 1: $\Delta_1 = a_{11} = 4 > 0$.
   - Minor 2: $\Delta_2 = \det(A) = (4)(3) - (2)(2) = 12 - 4 = 8 > 0$.
3. **Geometric Insight (2 Marks)**:
   Because both leading principal minors are positive, $A$ is **positive definite**. Geometrically, the quadratic form $q(x) = x^\top A x = 4x_1^2 + 4x_1 x_2 + 3x_2^2$ forms an upward-opening paraboloid with level sets that are concentric ellipses centered at $(0, 0)$.

---

### Q9. BITS Midterm 1 Regular Q5 — Eigenvalues of Rotation Matrix
**Question (6 Marks)**: For $A(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ with $\theta = 90^\circ$:  
(a) Find all eigenvalues of $A(90^\circ)$. [3 M]  
(b) Find the corresponding eigenvectors. [3 M]

**Model Answer**:
1. **Matrix Formulation (2 Marks)**:
   For $\theta = 90^\circ$, $\cos 90^\circ = 0, \sin 90^\circ = 1$:
   $$A(90^\circ) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
2. **Eigenvalues (2 Marks)**:
   Characteristic equation: $\det(A - \lambda I) = \det \begin{bmatrix} -\lambda & -1 \\ 1 & -\lambda \end{bmatrix} = \lambda^2 - (-1) = \lambda^2 + 1 = 0 \implies \mathbf{\lambda = \pm i}$.
3. **Eigenvectors (2 Marks)**:
   - **For $\lambda_1 = i$**:
     $$\begin{bmatrix} -i & -1 \\ 1 & -i \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies -i v_1 - v_2 = 0 \implies v_2 = -i v_1 \implies \mathbf{v_1 = \begin{bmatrix} 1 \\ -i \end{bmatrix}}$$
   - **For $\lambda_2 = -i$**:
     $$\begin{bmatrix} i & -1 \\ 1 & i \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies i v_1 - v_2 = 0 \implies v_2 = i v_1 \implies \mathbf{v_2 = \begin{bmatrix} 1 \\ i \end{bmatrix}}$$
   *Insight*: A pure 2D rotation by $90^\circ$ leaves no real vector pointing along its original line, which is why its eigenvalues are purely imaginary.

---

### Q10. Gradient of Linear and Quadratic Forms
**Question (6 Marks)**: Let $x \in \mathbb{R}^n$, $a \in \mathbb{R}^n$, and $A \in \mathbb{R}^{n \times n}$. Derive the gradient of:  
(a) $f(x) = a^\top x$ [3 M]  
(b) $g(x) = x^\top A x$ [3 M]

**Model Answer**:
1. **Linear Form Derivation (3 Marks)**:
   $f(x) = \sum_{j=1}^n a_j x_j$. Taking the partial derivative with respect to $x_k$:
   $$\frac{\partial f}{\partial x_k} = \frac{\partial}{\partial x_k} \left( \sum_{j=1}^n a_j x_j \right) = a_k$$
   Assembling into the gradient vector:
   $$\nabla_x (a^\top x) = \begin{bmatrix} a_1 \\ \vdots \\ a_n \end{bmatrix} = \mathbf{a}$$
2. **Quadratic Form Derivation (3 Marks)**:
   $g(x) = \sum_{i=1}^n \sum_{j=1}^n A_{ij} x_i x_j$. Differentiating with respect to $x_k$ using the product rule:
   $$\frac{\partial g}{\partial x_k} = \sum_{j=1}^n A_{kj} x_j + \sum_{i=1}^n A_{ik} x_i = (A x)_k + (A^\top x)_k$$
   Assembling into matrix notation:
   $$\nabla_x (x^\top A x) = \mathbf{(A + A^\top) x}$$
   If $A$ is symmetric ($A = A^\top$), $\nabla_x (x^\top A x) = \mathbf{2 A x}$.

---

### Q11. Concept of Eigenspace and Multiplicity
**Question (6 Marks)**: Define the eigenspace $\mathcal{E}_\lambda$. Differentiate between algebraic multiplicity and geometric multiplicity. Give an example of a defective matrix.

**Model Answer**:
1. **Definitions (2 Marks)**:
   - **Eigenspace $\mathcal{E}_\lambda$**: The set of all eigenvectors corresponding to eigenvalue $\lambda$, along with $\mathbf{0}$: $\mathcal{E}_\lambda = \mathcal{N}(A - \lambda I)$.
   - **Algebraic Multiplicity (AM)**: The multiplicity of $\lambda$ as a root of the characteristic polynomial $\det(A - \lambda I) = 0$.
   - **Geometric Multiplicity (GM)**: The dimension of the eigenspace $\dim(\mathcal{E}_\lambda) = \text{nullity}(A - \lambda I)$.
2. **Defective Matrix Example (2 Marks)**:
   Consider $A = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$.
   - $\det(A - \lambda I) = (2 - \lambda)^2 = 0 \implies \lambda = 2$ with $\text{AM}(2) = 2$.
   - $A - 2I = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \implies \text{rank}(A - 2I) = 1 \implies \text{GM}(2) = 2 - 1 = 1$.
3. **Conclusion (2 Marks)**:
   Since $\text{GM}(2) = 1 < \text{AM}(2) = 2$, $A$ lacks a complete basis of eigenvectors and is **defective** (non-diagonalizable).

---

### Q12. Directional Derivative and Steepest Descent
**Question (6 Marks)**: Define the directional derivative $D_v f(x)$ of a differentiable function $f: \mathbb{R}^n \to \mathbb{R}$ in unit direction $v$. Prove that $-\nabla f(x)$ is the direction of steepest descent.

**Model Answer**:
1. **Definition (2 Marks)**:
   $$D_v f(x) = \lim_{t \to 0} \frac{f(x + t v) - f(x)}{t} = \nabla f(x)^\top v = \|\nabla f(x)\|_2 \|v\|_2 \cos\theta$$
   where $\theta$ is the angle between $\nabla f(x)$ and $v$.
2. **Steepest Descent Proof (2 Marks)**:
   To find the direction of steepest descent, minimize $D_v f(x)$ subject to $\|v\|_2 = 1$:
   $$\min_{\|v\|=1} D_v f(x) = \min_{\theta} \|\nabla f(x)\|_2 \cos\theta$$
   Since $\cos\theta \in [-1, 1]$, the minimum is achieved when $\cos\theta = -1 \implies \theta = \pi$ ($180^\circ$).
3. **Result (2 Marks)**:
   $\theta = \pi$ means $v$ points in the exact opposite direction of the gradient:
   $$v^* = -\frac{\nabla f(x)}{\|\nabla f(x)\|_2}$$
   Hence, $-\nabla f(x)$ is the unique direction of **steepest descent**.

---

### Q13. Gram-Schmidt Orthogonalization Formula
**Question (6 Marks)**: State the Gram-Schmidt orthogonalization algorithm for three linearly independent vectors $\{v_1, v_2, v_3\}$ in an inner product space. Explain the projection term.

**Model Answer**:
1. **Step-by-Step Algorithm (3 Marks)**:
   - **Step 1**: Set $u_1 = v_1$, $e_1 = \frac{u_1}{\|u_1\|}$.
   - **Step 2**: Subtract the projection of $v_2$ onto $u_1$:
     $$u_2 = v_2 - \text{proj}_{u_1}(v_2) = v_2 - \frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle} u_1, \qquad e_2 = \frac{u_2}{\|u_2\|}$$
   - **Step 3**: Subtract the projections of $v_3$ onto $u_1$ and $u_2$:
     $$u_3 = v_3 - \frac{\langle v_3, u_1 \rangle}{\langle u_1, u_1 \rangle} u_1 - \frac{\langle v_3, u_2 \rangle}{\langle u_2, u_2 \rangle} u_2, \qquad e_3 = \frac{u_3}{\|u_3\|}$$
2. **Geometric Projection Explanation (3 Marks)**:
   The projection term $\text{proj}_u(v) = \frac{\langle v, u \rangle}{\|u\|^2} u$ represents the shadow of $v$ along $u$. Subtracting this shadow leaves the perpendicular error vector $u_2 \perp u_1$, guaranteeing pairwise orthogonality.

---

### Q14. Functional vs Geometric Margin in SVM
**Question (6 Marks)**: In Support Vector Machines, define functional margin and geometric margin of a training point $(x_i, y_i)$. Explain why scaling $w$ and $b$ affects functional margin but leaves geometric margin invariant.

**Model Answer**:
1. **Definitions (2 Marks)**:
   - **Functional Margin**: $\hat{\gamma}_i = y_i (w^\top x_i + b)$
   - **Geometric Margin**: $\gamma_i = \frac{y_i (w^\top x_i + b)}{\|w\|_2} = \frac{\hat{\gamma}_i}{\|w\|_2}$
2. **Scaling Analysis (2 Marks)**:
   Consider scaling the hyperplane parameters by scalar $\alpha > 0$: $w' = \alpha w, b' = \alpha b$.
   - $\hat{\gamma}_i' = y_i ((\alpha w)^\top x_i + \alpha b) = \alpha [y_i(w^\top x_i + b)] = \alpha \hat{\gamma}_i$. (Functional margin scales by $\alpha$).
   - $\gamma_i' = \frac{\alpha \hat{\gamma}_i}{\|\alpha w\|_2} = \frac{\alpha \hat{\gamma}_i}{\alpha \|w\|_2} = \frac{\hat{\gamma}_i}{\|w\|_2} = \gamma_i$. (Geometric margin is unchanged).
3. **SVM Formulation Consequence (2 Marks)**:
   Because the geometric margin is scale-invariant, we fix the arbitrary scale factor by imposing $\min_i \hat{\gamma}_i = 1$. The geometric margin at support vectors then becomes $\frac{1}{\|w\|_2}$, transforming margin maximization into $\min \frac{1}{2} \|w\|_2^2$.

---

### Q15. Soft-Margin SVM Slack Variables
**Question (6 Marks)**: Explain the role of slack variables $\xi_i$ in soft-margin SVM. Interpret the physical meaning of the three cases: $\xi_i = 0$, $0 < \xi_i \le 1$, and $\xi_i > 1$.

**Model Answer**:
1. **Role of Slack Variables (2 Marks)**:
   Slack variables $\xi_i \ge 0$ relax the rigid margin constraint to $y_i(w^\top x_i + b) \ge 1 - \xi_i$, allowing data points to fall inside the margin or be misclassified in non-linearly separable or noisy datasets.
2. **Three Cases Interpretation (3 Marks)**:
   - **$\xi_i = 0$**: The point is correctly classified and lies on or strictly outside the margin boundary ($y_i(w^\top x_i + b) \ge 1$). Zero penalty.
   - **$0 < \xi_i \le 1$**: The point is correctly classified ($\text{sign}(w^\top x_i + b) = y_i$) but falls **inside the margin** band ($0 \le y_i(w^\top x_i + b) < 1$).
   - **$\xi_i > 1$**: The point falls on the wrong side of the decision boundary ($y_i(w^\top x_i + b) < 0$) and is **misclassified**.
3. **Regularization Parameter $C$ (1 Mark)**:
   $C \sum \xi_i$ penalizes the sum of margin violations; large $C$ forces narrow margins with fewer violations, while small $C$ tolerates more violations for a wider margin.

---

## Tier 2: Core Analytical & Exam Numericals (Medium — 25 Questions)

### Q16. BITS Midterm 1 Regular Q1 — Row Echelon Form and Complete Solution
**Question (6 Marks)**: Reduce $[A \mid b]$ to row echelon form and find the complete solution for $A = \begin{bmatrix} 1 & 1 & 2 \\ 2 & 3 & 4 \\ 1 & 2 & 5 \end{bmatrix}, b = \begin{bmatrix} 5 \\ 11 \\ 9 \end{bmatrix}$.

**Model Answer**:
1. **Setup & Phase 1 EROs (2 Marks)**:
   $$[A \mid b] = \left[\begin{array}{ccc|c} 1 & 1 & 2 & 5 \\ 2 & 3 & 4 & 11 \\ 1 & 2 & 5 & 9 \end{array}\right] \xrightarrow{R_2 \leftarrow R_2 - 2R_1, \; R_3 \leftarrow R_3 - R_1} \left[\begin{array}{ccc|c} 1 & 1 & 2 & 5 \\ 0 & 1 & 0 & 1 \\ 0 & 1 & 3 & 4 \end{array}\right]$$
2. **Phase 2 ERO to REF (2 Marks)**:
   $$\xrightarrow{R_3 \leftarrow R_3 - R_2} \left[\begin{array}{ccc|c} 1 & 1 & 2 & 5 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 3 & 3 \end{array}\right]$$
   - $\text{rank}(A) = 3 = \text{rank}([A \mid b]) = n \implies$ **Unique solution**.
3. **Back-Substitution (2 Marks)**:
   - Row 3: $3 x_3 = 3 \implies x_3 = 1$
   - Row 2: $x_2 = 1$
   - Row 1: $x_1 + 1 + 2(1) = 5 \implies x_1 = 2$
   $$\mathbf{x} = \begin{bmatrix} 2 \\ 1 \\ 1 \end{bmatrix}$$

---

### Q17. BITS Midterm 1 Regular Q2 — Rank Analysis & Direction Vectors
**Question (6 Marks)**: For $A = \begin{bmatrix} 1 & 2 & 1 \\ 1 & 1 & 2 \\ 2 & 3 & 3 \end{bmatrix}, b = \begin{bmatrix} 4 \\ 3 \\ 7 \end{bmatrix}$, reduce to REF, find ranks, and express the complete general solution as particular solution plus direction vector.

**Model Answer**:
1. **Row Reduction (2 Marks)**:
   $$\left[\begin{array}{ccc|c} 1 & 2 & 1 & 4 \\ 1 & 1 & 2 & 3 \\ 2 & 3 & 3 & 7 \end{array}\right] \xrightarrow{R_2 - R_1, R_3 - 2R_1} \left[\begin{array}{ccc|c} 1 & 2 & 1 & 4 \\ 0 & -1 & 1 & -1 \\ 0 & -1 & 1 & -1 \end{array}\right] \xrightarrow{R_3 - R_2} \left[\begin{array}{ccc|c} 1 & 2 & 1 & 4 \\ 0 & 1 & -1 & 1 \\ 0 & 0 & 0 & 0 \end{array}\right]$$
2. **Rank Analysis (2 Marks)**:
   $\text{rank}(A) = 2, \text{rank}([A \mid b]) = 2 < 3 \implies$ Consistent with $3 - 2 = 1$ free variable ($x_3 = t$).
3. **Parametric Decomposition (2 Marks)**:
   - $x_2 - x_3 = 1 \implies x_2 = 1 + t$
   - $x_1 + 2(1 + t) + t = 4 \implies x_1 = 2 - 3t$
   $$\mathbf{x} = \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix} + t \begin{bmatrix} -3 \\ 1 \\ 1 \end{bmatrix}, \quad t \in \mathbb{R}$$
   $x_p = [2, 1, 0]^\top$ (particular), $v = [-3, 1, 1]^\top$ (null space direction).

---

### Q18. BITS Midterm 1 Regular Q4 — Orthogonal Basis & Inequality Verification
**Question (6 Marks)**: In $\mathbb{R}^3$ with standard dot product:  
(a) Find all vectors orthogonal to $w = (2, 0, 1)^\top$ and a basis $\{v_1, v_2\}$ for this set. [2 M]  
(b) Verify the Cauchy-Schwarz and Triangle inequalities for $v_1, v_2$. [4 M]

**Model Answer**:
1. **Orthogonal Space & Basis (2 Marks)**:
   $x = (x_1, x_2, x_3)^\top \perp w \iff 2x_1 + 0x_2 + x_3 = 0 \implies x_3 = -2x_1$.
   $$x = \begin{bmatrix} x_1 \\ x_2 \\ -2x_1 \end{bmatrix} = x_1 \begin{bmatrix} 1 \\ 0 \\ -2 \end{bmatrix} + x_2 \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$
   Basis: $v_1 = (1, 0, -2)^\top, v_2 = (0, 1, 0)^\top$.
2. **Cauchy-Schwarz Verification (2 Marks)**:
   $\langle v_1, v_2 \rangle = 1(0) + 0(1) + (-2)(0) = 0$.
   $\|v_1\| = \sqrt{1^2 + 0 + (-2)^2} = \sqrt{5}, \|v_2\| = \sqrt{0 + 1^2 + 0} = 1$.
   $|\langle v_1, v_2 \rangle| = 0 \le \sqrt{5} \cdot 1 = \sqrt{5}$. (Holds with strict inequality since $v_1, v_2$ independent).
3. **Triangle Inequality Verification (2 Marks)**:
   $v_1 + v_2 = (1, 1, -2)^\top \implies \|v_1 + v_2\| = \sqrt{1^2 + 1^2 + (-2)^2} = \sqrt{6} \approx 2.449$.
   $\|v_1\| + \|v_2\| = \sqrt{5} + 1 \approx 2.236 + 1 = 3.236$.
   Since $2.449 \le 3.236$, the Triangle inequality $\|v_1 + v_2\| \le \|v_1\| + \|v_2\|$ is verified.

---

### Q19. BITS Midterm EC-2 Q5 — Matrix Inner Product Distance & Angle
**Question (6 Marks)**: For $A = \begin{bmatrix} 2 & 0 & -1 \\ 0 & 2 & 0 \\ -1 & 0 & 2 \end{bmatrix}$ and inner product $\langle x, y \rangle_A = x^\top A y$:  
(a) Find $\alpha$ such that the angle between $x_1 = [1, \alpha, 1]^\top$ and $x_2 = [1, 1, 1]^\top$ is $45^\circ$. [3 M]  
(b) Compute the induced distance $d_A(x_1, x_2)$. [3 M]

**Model Answer**:
1. **Inner Product & Norms (2 Marks)**:
   $A x_2 = [1, 2, 1]^\top \implies \langle x_1, x_2 \rangle_A = 2\alpha + 2$.
   $\|x_1\|_A^2 = 2\alpha^2 + 2, \|x_2\|_A^2 = 4 \implies \|x_2\|_A = 2$.
2. **Angle Equation Solution (2 Marks)**:
   $$\cos 45^\circ = \frac{1}{\sqrt{2}} = \frac{2\alpha + 2}{2 \sqrt{2\alpha^2 + 2}} = \frac{\alpha + 1}{\sqrt{2(\alpha^2 + 1)}} \implies \sqrt{\alpha^2 + 1} = \alpha + 1$$
   $$\alpha^2 + 1 = \alpha^2 + 2\alpha + 1 \implies 2\alpha = 0 \implies \mathbf{\alpha = 0}$$
3. **Induced Distance (2 Marks)**:
   For $\alpha = 0$, $x_1 - x_2 = [0, -1, 0]^\top$.
   $$d_A(x_1, x_2)^2 = [0, -1, 0] \begin{bmatrix} 2 & 0 & -1 \\ 0 & 2 & 0 \\ -1 & 0 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ -1 \\ 0 \end{bmatrix} = (-1)(-2) = 2 \implies d_A(x_1, x_2) = \mathbf{\sqrt{2}}$$

---

### Q20. BITS Midterm EC-2 Q6 — Computational Graph and Autodiff Chain Rule
**Question (6 Marks)**: For $a = x_1^2 + x_2, b = x_1 x_2, f = a^2 + b$:  
(a) Compute $\partial f / \partial a$ and $\partial f / \partial b$. [1 M]  
(b) Compute partials of $a$ and $b$ w.r.t $x_1, x_2$. [1 M]  
(c) Compute $\nabla f(x_1, x_2)$ at $(1, 2)$. [4 M]

**Model Answer**:
1. **Intermediate Partials (2 Marks)**:
   $\frac{\partial f}{\partial a} = 2a, \frac{\partial f}{\partial b} = 1$.  
   $\frac{\partial a}{\partial x_1} = 2x_1, \frac{\partial a}{\partial x_2} = 1, \frac{\partial b}{\partial x_1} = x_2, \frac{\partial b}{\partial x_2} = x_1$.
2. **Chain Rule Formulation (2 Marks)**:
   $\frac{\partial f}{\partial x_1} = 2a(2x_1) + 1(x_2) = 4a x_1 + x_2$.  
   $\frac{\partial f}{\partial x_2} = 2a(1) + 1(x_1) = 2a + x_1$.
3. **Evaluation at $(1, 2)$ (2 Marks)**:
   At $(1, 2)$, $a = 1^2 + 2 = 3$.
   $$\frac{\partial f}{\partial x_1} = 4(3)(1) + 2 = 12 + 2 = \mathbf{14}$$
   $$\frac{\partial f}{\partial x_2} = 2(3) + 1 = 6 + 1 = \mathbf{7} \implies \nabla f(1, 2) = \begin{bmatrix} \mathbf{14} \\ \mathbf{7} \end{bmatrix}$$

---

### Q21. BITS Midterm EC-2 Q7 — Hessian & Critical Point Classification
**Question (6 Marks)**: For $f(x, y) = x^3 + y^3 - 3xy$:  
(a) Find all critical points. [2 M]  
(b) Compute the Hessian and classify each critical point using the second-derivative test. [4 M]

**Model Answer**:
1. **Critical Points (2 Marks)**:
   $\nabla f = \begin{bmatrix} 3x^2 - 3y \\ 3y^2 - 3x \end{bmatrix} = \mathbf{0} \implies y = x^2, x = y^2 \implies x^4 - x = 0 \implies x(x - 1)(x^2 + x + 1) = 0$.
   Real critical points: $P_1 = (0, 0)$ and $P_2 = (1, 1)$.
2. **Hessian Formulation (2 Marks)**:
   $$H(x, y) = \begin{bmatrix} 6x & -3 \\ -3 & 6y \end{bmatrix}$$
3. **Classification (2 Marks)**:
   - At $P_1 = (0, 0)$: $H(0, 0) = \begin{bmatrix} 0 & -3 \\ -3 & 0 \end{bmatrix} \implies \det(H) = -9 < 0$. Eigenvalues $\pm 3$ have opposite signs $\implies \mathbf{P_1 \text{ is a saddle point}}$.
   - At $P_2 = (1, 1)$: $H(1, 1) = \begin{bmatrix} 6 & -3 \\ -3 & 6 \end{bmatrix} \implies \Delta_1 = 6 > 0, \Delta_2 = 36 - 9 = 27 > 0$. $H \succ 0 \implies \mathbf{P_2 \text{ is a strict local minimum}}$.

---

### Q22. BITS EC-2 Makeup Q1 — 4x4 Matrix Inversion and Determinant
**Question (6 Marks)**: For $A = \begin{bmatrix} 1 & 1 & 0 & 2 \\ 1 & 0 & 1 & 1 \\ 1 & 0 & 0 & 1 \\ 2 & 1 & 0 & 2 \end{bmatrix}$:  
(a) Find REF and $\det(A)$. [3 M]  
(b) Solve $Ax = b$ for $b = [-2, 2, 1, -1]^\top$. [3 M]

**Model Answer**:
1. **Row Reduction to REF (2 Marks)**:
   Subtract $R_1$ from $R_2, R_3$ and $2R_1$ from $R_4$:
   $$\left[\begin{array}{cccc|c} 1 & 1 & 0 & 2 & -2 \\ 0 & -1 & 1 & -1 & 4 \\ 0 & -1 & 0 & -1 & 3 \\ 0 & -1 & 0 & -2 & 3 \end{array}\right] \xrightarrow{R_3 - R_2, R_4 - R_2} \left[\begin{array}{cccc|c} 1 & 1 & 0 & 2 & -2 \\ 0 & -1 & 1 & -1 & 4 \\ 0 & 0 & -1 & 0 & -1 \\ 0 & 0 & -1 & -1 & -1 \end{array}\right] \xrightarrow{R_4 - R_3} \left[\begin{array}{cccc|c} 1 & 1 & 0 & 2 & -2 \\ 0 & -1 & 1 & -1 & 4 \\ 0 & 0 & -1 & 0 & -1 \\ 0 & 0 & 0 & -1 & 0 \end{array}\right]$$
2. **Determinant (2 Marks)**:
   Triangular determinant: $\det(A) = (1)(-1)(-1)(-1) = \mathbf{-1} \ne 0$. (Full rank 4, invertible).
3. **Back-Substitution (2 Marks)**:
   - Row 4: $-x_4 = 0 \implies x_4 = 0$
   - Row 3: $-x_3 = -1 \implies x_3 = 1$
   - Row 2: $-x_2 + 1 - 0 = 4 \implies -x_2 = 3 \implies x_2 = -3$
   - Row 1: $x_1 + (-3) + 0 + 2(0) = -2 \implies x_1 = 1$
   $$\mathbf{x} = \begin{bmatrix} 1 \\ -3 \\ 1 \\ 0 \end{bmatrix}$$

---

### Q23. BITS EC-2 Makeup Q3 — Full SVD of a 3x2 Matrix
**Question (6 Marks)**: Find the complete Singular Value Decomposition ($U, \Sigma, V^\top$) of $A = \begin{bmatrix} 5 & 0 \\ 5 & 0 \\ 0 & -10 \end{bmatrix}$.

**Model Answer**:
1. **$A^\top A$ and Singular Values (2 Marks)**:
   $A^\top A = \begin{bmatrix} 50 & 0 \\ 0 & 100 \end{bmatrix}$. $\lambda_1 = 100, \lambda_2 = 50 \implies \sigma_1 = 10, \sigma_2 = 5\sqrt{2}$.
   $$\Sigma = \begin{bmatrix} 10 & 0 \\ 0 & 5\sqrt{2} \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$
2. **Right Singular Vectors $V$ (2 Marks)**:
   $v_1 = [0, 1]^\top, v_2 = [1, 0]^\top \implies V = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \implies V^\top = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$.
3. **Left Singular Vectors $U$ (2 Marks)**:
   $u_1 = \frac{1}{10} A v_1 = [0, 0, -1]^\top$.  
   $u_2 = \frac{1}{5\sqrt{2}} A v_2 = [1/\sqrt{2}, 1/\sqrt{2}, 0]^\top$.  
   $u_3 \perp u_1, u_2 \implies u_3 = [1/\sqrt{2}, -1/\sqrt{2}, 0]^\top$.
   $$U = \begin{bmatrix} 0 & 1/\sqrt{2} & 1/\sqrt{2} \\ 0 & 1/\sqrt{2} & -1/\sqrt{2} \\ -1 & 0 & 0 \end{bmatrix}$$

---

### Q24. BITS Comprehensive EC-3 Makeup Q2 — Gram-Schmidt with Matrix Inner Product
**Question (6 Marks)**: For $A = \begin{bmatrix} 3 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 2 \end{bmatrix}$ and inner product $\langle x, y \rangle_A = x^\top A y$, orthonormalize $v_1 = [1, 2, -1]^\top$ and $v_2 = [2, -1, 0]^\top$.

**Model Answer**:
1. **Normalize $v_1$ (2 Marks)**:
   $A v_1 = [5, 3, -2]^\top \implies \|v_1\|_A^2 = v_1^\top A v_1 = 1(5) + 2(3) + (-1)(-2) = 13$.
   $$e_1 = \frac{1}{\sqrt{13}} \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}$$
2. **Orthogonalize $v_2$ (2 Marks)**:
   $A v_2 = [5, 1, 0]^\top \implies \langle v_2, v_1 \rangle_A = v_2^\top A v_1 = 2(5) - 1(3) + 0(-2) = 7$.
   $$u_2 = v_2 - \frac{7}{13} v_1 = \begin{bmatrix} 2 \\ -1 \\ 0 \end{bmatrix} - \frac{7}{13} \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix} = \frac{1}{13} \begin{bmatrix} 19 \\ -27 \\ 7 \end{bmatrix}$$
3. **Normalize $u_2$ (2 Marks)**:
   Let $w = [19, -27, 7]^\top$. $A w = [30, -8, 14]^\top \implies w^\top A w = 19(30) - 27(-8) + 7(14) = 884$.
   $$\|u_2\|_A = \frac{\sqrt{884}}{13} \implies e_2 = \frac{1}{\sqrt{884}} \begin{bmatrix} 19 \\ -27 \\ 7 \end{bmatrix}$$

---

### Q25. BITS Comprehensive EC-3 Makeup Q4 — Momentum Gradient Descent 2 Iterations
**Question (6 Marks)**: For $J(w_0, w_1) = (w_0 - 3)^2 + (w_1 - 3)^2$, initial point $(2, 2)$, $\eta = 0.05, \beta = 0.5, v^{(0)} = (0, 0)$. Compute $w^{(1)}$ and $w^{(2)}$.

**Model Answer**:
1. **Iteration 1 (3 Marks)**:
   $\nabla J(2, 2) = [2(2 - 3), 2(2 - 3)]^\top = [-2, -2]^\top$.
   $$v^{(1)} = 0.5(0) + 0.05 \begin{bmatrix} -2 \\ -2 \end{bmatrix} = \begin{bmatrix} -0.1 \\ -0.1 \end{bmatrix}$$
   $$w^{(1)} = \begin{bmatrix} 2 \\ 2 \end{bmatrix} - \begin{bmatrix} -0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} \mathbf{2.1} \\ \mathbf{2.1} \end{bmatrix}$$
2. **Iteration 2 (3 Marks)**:
   $\nabla J(2.1, 2.1) = [2(-0.9), 2(-0.9)]^\top = [-1.8, -1.8]^\top$.
   $$v^{(2)} = 0.5 \begin{bmatrix} -0.1 \\ -0.1 \end{bmatrix} + 0.05 \begin{bmatrix} -1.8 \\ -1.8 \end{bmatrix} = \begin{bmatrix} -0.05 - 0.09 \\ -0.05 - 0.09 \end{bmatrix} = \begin{bmatrix} -0.14 \\ -0.14 \end{bmatrix}$$
   $$w^{(2)} = \begin{bmatrix} 2.1 \\ 2.1 \end{bmatrix} - \begin{bmatrix} -0.14 \\ -0.14 \end{bmatrix} = \begin{bmatrix} \mathbf{2.24} \\ \mathbf{2.24} \end{bmatrix}$$

---

### Q26. BITS Comprehensive EC-3 Makeup Q5 — PCA on 2D Dataset
**Question (6 Marks)**: For $X = \begin{bmatrix} 0 & 3 \\ 1 & 5 \\ 2 & 7 \end{bmatrix}$, compute covariance $S$, eigenvalues/eigenvectors, and the 1D principal subspace.

**Model Answer**:
1. **Mean & Covariance $S$ (2 Marks)**:
   $\mu = [1, 5]^\top$. Centered: $\tilde{X} = \begin{bmatrix} -1 & -2 \\ 0 & 0 \\ 1 & 2 \end{bmatrix}$.
   $$S = \frac{1}{2} \tilde{X}^\top \tilde{X} = \frac{1}{2} \begin{bmatrix} 2 & 4 \\ 4 & 8 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$
2. **Eigenvalues & Eigenvectors (2 Marks)**:
   $\det(S - \lambda I) = \lambda^2 - 5\lambda = 0 \implies \mathbf{\lambda_1 = 5, \lambda_2 = 0}$.
   $u_1 = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ (for $\lambda_1 = 5$), $u_2 = \frac{1}{\sqrt{5}} \begin{bmatrix} -2 \\ 1 \end{bmatrix}$ (for $\lambda_2 = 0$).
3. **1D Principal Subspace (2 Marks)**:
   $\mathcal{W} = \text{span}\left\{ \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix} \right\}$.
   Captures $\frac{5}{5 + 0} = \mathbf{100\%}$ of dataset variance with zero reconstruction loss.

---

### Q27. BITS Comprehensive EC-3 Makeup Q6 — SVM Soft Margin & Hinge Loss
**Question (6 Marks)**: For training points $X_1(3, 3)[+1], X_2(5, 5)[+1], X_3(3, 0)[-1], X_4(5, 0)[-1]$, test parameters $w = (1, -1)^\top, b = 0, C = 1$.  
(a) Compute functional margins $y_i(w^\top X_i + b)$. [2 M]  
(b) Compute hinge losses $\max(0, 1 - \text{margin})$. [2 M]  
(c) Compute total objective value. [2 M]

**Model Answer**:
1. **Functional Margins (2 Marks)**:
   $w^\top X_i + b = x_{i1} - x_{i2}$.
   - $X_1(3, 3), y_1 = +1 \implies (+1)(3 - 3) = \mathbf{0}$
   - $X_2(5, 5), y_2 = +1 \implies (+1)(5 - 5) = \mathbf{0}$
   - $X_3(3, 0), y_3 = -1 \implies (-1)(3 - 0) = \mathbf{-3}$
   - $X_4(5, 0), y_4 = -1 \implies (-1)(5 - 0) = \mathbf{-5}$
2. **Hinge Losses (2 Marks)**:
   $\ell_1 = \max(0, 1 - 0) = \mathbf{1}$  
   $\ell_2 = \max(0, 1 - 0) = \mathbf{1}$  
   $\ell_3 = \max(0, 1 - (-3)) = \mathbf{4}$  
   $\ell_4 = \max(0, 1 - (-5)) = \mathbf{6}$  
   Total slack penalty: $\sum \ell_i = 1 + 1 + 4 + 6 = \mathbf{12}$.
3. **Total Objective (2 Marks)**:
   $\frac{1}{2} \|w\|^2 = \frac{1}{2}(1^2 + (-1)^2) = 1$.
   $$\text{Total Objective} = 1 + (1)(12) = \mathbf{13}$$

---

### Q28. Moore-Penrose Pseudo-Inverse and Minimal Norm Solution
**Question (6 Marks)**: Define the Moore-Penrose pseudo-inverse $A^+$ via SVD. Prove that $x = A^+ b$ satisfies the normal equations $A^\top A x = A^\top b$.

**Model Answer**:
1. **Definition (2 Marks)**:
   If $A = U \Sigma V^\top$, then $A^+ = V \Sigma^+ U^\top$ where $\Sigma^+_{ii} = 1/\sigma_i$ for $\sigma_i > 0$ and $0$ otherwise.
2. **Normal Equations Proof (3 Marks)**:
   Substitute $x = V \Sigma^+ U^\top b$ into $A^\top A x$:
   $$A^\top A x = (V \Sigma^\top U^\top) (U \Sigma V^\top) (V \Sigma^+ U^\top b) = V \Sigma^\top \Sigma \Sigma^+ U^\top b$$
   Notice that $\Sigma^\top \Sigma \Sigma^+ = \Sigma^\top$ because for each non-zero singular value, $\sigma_i^2 \cdot \frac{1}{\sigma_i} = \sigma_i$.
   Therefore:
   $$V \Sigma^\top \Sigma \Sigma^+ U^\top b = V \Sigma^\top U^\top b = (U \Sigma V^\top)^\top b = A^\top b$$
3. **Conclusion (1 Mark)**:
   Thus, $A^\top A (A^+ b) = A^\top b$. $x = A^+ b$ is always a least-squares solution.

---

### Q29. Spectral Theorem Orthogonality Proof
**Question (6 Marks)**: Prove that eigenvectors corresponding to distinct eigenvalues of a real symmetric matrix $A = A^\top$ are orthogonal.

**Model Answer**:
1. **Setup (2 Marks)**:
   Let $A v_1 = \lambda_1 v_1$ and $A v_2 = \lambda_2 v_2$ where $\lambda_1 \ne \lambda_2$.
2. **Derivation (3 Marks)**:
   Consider the scalar quantity $v_1^\top A v_2$:
   - Evaluating $A v_2 = \lambda_2 v_2$:
     $$v_1^\top (A v_2) = v_1^\top (\lambda_2 v_2) = \lambda_2 (v_1^\top v_2)$$
   - Using matrix transpose and symmetry $A = A^\top$:
     $$v_1^\top A v_2 = (A^\top v_1)^\top v_2 = (A v_1)^\top v_2 = (\lambda_1 v_1)^\top v_2 = \lambda_1 (v_1^\top v_2)$$
   Equating both expressions:
   $$\lambda_1 (v_1^\top v_2) = \lambda_2 (v_1^\top v_2) \implies (\lambda_1 - \lambda_2)(v_1^\top v_2) = 0$$
3. **Conclusion (1 Mark)**:
   Since $\lambda_1 \ne \lambda_2$, $\lambda_1 - \lambda_2 \ne 0$. Therefore, $v_1^\top v_2 = 0 \implies v_1 \perp v_2$.

---

### Q30. Trace and Determinant Eigenvalue Identities
**Question (6 Marks)**: Prove that for any square matrix $A \in \mathbb{R}^{n \times n}$, $\text{trace}(A) = \sum_{i=1}^n \lambda_i$ and $\det(A) = \prod_{i=1}^n \lambda_i$.

**Model Answer**:
1. **Characteristic Polynomial Expansion (2 Marks)**:
   The characteristic polynomial can be written in two equivalent forms:
   $$p(\lambda) = \det(A - \lambda I) = (-1)^n \prod_{i=1}^n (\lambda - \lambda_i) = \prod_{i=1}^n (\lambda_i - \lambda)$$
2. **Determinant Identity (2 Marks)**:
   Evaluate $p(\lambda)$ at $\lambda = 0$:
   $$p(0) = \det(A - 0 \cdot I) = \det(A) = \prod_{i=1}^n (\lambda_i - 0) = \prod_{i=1}^n \lambda_i$$
3. **Trace Identity (2 Marks)**:
   Comparing the coefficient of $(-\lambda)^{n-1}$ in the determinant expansion $\det(A - \lambda I)$ gives the sum of diagonal elements $\sum a_{ii} = \text{trace}(A)$, while in the factored form it yields $\sum \lambda_i$. Hence, $\text{trace}(A) = \sum \lambda_i$.

---

### Q31. Projection Matrix Properties
**Question (6 Marks)**: For a matrix $A \in \mathbb{R}^{m \times n}$ with full column rank, let $P = A(A^\top A)^{-1} A^\top$. Prove that $P$ is symmetric and idempotent ($P^2 = P$).

**Model Answer**:
1. **Symmetry Proof (2 Marks)**:
   $$P^\top = [A (A^\top A)^{-1} A^\top]^\top = (A^\top)^\top [(A^\top A)^{-1}]^\top A^\top = A [(A^\top A)^\top]^{-1} A^\top = A (A^\top A)^{-1} A^\top = P$$
2. **Idempotence Proof (2 Marks)**:
   $$P^2 = P \cdot P = [A (A^\top A)^{-1} A^\top] [A (A^\top A)^{-1} A^\top] = A (A^\top A)^{-1} [(A^\top A) (A^\top A)^{-1}] A^\top = A (A^\top A)^{-1} [I] A^\top = P$$
3. **Geometric Meaning (2 Marks)**:
   $P$ is an **orthogonal projection matrix** onto the column space $\mathcal{C}(A)$. Applying the projection twice ($P^2$) produces no additional change because the vector already lies in $\mathcal{C}(A)$.

---

### Q32. Rank-Nullity Theorem Application
**Question (6 Marks)**: A linear transformation $T: \mathbb{R}^5 \to \mathbb{R}^3$ is represented by $A \in \mathbb{R}^{3 \times 5}$. If $\dim(\mathcal{N}(A)) = 2$, what is $\text{rank}(A)$? Can $T$ be injective? Can $T$ be surjective?

**Model Answer**:
1. **Rank Calculation (2 Marks)**:
   By the Rank-Nullity Theorem: $\text{rank}(A) + \text{nullity}(A) = n = 5$.
   $$\text{rank}(A) = 5 - \text{nullity}(A) = 5 - 2 = \mathbf{3}$$
2. **Injective Analysis (2 Marks)**:
   $T$ is injective if and only if $\mathcal{N}(A) = \{\mathbf{0}\} \iff \text{nullity}(A) = 0$. Since $\text{nullity}(A) = 2 \ne 0$, $T$ is **not injective**.
3. **Surjective Analysis (2 Marks)**:
   $T$ is surjective if and only if $\dim(\mathcal{C}(A)) = \text{codomain dimension} = 3$. Since $\text{rank}(A) = 3$, $T$ is **surjective**.

---

### Q33. Taylor Series of $\exp(\sin(x^2))$ — BITS EC-2 Makeup Q6
**Question (6 Marks)**: Find the degree-2 Taylor polynomial of $f(x) = \exp(\sin(x^2))$ at $x_0 = 0$. State its minimum and maximum values on $\mathbb{R}$.

**Model Answer**:
1. **Standard Expansions (2 Marks)**:
   Near $u = 0$, $\sin(u) = u - \frac{u^3}{6} + \dots$. Let $u = x^2 \implies \sin(x^2) = x^2 + O(x^6)$.
   Near $v = 0$, $\exp(v) = 1 + v + \frac{v^2}{2} + \dots$.
2. **Composition & Taylor Polynomial (2 Marks)**:
   $$f(x) = \exp(x^2 + O(x^6)) = 1 + x^2 + \frac{(x^2)^2}{2} + \dots = 1 + x^2 + O(x^4)$$
   Truncating to degree 2:
   $$T_2(x) = \mathbf{1 + x^2}$$
3. **Extremum Values (2 Marks)**:
   For $f(x) = \exp(\sin(x^2))$ on $\mathbb{R}$:
   - Since $\sin(x^2) \in [-1, 1]$:
     - Minimum value: $\exp(-1) = \mathbf{1/e} \approx 0.3679$ (achieved when $\sin(x^2) = -1$).
     - Maximum value: $\exp(1) = \mathbf{e} \approx 2.7183$ (achieved when $\sin(x^2) = 1$).

---

### Q34. Eckart-Young Theorem Error in Rank-1 Approximation
**Question (6 Marks)**: A matrix $A \in \mathbb{R}^{4 \times 3}$ has singular values $\sigma_1 = 12, \sigma_2 = 5, \sigma_3 = 2$. Compute the minimal approximation error $\|A - B\|$ for $\text{rank}(B) \le 1$ under both Frobenius and Spectral norms.

**Model Answer**:
1. **Eckart-Young Theorem (2 Marks)**:
   The optimal rank-1 approximation is $A_1 = \sigma_1 u_1 v_1^\top$.
2. **Frobenius Norm Error (2 Marks)**:
   $$\|A - A_1\|_F = \sqrt{\sum_{j=2}^3 \sigma_j^2} = \sqrt{\sigma_2^2 + \sigma_3^2} = \sqrt{5^2 + 2^2} = \sqrt{25 + 4} = \mathbf{\sqrt{29}} \approx 5.385$$
3. **Spectral Norm Error (2 Marks)**:
   $$\|A - A_1\|_2 = \sigma_{1+1} = \sigma_2 = \mathbf{5}$$

---

### Q35. Adam Optimizer Moment Bias Correction Derivation
**Question (6 Marks)**: In the Adam optimizer, explain why the raw moment estimates $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ are biased toward zero at $t = 1$. Derive the bias-corrected estimator $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$.

**Model Answer**:
1. **Unrolling the Recurrence (2 Marks)**:
   Given $m_0 = \mathbf{0}$, unroll $m_t$ back to $t = 0$:
   $$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$
2. **Expectation Derivation (2 Marks)**:
   Take expectations of both sides, assuming $g_i$ comes from a stationary distribution with $\mathbb{E}[g_i] = \mathbb{E}[g_t]$:
   $$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] = \mathbb{E}[g_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$
   The sum is a finite geometric series: $\sum_{i=1}^t \beta_1^{t-i} = \sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$.
   $$\mathbb{E}[m_t] = \mathbb{E}[g_t] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g_t] (1 - \beta_1^t)$$
3. **Bias Correction (2 Marks)**:
   Since $\mathbb{E}[m_t] \ne \mathbb{E}[g_t]$ (scaled by $1 - \beta_1^t < 1$), dividing by $(1 - \beta_1^t)$ yields an unbiased estimator:
   $$\mathbb{E}[\hat{m}_t] = \mathbb{E}\left[ \frac{m_t}{1 - \beta_1^t} \right] = \mathbb{E}[g_t]$$

---

### Q36. Mercer's Theorem and Kernel Matrix Positive Semidefiniteness
**Question (6 Marks)**: State Mercer's condition. Prove that for any dataset $\{x_1, \dots, x_N\}$, the kernel Gram matrix $K_{ij} = k(x_i, x_j)$ is symmetric positive semidefinite.

**Model Answer**:
1. **Mercer's Condition (2 Marks)**:
   A continuous symmetric function $k(x, z)$ is a valid Mercer kernel if and only if for any dataset, the resulting Gram matrix $K \in \mathbb{R}^{N \times N}$ is Symmetric Positive Semidefinite ($K \succeq 0$).
2. **Proof via Feature Mapping (3 Marks)**:
   By Mercer's theorem, there exists a feature mapping $\phi(x)$ such that $k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$.
   Let $\Phi = [\phi(x_1) \mid \dots \mid \phi(x_N)]^\top$. Then $K = \Phi \Phi^\top$.
   For any vector $c \in \mathbb{R}^N$:
   $$c^\top K c = c^\top (\Phi \Phi^\top) c = (\Phi^\top c)^\top (\Phi^\top c) = \|\Phi^\top c\|_2^2 \ge 0$$
3. **Conclusion (1 Mark)**:
   Since $c^\top K c \ge 0$ for all $c \in \mathbb{R}^N$ and $K_{ij} = K_{ji}$, $K$ is **Symmetric Positive Semidefinite**.

---

### Q37. Quadratic Convergence of Newton's Method
**Question (6 Marks)**: State the update rule for Newton's method in multidimensional optimization. Contrast its convergence rate and computational cost with Gradient Descent.

**Model Answer**:
1. **Newton's Update Rule (2 Marks)**:
   $$x^{(k+1)} = x^{(k)} - [H_f(x^{(k)})]^{-1} \nabla f(x^{(k)})$$
2. **Convergence Comparison (2 Marks)**:
   - **Gradient Descent**: Linear convergence rate $\|x^{(k+1)} - x^*\| \le c \|x^{(k)} - x^*\|$ ($c \in (0, 1)$). Sensitive to condition number $\kappa(H)$.
   - **Newton's Method**: **Quadratic convergence** $\|x^{(k+1)} - x^*\| \le M \|x^{(k)} - x^*\|^2$ near local minimum. Number of correct digits doubles every iteration.
3. **Computational Cost Trade-off (2 Marks)**:
   - GD cost per step: $O(n)$ for gradient evaluation.
   - Newton cost per step: $O(n^2)$ for Hessian computation $+ O(n^3)$ for Hessian inversion. Prohibitive when $n \gg 10^4$.

---

### Q38. Condition Number and Gradient Descent Ravines
**Question (6 Marks)**: Define the condition number of a symmetric positive definite matrix $A$. Explain why a high condition number causes gradient descent to oscillate wildly.

**Model Answer**:
1. **Definition (2 Marks)**:
   $$\kappa(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)} \ge 1$$
2. **Geometric Mechanism (2 Marks)**:
   The level curves of $f(x) = \frac{1}{2} x^\top A x$ are ellipsoids whose axis lengths are proportional to $1/\sqrt{\lambda_i}$. When $\kappa(A) \gg 1$, the ellipsoid is an extremely elongated ravine.
3. **Oscillation Analysis (2 Marks)**:
   The gradient $\nabla f(x) = A x$ is dominated by the component along the eigenvector of $\lambda_{\max}$, which points almost perpendicular to the long axis of the valley. Consequently, gradient descent takes giant steps back and forth across the walls of the ravine while making minuscule progress toward the minimum at the base.

---

### Q39. Gram Matrix Properties in SVD
**Question (6 Marks)**: For $A \in \mathbb{R}^{m \times n}$, prove that $A^\top A$ and $A A^\top$ have identical non-zero eigenvalues.

**Model Answer**:
1. **Setup (2 Marks)**:
   Let $\lambda \ne 0$ be an eigenvalue of $A^\top A$ with eigenvector $v \ne \mathbf{0}$:
   $$(A^\top A) v = \lambda v$$
2. **Derivation (3 Marks)**:
   Premultiply both sides by $A$:
   $$A (A^\top A v) = A (\lambda v) \implies (A A^\top) (A v) = \lambda (A v)$$
   Let $u = A v$. We check that $u \ne \mathbf{0}$:
   $$\|u\|_2^2 = u^\top u = (A v)^\top (A v) = v^\top (A^\top A v) = v^\top (\lambda v) = \lambda \|v\|_2^2$$
   Since $\lambda \ne 0$ and $v \ne \mathbf{0}$, $\|u\|_2^2 > 0 \implies u \ne \mathbf{0}$.
3. **Conclusion (1 Mark)**:
   Thus, $u = A v$ is an eigenvector of $A A^\top$ with the exact same eigenvalue $\lambda$. By symmetry, every non-zero eigenvalue of $A A^\top$ is also an eigenvalue of $A^\top A$.

---

### Q40. Backpropagation Adjoint Formulation
**Question (6 Marks)**: In reverse-mode automatic differentiation, state the adjoint update rule for an intermediate variable $x$ that feeds into multiple downstream nodes $y_1, \dots, y_k$. Prove the multivariable chain rule.

**Model Answer**:
1. **Adjoint Definition (2 Marks)**:
   The adjoint of node $x$ is defined as $\bar{x} = \frac{\partial L}{\partial x}$, where $L$ is the scalar output.
2. **Proof via Total Derivative (3 Marks)**:
   Let $L = F(y_1(x), y_2(x), \dots, y_k(x))$. By the multivariable chain rule:
   $$\frac{\partial L}{\partial x} = \sum_{j=1}^k \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x} \implies \bar{x} = \sum_{j=1}^k \bar{y}_j \frac{\partial y_j}{\partial x}$$
3. **Computational Insight (1 Mark)**:
   This ensures that all downstream gradient flows converging onto node $x$ are summed, maintaining exact equivalence with analytical differentiation.

---

## Tier 3: Advanced Comprehensive Examination Problems (Tough — 10 Questions)

### Q41. Prof. Mandar Datar KKT Analytical Solution — Complete 4-Case Analysis
**Question (6 Marks)**: Solve analytically using KKT conditions:
$$\max -x_1^2 - 2x_2^2 \quad \text{s.t.} \quad x_1 + x_2 \ge 3, \quad x_2 - x_1^2 \ge 1$$

**Model Answer**:
1. **Standard Form & Lagrangian (2 Marks)**:
   $\min x_1^2 + 2x_2^2$ s.t. $g_1 = -x_1 - x_2 + 3 \le 0, g_2 = -x_2 + x_1^2 + 1 \le 0$.
   $$L = x_1^2 + 2x_2^2 + \mu_1(-x_1 - x_2 + 3) + \mu_2(-x_2 + x_1^2 + 1)$$
   - Stationarity: $2x_1(1 + \mu_2) - \mu_1 = 0, \quad 4x_2 - \mu_1 - \mu_2 = 0$.
   - Complementary slackness: $\mu_1 g_1 = 0, \mu_2 g_2 = 0, \mu_1 \ge 0, \mu_2 \ge 0$.
2. **Cases 1, 2, 3 Elimination (2 Marks)**:
   - Case 1 ($\mu_1 = \mu_2 = 0$): $x_1 = x_2 = 0 \implies g_1 = 3 > 0$ (Infeasible).
   - Case 2 ($\mu_1 > 0, \mu_2 = 0$): $x_1 = 2x_2, x_1 + x_2 = 3 \implies (2, 1) \implies g_2 = -1 + 4 + 1 = 4 > 0$ (Infeasible).
   - Case 3 ($\mu_1 = 0, \mu_2 > 0$): $x_1 = 0, x_2 = 1 \implies g_1 = 2 > 0$ (Infeasible).
3. **Case 4 Resolution (2 Marks)**:
   Both active ($\mu_1 > 0, \mu_2 > 0$): $x_2 = 3 - x_1 = x_1^2 + 1 \implies x_1^2 + x_1 - 2 = 0 \implies x_1 \in \{-2, 1\}$.
   - If $x_1 = -2, x_2 = 5 \implies \mu_1 = -4(1 + \mu_2) < 0$ (Dual violation).
   - If $x_1 = 1, x_2 = 2 \implies \mu_1 - 2\mu_2 = 2$ and $\mu_1 + \mu_2 = 8 \implies \mathbf{\mu_2^* = 2 > 0, \mu_1^* = 6 > 0}$.
   Optimal solution: $\mathbf{x^* = (1, 2)}$, $\mathbf{\mu^* = (6, 2)}$, minimum value $= 9$.

---

### Q42. Full Derivation of SVM Wolfe Dual from Primal Lagrangian
**Question (6 Marks)**: Starting from the soft-margin SVM primal problem, write the Lagrangian and derive the Wolfe Dual optimization problem step-by-step.

**Model Answer**:
1. **Primal & Lagrangian (2 Marks)**:
   $$\min_{w, b, \xi} \frac{1}{2} w^\top w + C \sum \xi_i \quad \text{s.t.} \quad y_i(w^\top x_i + b) - 1 + \xi_i \ge 0, \; \xi_i \ge 0$$
   $$L(w, b, \xi, \alpha, \mu) = \frac{1}{2} w^\top w + C \sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i [y_i(w^\top x_i + b) - 1 + \xi_i] - \sum_{i=1}^N \mu_i \xi_i$$
2. **Stationarity Elimination (2 Marks)**:
   - $\nabla_w L = w - \sum \alpha_i y_i x_i = \mathbf{0} \implies w = \sum_{i=1}^N \alpha_i y_i x_i$
   - $\frac{\partial L}{\partial b} = -\sum \alpha_i y_i = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$
   - $\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \implies \alpha_i + \mu_i = C \implies 0 \le \alpha_i \le C$ (since $\mu_i \ge 0$)
3. **Dual Assembly (2 Marks)**:
   Substitute $w = \sum \alpha_i y_i x_i$ into $L$:
   $$\frac{1}{2} w^\top w - w^\top w = -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^\top x_j$$
   $$\sum \xi_i (C - \alpha_i - \mu_i) = 0, \qquad \sum \alpha_i (1) = \sum \alpha_i$$
   $$\max_\alpha \quad \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^\top x_j \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \; \sum_{i=1}^N \alpha_i y_i = 0$$

---

### Q43. Exact Line Search on Quadratic Objective
**Question (6 Marks)**: For $f(x) = \frac{1}{2} x^\top A x - b^\top x$ where $A \succ 0$, derive the closed-form exact line search step size $\alpha_k^* = \arg\min_{\alpha > 0} f(x^{(k)} - \alpha g_k)$ where $g_k = \nabla f(x^{(k)})$.

**Model Answer**:
1. **Line Search Function Setup (2 Marks)**:
   Let $x(\alpha) = x - \alpha g$ where $g = \nabla f(x) = A x - b$.
   $$\phi(\alpha) = f(x - \alpha g) = \frac{1}{2} (x - \alpha g)^\top A (x - \alpha g) - b^\top (x - \alpha g)$$
2. **Expansion and Differentiation (2 Marks)**:
   $$\phi(\alpha) = \frac{1}{2} x^\top A x - \alpha g^\top A x + \frac{1}{2} \alpha^2 g^\top A g - b^\top x + \alpha b^\top g$$
   $$= f(x) - \alpha g^\top (A x - b) + \frac{1}{2} \alpha^2 g^\top A g = f(x) - \alpha g^\top g + \frac{1}{2} \alpha^2 g^\top A g$$
   Differentiate with respect to $\alpha$ and set to zero:
   $$\phi'(\alpha) = -g^\top g + \alpha g^\top A g = 0$$
3. **Closed-Form Step Size (2 Marks)**:
   $$\mathbf{\alpha^* = \frac{g^\top g}{g^\top A g} = \frac{\|g\|_2^2}{g^\top A g}}$$
   Since $A \succ 0$, $g^\top A g > 0$ for all $g \ne \mathbf{0}$, ensuring $\alpha^* > 0$.

---

### Q44. Eckart-Young-Mirsky Theorem Proof Outline
**Question (6 Marks)**: Prove that for any matrix $B$ with $\text{rank}(B) \le k$, $\|A - B\|_2 \ge \sigma_{k+1}(A)$.

**Model Answer**:
1. **Subspace Dimensions (2 Marks)**:
   Let $A = U \Sigma V^\top \in \mathbb{R}^{m \times n}$ with $\sigma_1 \ge \dots \ge \sigma_r > 0$.
   Let $V_{k+1} = \text{span}\{v_1, \dots, v_{k+1}\} \subset \mathbb{R}^n \implies \dim(V_{k+1}) = k+1$.
   For any matrix $B$ with $\text{rank}(B) \le k$, by Rank-Nullity: $\dim(\mathcal{N}(B)) = n - \text{rank}(B) \ge n - k$.
2. **Non-Trivial Intersection (2 Marks)**:
   $$\dim(V_{k+1} \cap \mathcal{N}(B)) = \dim(V_{k+1}) + \dim(\mathcal{N}(B)) - \dim(V_{k+1} + \mathcal{N}(B)) \ge (k+1) + (n-k) - n = 1$$
   Thus, there exists a unit vector $z \in V_{k+1} \cap \mathcal{N}(B)$ with $\|z\|_2 = 1$ and $B z = \mathbf{0}$.
3. **Bounding the Norm (2 Marks)**:
   Since $z \in V_{k+1}$, $z = \sum_{i=1}^{k+1} c_i v_i$ with $\sum c_i^2 = 1$.
   $$\|A - B\|_2^2 \ge \|(A - B)z\|_2^2 = \|A z - \mathbf{0}\|_2^2 = \|A z\|_2^2 = \sum_{i=1}^{k+1} c_i^2 \sigma_i^2 \ge \sigma_{k+1}^2 \sum_{i=1}^{k+1} c_i^2 = \sigma_{k+1}^2$$
   Taking square roots: $\|A - B\|_2 \ge \mathbf{\sigma_{k+1}}$. Since $\|A - A_k\|_2 = \sigma_{k+1}$, $A_k$ is the exact minimizer!

---

### Q45. Derivation of PCA Maximum Variance via Rayleigh Quotient
**Question (6 Marks)**: Formulate PCA as the maximization of the Rayleigh quotient $R_S(u) = \frac{u^\top S u}{u^\top u}$. Prove that the stationary points of $R_S(u)$ are eigenvectors of $S$.

**Model Answer**:
1. **Rayleigh Quotient Formulation (2 Marks)**:
   The projected variance along unnormalized direction $u \ne \mathbf{0}$ is $R_S(u) = \frac{u^\top S u}{u^\top u}$.
2. **Derivative via Quotient Rule (2 Marks)**:
   $$\nabla_u R_S(u) = \frac{\nabla(u^\top S u) (u^\top u) - (u^\top S u) \nabla(u^\top u)}{(u^\top u)^2} = \frac{2 S u (u^\top u) - (u^\top S u) (2u)}{(u^\top u)^2}$$
3. **Stationarity & Eigenvalues (2 Marks)**:
   Set $\nabla_u R_S(u) = \mathbf{0}$:
   $$\frac{2}{u^\top u} \left[ S u - \frac{u^\top S u}{u^\top u} u \right] = \mathbf{0} \implies S u = R_S(u) u$$
   Let $\lambda = R_S(u)$. Then $S u = \lambda u$, which is the exact **eigenvalue equation**.
   The maximum variance is attained at the principal eigenvector $u_1$ with maximal eigenvalue $\lambda_1 = \lambda_{\max}(S)$.

---

### Q46. SVD Sensitivity & Condition Number of a Linear System
**Question (6 Marks)**: For a square invertible linear system $A x = b$, let $A = U \Sigma V^\top$. If $b$ is perturbed by $\delta b$, prove that relative error satisfies $\frac{\|\delta x\|_2}{\|x\|_2} \le \frac{\sigma_1}{\sigma_n} \frac{\|\delta b\|_2}{\|b\|_2}$.

**Model Answer**:
1. **Norm Bounds (2 Marks)**:
   $A x = b \implies \|b\| = \|A x\| \le \|A\|_2 \|x\| = \sigma_1 \|x\| \implies \frac{1}{\|x\|} \le \frac{\sigma_1}{\|b\|}$.
2. **Perturbation Bounds (2 Marks)**:
   $A(x + \delta x) = b + \delta b \implies A \delta x = \delta b \implies \delta x = A^{-1} \delta b$.
   $$\|\delta x\| = \|A^{-1} \delta b\| \le \|A^{-1}\|_2 \|\delta b\| = \frac{1}{\sigma_n} \|\delta b\|$$
3. **Combining Bounds (2 Marks)**:
   Multiply the two inequalities:
   $$\frac{\|\delta x\|}{\|x\|} \le \left(\frac{1}{\sigma_n} \|\delta b\|\right) \left(\frac{\sigma_1}{\|b\|}\right) = \left(\frac{\sigma_1}{\sigma_n}\right) \frac{\|\delta b\|}{\|b\|} = \mathbf{\kappa(A) \frac{\|\delta b\|_2}{\|b\|_2}}$$
   where $\kappa(A) = \frac{\sigma_1}{\sigma_n}$ is the matrix condition number.

---

### Q47. Backpropagation through a Softmax Layer
**Question (6 Marks)**: For Softmax $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ and Cross-Entropy loss $L = -\sum_k y_k \log p_k$ with one-hot target $y$, prove that $\frac{\partial L}{\partial z_i} = p_i - y_i$.

**Model Answer**:
1. **Jacobian of Softmax (2 Marks)**:
   - For $i = j$: $\frac{\partial p_i}{\partial z_i} = p_i (1 - p_i)$
   - For $i \ne j$: $\frac{\partial p_j}{\partial z_i} = -p_j p_i$
2. **Chain Rule with Cross-Entropy (2 Marks)**:
   $$\frac{\partial L}{\partial z_i} = \sum_{j} \frac{\partial L}{\partial p_j} \frac{\partial p_j}{\partial z_i} = \sum_j \left( -\frac{y_j}{p_j} \right) \frac{\partial p_j}{\partial z_i}$$
3. **Algebraic Cancellation (2 Marks)**:
   Separate into $j = i$ and $j \ne i$:
   $$\frac{\partial L}{\partial z_i} = -\frac{y_i}{p_i} [p_i (1 - p_i)] + \sum_{j \ne i} \left( -\frac{y_j}{p_j} \right) (-p_j p_i) = -y_i(1 - p_i) + \sum_{j \ne i} y_j p_i$$
   $$= -y_i + y_i p_i + p_i \sum_{j \ne i} y_j = -y_i + p_i \sum_{\text{all } j} y_j$$
   Since $\sum_j y_j = 1$ (one-hot probability distribution):
   $$\mathbf{\frac{\partial L}{\partial z_i} = p_i - y_i}$$

---

### Q48. Nesterov Accelerated Gradient Look-Ahead Mechanism
**Question (6 Marks)**: Contrast standard Polyak Momentum with Nesterov Accelerated Gradient (NAG). Mathematically formulate the look-ahead gradient step and explain how it prevents overshoot.

**Model Answer**:
1. **Formulation Comparison (2 Marks)**:
   - **Polyak Momentum**:
     $$v_{t+1} = \beta v_t + \eta \nabla f(w_t), \quad w_{t+1} = w_t - v_{t+1}$$
   - **Nesterov Accelerated Gradient (NAG)**:
     $$v_{t+1} = \beta v_t + \eta \nabla f(w_t - \beta v_t), \quad w_{t+1} = w_t - v_{t+1}$$
2. **Look-Ahead Mechanism (2 Marks)**:
   Instead of computing the gradient at current position $w_t$, NAG computes the gradient at the **predicted future position** $w_t - \beta v_t$ (the point where the existing momentum is already about to carry the parameter).
3. **Overshoot Prevention Insight (2 Marks)**:
   If the momentum vector is about to carry the parameter up an opposing hill, the look-ahead gradient $\nabla f(w_t - \beta v_t)$ detects the oncoming incline in advance and applies an opposing corrective brake *before* the overshoot occurs.

---

### Q49. Constrained Least Squares via Lagrange Multipliers
**Question (6 Marks)**: Derive the solution to the equality-constrained least squares problem $\min_x \|A x - b\|_2^2$ subject to $C x = d$, assuming $A^\top A \succ 0$ and $C$ has full row rank.

**Model Answer**:
1. **Lagrangian Formulation (2 Marks)**:
   $$L(x, \lambda) = (A x - b)^\top (A x - b) + \lambda^\top (C x - d)$$
2. **Stationarity Conditions (2 Marks)**:
   - $\nabla_x L = 2 A^\top(A x - b) + C^\top \lambda = \mathbf{0} \implies 2 A^\top A x + C^\top \lambda = 2 A^\top b$
   - $\nabla_\lambda L = C x - d = \mathbf{0} \implies C x = d$
3. **Block Matrix System (KKT Matrix) (2 Marks)**:
   Assemble into the block linear system:
   $$\begin{bmatrix} 2 A^\top A & C^\top \\ C & 0 \end{bmatrix} \begin{bmatrix} x \\ \lambda \end{bmatrix} = \begin{bmatrix} 2 A^\top b \\ d \end{bmatrix}$$
   Solving for $x$: $x = (A^\top A)^{-1} [A^\top b - \frac{1}{2} C^\top \lambda]$. Premultiply by $C$ and use $C x = d$ to determine $\lambda$, then back-substitute to find the exact constrained minimum $x^*$.

---

### Q50. Non-Linear Separation and Polynomial Kernel Mapping
**Question (6 Marks)**: Consider data in $\mathbb{R}^1$: $x_1 = -1 [y_1 = -1], x_2 = 0 [y_2 = +1], x_3 = 1 [y_3 = -1]$. Show that this data is not linearly separable in $\mathbb{R}^1$. Apply the polynomial feature map $\phi(x) = [x, x^2]^\top$ to show that the data becomes linearly separable in $\mathbb{R}^2$, and write the separating hyperplane.

**Model Answer**:
1. **Non-Separability in 1D (2 Marks)**:
   Class $+1$ is at $x = 0$, while class $-1$ is at $x = -1$ and $x = +1$.
   A linear decision boundary in 1D is a single point $w x + b = 0 \implies x = -b/w$. A single threshold cannot separate a middle point ($0$) from points on both its left ($-1$) and right ($+1$). Hence, linearly non-separable.
2. **2D Feature Space Mapping (2 Marks)**:
   Apply $\phi(x) = [x, x^2]^\top$:
   - $\phi(x_1) = \phi(-1) = [-1, 1]^\top, \quad y_1 = -1$
   - $\phi(x_2) = \phi(0) = [0, 0]^\top, \quad y_2 = +1$
   - $\phi(x_3) = \phi(1) = [1, 1]^\top, \quad y_3 = -1$
3. **Separating Hyperplane (2 Marks)**:
   In $\mathbb{R}^2$, class $-1$ points have $z_2 = 1$, while class $+1$ point has $z_2 = 0$.
   A horizontal line $z_2 = 0.5$ separates them perfectly!
   Hyperplane equation: $w^\top z + b = 0$ with $w = [0, -1]^\top, b = 0.5$:
   - For $\phi(x_2) = (0, 0)$: $0(0) - 1(0) + 0.5 = 0.5 > 0$ (Class $+1$).
   - For $\phi(x_1) = (-1, 1)$: $0(-1) - 1(1) + 0.5 = -0.5 < 0$ (Class $-1$).
   - For $\phi(x_3) = (1, 1)$: $0(1) - 1(1) + 0.5 = -0.5 < 0$ (Class $-1$).
   In the original 1D input space, this corresponds to the non-linear decision rule $0.5 - x^2 = 0 \implies x^2 = 0.5$.
