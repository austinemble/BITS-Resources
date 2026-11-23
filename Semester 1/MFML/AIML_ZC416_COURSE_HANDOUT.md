# BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI
## WORK INTEGRATED LEARNING PROGRAMMES
### Course Handout: AIML ZC416 / AIML ZG514 — Mathematical Foundations for Machine Learning
**Academic Year**: 2025–2026 | **Semester**: I / II

---

## Part A: Content Design & Course Structure

| Attribute | Details |
| :--- | :--- |
| **Course Title** | Mathematical Foundations for Machine Learning |
| **Course Code** | AIML ZC416 / AIML ZG514 |
| **Credit Units** | 4 |
| **Credit Model** | 3 – 1 – 0 |
| **Lead Instructor** | Prof. Mandar Vijay Datar |
| **Instructors** | Prof. Saurabh, Prof. Upendar Mendu, Prof. Michael Alphonse, Prof. Deepak G. Skariah, Prof. G. Venkiteswaran, Prof. Madhusudhanan B., Prof. Navneet Goyal, Prof. Nedungadi Aatira Gopalkrishnan |
| **Target Audience** | M.Tech in Artificial Intelligence & Machine Learning / Data Science |

---

### Course Description
This course provides a rigorous, self-contained mathematical grounding in linear algebra, vector calculus, analytic geometry, and optimization theory specifically tailored for modern machine learning and artificial intelligence. Starting from systems of linear equations and Gaussian elimination, the curriculum progresses through vector spaces, inner product spaces, matrix decompositions (Eigen-decomposition, Spectral Theorem, and Singular Value Decomposition), matrix calculus (Jacobians, Hessians, and automatic differentiation), multivariate Taylor expansions, unconstrained optimization algorithms (Gradient Descent, Line Search, Momentum, RMSProp, Adam), constrained optimization (Lagrange Multipliers, Duality, and Karush-Kuhn-Tucker conditions), and culminates in core machine learning foundations including Principal Component Analysis (PCA), low-rank matrix approximations, and Support Vector Machines (SVM).

---

### Course Objectives (COs)
- **CO1**: Formulate, analyze, and solve systems of linear equations using Gaussian elimination, echelon forms, rank theorems, and fundamental matrix subspaces.
- **CO2**: Master vector spaces, linear transformations, inner products, induced norms, orthogonal projections, and spectral matrix decompositions (Eigenvalues and SVD).
- **CO3**: Derive vector and matrix derivatives using multivariable calculus, computational graphs, and reverse-mode automatic differentiation.
- **CO4**: Analyze unconstrained and constrained optimization problems, apply first- and second-order optimization algorithms, and solve constrained problems via Lagrange multipliers and KKT conditions.
- **CO5**: Connect foundational mathematical principles directly to core machine learning algorithms including PCA dimensionality reduction and Support Vector Machine classification.

---

### Prescribed Textbooks & Reference Books

#### Prescribed Textbooks (T1)
- **T1**: Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, *Mathematics for Machine Learning*, Cambridge University Press, 2020.

#### Reference Books (R1, R2, R3)
- **R1**: Stephen Boyd and Lieven Vandenberghe, *Introduction to Applied Linear Algebra – Vectors, Matrices, and Least Squares*, Cambridge University Press, 2018.
- **R2**: Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.
- **R3**: Gilbert Strang, *Linear Algebra and Its Applications*, 4th/5th Edition, Cengage Learning, 2016.

---

## Part B: Contact Session & Modular Schedule

| Contact Session | Topic Title | Modular Scope & Key Concepts | Textbook Chapter (T1) |
| :---: | :--- | :--- | :---: |
| **CS 01** | Systems of Linear Equations | Linear systems $Ax = b$, coefficient & augmented matrices $[A \mid b]$, elementary row operations, Row Echelon Form (REF), Reduced Row Echelon Form (RREF), pivot variables vs free variables, unique/infinite/no solution criteria. | T1 Ch 2.1–2.3 |
| **CS 02** | Vector Spaces & Subspaces | Groups, vector space axioms, closure, vector subspaces, span, linear combinations, null spaces $\mathcal{N}(A)$, column spaces $\mathcal{C}(A)$. | T1 Ch 2.4 |
| **CS 03** | Linear Independence, Basis & Dimension | Testing linear independence, minimal spanning sets, basis definitions, dimension theorem, rank-nullity theorem $\text{rank}(A) + \text{nullity}(A) = n$. | T1 Ch 2.5–2.6 |
| **CS 04** | Linear Mappings & Matrices | Linear mappings between vector spaces, transformation matrices, change of basis, kernel and image, invertible transformations. | T1 Ch 2.7–2.8 |
| **CS 05** | Analytic Geometry & Inner Products | Dot products, general bilinear forms, symmetric positive definite (SPD) matrices $x^\top A y$, induced norms $\|x\|_A$, distances, angles, Cauchy-Schwarz and triangle inequalities. | T1 Ch 3.1–3.4 |
| **CS 06** | Orthogonality & Projections | Orthogonal vectors, orthogonal complements, orthonormal bases, Gram-Schmidt orthogonalization process, orthogonal projection onto subspaces. | T1 Ch 3.5–3.8 |
| **CS 07** | Eigenvalues & Eigenvectors | Determinants, characteristic equation $\det(A - \lambda I) = 0$, algebraic vs geometric multiplicity, eigenspaces, diagonalization $A = P D P^{-1}$, Spectral Theorem for symmetric matrices. | T1 Ch 4.1–4.4 |
| **CS 08** | Singular Value Decomposition (SVD) | Singular values $\sigma_i = \sqrt{\lambda_i(A^\top A)}$, full SVD $A = U \Sigma V^\top$, geometric transformation (rotation-scaling-rotation), left/right singular vectors, pseudo-inverse $A^+$. | T1 Ch 4.5 |
| **CS 09** | Vector & Matrix Calculus | Partial derivatives, gradients of scalar functions $\nabla f$, Jacobian matrices $J_f(x)$, Hessian matrices $H(x)$, chain rule for multivariable compositions. | T1 Ch 5.1–5.4 |
| **CS 10** | Automatic Differentiation & Backprop | Computational graphs, forward mode vs reverse mode AD, error backpropagation through matrices, matrix derivative identities. | T1 Ch 5.5–5.6 |
| **CS 11** | Multivariate Taylor Series | Linear approximations, quadratic approximations, second-order Taylor expansion $f(x) \approx f(x_0) + \nabla f(x_0)^\top(x - x_0) + \frac{1}{2}(x - x_0)^\top H(x_0)(x - x_0)$, stationary points & classification. | T1 Ch 5.7–5.8 |
| **CS 12** | Unconstrained Optimization | First-order necessary conditions, gradient descent algorithm, step-size selection, exact line search, backtracking & binary search, convergence behaviors. | T1 Ch 7.1–7.2 |
| **CS 13** | Advanced First-Order Optimizers | Ill-conditioned curvature & ravines, Momentum gradient descent, Nesterov accelerated gradient, adaptive learning rates: AdaGrad, RMSProp, Adam algorithm. | T1 Ch 7.2 + Notes |
| **CS 14** | Constrained Optimization & KKT | Equality constraints & Lagrange multipliers, inequality constraints, primal problem, Lagrangian function, Karush-Kuhn-Tucker (KKT) necessary and sufficient conditions, complementary slackness. | T1 Ch 7.3 |
| **CS 15** | Principal Component Analysis (PCA) | Dimensionality reduction intuition, sample mean and sample covariance matrix $S$, maximal variance formulation, minimal reconstruction error formulation, projecting data onto principal eigenvectors. | T1 Ch 10.1–10.4 |
| **CS 16** | Low-Rank Approximations & SVM | Eckart-Young-Mirsky low-rank theorem via truncated SVD, Support Vector Machines (SVM), optimal margin hyperplane, soft margin with slack variables $\xi_i$, dual formulation, kernel trick. | T1 Ch 4.5 & Ch 12 |

---

## Part C: Evaluation Scheme

| Component | Nature | Weightage | Duration | Scope |
| :--- | :---: | :---: | :---: | :--- |
| **Quiz 1 / Online Assignment 1** | Online / Objective | 10% | 1 Hour | Contact Sessions 1–4 (Linear Systems & Vector Spaces) |
| **Mid-Semester Examination (EC-2)** | **Closed Book** | 30% | 2 Hours | Contact Sessions 1–8 (Linear Algebra, Orthogonality, Eigenvalues, SVD, Calculus) |
| **Quiz 2 / Online Assignment 2** | Online / Numerical | 10% | 1 Hour | Contact Sessions 9–12 (Optimization & Multivariate Calculus) |
| **Assignment 3 (Experiential/Coding)** | Python / Jupyter | 10% | 2 Weeks | Matrix Implementations, Gradient Descent with Adam, PCA, and SVM from scratch |
| **Comprehensive Examination (EC-3)** | **Open Book** | 40% | 2.5 Hours | Comprehensive (Contact Sessions 1–16) |
