# Module 6: Gaussian Mixture Models and Expectation–Maximization

**Course**: AIML ZC418 — Introduction to Statistical Methods  
**Prescribed References**: T1 Kaptein et al. (mixture models, latent variables); T2 Devore (multivariate normal, likelihood); T3 (optional: regime-switching as mixture-like structure in series)  
**Syllabus module**: **M06** — Gaussian Mixture Models and Expectation–Maximization (see `Notes/AI Notes/module_map.yaml`)  
**Canonical video**: `Inbox/Prof. G.Krishna Kumari/Recordings/Stats_G.Krishna Kumari_8.mp4`  
**Contact sessions** (reference only): Contact Session 15 (`ISM_CS-15` GMM/EM).

> **Exam note (Lead Instructor)**: **Closed book** — GMM density form, responsibilities \(\gamma_{ik}\), EM E-step/M-step updates for univariate two-component GMM, link to soft clustering. **Open book** — interpret mixture weights, covariance shapes, convergence, and connection to \(k\)-means / hierarchical clustering.

---

## 1. Executive Summary & Foundational Definitions

A **Gaussian Mixture Model (GMM)** represents the data density as a **convex combination** of Gaussian components:
$$p(\mathbf{x})=\sum_{k=1}^K \pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k), \quad \sum_{k=1}^K\pi_k=1,\;\pi_k\ge 0$$

Each component is a **latent cluster**; an observation belongs to component \(k\) with probability \(\pi_k\) (soft assignment). **Expectation–Maximization (EM)** iteratively fits \(\{\pi_k,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k\}\) by alternating **E-step** (expected complete-data log-likelihood via responsibilities) and **M-step** (closed-form parameter updates for Gaussians).

GMMs generalize \(k\)-means to **probabilistic**, elliptical clusters and underpin soft clustering, semi-supervised priors, and density estimation in anomaly detection.

```
Data x_i ──> latent Z_i ∈ {1,…,K} ──> component k with prob π_k
                │                              │
                └── EM: q(Z_i=k) = γ_{ik} ──> update π_k, μ_k, Σ_k
```

---

## 2. Formal Model (Mixture of Gaussians)

### 2.1 Generative story

1. Choose component \(Z\in\{1,\ldots,K\}\) with \(P(Z=k)=\pi_k\).
2. Sample \(\mathbf{X}\mid Z=k \sim \mathcal{N}(\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\).

**Marginal density**:
$$p(\mathbf{x})=\sum_{k=1}^K \pi_k \,\frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_k|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)\right)$$

### 2.2 Complete-data log-likelihood

If \(Z_i\) were observed, with one-hot \(z_{ik}\):
$$\ell_c(\theta)=\sum_{i=1}^n\sum_{k=1}^K z_{ik}\big[\log\pi_k+\log\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\big]$$

**Observed-data** likelihood \(\prod_i p(\mathbf{x}_i)\) has **no** simple closed-form MLE — EM necessary.

### 2.3 Responsibilities (soft memberships)

$$\gamma_{ik}=P(Z_i=k\mid\mathbf{x}_i,\theta^{(t)})=\frac{\pi_k^{(t)}\,\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k^{(t)},\boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)}\,\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_j^{(t)},\boldsymbol{\Sigma}_j^{(t)})}$$

---

## 3. EM Algorithm for GMM

**Initialize** \(\pi_k^{(0)},\boldsymbol{\mu}_k^{(0)},\boldsymbol{\Sigma}_k^{(0)}\) (e.g. \(k\)-means centroids).

Repeat until log-likelihood stabilizes:

### E-step

Compute \(\gamma_{ik}\) for all \(i,k\) using current parameters.

### M-step

$$N_k=\sum_{i=1}^n\gamma_{ik}$$

$$\pi_k^{\text{new}}=\frac{N_k}{n}$$

$$\boldsymbol{\mu}_k^{\text{new}}=\frac{1}{N_k}\sum_{i=1}^n\gamma_{ik}\mathbf{x}_i$$

$$\boldsymbol{\Sigma}_k^{\text{new}}=\frac{1}{N_k}\sum_{i=1}^n\gamma_{ik}(\mathbf{x}_i-\boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i-\boldsymbol{\mu}_k^{\text{new}})^T$$

(Univariate scalar \(x_i\): \(\sigma_k^2\) is weighted average of squared deviations.)

**Monotonicity**: each EM cycle does not decrease observed-data log-likelihood; converges to a **local** optimum (initialization matters).

### 3.1 Univariate two-component specialization

\(K=2\); parameters \(\pi,\mu_1,\mu_2,\sigma_1^2,\sigma_2^2\). E-step uses Gaussian PDFs in 1D; M-step as above with scalar means/variances.

---

## 4. Connection to Clustering

| Method | Assignment | Shape | Uncertainty |
| :--- | :--- | :--- | :--- |
| **\(k\)-means** | Hard (nearest centroid) | Spherical (Euclidean) | None |
| **GMM + EM** | Soft \(\gamma_{ik}\) | Elliptical (\(\boldsymbol{\Sigma}_k\)) | Yes |
| **Hierarchical** | Tree merge/split | Geometry depends on linkage | Partial |

**EM as soft \(k\)-means**: if \(\boldsymbol{\Sigma}_k=\sigma^2\mathbf{I}\), \(\sigma^2\to 0\), responsibilities harden toward nearest mean (qualitative limit).

**Choice of \(K\)**: BIC/ICL penalize too many components; elbow in log-likelihood (open-book interpretation).

---

## 5. Identifiability and Practical Issues

1. **Label switching**: permuting component labels gives same mixture — resolve by ordering \(\mu_k\) or fixing \(\pi_1\le\pi_2\).
2. **Singular covariances**: a component collapsing on one point → regularize \(\boldsymbol{\Sigma}_k+\epsilon\mathbf{I}\) in software (concept for exams).
3. **Local maxima**: multiple random restarts (standard in `sklearn.mixture.GaussianMixture`).

---

## 6. AI / ML Bridge (Lead ISM Lectures)

| Idea | AIML application |
| :--- | :--- |
| GMM density | Semi-supervised learning, generative classifiers, **anomaly** scores \(-\log p(\mathbf{x})\) |
| Soft clustering | Customer segmentation, speaker diarization priors |
| EM pattern | Same skeleton as **Baum–Welch** (HMM), **variational EM** in latent-variable models |
| Mixture of experts | MoE networks route inputs to specialists — conceptual parallel to \(\gamma_{ik}\) |

**Unsupervised pipeline**: exploratory GMM on standardized features → validate clusters with silhouette / domain labels — complements Lab work on clustering without assuming spherical groups.

---

## 7. Solved Numerical Example (Exam Pattern)

**Problem Statement (BITS Pattern, ~6 Marks)**: Univariate GMM, \(K=2\), one observation \(x_1=2\). Current parameters: \(\pi_1=0.6\), \(\pi_2=0.4\); \(\mu_1=0\), \(\mu_2=3\); \(\sigma_1=\sigma_2=1\). Standard normal densities at \(x=2\): \(\phi(2\mid\mu_1)=0.054\), \(\phi(2\mid\mu_2)=0.242\) (given). Compute \(\gamma_{11},\gamma_{12}\) and the **one-observation** M-step updates for \(\pi_k\) and \(\mu_k\) if this is the only point in the dataset (illustrative EM round).

### Solution

**E-step** (unnormalized):
$$u_1=\pi_1\phi(2\mid\mu_1)=0.6(0.054)=0.0324, \quad u_2=0.4(0.242)=0.0968$$

$$N=u_1+u_2=0.1292$$

$$\gamma_{11}=\frac{0.0324}{0.1292}\approx \mathbf{0.251}, \quad \gamma_{12}=\frac{0.0968}{0.1292}\approx \mathbf{0.749}$$

Observation 2 is **closer** to component 2 (\(\mu_2=3\)) — higher responsibility on cluster 2.

**M-step** (\(n=1\)): \(N_1=\gamma_{11}\), \(N_2=\gamma_{12}\).

$$\pi_1^{\text{new}}=0.251, \quad \pi_2^{\text{new}}=0.749$$

$$\mu_1^{\text{new}}=\frac{\gamma_{11}\,x_1}{N_1}=2 \quad (\text{only point}), \quad \mu_2^{\text{new}}=2$$

With one point, means equal \(x_1\); with many points, \(\mu_k\) is weighted average — exam intent is **E-step arithmetic** and M-step structure.

---

## 8. Formula Sheet (Closed-Book Priority)

| Item | Formula |
| :--- | :--- |
| Mixture density | \(p(\mathbf{x})=\sum_k \pi_k\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\) |
| Responsibility | \(\gamma_{ik}\propto \pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\), normalize |
| M-step \(\pi_k\) | \(N_k/n\), \(N_k=\sum_i\gamma_{ik}\) |
| M-step \(\boldsymbol{\mu}_k\) | \(\sum_i\gamma_{ik}\mathbf{x}_i/N_k\) |
| EM guarantee | Log-likelihood non-decreasing each iteration |

---

## 9. Common Pitfalls & Practical Guidelines

1. **Feature scaling**: GMM with full \(\boldsymbol{\Sigma}_k\) is scale-sensitive — standardize before fitting.
2. **Too many \(K\)**: overfits; use BIC / held-out log-likelihood.
3. **Confusing with \(k\)-means**: hard vs soft assignment — state which algorithm exam question assumes.
4. **EM stop rule**: max iterations or \(\Delta\ell<\eta\) — report that EM finds local, not global, optimum.

**PPT cross-reference**: **ISM_CS-15** — GMM definition, EM flowchart, clustering comparison (Prof. G. Krishna Kumari lead session).

---

## 10. EM Algorithm Summary (Exam Flowchart)

```
Initialize θ⁽⁰⁾
loop t = 0,1,2,…
  E-step: γᵢₖ from θ⁽ᵗ⁾
  M-step: πₖ, μₖ, Σₖ from weighted sufficient statistics
  until ℓ(θ⁽ᵗ⁺¹⁾) − ℓ(θ⁽ᵗ⁾) small
output θ, assign argmax_k γᵢₖ for hard labels
```