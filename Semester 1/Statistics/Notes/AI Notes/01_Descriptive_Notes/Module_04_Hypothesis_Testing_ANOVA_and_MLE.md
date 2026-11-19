# Module 4: Hypothesis Testing, ANOVA, and Maximum Likelihood

**Course**: AIML ZC418 — Introduction to Statistical Methods  
**Prescribed References**: T1 Kaptein et al.; T2 Devore (Chapters on sampling, CI, tests, ANOVA); T3 (secondary for forecast residuals used in model checks)  
**Primary Teacher**: Prof. G. Krishna Kumari (Lead ISM)  
**Lecture alignment**: ISM Contact Sessions **CS-7, CS-9, CS-10** (`ISM_CS-7` sampling & CLT; `ISM_CS-9` mean/proportion tests; `ISM_CS-10` MLE & one/two-way ANOVA)

> **Exam note (Lead Instructor)**: **Closed-book mid** — test statistics, critical-value logic, Type I/II error definitions, CLT/CI formulas, ANOVA table template. **Open-book comprehensive** — state \(H_0,H_1\), assumptions, **practical significance**, and interpret \(p\)-values / confidence intervals in context.

---

## 1. Executive Summary & Foundational Definitions

**Statistical inference** draws population conclusions from samples. **Sampling distributions** describe statistics \(\bar{X}, S^2, \hat{p}\) as random variables. The **Central Limit Theorem (CLT)** justifies normal approximations for \(\bar{X}\). **Confidence intervals (CIs)** quantify uncertainty; **hypothesis tests** formalize decisions using **\(p\)-values** and error rates.

**ANOVA** partitions total variance to compare several group means with one \(F\)-test (extensions: two factors). **Maximum likelihood estimation (MLE)** chooses parameters maximizing the data likelihood — the backbone of logistic regression, GLMs, and many neural loss functions.

```
Sample D ──> Statistic T(D) ──> Sampling dist. (often ≈ Normal/t/χ²/F)
                │
                ├── CI: plausible θ via pivot
                └── Test: H₀ vs H₁; reject if T in critical region (or p < α)
```

---

## 2. Sampling, CLT, and Estimation

### 2.1 Sampling methods

- **Simple random sample (SRS)**: each population unit equally likely in sample.
- **Stratified sampling**: partition population into strata; sample within each → lower variance when strata are homogeneous (CS-7 emphasis).

### 2.2 Sampling distribution of the mean

For i.i.d. observations with \(E[X_i]=\mu\), \(\mathrm{Var}(X_i)=\sigma^2\):
$$E[\bar{X}]=\mu, \qquad \mathrm{Var}(\bar{X})=\frac{\sigma^2}{n}, \qquad SE(\bar{X})=\frac{\sigma}{\sqrt{n}}$$

If population is \(\mathcal{N}(\mu,\sigma^2)\), \(\bar{X}\) is **exactly** normal for any \(n\).

### 2.3 Central Limit Theorem

For large \(n\), regardless of population shape (mild conditions: finite \(\sigma^2\)):
$$\frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \approx \mathcal{N}(0,1)$$

If \(\sigma\) unknown, replace \(\sigma\) by \(S\): \(\dfrac{\bar{X}-\mu}{S/\sqrt{n}}\sim t_{n-1}\) **exactly** when data are normal.

### 2.4 Confidence intervals for \(\mu\)

| Case | \((1-\alpha)\) CI for \(\mu\) |
| :--- | :--- |
| \(\sigma\) known | \(\bar{x} \pm z_{\alpha/2}\,\dfrac{\sigma}{\sqrt{n}}\) |
| \(\sigma\) unknown, normal data | \(\bar{x} \pm t_{\alpha/2,\,n-1}\,\dfrac{s}{\sqrt{n}}\) |

**Interpretation (comprehensive exam)**: If we repeated sampling, \((1-\alpha)\) of constructed intervals contain the true \(\mu\); **not** “probability \(\mu\) lies in this interval.”

---

## 3. Hypothesis Testing Framework

### 3.1 Definitions

- **Null hypothesis** \(H_0\): status quo / no effect.
- **Alternative** \(H_1\): effect of interest (one- or two-sided).
- **Significance level** \(\alpha\): \(\Pr(\text{reject }H_0 \mid H_0\text{ true})\) — **Type I error**.
- **Type II error** \(\beta\): \(\Pr(\text{fail to reject }H_0 \mid H_1\text{ true})\); **Power** \(=1-\beta\).

**Decision rule**: reject \(H_0\) if \(p\text{-value}<\alpha\) or test statistic falls in rejection region.

**\(p\)-value**: probability, under \(H_0\), of observing data at least as extreme as the sample.

### 3.2 One-sample \(z\)-test (known \(\sigma\))

\(H_0:\mu=\mu_0\) vs \(H_1:\mu\neq\mu_0\):
$$z=\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}$$

Compare \(|z|\) to \(z_{\alpha/2}\) or use two-tailed normal \(p\)-value.

### 3.3 One-sample \(t\)-test (unknown \(\sigma\))

$$t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}} \sim t_{n-1} \quad \text{under }H_0 \text{ (normal population)}$$

### 3.4 Two-sample mean tests

**Pooled \(t\)** (equal variances, independent samples):
$$t=\frac{(\bar{x}_1-\bar{x}_2)-(\mu_1-\mu_2)_0}{s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}, \quad s_p^2=\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}$$

Degrees of freedom: \(n_1+n_2-2\). Welch’s \(t\) when variances unequal (interpretation questions often stress checking assumptions).

### 3.5 Tests for proportions

Single proportion \(\hat{p}=x/n\), \(H_0:p=p_0\):
$$z=\frac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}} \approx \mathcal{N}(0,1) \quad \text{(large }n\text{)}$$

Two proportions: use pooled \(\hat{p}\) under \(H_0:p_1=p_2\) for SE.

**Chi-square goodness-of-fit / independence** (several categories): compare observed \(O_i\) to expected \(E_i\) under \(H_0\):
$$\chi^2=\sum\frac{(O_i-E_i)^2}{E_i} \approx \chi^2_{\text{df}}$$

---

## 4. One-Way and Two-Way ANOVA

### 4.1 One-way ANOVA (single factor, \(k\) levels)

Model: \(X_{ij}=\mu+\tau_i+\varepsilon_{ij}\), \(\varepsilon_{ij}\sim\mathcal{N}(0,\sigma^2)\), \(\sum_i n_i\tau_i=0\).

**\(H_0\)**: \(\tau_1=\cdots=\tau_k=0\) (all group means equal).

Partition sum of squares:
$$SS_{\text{total}}=SS_{\text{treat}}+SS_{\text{error}}$$

$$SS_{\text{treat}}=\sum_{i=1}^k n_i(\bar{x}_{i\cdot}-\bar{x}_{\cdot\cdot})^2, \quad SS_{\text{error}}=\sum_{i,j}(x_{ij}-\bar{x}_{i\cdot})^2$$

**F statistic**:
$$F=\frac{MS_{\text{treat}}}{MS_{\text{error}}}=\frac{SS_{\text{treat}}/(k-1)}{SS_{\text{error}}/(N-k)} \sim F_{k-1,\,N-k} \quad \text{under }H_0$$

Reject \(H_0\) for large \(F\) (right-tailed).

### 4.2 Two-way ANOVA (factors A, B)

Main effects for A and B plus interaction \(A\times B\) when replicated. **Interaction significant** \(\Rightarrow\) do not interpret main effects in isolation (CS-10 PPT pattern).

### 4.3 ANOVA table (one-way template)

| Source | df | SS | MS | F |
| :--- | :--- | :--- | :--- | :--- |
| Treatment | \(k-1\) | \(SS_{\text{treat}}\) | \(MS_{\text{treat}}\) | \(MS_{\text{treat}}/MS_{\text{error}}\) |
| Error | \(N-k\) | \(SS_{\text{error}}\) | \(MS_{\text{error}}\) | — |
| Total | \(N-1\) | \(SS_{\text{total}}\) | — | — |

---

## 5. Maximum Likelihood Estimation (Intro)

For i.i.d. data \(\mathbf{x}=(x_1,\ldots,x_n)\) from \(f(x;\theta)\):

**Likelihood**: \(L(\theta)=\prod_{i=1}^n f(x_i;\theta)\).

**Log-likelihood**: \(\ell(\theta)=\sum_i \log f(x_i;\theta)\).

**MLE** \(\hat{\theta}\): maximizes \(L(\theta)\) or \(\ell(\theta)\); often solve \(\dfrac{d\ell}{d\theta}=0\).

**Examples (exam favorites)**:
- Bernoulli\((p)\): \(\hat{p}=\bar{x}\).
- Poisson\((\lambda)\): \(\hat{\lambda}=\bar{x}\).
- Normal\((\mu,\sigma^2)\): \(\hat{\mu}=\bar{x}\), \(\hat{\sigma}^2=\dfrac{1}{n}\sum(x_i-\bar{x})^2\) (MLE; unbiased version uses \(n-1\)).

**Asymptotics**: under regularity, \(\hat{\theta}\approx\mathcal{N}(\theta,\,I(\theta)^{-1}/n)\) for large \(n\) (Fisher information \(I(\theta)\)).

---

## 6. AI / ML Bridge (Lead ISM Lectures)

| Technique | ML connection |
| :--- | :--- |
| CLT / large \(n\) | Justifies Gaussian noise in SGD mini-batch mean gradients (qualitative) |
| Hypothesis tests | A/B testing for model deployments; \(p\)-value **not** effect size |
| ANOVA | Comparing multiple algorithms / hyperparameter levels on benchmark folds |
| MLE | Cross-entropy = Bernoulli/Binomial MLE; Gaussian MLE = least squares in linear regression |
| Type I/II errors | False positive discoveries in feature selection; low power in small validation sets |

**Practical guideline**: statistical significance \(\neq\) model improvement large enough to matter in production — report CIs and domain metrics.

---

## 7. Solved Numerical Example (Exam Pattern)

**Problem Statement (BITS Pattern, ~6 Marks)**: Three training curricula (A, B, C) are compared on quiz scores. Summary: \(\bar{x}_A=72\), \(\bar{x}_B=78\), \(\bar{x}_C=66\); \(n_A=n_B=n_C=5\); pooled \(s_p=4\). Test \(H_0\): equal mean scores at \(\alpha=0.05\). Report \(F\) and decision. (Simplified: given \(SS_{\text{treat}}=360\), \(SS_{\text{error}}=240\).)

### Solution

\(k=3\), \(N=15\), df\(_{\text{treat}}=2\), df\(_{\text{error}}=12\).

$$MS_{\text{treat}}=\frac{360}{2}=180, \quad MS_{\text{error}}=\frac{240}{12}=20$$

$$F=\frac{180}{20}=9$$

Critical value \(F_{0.05,\,2,\,12}\approx 3.89\) (standard table). Since \(F=9>3.89\), **reject \(H_0\)**.

**Interpretation**: strong evidence that at least one curriculum mean differs; post-hoc comparisons (Tukey) needed to pinpoint which pairs — ANOVA alone does not identify pairs (common comprehensive question).

---

## 8. Formula Sheet (Closed-Book Priority)

| Item | Formula |
| :--- | :--- |
| SE of mean | \(\sigma/\sqrt{n}\) or \(s/\sqrt{n}\) |
| \(t\) statistic | \((\bar{x}-\mu_0)/(s/\sqrt{n})\) |
| One-way F | \(MS_{\text{treat}}/MS_{\text{error}}\) |
| Proportion SE | \(\sqrt{p(1-p)/n}\) |
| MLE Bernoulli | \(\hat{p}=\bar{x}\) |

---

## 9. Common Pitfalls & Practical Guidelines

1. **\(p\)-value misuse**: small \(p\) does not prove causation or practical importance.
2. **Assumptions**: \(t\) and ANOVA need approximate normality and (for ANOVA) homogeneity of variance; robustness improves with balanced \(n\).
3. **Multiple testing**: many comparisons inflate Type I error — adjust or use ANOVA first.
4. **One-sided vs. two-sided**: exam marks lost for wrong tail doubling.

**PPT cross-reference**: **ISM_CS-7** (stratified sampling, CLT, CI); **ISM_CS-9** (z/t, proportions, χ²); **ISM_CS-10** (MLE, ANOVA one & two factor).