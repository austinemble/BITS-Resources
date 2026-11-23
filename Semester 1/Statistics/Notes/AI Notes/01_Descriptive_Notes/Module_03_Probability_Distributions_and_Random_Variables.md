# Module 3: Probability Distributions and Random Variables

**Course**: AIML ZC418 — Introduction to Statistical Methods  
**Prescribed References**: T1 Kaptein et al. (*Statistics for Data Scientists*, Springer 2022); T2 Devore (*Probability and Statistics for Engineering and the Sciences*, 8th ed.); T3 Brockwell & Davis (time-series supplement where joint/moment ideas recur)  
**Syllabus module**: **M03** — Probability Distributions and Random Variables (see `Notes/AI Notes/module_map.yaml`)  
**Canonical video**: `Inbox/Prof. G.Krishna Kumari/Recordings/Stats_G.Krishna Kumari_5.mp4`  
**Contact sessions** (reference only): Contact Sessions 5–6 (`ISM_CS-5`, `ISM_CS-6` PPTs — random variables, standard distributions, sampling-related laws).

> **Exam note (Lead Instructor)**: **Mid-semester (closed book)** — memorize PMF/PDF, means, variances, mgf identities, and transformation rules for standard families. **Comprehensive (open book)** — emphasize **model choice**, parameter interpretation, and when each distribution applies to data / ML pipelines.

---

## 1. Executive Summary & Foundational Definitions

Random variables (RVs) formalize uncertain outcomes as numbers. **Discrete** RVs take countable values with a probability mass function (PMF); **continuous** RVs take values on intervals with a probability density function (PDF). Expectation, variance, and covariance summarize location and spread; **moment generating functions (mgf)** encode all moments when they exist.

Standard distributions model the bulk of AIML/ISM coursework: **Bernoulli/Binomial** for counts of successes, **Poisson** for rare events per interval, **Normal** as the CLT limit and noise model in regression, and **t, F, χ²** as sampling distributions for inference (Module 4).

```
Experiment ω ──> RV X(ω) ──> Law: PMF f_X or PDF f_X
                              │
                              ├── E[X], Var(X), Cov(X,Y)
                              ├── MGF M_X(t) = E[e^{tX}]
                              └── Transform g(X) ──> new law (Jacobian / discrete map)
```

---

## 2. Random Variables, Expectation, and Joint Laws

### 2.1 Formal definitions

- **Discrete RV**: \(X\) takes values \(x_i\) with \(P(X=x_i)=p_X(x_i)\), \(\sum_i p_X(x_i)=1\).
- **Continuous RV**: \(P(a\le X\le b)=\int_a^b f_X(x)\,dx\) with \(f_X(x)\ge 0\), \(\int_{-\infty}^{\infty} f_X(x)\,dx=1\).
- **CDF**: \(F_X(x)=P(X\le x)\); for continuous \(X\), \(f_X(x)=\frac{d}{dx}F_X(x)\) where differentiable.

**Expectation**:
$$E[X]=\sum_i x_i p_X(x_i) \quad \text{(discrete)}; \qquad E[X]=\int_{-\infty}^{\infty} x f_X(x)\,dx \quad \text{(continuous)}$$

**Variance**: \(\mathrm{Var}(X)=E[(X-\mu)^2]=E[X^2]-(E[X])^2\) with \(\mu=E[X]\).

**Covariance** (joint RVs \(X,Y\)):
$$\mathrm{Cov}(X,Y)=E[(X-\mu_X)(Y-\mu_Y)]=E[XY]-E[X]E[Y]$$
Independent \(\Rightarrow\) \(\mathrm{Cov}(X,Y)=0\) (converse false).

### 2.2 Joint, marginal, and conditional

For discrete \((X,Y)\): joint PMF \(p_{X,Y}(x,y)\); marginals \(p_X(x)=\sum_y p_{X,Y}(x,y)\).

For continuous: joint PDF \(f_{X,Y}(x,y)\); marginals \(f_X(x)=\int f_{X,Y}(x,y)\,dy\).

**Conditional PDF** (continuous): \(f_{Y|X}(y|x)=\dfrac{f_{X,Y}(x,y)}{f_X(x)}\) when \(f_X(x)>0\).

### 2.3 Moment generating function (mgf)

**Definition**: \(M_X(t)=E[e^{tX}]\) for \(t\) in a neighborhood of \(0\).

**Properties** (when mgf exists):
- \(M_X^{(k)}(0)=E[X^k]\).
- If \(X,Y\) independent: \(M_{X+Y}(t)=M_X(t)M_Y(t)\).
- Uniqueness: matching mgfs on an interval \(\Rightarrow\) same distribution.

**Examples**:
- Bernoulli\((p)\): \(M_X(t)=1-p+pe^t\).
- Poisson\((\lambda)\): \(M_X(t)=\exp\big(\lambda(e^t-1)\big)\).
- Normal\((\mu,\sigma^2)\): \(M_X(t)=\exp\big(\mu t+\tfrac{1}{2}\sigma^2 t^2\big)\).

---

## 3. Standard Discrete Distributions

| Distribution | PMF / support | \(E[X]\) | \(\mathrm{Var}(X)\) | Typical use |
| :--- | :--- | :--- | :--- | :--- |
| **Bernoulli**(\(p\)) | \(P(X=1)=p\), \(P(X=0)=1-p\) | \(p\) | \(p(1-p)\) | Single trial indicator |
| **Binomial**(\(n,p\)) | \(\binom{n}{x}p^x(1-p)^{n-x}\), \(x=0,\ldots,n\) | \(np\) | \(np(1-p)\) | # successes in \(n\) i.i.d. Bernoulli |
| **Poisson**(\(\lambda\)) | \(\dfrac{e^{-\lambda}\lambda^x}{x!}\), \(x=0,1,\ldots\) | \(\lambda\) | \(\lambda\) | Rare events per unit; \(n\) large, \(p\) small, \(np\approx\lambda\) |

**Binomial–Poisson link**: if \(X\sim\text{Binomial}(n,p)\) with \(n\to\infty\), \(p\to 0\), \(np\to\lambda\), then \(X\approx\text{Poisson}(\lambda)\).

---

## 4. Standard Continuous & Sampling Distributions

### 4.1 Normal (Gaussian)

\(X\sim\mathcal{N}(\mu,\sigma^2)\):
$$f_X(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Standardization: \(Z=\dfrac{X-\mu}{\sigma}\sim\mathcal{N}(0,1)\).

If \(X_i\) i.i.d. \(\mathcal{N}(\mu,\sigma^2)\), then \(\bar{X}\sim\mathcal{N}\!\left(\mu,\dfrac{\sigma^2}{n}\right)\).

### 4.2 Student’s \(t\)

If \(Z\sim\mathcal{N}(0,1)\), \(U\sim\chi^2_\nu\) independent, then
$$T=\frac{Z}{\sqrt{U/\nu}}\sim t_\nu$$
Used when \(\sigma\) unknown and estimated from data (small samples).

### 4.3 Chi-square \(\chi^2_\nu\)

Sum of squares of \(\nu\) i.i.d. \(\mathcal{N}(0,1)\) RVs: \(\chi^2_\nu = \sum_{i=1}^\nu Z_i^2\).

If \(S^2\) is sample variance from normal data, \(\dfrac{(n-1)S^2}{\sigma^2}\sim\chi^2_{n-1}\).

### 4.4 \(F\) distribution

If \(U_1\sim\chi^2_{d_1}\), \(U_2\sim\chi^2_{d_2}\) independent,
$$F=\frac{U_1/d_1}{U_2/d_2}\sim F_{d_1,d_2}$$
Appears in ANOVA ratio of mean squares (Module 4).

---

## 5. Transformation of Random Variables

### 5.1 Discrete case

If \(Y=g(X)\) is one-to-one with inverse \(x=g^{-1}(y)\):
$$p_Y(y)=p_X\big(g^{-1}(y)\big)$$

### 5.2 Continuous case (monotone \(g\))

Let \(Y=g(X)\), \(x=g^{-1}(y)\). Then
$$f_Y(y)=f_X\big(g^{-1}(y)\big)\left|\frac{dx}{dy}\right|$$

**Linear**: \(Y=aX+b\) (\(a\neq 0\)) \(\Rightarrow\) \(f_Y(y)=\dfrac{1}{|a|}f_X\!\left(\dfrac{y-b}{a}\right)\), \(E[Y]=aE[X]+b\), \(\mathrm{Var}(Y)=a^2\mathrm{Var}(X)\).

**Exam sketch**: Box–Muller and log-normal arise from nonlinear transforms of normals; you need the Jacobian \(|\det J|\), not full derivation, unless asked.

---

## 6. AI / ML Bridge (Lead ISM Lectures)

| Statistical object | ML role |
| :--- | :--- |
| Bernoulli / Binomial | Classification labels, binomial loss / log-loss for \(0/1\) outcomes |
| Poisson | Count targets (clicks, defects); Poisson regression link \(\log \lambda=\mathbf{w}^T\mathbf{x}\) |
| \(\mathcal{N}(\mu,\sigma^2)\) | Gaussian likelihood in linear regression; weight init (Xavier uses variance reasoning) |
| Joint \((X,Y)\) | Feature–label pairs; \(\mathrm{Cov}(X,Y)\) drives correlation-based feature screening |
| mgf / moments | Moment-matching in variational inference; stability of log-sum-exp in softmax |

**Naïve Bayes link (Module 2)**: discrete likelihoods often Bernoulli/Binomial per feature; continuous features often modeled as Gaussian per class — distribution choice is a **modeling assumption**, not optional decoration.

---

## 7. Solved Numerical Example (Exam Pattern)

**Problem Statement (BITS Pattern, ~6 Marks)**: A server log shows **defect alerts** with historical average **4 per hour**. (i) Model hourly count \(X\). (ii) Find \(P(X\le 2)\). (iii) In a **Binomial** approximation with \(n=100\) trials and success probability \(p\) matched to the same mean, compute \(P(X\le 2)\) and comment.

**Given / assumptions**: Defects per hour are independent, rare, and rate-stable \(\Rightarrow\) **Poisson**(\(\lambda=4\)).

### Solution

**(i)** \(X\sim\text{Poisson}(\lambda=4)\).

**(ii)**
$$P(X\le 2)=e^{-4}\left(1+4+\frac{4^2}{2!}\right)=e^{-4}(1+4+8)=13\,e^{-4}\approx 0.238$$

**(iii)** Binomial with \(np=4\): take \(n=100\), \(p=0.04\). Using Poisson as reference:
$$P(X\le 2)\approx \binom{100}{0}(0.96)^{100}+\binom{100}{1}(0.04)(0.96)^{99}+\binom{100}{2}(0.04)^2(0.96)^{98}\approx 0.242$$

**Interpretation**: Poisson and matched Binomial agree closely — validates **Poisson limit** for rare events. In ML monitoring, choosing Poisson for count streams avoids arbitrary bin \(n\).

---

## 8. Formula Sheet (Closed-Book Priority)

| Item | Formula |
| :--- | :--- |
| Binomial PMF | \(\binom{n}{x}p^x(1-p)^{n-x}\) |
| Poisson PMF | \(e^{-\lambda}\lambda^x/x!\) |
| Normal PDF | \(\dfrac{1}{\sqrt{2\pi}\sigma}\exp\big(-(x-\mu)^2/(2\sigma^2)\big)\) |
| \(Z\) standardize | \(Z=(X-\mu)/\sigma\) |
| mgf composition | Independent sum \(\Rightarrow\) product of mgfs |
| Transform (cont.) | \(f_Y(y)=f_X(g^{-1}(y))|dx/dy|\) |

---

## 9. Common Pitfalls & Practical Guidelines

1. **Poisson vs. Binomial**: use Poisson when \(n\) large, \(p\) small, \(\lambda=np\) moderate; exam traps swap mean \(np\) with variance \(np(1-p)\).
2. **Continuous \(P(X=a)\)**: always **0** for continuous RVs; probabilities are **interval** integrals.
3. **Independence vs. zero covariance**: uncorrelated \(\not\Rightarrow\) independent unless joint normal (special case).
4. **t vs. Normal**: use \(t\) when \(\sigma\) unknown from same sample; large \(n\) \(\Rightarrow\) \(t\approx z\).

**PPT cross-reference**: Prof. G. Krishna Kumari — **ISM_CS-5** (RV, expectation, joint laws), **ISM_CS-6** (Bernoulli through χ² families).