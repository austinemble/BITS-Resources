# AIML ZC418: Introduction to Statistical Methods — Comprehensive Examination Question-Answer Bank

**Course**: AIML ZC418 — Introduction to Statistical Methods  
**Standard**: 50 High-Yield University Examination Questions (**15 Easy**, **25 Medium**, **10 Tough**)  
**Marking Scheme**: 6 Marks Each (Point-wise model solutions: **2 Marks Concept/Definition**, **2 Marks Computation/Derivation**, **2 Marks Interpretation/Insight**)  
**Primary Teacher Followed**: Prof. G. Krishna Kumari (Lead ISM, BITS Pilani WILP)  
**Syllabus Modules**: M1–M6 (Probability → Bayes → Distributions → Hypothesis/ANOVA/MLE → Correlation/Regression/Time Series → GMM/EM)

---

# Part I: Basic Probability & Descriptive Statistics (Module 1)

### Q1. [Easy] Define the five-number summary and Tukey's IQR outlier rule. Apply the rule to $[12, 14, 11, 15, 13, 22, 14, 11, 16, 13]$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept / Definition (2 Marks)**:
   - Five-number summary: $\{\min, Q_1, \text{median}, Q_3, \max\}$.
   - $\mathrm{IQR}=Q_3-Q_1$; outlier if $x<Q_1-1.5\,\mathrm{IQR}$ or $x>Q_3+1.5\,\mathrm{IQR}$.
2. **Computation (2 Marks)**:
   - Sorted: $11,11,12,13,13,14,14,15,16,22$ → $Q_1=12$, median $=13.5$, $Q_3=15$.
   - $\mathrm{IQR}=3$; fences $[7.5,\ 19.5]$.
3. **Interpretation (2 Marks)**:
   - Only **22** exceeds $19.5$ → single high outlier (e.g. one exceptional salary distorts mean); report **median** for centre.

---

### Q2. [Easy] State Kolmogorov's three axioms of probability. Verify that $P(A^c)=1-P(A)$ follows from them. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Core Concept (2 Marks)**:
   - (i) $P(A)\ge 0$; (ii) $P(S)=1$; (iii) for disjoint $\{A_i\}$, $P(\bigcup_i A_i)=\sum_i P(A_i)$.
2. **Derivation (2 Marks)**:
   - $A$ and $A^c$ disjoint, $A\cup A^c=S$ ⇒ $P(S)=P(A)+P(A^c)=1$.
3. **Interpretation (2 Marks)**:
   - Complement rule is the workhorse for “at least one” problems via $1-P(\text{none})$.

---

### Q3. [Medium] A class has 30 ISM students: 18 registered for ML elective and 12 for Statistics elective; 8 take **both**. Are “ML only” and “Statistics only” mutually exclusive? Are the events “takes ML” and “takes Statistics” independent? Compute relevant probabilities. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definitions (2 Marks)**:
   - Mutually exclusive: $P(A\cap B)=0$. Independent: $P(A\cap B)=P(A)P(B)$.
2. **Computation (2 Marks)**:
   - $P(\text{ML})=18/30=0.6$, $P(\text{Stat})=12/30=0.4$, $P(\text{both})=8/30\approx0.267$.
   - $P(\text{ML})P(\text{Stat})=0.24\neq0.267$ ⇒ **not independent**. ML-only and Stat-only disjoint ⇒ **mutually exclusive**.
3. **Interpretation (2 Marks)**:
   - Overlap (double majors) breaks independence; lecture analogy: same professor two slots = exclusive, parallel sections = independent.

---

### Q4. [Medium] From a deck of 52 cards, what is $P(\text{King or Queen})$? Given a face card is drawn, what is $P(\text{King}\mid\text{face})$? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Concept (2 Marks)**:
   - King and Queen disjoint ⇒ $P(K\cup Q)=P(K)+P(Q)$.
   - Conditional: $P(K\mid F)=P(K\cap F)/P(F)$.
2. **Computation (2 Marks)**:
   - $P(K\cup Q)=4/52+4/52=8/52=2/13$.
   - Face cards 12; $P(K\mid F)=(4/52)/(12/52)=1/3$.
3. **Interpretation (2 Marks)**:
   - Conditioning shrinks sample space to faces — each rank equally likely among 12 face cards.

---

### Q5. [Easy] Differentiate mean vs median for **right-skewed** income data. Why might a lecturer call a “everyone scored $\ge 90$” histogram suspicious? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition (2 Marks)**:
   - Right skew: long right tail; $\bar{x}>\text{median}$ typically.
2. **Computation / contrast (2 Marks)**:
   - One crore package inflates $\bar{x}$; median near bulk of employees unchanged.
3. **Interpretation (2 Marks)**:
   - Symmetric-looking exam marks with floor 90 suggests **truncation / poor discrimination**, not true normality.

---

# Part II: Conditional Probability, Bayes & Naïve Bayes (Module 2)

### Q6. [Easy] Write the definition of $P(A\mid B)$. For fair die, find $P(\text{even}\mid\text{number}>3)$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition (2 Marks)**:
   - $P(A\mid B)=P(A\cap B)/P(B)$, $P(B)>0$.
2. **Computation (2 Marks)**:
   - $B=\{4,5,6\}$, $A\cap B=\{4,6\}$ ⇒ $P(A\mid B)=(2/6)/(3/6)=2/3$.
3. **Interpretation (2 Marks)**:
   - Conditioning on $B$ renormalizes probabilities to the restricted outcome set.

---

### Q7. [Medium] Prove the multiplication rule $P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A)$. When does $P(A\mid B)=P(A)$? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Concept (2 Marks)**:
   - Joint probability as chain of conditional and marginal.
2. **Derivation (2 Marks)**:
   - From $P(A\mid B)=P(A\cap B)/P(B)$ cross-multiply both forms.
   - $P(A\mid B)=P(A)$ iff $P(A\cap B)=P(A)P(B)$ — **independence**.
3. **Interpretation (2 Marks)**:
   - Bayes flips conditioning direction — essential for diagnostic tests with rare diseases.

---

### Q8. [Medium] **Bayes diagnostic**: disease prevalence $1\%$, test sensitivity $95\%$, specificity $90\%$. Patient tests positive. Find $P(\text{disease}\mid+)$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Setup (2 Marks)**:
   - $P(D)=0.01$, $P(+\mid D)=0.95$, $P(+\mid D^c)=0.10$.
2. **Computation (2 Marks)**:
   - $P(+)=(0.95)(0.01)+(0.10)(0.99)=0.0095+0.099=0.1085$.
   - $P(D\mid+)=0.0095/0.1085\approx0.088$ (**~8.8%**).
3. **Interpretation (2 Marks)**:
   - **Base-rate paradox**: most positives are false alarms when disease is rare — confirm with retest.

---

### Q9. [Tough] Derive Bayes' theorem from the definition of conditional probability and the multiplication rule. Extend to a three-hypothesis partition $\{H_1,H_2,H_3\}$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition chain (2 Marks)**:
   - $P(H_i\mid E)=\frac{P(E\mid H_i)P(H_i)}{P(E)}$ from $P(H_i\mid E)=P(H_i\cap E)/P(E)$.
2. **Derivation (2 Marks)**:
   - $P(H_i\cap E)=P(E\mid H_i)P(H_i)$; $P(E)=\sum_j P(E\mid H_j)P(H_j)$ (total probability).
3. **Interpretation (2 Marks)**:
   - Posterior $\propto$ likelihood $\times$ prior; evidence $P(E)$ is normalizing constant over competing hypotheses.

---

### Q10. [Medium] **Naïve Bayes text**: $P(\text{Spam})=0.4$, $P(\text{word “offer”}\mid\text{Spam})=0.8$, $P(\text{word “offer”}\mid\text{Ham})=0.1$; no other features. Classify an email with “offer”. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Model (2 Marks)**:
   - $\hat{y}=\arg\max_c P(c)P(x\mid c)$ (single feature NB).
2. **Computation (2 Marks)**:
   - Spam score $=0.4\times0.8=0.32$; Ham $=0.6\times0.1=0.06$ → **Spam**.
3. **Interpretation (2 Marks)**:
   - Likelihood ratio overwhelms modest prior; in practice use log scores and many conditionally independent features.

---

# Part III: Random Variables & Probability Distributions (Module 3)

### Q11. [Easy] For Bernoulli$(p)$, state PMF, $\mathbb{E}[X]$, $\mathrm{Var}(X)$. Verify variance formula for $p=0.3$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition (2 Marks)**:
   - $P(X=1)=p$, $P(X=0)=1-p$; $\mathbb{E}[X]=p$, $\mathrm{Var}(X)=p(1-p)$.
2. **Computation (2 Marks)**:
   - $p=0.3$ ⇒ $\mathrm{Var}=0.21$, $\sigma\approx0.458$.
3. **Interpretation (2 Marks)**:
   - Building block of binomial; max variance at $p=0.5$.

---

### Q12. [Medium] $X\sim\text{Binomial}(10,0.2)$. Find $P(X=2)$ and $\mathbb{E}[X]$. When is Poisson$(\lambda=np)$ a reasonable approximation? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Formulas (2 Marks)**:
   - $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$; $\mathbb{E}[X]=np$.
2. **Computation (2 Marks)**:
   - $P(X=2)=\binom{10}{2}(0.2)^2(0.8)^8\approx0.302$; $\mathbb{E}[X]=2$.
3. **Interpretation (2 Marks)**:
   - Poisson approx when $n$ large, $p$ small, $\lambda=np$ moderate (defects per lot).

---

### Q13. [Medium] Calls arrive at $\lambda=4$ per hour (Poisson). Find $P(X=0)$ and $P(X\ge2)$ in one hour. (6 Marks)

#### Model Solution (6/6 Marks):
1. **PMF (2 Marks)**:
   - $P(X=k)=e^{-\lambda}\lambda^k/k!$.
2. **Computation (2 Marks)**:
   - $P(0)=e^{-4}\approx0.018$; $P(\ge2)=1-P(0)-P(1)=1-e^{-4}(1+4)\approx0.938$.
3. **Interpretation (2 Marks)**:
   - Mean=variance=4 — check Poisson plausibility if sample mean ≈ sample variance.

---

### Q14. [Medium] If $X\sim\mathcal{N}(100,15^2)$, find $P(X<115)$ using $Z$-standardization. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Transform (2 Marks)**:
   - $Z=(X-\mu)/\sigma=(115-100)/15=1$.
2. **Computation (2 Marks)**:
   - $P(X<115)=P(Z<1)\approx0.8413$.
3. **Interpretation (2 Marks)**:
   - One-sigma rule: ~68–95–99.7% for normal bulk; CLT motivates normal for averages.

---

### Q15. [Tough] If $Y=g(X)$ with $X$ continuous PDF $f_X$ and monotone $g$, state the transformation formula for $f_Y$. Apply to $X\sim\mathcal{U}(0,1)$, $Y=-2\ln X$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **General formula (2 Marks)**:
   - $f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|$.
2. **Computation (2 Marks)**:
   - $Y\sim\text{Exponential}(2)$: support $y>0$, $f_Y(y)=e^{-y/2}/2$.
3. **Interpretation (2 Marks)**:
   - Links uniform to chi-square/exponential — used in simulation and goodness-of-fit.

---

### Q16. [Easy] State the roles of **$t$**, **$\chi^2$**, and **$F$** distributions in inference (one line each). (6 Marks)

#### Model Solution (6/6 Marks):
1. **$t$ (2 Marks)**:
   - Sampling distribution of standardized mean when $\sigma$ unknown; heavier tails than $Z$.
2. **$\chi^2$ (2 Marks)**:
   - Sum of squared standard normals; GOF / variance tests.
3. **$F$ (2 Marks)**:
   - Ratio of scaled chi-squares; **ANOVA** tests equal means via MSB/MSW.

---

### Q17. [Medium] Joint PMF: $p(0,0)=0.2$, $p(0,1)=0.3$, $p(1,0)=0.1$, $p(1,1)=0.4$. Find marginals, $\mathbb{E}[X]$, $\mathrm{Cov}(X,Y)$. Are $X,Y$ independent? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Marginals (2 Marks)**:
   - $p_X(0)=0.5$, $p_X(1)=0.5$; $p_Y(0)=0.3$, $p_Y(1)=0.7$.
2. **Moments (2 Marks)**:
   - $\mathbb{E}[X]=0.5$, $\mathbb{E}[Y]=0.7$; $\mathbb{E}[XY]=0.4$ ⇒ $\mathrm{Cov}=0.4-0.35=0.05$.
3. **Independence (2 Marks)**:
   - $p(1,1)=0.4\neq0.5\times0.7=0.35$ ⇒ **dependent**.

---

### Q18. [Tough] Prove $\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\mathrm{Cov}(X,Y)$. When does variance of a sum equal sum of variances? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition (2 Marks)**:
   - $\mathrm{Var}(Z)=\mathbb{E}[(Z-\mathbb{E}Z)^2]$.
2. **Derivation (2 Marks)**:
   - Expand $(X+Y-\mu_X-\mu_Y)^2$; cross-term $2\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$.
3. **Interpretation (2 Marks)**:
   - Independence ⇒ $\mathrm{Cov}=0$; portfolio risk increases if positively correlated assets.

---

# Part IV: Sampling, Estimation, Hypothesis Testing, MLE & ANOVA (Module 4)

### Q19. [Easy] Differentiate **random sampling** vs **stratified sampling**. When is stratification preferred? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definitions (2 Marks)**:
   - Random: each unit equal selection chance; Stratified: sample within homogeneous strata.
2. **Design (2 Marks)**:
   - Stratified proportional allocation: $n_h=n\cdot N_h/N$.
3. **Interpretation (2 Marks)**:
   - Use strata when subpopulations differ sharply (campus × programme) — reduces variance of estimators.

---

### Q20. [Medium] State the Central Limit Theorem informally. $n=64$, $\sigma=8$, $\mu$ unknown. Why can $\bar{X}$ justify normal-based CI even if parent is skewed? (6 Marks)

#### Model Solution (6/6 Marks):
1. **CLT (2 Marks)**:
   - $\bar{X}$ approx $\mathcal{N}(\mu,\sigma^2/n)$ for large $n$ (i.i.d., finite variance).
2. **Computation (2 Marks)**:
   - $\mathrm{SE}(\bar{X})=\sigma/\sqrt{n}=8/8=1$.
3. **Interpretation (2 Marks)**:
   - Averaging smooths skew; $n=64$ usually sufficient for engineering/economic measurements.

---

### Q21. [Tough] Construct a 95% CI for $\mu$ when $n=25$, $\bar{x}=48$, $s=10$. Use $t_{0.025,24}\approx2.064$. Interpret in business language. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Formula (2 Marks)**:
   - $\bar{x}\pm t_{\alpha/2,\nu}\frac{s}{\sqrt{n}}$.
2. **Computation (2 Marks)**:
   - Margin $=2.064\times10/5=4.128$ → **CI $[43.87,\ 52.13]$**.
3. **Interpretation (2 Marks)**:
   - We are 95% confident the true process mean lies in this band; not “95% probability $\mu$ is in interval” (frequentist reading).

---

### Q22. [Easy] Define Type I and Type II errors with a forensic analogy. What is test **power**? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Type I (2 Marks)**:
   - Convict innocent ($\alpha$); reject true $H_0$.
2. **Type II (2 Marks)**:
   - Acquit guilty ($\beta$); fail to reject false $H_0$.
3. **Power (2 Marks)**:
   - $1-\beta$ — probability of detecting real effect; increase via larger $n$ or focused one-tailed test.

---

### Q23. [Medium] One-sample $t$-test: $H_0:\mu=50$, $n=16$, $\bar{x}=53$, $s=4$. Test at $\alpha=0.05$ ($t_{0.025,15}\approx2.131$). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Statistic (2 Marks)**:
   - $t=\frac{53-50}{4/4}=3$.
2. **Decision (2 Marks)**:
   - $|t|=3>2.131$ ⇒ **reject** $H_0$.
3. **Interpretation (2 Marks)**:
   - Evidence mean exceeds 50; report effect $\bar{x}-\mu_0=3$ in original units (e.g. mm).

---

### Q24. [Tough] Two proportions: $n_1=200$, $\hat{p}_1=0.72$; $n_2=150$, $\hat{p}_2=0.60$. Test $H_0:p_1=p_2$ at $\alpha=0.05$ using pooled $\hat{p}$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Pooled estimator (2 Marks)**:
   - $\hat{p}=\frac{144+90}{350}=\frac{234}{350}\approx0.669$.
2. **$z$-statistic (2 Marks)**:
   - $z=\frac{0.12}{\sqrt{\hat{p}(1-\hat{p})(1/200+1/150)}}\approx\frac{0.12}{0.057}\approx2.11$.
3. **Interpretation (2 Marks)**:
   - $|z|>1.96$ ⇒ reject equal rates; practical check: 12 pp gap on conversion KPI.

---

### Q25. [Medium] Write the likelihood for i.i.d. $\mathcal{N}(\mu,\sigma^2)$ samples. Show $\hat{\mu}_{\text{MLE}}=\bar{x}$. Note MLE of $\sigma^2$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Likelihood (2 Marks)**:
   - $L(\mu,\sigma^2)\propto\exp\big(-\frac{1}{2\sigma^2}\sum(x_i-\mu)^2\big)$.
2. **MLE (2 Marks)**:
   - Minimize $\sum(x_i-\mu)^2$ ⇒ $\hat{\mu}=\bar{x}$; $\hat{\sigma}^2_{\text{MLE}}=\frac{1}{n}\sum(x_i-\bar{x})^2$.
3. **Interpretation (2 Marks)**:
   - MLE $\sigma^2$ biased low vs unbiased $s^2$ — connects to regression Gaussian noise.

---

### Q26. [Tough] One-way ANOVA table skeleton: $k=3$ groups, $n_i=10$ each. Define SSB, SSW, MSB, MSW, and $F$. State $H_0$ and df. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Partition (2 Marks)**:
   - $\mathrm{SST}=\mathrm{SSB}+\mathrm{SSW}$; SSB between group means, SSW within groups.
2. **$F$ statistic (2 Marks)**:
   - $F=\mathrm{MSB}/\mathrm{MSW}$, df $(k-1,\ N-k)=(2,27)$.
3. **Interpretation (2 Marks)**:
   - $H_0:\mu_1=\mu_2=\mu_3$; reject large $F$ — training methods differ on average outcome.

---

### Q27. [Easy] What assumption underlies standard one-way ANOVA besides independent groups? How do you check it informally? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Assumption (2 Marks)**:
   - Equal variance across groups (homoscedasticity); normality within groups for exact $F$.
2. **Check (2 Marks)**:
   - Residual plots vs fitted; boxplots of residuals by group; Levene test if available.
3. **Interpretation (2 Marks)**:
   - Robust to mild non-normality with balanced $n$; unequal variance invalidates pooled MSW logic.

---

### Q28. [Medium] Two-factor ANOVA: when is the **interaction** term essential? Give a factorial example (e.g. fertilizer × irrigation). (6 Marks)

#### Model Solution (6/6 Marks):
1. **Concept (2 Marks)**:
   - Interaction: effect of factor A depends on level of B.
2. **Example (2 Marks)**:
   - Irrigation helps only with fertilizer present — lines of cell means not parallel.
3. **Interpretation (2 Marks)**:
   - Significant $A\times B$ forbids interpreting main effects alone; simple effects analysis needed.

---

# Part V: Correlation, Regression & Time Series (Module 5)

### Q29. [Medium] Compute Pearson $r$ for $(x,y)$: $(1,2),(2,3),(3,5)$. Fit $\hat{y}=\hat{\beta}_0+\hat{\beta}_1 x$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Means (2 Marks)**:
   - $\bar{x}=2$, $\bar{y}=10/3$; sums of deviations for $s_{xy}$, $s_x^2$.
2. **Computation (2 Marks)**:
   - $\hat{\beta}_1=3/2=1.5$, $\hat{\beta}_0=10/3-3=1/3$; $r\approx0.993$.
3. **Interpretation (2 Marks)**:
   - Strong linear trend; $R^2\approx0.986$ — almost all $y$ variation explained by $x$ (toy data).

---

### Q30. [Tough] Explain why $r=0$ does not imply **no relationship**. Sketch a U-shaped scenario. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition limit (2 Marks)**:
   - $r$ measures **linear** association only.
2. **Example (2 Marks)**:
   - $y=(x-2)^2$ symmetric about $x=2$ ⇒ $r=0$ possible.
3. **Interpretation (2 Marks)**:
   - Always plot scatter; polynomial/regime models may be needed.

---

### Q31. [Easy] Decompose a time series into four components per Brockwell–Davis / lecture standard. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Components (2 Marks)**:
   - Trend, seasonal, cyclical, irregular (random).
2. **Model views (2 Marks)**:
   - Additive $Y=T+S+C+I$ or multiplicative for growing seasonality.
3. **Interpretation (2 Marks)**:
   - Forecasting requires modelling structure; irregular is what ARMA tries to capture.

---

### Q32. [Medium] Write MA$(1)$: $X_t=\varepsilon_t+\theta\varepsilon_{t-1}$. Express $\mathrm{Var}(X_t)$ and lag-1 ACF $\rho_1$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Setup (2 Marks)**:
   - $\varepsilon_t$ white noise, $\mathrm{Var}(\varepsilon)=\sigma^2$.
2. **Computation (2 Marks)**:
   - $\mathrm{Var}(X_t)=(1+\theta^2)\sigma^2$; $\rho_1=\frac{\theta}{1+\theta^2}$.
3. **Interpretation (2 Marks)**:
   - Finite memory; ACF cuts after lag 1 — used for short forecast smoothing.

---

### Q33. [Medium] AR$(1)$ $X_t=0.6X_{t-1}+\varepsilon_t$. Is it stationary? Compute unconditional mean and variance. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Stationarity (2 Marks)**:
   - Root of $1-0.6B$ is $1/0.6>1$ ⇒ **stationary**.
2. **Moments (2 Marks)**:
   - Mean 0 if intercept 0; $\mathrm{Var}(X)=\sigma^2/(1-0.36)=\sigma^2/0.64$.
3. **Interpretation (2 Marks)**:
   - Shocks persist geometrically; ACF decays as $0.6^k$.

---

### Q34. [Medium] First difference of series with linear trend $X_t=5+2t+\varepsilon_t$. Why apply ARIMA with $d=1$? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Differencing (2 Marks)**:
   - $\nabla X_t=X_t-X_{t-1}\approx2+\varepsilon_t-\varepsilon_{t-1}$ removes deterministic trend.
2. **Stationarity (2 Marks)**:
   - Original non-stationary mean increases with $t$.
3. **Interpretation (2 Marks)**:
   - ARIMA$(p,1,q)$ models changes — standard for financial/index levels.

---

### Q35. [Tough] Outline **Box–Jenkins** identification–estimation–diagnostic cycle for ARIMA$(1,1,1)$. What should residual ACF look like? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Identify (2 Marks)**:
   - Plot, ADF/unit-root intuition, ACF/PACF on differenced series.
2. **Estimate / check (2 Marks)**:
   - MLE or CSS for $\phi,\theta$; compare AIC/BIC across candidates.
3. **Diagnostics (2 Marks)**:
   - Residuals ≈ white noise; Ljung–Box on residual ACF — no significant spikes.

---

### Q36. [Easy] **Simple exponential smoothing**: $\hat{X}_{t+1}=\alpha X_t+(1-\alpha)\hat{X}_t$. Role of $\alpha$? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition (2 Marks)**:
   - Forecast = convex combo of latest observation and previous forecast.
2. **$\alpha$ effect (2 Marks)**:
   - Large $\alpha$ → react fast; small $\alpha$ → smooth memory.
3. **Interpretation (2 Marks)**:
   - Holt adds trend; Holt–Winters adds season length $m$ (CS14).

---

### Q37. [Medium] 3-period **simple moving average** of $[10,12,11,15,14]$: compute forecasts at $t=3,4,5$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Rule (2 Marks)**:
   - $\hat{X}_{t+1}=\frac{1}{3}(X_t+X_{t-1}+X_{t-2})$.
2. **Computation (2 Marks)**:
   - At $t=3$: $(10+12+11)/3=11$; $t=4$: $(12+11+15)/3\approx12.67$; $t=5$: $(11+15+14)/3=13.33$.
3. **Interpretation (2 Marks)**:
   - Lags turning points; weighted MA (lecture Excel) down-weights older points.

---

### Q38. [Easy] Define **weak stationarity** for $\{X_t\}$. Why is it required before applying ARMA? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Definition (2 Marks)**:
   - Constant mean/variance; $\mathrm{Cov}(X_t,X_{t-h})$ depends only on lag $h$.
2. **Reason (2 Marks)**:
   - ARMA parameters assume stable autocovariance structure over time.
3. **Interpretation (2 Marks)**:
   - Differencing / detrending enforces stationarity on transforms.

---

### Q39. [Medium] Contrast **SARIMA** vs **Holt–Winters** for monthly sales with yearly season. (6 Marks)

#### Model Solution (6/6 Marks):
1. **SARIMA (2 Marks)**:
   - $(P,D,Q)_m$ seasonal operators; stochastic seasonal persistence.
2. **Holt–Winters (2 Marks)**:
   - Deterministic seasonal indices with level/trend smoothing.
3. **Interpretation (2 Marks)**:
   - SARIMA flexible; HW interpretable and fast — compare on hold-out RMSE.

---

### Q40. [Tough] State the AR$(p)$ stationarity condition in terms of characteristic polynomial roots. For AR$(2)$ $\phi_1+\phi_2<1$, $\phi_2-\phi_1<1$, $|\phi_2|<1$, explain intuition. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Formal (2 Marks)**:
   - All roots of $1-\phi_1 z-\cdots-\phi_p z^p$ **outside** unit circle.
2. **AR(2) inequalities (2 Marks)**:
   - Prevent unit-root / explosive recursion; stable impulse responses.
3. **Interpretation (2 Marks)**:
   - Violation ⇒ variance blows up — must difference or re-specify.

---

# Part VI: Gaussian Mixture Models & EM (Module 6)

### Q41. [Easy] Write the GMM density for $K=2$ univariate components. What do $\pi_1,\pi_2$ represent? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Density (2 Marks)**:
   - $p(x)=\pi_1\mathcal{N}(x|\mu_1,\sigma_1^2)+\pi_2\mathcal{N}(x|\mu_2,\sigma_2^2)$.
2. **Constraints (2 Marks)**:
   - $\pi_1+\pi_2=1$, $\pi_k\ge0$.
3. **Interpretation (2 Marks)**:
   - Population fraction from each latent sub-population before observing $x$.

---

### Q42. [Medium] Write the **E-step** responsibility $\gamma_{ik}$ for GMM. Why must denominators sum over all components $k$? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Formula (2 Marks)**:
   - $\gamma_{ik}=\frac{\pi_k\mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_j\pi_j\mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}$.
2. **Normalization (2 Marks)**:
   - Denominator = marginal $p(\mathbf{x}_i)$ under mixture ⇒ $\sum_k\gamma_{ik}=1$.
3. **Interpretation (2 Marks)**:
   - Soft cluster membership — bridges Bayes posterior and $k$-means hard labels.

---

### Q43. [Tough] Derive the **M-step** updates for $\boldsymbol{\mu}_k$ and $\pi_k$ from complete-data log-likelihood with responsibilities fixed. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Objective (2 Marks)**:
   - $Q(\theta)=\sum_i\sum_k\gamma_{ik}[\log\pi_k+\log\mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)]$.
2. **Derivation (2 Marks)**:
   - $\pi_k=\frac{1}{n}\sum_i\gamma_{ik}$; $\boldsymbol{\mu}_k=\frac{\sum_i\gamma_{ik}\mathbf{x}_i}{\sum_i\gamma_{ik}}$ (Lagrange multiplier on $\sum\pi_k=1$).
3. **Interpretation (2 Marks)**:
   - Weighted $k$-means centroid; weights are current posterior beliefs.

---

### Q44. [Easy] Why does plain MLE for GMM need EM instead of a single closed-form solution? (6 Marks)

#### Model Solution (6/6 Marks):
1. **Latent variables (2 Marks)**:
   - Missing component labels $Z_i$ — incomplete data.
2. **Likelihood (2 Marks)**:
   - $\log\sum_k\pi_k\mathcal{N}(\cdot)$ has log-of-sum; no analytic root for all parameters jointly.
3. **Interpretation (2 Marks)**:
   - EM maximizes observed log-likelihood iteratively with guaranteed non-decrease.

---

### Q45. [Medium] Compare **$k$-means** vs **GMM+EM** for customer segmentation. (6 Marks)

#### Model Solution (6/6 Marks):
1. **$k$-means (2 Marks)**:
   - Hard labels, spherical clusters (implicit equal covariance), fast Lloyd.
2. **GMM (2 Marks)**:
   - Elliptical $\boldsymbol{\Sigma}_k$, probabilistic $\gamma_{ik}$, density estimates for anomalies.
3. **Interpretation (2 Marks)**:
   - Use GMM when overlap and soft scores matter (credit risk tiers); $k$-means for speed.

---

# Part VII: Integrated Exam Numericals & Synthesis (Cross-Module)

### Q46. [Medium] **Numerical Bayes**: Factory A supplies 60% of chips with 2% defect rate; factory B 40% with 5% defect. Random chip is defective. Find source posterior for A. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Priors/likelihoods (2 Marks)**:
   - $P(A)=0.6$, $P(D\mid A)=0.02$, $P(D\mid B)=0.05$.
2. **Computation (2 Marks)**:
   - $P(D)=0.012+0.02=0.032$; $P(A\mid D)=0.012/0.032=0.375$.
3. **Interpretation (2 Marks)**:
   - Despite higher quality from A, more defects still from B due to **base rates** — traceability matters.

---

### Q47. [Easy] Standardize scores $[70,80,90]$ with $\bar{x}=80$, $s=10$. Interpret $z$ for 90. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Formula (2 Marks)**:
   - $z=(x-\bar{x})/s$.
2. **Computation (2 Marks)**:
   - $z(90)=1$; others $-1,0$.
3. **Interpretation (2 Marks)**:
   - One standard deviation above class mean — link to percentile if normal approx holds.

---

### Q48. [Medium] **$\chi^2$ GOF**: 100 rolls, counts for faces 1–6: $(12,18,14,22,16,18)$ vs fair die. Compute $\chi^2$ (6 d.f.) and state $H_0$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Expected (2 Marks)**:
   - $E_i=100/6\approx16.67$ each; $H_0$: fair uniform.
2. **Statistic (2 Marks)**:
   - $\chi^2=\sum(O-E)^2/E\approx5.0$ (approximate).
3. **Interpretation (2 Marks)**:
   - Compare to $\chi^2_{0.05,5}\approx11.07$ — fail to reject fairness at 5%.

---

### Q49. [Medium] **Regression residual**: Given $\hat{y}=2+1.5x$, point $(4,10)$. Compute residual and its sign meaning. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Prediction (2 Marks)**:
   - $\hat{y}(4)=2+6=8$.
2. **Residual (2 Marks)**:
   - $e=10-8=+2$ (under-predicted).
3. **Interpretation (2 Marks)**:
   - Contributes to SSE; check for influential outliers if $|e|$ large vs $\sigma$.

---

### Q50. [Medium] **EM one E-step**: 1D GMM $\pi_1=\pi_2=0.5$, $\mu_1=0,\mu_2=4$, $\sigma_1=\sigma_2=1$. Observation $x=2$. Compute $\gamma_{11},\gamma_{12}$. (6 Marks)

#### Model Solution (6/6 Marks):
1. **Densities (2 Marks)**:
   - $\mathcal{N}(2|0,1)\approx0.054$; $\mathcal{N}(2|4,1)\approx0.242$.
2. **Responsibilities (2 Marks)**:
   - Numerator terms: $0.5\times0.054$, $0.5\times0.242$ ⇒ $\gamma_{11}=0.054/(0.054+0.242)\approx0.18$, $\gamma_{12}\approx0.82$.
3. **Interpretation (2 Marks)**:
   - Point closer to second mode — M-step would pull $\mu_2$ slightly toward 2.

---

## Difficulty Index (Verification)

| Level | Question IDs | Count |
| :--- | :--- | :---: |
| **Easy** | 1, 2, 5, 6, 11, 16, 19, 22, 27, 31, 36, 38, 41, 44, 47 | **15** |
| **Medium** | 3, 4, 7, 8, 10, 12, 13, 14, 17, 20, 23, 25, 28, 29, 32, 33, 34, 37, 39, 42, 45, 46, 48, 49, 50 | **25** |
| **Tough** | 9, 15, 18, 21, 24, 26, 30, 35, 40, 43 | **10** |

---

## Examiner Quick Map (Modules ↔ Questions)

| Module | Questions |
| :--- | :--- |
| M1 | Q1–Q5 |
| M2 | Q6–Q10 |
| M3 | Q11–Q18 |
| M4 | Q19–Q28 |
| M5 | Q29–Q40 |
| M6 | Q41–Q45 |
| Synthesis | Q46–Q50 |

---

*Pair with `Master_Formula_Sheet_and_Study_Guide.md` for formulas, critical values, flowcharts, and viva checklist.*