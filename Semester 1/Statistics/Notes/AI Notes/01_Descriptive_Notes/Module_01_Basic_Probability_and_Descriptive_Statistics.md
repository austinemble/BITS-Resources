# Module 1: Basic Probability & Descriptive Statistics

**Course**: AIML ZC418 — Introduction to Statistical Methods  
**Prescribed References**: Kaptein et al. (T1, Ch. 2–3); Devore (T2, Ch. 1–2)  
**Primary Teacher**: Prof. G. Krishna Kumari — **Contact Session 1–2** (orientation + axioms)  
**Slides**: `ISM_CS-1_PPT`, `ISM_CS-2` / `S1-26_ISM_CS-2`

> **Exam note (Prof. Krishna Kumari)**: Mid-semester (CS1–8) is **closed book** — memorise five-number summary, IQR outlier rule, and probability axioms. Comprehensive stresses **interpretation** of plots and business context.

---

## 1. Executive Summary

Before building ML models, you must characterise **how data behave** (location, spread, shape) and quantify **uncertainty** with probability. Module 1 links analyst work (summaries, box plots) to scientist work (detecting skew, outliers, and unrealistic exam-grade distributions) — the same pipeline used in exploratory data analysis prior to feature engineering.

---

## 2. Measures of Central Tendency & Five-Number Summary

For observations $x_1,\ldots,x_n$:

| Measure | Formula | Robust to outliers? |
| :--- | :--- | :---: |
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ | No |
| Median | Middle order statistic | Yes |
| Mode | Most frequent value | — |

**Five-number summary**: $\{\min, Q_1, \text{median}, Q_3, \max\}$.

**IQR**: $\mathrm{IQR} = Q_3 - Q_1$.

**Outlier rule (Tukey, lecture standard)**:
$$x < Q_1 - 1.5\,\mathrm{IQR} \quad \text{or} \quad x > Q_3 + 1.5\,\mathrm{IQR}$$

**Variance / standard deviation** (sample):
$$s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2, \qquad s = \sqrt{s^2}$$

High $s$ ⇒ points scattered far from $\bar{x}$ — often triggers investigation before modelling (lecture: salary outliers when one ₹1 Cr package distorts mean).

### 2.1 Skewness intuition

- **Right skew**: tail on right; $\text{mean} > \text{median}$ (lecture: “everyone scored ≥90” implausible exam design).
- **Left skew**: tail on left; $\text{mean} < \text{median}$.
- **Symmetric**: mean ≈ median; normal curve reference.

---

## 3. Probability Foundations

### 3.1 Classical definition
For equally likely outcomes in sample space $S$:
$$P(A) = \frac{|A|}{|S|} = \frac{\text{# favourable}}{\text{# total}}$$

**Deck examples (CS2)**:
- One specific card (diamond queen): $\frac{1}{52}$.
- Any queen: $\frac{4}{52}$.
- Diamond queen given queen drawn: $\frac{1}{4}$ (conditional narrative).

### 3.2 Kolmogorov axioms

1. $P(A) \ge 0$
2. $P(S) = 1$
3. For disjoint $A_i$: $P\left(\bigcup_i A_i\right) = \sum_i P(A_i)$

**Complement**: $P(A^c) = 1 - P(A)$.

### 3.3 Mutually exclusive vs independent

| Concept | Definition (lecture) | Formula check |
| :--- | :--- | :--- |
| **Mutually exclusive** | One event **blocks** the other (same professor, two Sunday ISM slots) | $P(A\cap B)=0$ |
| **Independent** | Parallel sections; one does not affect the other | $P(A\cap B)=P(A)P(B)$ |

---

## 4. Counting: Permutations & Combinations

- **Permutation** ${}_nP_r$: order matters → ${}_nP_r = \frac{n!}{(n-r)!}$
- **Combination** ${}_nC_r$: order irrelevant → ${}_nC_r = \binom{n}{r} = \frac{n!}{r!(n-r)!}$

PPT practice: choosing committees, card draws — express favourable count over ${}_nC_k$ total selections.

---

## 5. Solved Numerical (Exam Pattern — Five-Point Summary)

**Data** (CS1 style): 12, 14, 15, 15, 18, 22, 25, 28, 35, 40 ( $n=10$ ).

1. Order: already sorted.
2. $Q_2$ (median): average of 5th & 6th → $(18+22)/2 = 20$.
3. Lower half → $Q_1 = 15$; upper half → $Q_3 = 28$.
4. $\mathrm{IQR} = 13$; fences: $15 - 19.5 = -4.5$, $28 + 19.5 = 47.5$ → **no outliers**.
5. $\bar{x} = 234/10 = 23.4$; $s \approx 9.0$ (use calculator in exam).

**Interpretation (2 marks)**: Mild right tail (40) pulls mean above median slightly; box plot would show upper whisker longer.

---

## 6. AI / ML Bridge

| ISM concept | ML usage |
| :--- | :--- |
| Mean, variance, IQR | Feature scaling decisions, detecting label noise |
| Outliers | Robust metrics, anomaly detection precursors |
| Skewed grades / sales | Train/test leakage checks, unrealistic priors |
| Probability axioms | Probabilistic classifiers, calibration |

---

## 7. Lab & PPT Alignment

- **Lab 1**: Display data & statistical summary (Python optional; exam is manual).
- Practice problems at end of **ISM_CS-1** and **ISM_CS-2** — at least one mid-sem question may be **direct from practice set** (faculty announcement).