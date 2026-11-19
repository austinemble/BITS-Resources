# Module 5: Correlation, Regression, and Time Series

**Course**: AIML ZC418 — Introduction to Statistical Methods  
**Prescribed References**: T1 Kaptein et al.; T2 Devore (correlation & simple regression); **T3 Brockwell & Davis** (*Introduction to Time Series and Forecasting*, 2nd ed. — primary for CS-12–14)  
**Primary Teacher**: Prof. G. Krishna Kumari (Lead ISM)  
**Lecture alignment**: ISM Contact Sessions **CS-11 to CS-14** (`ISM_CS-11` correlation/regression; `ISM_CS-12` components & MA/WMA; `ISM_CS-13` AR/ARMA/ARIMA; `ISM_CS-14` SARIMA, VAR family, exponential smoothing)

> **Exam note (Lead Instructor)**: **Closed book** — correlation formula, regression line coefficients, MA/WMA weights, AR(1) and ARIMA \((p,d,q)\) definitions, smoothing level \(\alpha\). **Open book** — diagnose series (trend/seasonality), interpret ACF/PACF, justify model order, evaluate forecasts.

---

## 1. Executive Summary & Foundational Definitions

**Correlation** measures linear association between two variables; **simple linear regression** models \(Y\) as a linear function of \(X\) with least-squares fit. **Time series** \(\{Y_t\}\) are ordered observations where dependence across time matters; standard decompositions separate **trend**, **seasonality**, and **irregular** components.

**Smoothing** (MA, WMA, exponential) produces short-horizon forecasts for smooth series. **AR/MA/ARIMA/SARIMA** model autocorrelation structure; **VAR** extends to multivariate series. These models underpin demand forecasting, anomaly detection, and sequential features in AIML pipelines.

```
Cross-section: (X,Y) pairs ──> r, ŷ = β₀ + β₁x
Time index t: Y_t ──> decompose ──> smooth / ARMA / ARIMA(p,d,q) ──> forecast Ŷ_{t+h}
```

---

## 2. Correlation

### 2.1 Pearson correlation coefficient

For pairs \((x_i,y_i)\), \(i=1,\ldots,n\):

$$r=\frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^n (x_i-\bar{x})^2}\sqrt{\sum_{i=1}^n (y_i-\bar{y})^2}}=\frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}$$

**Range**: \(-1\le r\le 1\). \(r=\pm 1\) iff perfect linear relationship.

**Formal property**: scale-invariant; **does not** imply causation.

### 2.2 Inference (large \(n\))

Test \(H_0:\rho=0\) vs \(H_1:\rho\neq 0\). For bivariate normal data, under \(H_0\):
$$t=\frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t_{n-2}$$

---

## 3. Simple Linear Regression

### 3.1 Model

$$Y_i=\beta_0+\beta_1 X_i+\varepsilon_i, \quad \varepsilon_i \stackrel{\text{i.i.d.}}{\sim}\mathcal{N}(0,\sigma^2)$$

### 3.2 Least-squares estimators

$$\hat{\beta}_1=\frac{S_{xy}}{S_{xx}}=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}, \qquad \hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}$$

Fitted line: \(\hat{y}=\hat{\beta}_0+\hat{\beta}_1 x\).

### 3.3 Goodness of fit

**Coefficient of determination**:
$$R^2=1-\frac{SS_{\text{res}}}{SS_{\text{tot}}}=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$$

For simple regression, \(R^2=r^2\).

**Residuals** \(e_i=y_i-\hat{y}_i\) — used to check linearity and constant variance (comprehensive emphasis).

---

## 4. Time Series Components

A classical **additive** decomposition:
$$Y_t = T_t + S_t + R_t$$

- **Trend** \(T_t\): long-run level change.
- **Seasonal** \(S_t\): periodic pattern (period \(s\), e.g. 12 months).
- **Irregular** \(R_t\): random noise.

**Multiplicative** form: \(Y_t=T_t\cdot S_t\cdot R_t\) when seasonal amplitude scales with level (CS-12).

**Stationarity**: mean and variance constant over time; autocovariances depend only on lag \(h\) — required for ARMA/ARIMA modeling after differencing.

---

## 5. Moving Averages and Smoothing

### 5.1 Simple moving average (MA) forecast

Order-\(m\) MA for forecasting at \(t+1\):
$$\hat{Y}_{t+1}=\frac{1}{m}\sum_{j=0}^{m-1} Y_{t-j}$$

Low-pass filter; larger \(m\) → smoother, slower to react.

### 5.2 Weighted moving average (WMA)

$$\hat{Y}_{t+1}=\sum_{j=0}^{m-1} w_j Y_{t-j}, \quad \sum_j w_j=1, \quad w_j\ge 0$$

Recent observations often get higher weights (PPT examples: \(w=[0.5,0.3,0.2]\)).

### 5.3 Simple exponential smoothing (SES)

One-parameter **level** \(\ell_t\):
$$\ell_t=\alpha Y_t+(1-\alpha)\ell_{t-1}, \quad 0<\alpha<1$$

One-step ahead forecast: \(\hat{Y}_{t+1}=\ell_t\). **Holt** (double) adds trend; **Holt–Winters** (triple) adds seasonality (CS-14).

---

## 6. AR, MA, ARMA, ARIMA, SARIMA (Overview)

### 6.1 Autoregressive AR\((p)\)

$$Y_t=\phi_1 Y_{t-1}+\cdots+\phi_p Y_{t-p}+\varepsilon_t, \quad \varepsilon_t\sim\mathcal{N}(0,\sigma^2)$$

**Stationarity**: roots of \(1-\phi_1 z-\cdots-\phi_p z^p=0\) outside unit circle.

### 6.2 Moving average MA\((q)\)

$$Y_t=\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}$$

### 6.3 ARMA\((p,q)\) and ARIMA\((p,d,q)\)

**ARMA**: \(Y_t\) mean-centered stationary series combining AR and MA parts.

**ARIMA**: apply \(d\) differences \(\nabla^d Y_t\) to achieve stationarity, then ARMA on differenced series:
$$\nabla Y_t=Y_t-Y_{t-1}, \quad \nabla^d \text{ removes polynomial trend of order } d$$

**Model box**: ARIMA\((p,d,q)\) — autoregressive order \(p\), integration \(d\), MA order \(q\).

### 6.4 Seasonal ARIMA SARIMA\((p,d,q)\times(P,D,Q)_s\)

Seasonal differencing at lag \(s\); seasonal AR/MA terms at lags \(s, 2s, \ldots\) (monthly \(s=12\)).

### 6.5 VAR / VARMAX (awareness)

**Vector AR**: multivariate extension — each series regressed on own and others’ lags. **VARMAX** adds exogenous covariates (e.g. promotions). Listed in syllabus CS-14; full estimation is open-book interpretive.

### 6.6 ACF and PACF (model identification)

| Model | ACF typical shape | PACF typical shape |
| :--- | :--- | :--- |
| AR\((p)\) | Tails off | Cuts off after lag \(p\) |
| MA\((q)\) | Cuts off after lag \(q\) | Tails off |

Exam focus: **pattern recognition**, not full Yule–Walker inversion by hand unless small \(p\).

---

## 7. AI / ML Bridge (Lead ISM Lectures)

| Tool | AIML use |
| :--- | :--- |
| Regression / \(R^2\) | Baseline supervised learner; feature–target relationships before complex models |
| Correlation | Multicollinearity screening; naive feature ranking (causation caution) |
| MA / SES | Baseline forecast competitions; demand pipelines (Prophet-style decomposition parallels) |
| ARIMA | Strong seasonal business series; residual diagnostics for LSTM/transformer inputs |
| Stationarity / differencing | Stabilizing neural training on raw price series |

**Lab alignment**: Lab 5 (regression summary), Lab 6 (time series) — Python `statsmodels` ARIMA/SARIMA fits mirror theory tables.

---

## 8. Solved Numerical Example (Exam Pattern)

**Problem Statement (BITS Pattern, ~6 Marks)**: Monthly sales (\(n=5\)): 120, 132, 128, 140, 150. (i) Compute WMA forecast for month 6 with weights \(w_0=0.5, w_1=0.3, w_2=0.2\) on the **three most recent** months. (ii) For SES with \(\alpha=0.4\) and \(\ell_4=130\) (given after month 4), update \(\ell_5\) and forecast month 6.

### Solution

Label months 1…5; most recent = 150 (month 5), then 140, 128.

**(i) WMA** (\(Y_5=150\), \(Y_4=140\), \(Y_3=128\)):
$$\hat{Y}_6=0.5(150)+0.3(140)+0.2(128)=75+42+25.6=\mathbf{142.6}$$

**(ii) SES**: \(\ell_5=\alpha Y_5+(1-\alpha)\ell_4=0.4(150)+0.6(130)=60+78=\mathbf{138}\).

One-step forecast \(\hat{Y}_6=\ell_5=\mathbf{138}\).

**Comment**: WMA reacts more to latest spike (150); SES level is between recent and smoothed history — model choice affects responsiveness vs. variance (comprehensive interpretation).

---

## 9. Formula Sheet (Closed-Book Priority)

| Item | Formula |
| :--- | :--- |
| Pearson \(r\) | \(S_{xy}/\sqrt{S_{xx}S_{yy}}\) |
| \(\hat{\beta}_1\) | \(S_{xy}/S_{xx}\) |
| \(\hat{\beta}_0\) | \(\bar{y}-\hat{\beta}_1\bar{x}\) |
| MA\((m)\) | \(\hat{Y}_{t+1}=\frac{1}{m}\sum_{j=0}^{m-1}Y_{t-j}\) |
| SES update | \(\ell_t=\alpha Y_t+(1-\alpha)\ell_{t-1}\) |
| AR(1) | \(Y_t=\phi Y_{t-1}+\varepsilon_t\) |

---

## 10. Common Pitfalls & Practical Guidelines

1. **Correlation with trend**: spurious high \(r\) on non-stationary series — difference or detrend first.
2. **Extrapolation**: regression outside \(X\) range unreliable.
3. **ARIMA orders**: overfitting \((p,d,q)\) on short series — use information criteria (AIC/BIC) in labs, justify in exams.
4. **Seasonal period \(s\)**: wrong \(s\) mis-specifies SARIMA.

**PPT cross-reference**: **ISM_CS-11** (correlation, regression ANOVA-style summary awareness); **ISM_CS-12/13/14** (MA, WMA, ACF/PACF, ARIMA, Holt–Winters).