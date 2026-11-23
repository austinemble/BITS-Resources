# Statistics Notes Re-Audit (Pass 1 → Refresh 2026-11-23)

**Course**: AIML ZC418 — Introduction to Statistical Methods  
**Target Examination**: Comprehensive **5 Dec 2026** (Open Book)  
**Primary Stream**: Prof. G. Krishna Kumari (Lead ISM) | **Companion Faculty**: Prof. Vipin Khurana, Prof. Tathe Kartik Vilasrao

## Verdict: **Comprehensive-Ready & Exam-Verified** (All Audit Points Resolved)

| Module | Desk Depth | Gap Closed (Pass 1 Refresh) | Solved Numerics Present |
| :--- | :---: | :--- | :--- |
| **M01** | ✅ | Standardized `**Syllabus module**` header; Five-number summary | IQR, Tukey outlier fences |
| **M02** | ✅ | Standardized header; Medical Bayes Q8 table in descriptive | Disease screening sensitivity/specificity table |
| **M03** | ✅ | Standardized header; Distributions + table usage guidelines | Z-score, t, chi-square, F distribution properties |
| **M04** | ✅ | **Full One-Way ANOVA partition worked numeric** (Section 4.4) | 3-Machine shaft diameter test, F=11.40 vs F_crit=4.26 |
| **M05** | ✅ | **Holt Linear Trend & Holt-Winters Add/Mult equations** (CS-14 Excel sync) | 3-period MA, WMA, SES, Holt, Holt-Winters formulas |
| **M06** | ✅ | Standardized header; GMM density, responsibilities, EM steps | 2-component univariate GMM E-step / M-step updates |
| **QA Bank**| ✅ | 50Q exam-aligned bank covering M01–M06 | Complete 6-mark derivations and model answers |

---

## Audit Points Resolved This Pass

1. **Header Convention Gap Standardized**:
   - Replaced legacy `Primary Teacher / CS` headers in all 6 modules (M01–M06) with the canonical `**Syllabus module**`, `**Canonical video**`, and `**Contact sessions**` blocks matching ML, DNN, and MFML standards.
2. **One-Way ANOVA Partition Solved Example Added**:
   - Enriched `Module_04_Hypothesis_Testing_ANOVA_and_MLE.md` with Section 4.4 containing a full 6-mark worked numerical example: 3-group ANOVA, calculation of $G$, $CF$, $SS_{\text{total}}$, $SS_{\text{treat}}$, $SS_{\text{error}}$, Mean Squares, and hypothesis testing against $F_{0.05}(2, 9) = 4.26$.
3. **Streamlit Cloud Blocker Resolved (T1 Kaptein)**:
   - Replaced the absolute symlink in `Books/T1_Kaptein_Statistics_for_Data_Scientists.pdf` with the real 15MB PDF directly in the repository, ensuring complete portability on Streamlit Cloud without relying on untracked `Inbox/` folders.
4. **Holt & Holt–Winters Time Series Models Added**:
   - Updated `Module_05_Correlation_Regression_and_Time_Series.md` with explicit level, trend, and seasonal index recurrence equations for Holt's linear trend and Holt-Winters additive/multiplicative models calibrated via SSE minimization.
5. **Portal Map Validation**:
   - Verified `module_map.yaml`: **0 broken links** across all 6 modules, slides, and recordings.
