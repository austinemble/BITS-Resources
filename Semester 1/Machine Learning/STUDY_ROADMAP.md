# Machine Learning (ZG565) — Cram Roadmap from Zero

**Comprehensive**: **6 Dec 2026 (AN)** · Open book · 2½ h · 40%  
**Today**: 19 Nov 2026 · **~17 days** · Assume **~1.5 h/day** ML (rest split across other courses).

## What “pass” looks like (open book)

- Write **manual** numerics: decision trees (entropy/Gini), k-NN/LWR distances, linear/logistic GD steps, SVM margin/dual, NB log-posterior, evaluation metrics (confusion, macro-F1, cancer paradox).
- Use `ML_Master_Formula_Sheet.md` + QA bank **Part VIII (Q51–58)** as daily drills.

## Phase 0 — Day 1 (orientation, 2 h)

| Block | Do this |
| :--- | :--- |
| 30 m | Skim `Syllabus.md` module list (Part A). |
| 45 m | Read **M01** + **M02** (problem formulation, splits, scaling). |
| 45 m | QA: Q1–Q5 (write answers, no peeking first). |

## Phase 1 — Days 2–7 (core supervised, ~1.5 h/day)

| Day | Module notes | QA drill | Must memorise |
| :--- | :--- | :--- | :--- |
| 2 | M03 Regression | Q10–15 | GD update, $R^2$, regularization |
| 3 | M04 Classification | Q16–20 | Sigmoid, log-loss steps |
| 4 | M05 Trees + MDL | Q21–28, **Q54** | Play Tennis entropy, MDL table |
| 5 | M06 Instance learning | **Q51**, Q29–32 | Jaccard, LWR gradient |
| 6 | M07 SVM | Q33–38 | Soft margin, 3-point dual |
| 7 | M08 Bayes | Q39–44 | Laplace, log-sum trick |

## Phase 2 — Days 8–11 (ensemble + eval)

| Day | Module | QA | Note |
| :--- | :--- | :--- | :--- |
| 8 | M09 (theory-labelled) | Q45–48 | MDI, CV; skip deep AdaBoost proof if short on time |
| 9 | M11 Metrics | **Q56–58** | Paradox, precision/recall |
| 10 | M10 (textbook bridge) | — | 30 m Bishop/Tan skim only |
| 11 | Mixed paper | Q49–50 + random 6-mark | 2 h timed |

## Phase 3 — Days 12–16 (exam simulation)

- **Day 12–13**: Full 50Q bank — open book, 3 h per half.
- **Day 14**: Weak-list only (from mistakes).
- **Day 15**: Formula sheet blank → refill from memory, then check.
- **Day 16 (5 Dec eve)**: M05 + M06 + M11 numerics once; sleep.

## Phase 4 — Exam day

- Pack: printed formula sheet, syllabus module index, 2–3 solved tree/SVM/NB pages you wrote by hand.
- First 15 m: skim formula sheet + Q56–58 templates.

## If you only have 7 days total

Priority order: **M05 → M06 → M07 → M08 → M11 → M03 → M04 → M01 → M02 → M09 → M10**.

## Artifacts map

```
Notes/AI Notes/01_Descriptive_Notes/Module_0*.md
Notes/AI Notes/02_Question_Answer_Notes/ML_Comprehensive_Exam_QA_Bank.md
Notes/AI Notes/ML_Master_Formula_Sheet.md
Notes/AI Notes/audit_ML_pass5.md
```