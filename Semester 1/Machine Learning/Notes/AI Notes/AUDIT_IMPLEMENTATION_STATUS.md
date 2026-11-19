# Audit Implementation Status — AIML ZG565 AI Notes

**Last updated**: 2026-11-19 (pass 4 — module-wise portal metadata)  
**Source audits**: `audit_M01_M05.md`, `audit_M06_M11.md`, `audit_QA_and_resources.md`

## Status: **Modules 01–11 aligned with audits**

| Module | Lecture / source alignment | Exam content |
| :--- | :--- | :--- |
| **M01** | Sessions 2 + CS1 deck | $\langle T,P,E\rangle$, splits, no Python |
| **M02** | Sessions 3–4 + CS2 | Stevens, IQR, scaling, vectorization |
| **M03** | Sessions 3–6 | NE vs GD, meal–tip, regularization, $R^2$ |
| **M04** | Sessions 5–6 | Logistic GD, ROC mnemonics; LDA → supplement |
| **M05** | Sessions 6–8, slide ~45 scope | Play Tennis + continuous Gini; MDL; manual trees |
| **M06** | Session 10 | Dissimilarity, Jaccard, Gower, LWR GD, mixed distance |
| **M07** | Sessions 11, 11b, 12 | Margin/slack/KKT; analytical 3-point dual |
| **M08** | Sessions 12–13 | MLE/MAP, NB smoothing, Gibbs, log-space NB |
| **M09** | Webinar 2 + textbook bridge | MDI, GridSearchCV; theory labelled |
| **M10** | Textbook bridge only | Bishop/Tan study path stated |
| **M11** | Session 10 + Webinar 2 | 3-way split, paradox, macro/weighted F1 |
| **QA / formulas** | `audit_QA_and_resources.md` | Part VIII Q51–58; operational rules §3 |

## Conventions applied repo-wide

- Python blocks renamed **Implementation note** where exams are manual (M04–11 except prior M02/M03).
- Non-lecture material labelled: textbook bridge, supplement, or advanced.

## Pass 4 (module-wise notes)

- Added `module_map.yaml` — Streamlit primary key = **M01–M11**, not contact session.
- All descriptive headers: **Syllabus module** + canonical `Videos/Lecture_*.mp4`.
- **M11**: cancer-screening accuracy paradox (Exam Q56) in descriptive notes.
- **M06**: LWR Problem Type 3 / Q51 worked block in descriptive notes.
- Formula sheet: Fisher LDA marked supplement.

## Optional future enhancements (not blocking)

- Mirror full `KNN-BITS-Problem3` gadget LWR table if source PDF is transcribed.