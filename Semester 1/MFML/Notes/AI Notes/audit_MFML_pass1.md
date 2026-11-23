# MFML Notes Audit (Pass 1 → Refresh 2026-11-23)

**Course**: AIML ZC416 · **Modules M01–M14** · **50-QA Bank** · **Master Formula Sheet**  
**Primary Stream**: Prof. Mandar Vijay Datar | **Companion Stream**: Prof. Saurabh | **Exam Reference**: Prof. Michael Alphonse

## Verdict: **Comprehensive-Ready & Exam-Verified** (All Audit Points Resolved)

| Module | Lines | Assets | Solved Numerics | Syllabus Module Header | Portal Wired (Slides & Videos) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M01 Linear Systems** | 248 | 2 PNG | ✅ REF/Gauss (Midterm Regular Q1, Q2) | ✅ Standardized | ✅ CS 1–2 slides + videos |
| **M02 Vector Spaces** | 212 | 1 PNG | ✅ Basis/Dim (Midterm Regular Q3, EC-2 Q3) | ✅ Standardized | ✅ CS 3–4 slides + videos |
| **M03 Inner Products** | 241 | 2 PNG | ✅ SPD/Gram-Schmidt (EC-2 Q5, EC-3M Q2) | ✅ Standardized | ✅ CS 5–6 slides + videos |
| **M04 Eigenvalues** | 182 | 2 PNG | ✅ Diagonalization (Regular Q5, EC-3M Q1) | ✅ Standardized | ✅ CS 7–8 slides + videos |
| **M05 SVD** | 206 | 2 PNG | ✅ SVD $3 \times 2$ (EC-2M Q3) | ✅ Standardized | ✅ CS 9 slides + video |
| **M06 Vector Calculus** | 206 | 1 PNG | ✅ Jacobian/Hessian (Regular Q6, EC-2M Q5) | ✅ Standardized | ✅ CS 10 slides + video |
| **M07 Auto-Diff** | 218 | 1 PNG | ✅ Backprop Graph (EC-2 Q6) | ✅ Standardized | ✅ CS 11 slides + video |
| **M08 Taylor Series** | 207 | 1 PNG | ✅ Multivariable Taylor (Regular Q7, EC-2 Q7) | ✅ Standardized | ✅ CS 12 slides + video |
| **M09 Gradient Descent**| 208 | 1 PNG | ✅ Binary Line Search (EC-3M Q3) | ✅ Standardized | ✅ CS 13 slides + video |
| **M10 KKT Conditions** | 251 | 1 PNG | ✅ Mandar KKT 4-Case Problem (`Problem-KKT`) | ✅ Standardized | ✅ CS 14 slides + problem set |
| **M11 Optimizers** | 181 | 1 PNG | ✅ Momentum Updates (EC-3M Q4) | ✅ Standardized | ✅ CS 13 slides + video |
| **M12 PCA** | 228 | 1 PNG | ✅ 2D Covariance Eigenvectors (EC-3M Q5) | ✅ Standardized | ✅ CS 15 slides + video |
| **M13 Low-Rank SVD** | 179 | 1 PNG | ✅ Eckart–Young Approximation | ✅ Standardized | ✅ CS 15 slides + video |
| **M14 SVM & Kernels** | 274 | 2 PNG | ✅ Soft-Margin / Dual (EC-3M Q6) | ✅ Standardized | ✅ CS 16 slides + video |
| **QA Bank (50 Q)** | 923 | — | Model 2+2+2 + **Master Exam Index** | ✅ Aligned | ✅ All EC-2 & EC-3 papers |
| **Formula Sheet** | 215 | 6 Diags | 14-Module Formulas + Decision Trees | ✅ Aligned | ✅ Viva defense checklist |

---

## Workflow Phase Verification Checklist

| Phase | Status | Verification Details |
| :--- | :---: | :--- |
| **1 Syllabus** | ✅ | `AIML_ZC416_COURSE_HANDOUT.md` & `Syllabus.md` complete |
| **2 Books** | ✅ | `BOOKS_VERIFICATION.md` — T1, R1, R2 on disk in `Books/` |
| **3 Faculty** | ✅ | `FACULTY_AUDIT.md` — Prof. Mandar Datar, Prof. Saurabh, Prof. Michael Alphonse |
| **4 Curriculum** | ✅ | `CURRICULUM_MAPPING.md` (CS 1–16 session breakdown) |
| **5 Keyframes** | ✅ | 19 PNGs under `assets/`, high-res 200 DPI vector extracts, all resolve |
| **6 Dual Notes** | ✅ | 14 Descriptive Modules + 50-QA Bank + Master Formula Sheet |
| **7 Assets** | ✅ | Companion PDFs and problem sets in `Inbox/` fully wired to `module_map.yaml` |
| **8 Portal** | ✅ | `module_map.yaml` enriched with `descriptive_note`, `slides`, `lectures` (0 broken links verified) |

---

## Audit Points Resolved This Pass

1. **`module_map.yaml` Portal Wiring**:
   - Fixed module mapping schema: replaced `file:` with `descriptive_note:` so Streamlit Tab 2 displays the note content.
   - Enriched all 14 modules with companion slide decks (`Annotated Slides/`) and lecture recordings (`Recordings/`).
   - Verified automated path validation via `portal_lib.py`: **0 broken links**.
2. **Standardized Syllabus Module Headers**:
   - Updated all 14 descriptive modules with canonical `**Syllabus module**`, `**Canonical video**`, and `**Contact sessions**` blocks, matching ML and DNN conventions.
3. **Official BITS Exam Cross-Validation**:
   - Added Master Cross-Validation Index to `50_Exam_Question_Answer_Bank.md` mapping every problem on `EC-2R`, `EC-2M`, `EC-3R`, and `EC-3M` directly to the QA bank and descriptive modules.
4. **Ingestion Status Document Refreshed**:
   - Updated `AUDIT_INGESTION_STATUS.md` to reflect all 8 phases completed.
