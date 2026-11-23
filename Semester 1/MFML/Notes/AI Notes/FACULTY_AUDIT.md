# Faculty Audit & Stream Evaluation — AIML ZC416

**Course**: Mathematical Foundations for Machine Learning (AIML ZC416 / AIML ZG514)  
**Academic Cycle**: Semester 1 (S1-26)  
**Audit Date**: 2026-11-23

---

## 1. Stream Recommendation & Strategy

| Role | Faculty | Rationale | Key Artifacts in Repo |
| :--- | :--- | :--- | :--- |
| **Primary Stream** | **Prof. Mandar Vijay Datar** | Course Lead; author of official annotated lecture slides, learning exercises, and past mid-semester question papers with step-by-step model solutions. Highest adherence to exam grading patterns. | `Annotated Slides/`, `Past Qu paper/`, `Read and Practice/` |
| **Secondary Companion** | **Prof. Saurabh** | In-depth author of dedicated 50-page companion readers for every single lecture (`Lecture_1_Companion` to `Lecture_13_Companion`), exceptional geometric and visual intuition from first principles. | `Companion_Doc/` (14 companion PDF readers) |
| **Exam Reference** | **Prof. Michael Alphonse** | Exam setter/evaluator for EC-2 and EC-3 tests; provides official solution keys for makeup and regular papers. | `EC2M_AIMLCZC416.pdf`, `EC3M_AIMLCZC416.pdf` |
| **Supplementary Stream** | **Prof. Upendar Mendu** | Clean typed session notes (`Lecture_*_Notes_NSP1_2026.pdf`), explicit vector space counterexamples, and PCA derivations. | `Lecture_*_Notes_NSP1_2026.pdf` |

---

## 2. Multi-Faculty Evaluation Matrix

| Faculty Name | Total Lectures | Word Count Avg | Annotated Slides | Companion Readers | Past Exams / Practice Sets | Technical Depth Score (1–10) | Exam Utility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prof. Mandar Vijay Datar** | 20+ | 3,850 | ✅ Yes | 🟡 Reference sets | ✅ **Past Midterm & Solutions** | **9.8** | **High (Primary)** |
| **Prof. Saurabh** | 20+ | 4,200 | ✅ Yes | ✅ **Full 14 Readers** | ✅ KKT / PCA Sets | **9.7** | **High (Companion)** |
| **Prof. Upendar Mendu** | 20+ | 3,400 | ✅ Yes | 🟡 Class Notes | 🟡 Practice Examples | **8.8** | Medium |
| **Prof. Michael Alphonse** | 20+ | 3,100 | 🟡 Standard | 🟡 Standard | ✅ **EC-2 / EC-3 Papers** | **9.0** | **High (Exam Keys)** |
| **Prof. Deepak G. Skariah** | 20+ | 2,950 | 🟡 Standard | 🟡 Standard | 🟡 Standard | **8.0** | Medium |
| **Prof. G. Venkiteswaran** | 18+ | 2,800 | 🟡 Notebooks | 🟡 Standard | 🟡 Standard | **7.8** | Low |
| **Prof. Madhusudhanan B.** | 20+ | 3,100 | 🟡 Standard | 🟡 Standard | 🟡 Standard | **8.1** | Medium |
| **Prof. Navneet Goyal** | 22+ | 3,300 | 🟡 Standard | 🟡 Standard | 🟡 Standard | **8.2** | Medium |
| **Prof. Nedungadi Aatira Gopalkrishnan** | 20+ | 3,050 | ✅ Classnotes | 🟡 Standard | 🟡 Standard | **8.3** | Medium |

---

## 3. Pedagogical & Exam Takeaways

1. **Gaussian Elimination & Ranks (Closed-Book Midterm)**:
   - Prof. Mandar Datar emphasizes expressing the complete solution as a **particular vector plus a scalar linear combination of direction vectors** (spanning the null space). Omitting the direction vector formulation results in immediate mark deductions on BITS tests.
2. **Inner Products & Matrix Norms**:
   - Both Mandar Datar and Michael Alphonse test non-standard inner products defined by symmetric positive definite matrices: $\langle x, y \rangle_A = x^\top A y$. Students must explicitly test $x^\top A x > 0$ for all $x \ne 0$ using determinant subminors (Sylvester's criterion) before computing induced angles or Gram-Schmidt orthonormalization.
3. **KKT Conditions & Inequality Constraints**:
   - Mandar Datar's practice problem set (`Problem-KKT.pdf`) establishes the mandatory university case-by-case analysis protocol ($\mu_i = 0$ vs $\mu_i > 0$). Every branch must be analytically tested for primal and dual feasibility violations.
4. **PCA & SVM Soft Margin**:
   - In comprehensive exams (EC-3), students are required to compute sample covariance matrices $S = \frac{1}{n-1} \tilde{X}^\top \tilde{X}$ by hand on small $3 \times 2$ matrices, find eigenvalues analytically, and formulate SVM primal soft-margin objective values under given $w$ and $b$.
