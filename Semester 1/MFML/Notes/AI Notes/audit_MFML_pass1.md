# MFML Notes Audit (Pass 1) — 2026-11-23

**Course**: AIML ZC416 · **Modules M01–M14** · **50-QA bank** · Primary stream: **Prof. Mandar Datar**

## Verdict: **Strong for closed-book midterm + open-book comp** — verify Tier 3 QA against official solution keys

| Module | Lines | Assets | Solved numerics | Lecture transcript cited in note? |
| :--- | :---: | :---: | :---: | :---: |
| M01 Linear systems | 246 | 2 PNG | ✅ REF/Gauss | ❌ (T1 + companions) |
| M02 Vector spaces | 210 | 1 | ✅ | ❌ |
| M03 Inner products | 239 | 2 | ✅ SPD / Gram-Schmidt | ❌ |
| M04 Eigen | 180 | 2 | ✅ diagonalization | ❌ |
| M05 SVD | 204 | 2 | ✅ | ❌ |
| M06 Vector calculus | 204 | 1 | ✅ Jacobian/Hessian | ❌ |
| M07 Auto-diff | 216 | 1 | ✅ backprop graph | ❌ |
| M08 Taylor | 205 | 1 | ✅ | ❌ |
| M09 GD | 206 | 1 | ✅ | ❌ |
| M10 KKT | 249 | 1 | ✅ **Problem-KKT style** | ❌ (PDF named in header) |
| M11 Adam family | 179 | 1 | ✅ | ❌ |
| M12 PCA | 226 | 1 | ✅ 3×2 covariance | ❌ |
| M13 Low-rank | 177 | 1 | ✅ Eckart–Young | ❌ |
| M14 SVM | 272 | 2 | ✅ soft margin / dual | ❌ |
| **QA** | 50 Q | — | Model 2+2+2 | Faculty named in bank header |

## Workflow phase checklist

| Phase | Status |
| :--- | :--- |
| 1 Syllabus | ✅ `Syllabus.md` |
| 2 Books | ✅ `BOOKS_VERIFICATION.md` |
| 3 Faculty | ✅ `FACULTY_AUDIT.md` |
| 4 Curriculum | ✅ `CURRICULUM_MAPPING.md` (CS1–16) |
| 5 Keyframes | ✅ 18 PNGs, all referenced paths resolve |
| 6 Dual notes | ✅ Descriptive + QA + formula sheet |
| 7 Assets | 🟡 Companion PDFs in `Inbox/`, not wired to `module_map` |
| 8 Portal | 🟡 Course listed; **no slides/lectures in `module_map.yaml`** |

## `module_map.yaml` contact-session notes

- **CS 13** covers **M09–M11** (optimizers block) — modules M09 and M11 both tagged `[13]` intentionally.
- **CS 15** covers **M12–M13** — both tagged `[15]`.
- **CS 14** → M10 only (Lagrange/KKT).

## Exam-critical faculty rules (from audit)

1. **M01**: solution = particular + span of direction vectors (null space).
2. **M03**: non-standard inner product $x^\top A y$ — prove SPD before orthonormalization.
3. **M10**: case split $\mu_i = 0$ vs $\mu_i > 0$ for every inequality constraint.
4. **M12–M14**: hand covariance → eigenvalues; SVM dual with given $(w,b,C)$.

## Blocking vs optional

| Priority | Item |
| :--- | :--- |
| **Should fix** | Wire `module_map.yaml` slides/transcripts for Streamlit hub |
| **Should fix** | Cross-validate QA Q41–50 vs `Inbox/.../EC-2M`, `EC-3M` keys |
| Optional | Per-module Mandar transcript citations |
| Optional | M15/M16 only if syllabus adds contact sessions beyond 16 |

## Compared to prior scaffold (2026-11-19)

The old single-page `Module_01_Linear_Systems_Gauss_Elimination.md` (~120 lines) is **superseded** by the full M01 note (246 lines) + QA bank — do not maintain both.