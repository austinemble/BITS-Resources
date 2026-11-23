# Mathematical Foundations for ML (ZC416) — Cram Roadmap from Zero

**Status**: AI descriptive notes **not complete** — use Mandar transcripts (CS1–8 linear algebra) + `Inbox/` slides first.  
**Role**: Backbone for ML/DNN — study **in parallel** with ML module order below.

## Course spine (from Mandar + `session_map` keywords)

| Block | Contact sessions (approx) | Topics |
| :--- | :--- | :--- |
| **Linear algebra** | 1–8 | Systems, REF/RREF, rank, Gauss elimination, vector spaces, eigen/SVD intro |
| **Calculus / opt** | 9–14 | Gradients, Taylor, constrained opt (Lagrange), SGD, momentum/Adam |
| **ML links** | 12–16 | PCA, SVM/KKT (`Lecture_14.pdf`, `Problem-KKT.pdf`) |

## 17-day plan (~1 h/day until ML/DNN comps)

| Week | Focus | Source |
| :--- | :--- | :--- |
| **W1** | Linear systems → REF/RREF → rank → unique/infinite/no solution | `Notes/AI Notes/01_Descriptive_Notes/Module_01_*.md`; Mandar `20260801` transcript |
| **W1–2** | Matrix mult (non-commutative), transpose, special matrices | Same + practice echelon forms |
| **W2** | Eigenvalues, SVD intuition (for PCA later) | Mandar later sessions; session_map CS4–5 |
| **W2–3** | Gradients, chain rule, backprop matrix view | Tie to **ML M03** + **DNN M05** |
| **W3** | Lagrange/KKT | `Inbox/.../Problem-KKT.pdf`, `Lecture_14.pdf` |
| **Ongoing** | Every ML study day: 15 m linear algebra drill (one REF or one $Ax=b$) | — |

## If MFML exam is same window as ML

Add in **week 3**: full KKT template + PCA variance ratio problem from slides.

## Next ingestion (repo)

1. Add `Syllabus.md` from WILP handout (not in repo yet).
2. Build `Notes/AI Notes/` modules M01–M16 from transcripts (M01 started).
3. `module_map.yaml` when syllabus CS list is fixed.

## First module note

`Notes/AI Notes/01_Descriptive_Notes/Module_01_Linear_Systems_Gauss_Elimination.md`