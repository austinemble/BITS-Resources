# Module 1: Linear Systems, Matrices & Gauss Elimination

**Course**: AIML ZC416 — Mathematical Foundations for ML  
**Primary Teacher**: Prof. Mandar Datar — Contact Session 2 (transcript `20260801_103015`)  
**Block**: Linear algebra (CS 1–8)

---

## 1. Executive Summary

Real ML models (regression, deep nets, PCA) reduce to **linear algebra**: solving $A\mathbf{x}=\mathbf{b}$, row operations, **rank**, and **REF/RREF**. Mandar’s spine: geometry in 2D/3D → general $m\times n$ → **Gauss elimination** on augmented $[A\mid\mathbf{b}]$ → classify **consistent / inconsistent** and **unique / infinite** solutions.

---

## 2. Linear System

**Linear equation**: variables appear only to power 1 (e.g. $30x + 50y = 1000$).  
**System**: multiple equations in the same unknowns.

**Matrix form**:
$$A\mathbf{x} = \mathbf{b}$$

Example ($3$ equations, $3$ unknowns):
$$A=\begin{bmatrix}2&3&4\\1&-3&7\\2&1&5\end{bmatrix},\quad \mathbf{x}=\begin{bmatrix}x\\y\\z\end{bmatrix},\quad \mathbf{b}=\begin{bmatrix}7\\1\\8\end{bmatrix}$$

---

## 3. Geometric Intuition (2D)

Each equation = line. Possibilities:

| Geometry | Algebra |
| :--- | :--- |
| One intersection | **Unique** solution |
| Parallel lines | **No** solution |
| Coincident lines | **Infinite** solutions |

Same story in 3D with **planes**; generalise to $m$ equations, $n$ unknowns.

---

## 4. Elementary Row Operations (ERO)

On augmented matrix $[A\mid\mathbf{b}]$ only:

1. **Swap** rows $R_i \leftrightarrow R_j$
2. **Scale** row: $R_i \leftarrow c R_i$, $c\neq 0$
3. **Add multiple of one row to another**: $R_i \leftarrow R_i + c R_j$ (not “multiple × multiple” of two rows)

Matrix product **not commutative**: $AB \neq BA$ (and sometimes one product undefined).

---

## 5. Row Echelon Form (REF)

Scan **row-wise**. REF rules:

- All-zero rows at the **bottom**
- First non-zero per row = **pivot**; pivots shift **right** as you go down (staircase)
- Pivot columns: entries **below** pivot are 0 (achieved by ERO)

**Rank** $= \#$ non-zero rows in REF of $A$.

**Reduced REF (RREF)**: in each pivot column, **all** entries except pivot are 0 (pivot may be any non-zero; 1 is convenient).

---

## 6. Gauss Elimination Algorithm

1. Write $A\mathbf{x}=\mathbf{b}$
2. Form **augmented** matrix $[A\mid\mathbf{b}]$
3. ERO → REF (often swap row with 1 in pivot column for easier arithmetic)
4. Compare ranks:
   - $\operatorname{rank}([A\mid\mathbf{b}]) \neq \operatorname{rank}(A)$ → **inconsistent** (no solution)
   - Equal ranks and $\operatorname{rank}(A)=n$ → **unique**
   - Equal ranks and $\operatorname{rank}(A)<n$ → **infinite**
5. **Back substitution** on triangular system from REF

**Theorem**: $\operatorname{rank}(A) \le \min(m,n)$.

---

## 7. Mini Worked Rank Check

If REF of $[A\mid\mathbf{b}]$ has 3 non-zero rows and $A$ part also has rank 3 with $n=3$ unknowns → unique solution (then back-substitute for $z$, then $y$, then $x$).

---

## 8. ML Bridge

| Idea | Where used |
| :--- | :--- |
| $A\mathbf{x}=\mathbf{b}$ | Normal equations / linear regression |
| Rank deficiency | Multicollinearity, non-unique weights |
| REF / RREF | Gaussian elimination in numerics; RREF → **inverse** (Gauss–Jordan) |
| Pivoting | Numerical stability (LU with pivot) |

---

## 9. Practice (exam-style)

1. Bring $\begin{bmatrix}1&2&-4&3\\3&2&1&7\\2&1&5&8\end{bmatrix}$ to REF; state rank of $A$ and solution type.  
2. Multiply $2\times 3$ by $3\times 2$; explain why $BA$ size differs from $AB$.  
3. Distinguish REF vs RREF on a $3\times 4$ upper-triangular example.

**Source recording**: `Inbox/Prof. Mandar Datar/Recordings/...20260801_103015...md`