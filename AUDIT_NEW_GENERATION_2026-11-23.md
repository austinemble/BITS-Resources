# Cross-Course Audit — Newly Generated Notes (2026-11-23)

**Scope**: Content added or refreshed since **2026-11-17** — pass-5 ML/DNN/Stats fixes, **MFML M01–M14 full tree**, study roadmaps, portal PDF fix.

**Workflow lens**: `course-notes-generator` Phases 2–6 (books → notes + QA + assets).

---

## Executive verdict

| Course | Modules | QA | Phase 2 books | Lecture / transcript ground | Exam readiness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ML** (ZG565) | 11 ✅ | 58 Q ✅ | ✅ | Strong (Sharon/Pritee + webinar labels) | **Comprehensive-ready** |
| **DNN** (ZG511) | 12 ✅ | 50 Q ✅ | ✅ | Strong (Shakshi + Mithun numerics) | **Comprehensive-ready** |
| **Stats** (ZC418) | 6 ✅ | 50 Q ✅ | ⚠️ R1 missing; T1 symlink | Krishna CS1–7; CS8+ thin | **Ready w/ QA + M1–M5 focus** |
| **MFML** (ZC416) | 14 ✅ | 50 Q ✅ | ✅ T1,R1,R2 | **T1 + faculty artifacts**; **no transcript citations in module bodies** | **Strong theory/QA**; portal hub incomplete |

---

## Pass-5 delta audit (ML / DNN / Stats)

### Machine Learning — `audit_ML_pass5.md` still valid
- **Verified in repo**: MDL Q54 table (M05), cancer Q56 (M11), LWR Q51 (M06), `**Syllabus module**` on all 11 modules.
- **Open**: optional Q53 CART mirror; M09–M10 textbook-bridge only (labelled).

### Deep Learning — `audit_DNN_pass2.md` still valid
- No new descriptive edits since pass 2; M10 attention numeric + M11 ViT params + M12 Adam proof present.
- **Open**: optional extra numerics from `DNN_M5/M6_Numerical.pdf`.

### Statistics — extends `audit_Stats_pass1.md`
| Check | Status |
| :--- | :--- |
| M02 Bayes medical **Q8** table | ✅ Present |
| M05 **3-period MA** numeric | ✅ Present |
| `**Syllabus module**` headers | ❌ Still use **Primary Teacher / CS** style (unlike ML/DNN) |
| `module_map.yaml` paths | ✅ 0 broken refs (automated) |
| Keyframes | ⚠️ Mostly M02 asset; M3–M6 sparse |
| R1 Miller & Freund | ❌ Missing |

### Study roadmaps (`STUDY_ROADMAP.md` × 4)
- Internally consistent with handout dates (DNN/Stats **5 Dec**, ML **6 Dec**).
- Assume ~6 h/day split — realistic for crunch mode; MFML at 1 h/day is tight for 14 modules unless paired with QA only.

---

## MFML — new generation (`audit_MFML_pass1.md` for detail)

### What shipped (high quality)
- **14 descriptive modules** (~177–272 lines each; **3,013** lines total).
- **50-question QA bank** with BITS 2+2+2 mark structure; aligns with T1 chapter order.
- **18 PNG keyframes** under `assets/` (real images, ~300–400 KB each).
- **Syllabus.md**, `FACULTY_AUDIT.md`, `BOOKS_VERIFICATION.md`, `CURRICULUM_MAPPING.md` (CS1–16 → modules).
- **Books gate**: all prescribed texts on disk.

### Gaps & risks

| Issue | Severity | Recommendation |
| :--- | :---: | :--- |
| `AUDIT_INGESTION_STATUS.md` says “M01 only” | Low (doc stale) | Update to reflect M01–M14 |
| **No `lectures` / `slides` in `module_map.yaml`** | **Medium** | Portal module hub shows notes only — add companion PDF paths per `CURRICULUM_MAPPING.md` |
| **Zero transcript/Inbox refs in module markdown** | Medium | Cross-check numerics vs Mandar `Problem-KKT.pdf` / past papers; add `**Contact sessions**` header block like ML |
| CS **13** spans M09–M11; map lists M09 & M11 both `[13]` | Low | Document in map; not a content error |
| CS **15** spans M12–M13 | Low | Same |
| Duplicate early scaffold `Module_01_Linear_Systems_Gauss_Elimination` | — | **Removed**; canonical name is `…_Gaussian_Elimination.md` |
| Videos in `Inbox/` (gitignored) | Low | Materialize + map if hub videos needed |

### Spot-check content quality
- **M01**: Gauss elimination, rank, null-space direction vectors — matches FACULTY midterm emphasis ✅
- **M10**: KKT + Mandar practice-set numeric block ✅
- **M14**: SVM primal/dual/kernel — ties to T1 Ch 12 + R2 ✅
- **Ground-truth risk**: modules read **textbook-synthesised**; prioritize verifying Q41–50 against `EC-2/EC-3` solutions in Inbox when cramming.

---

## Portal / GitHub hygiene (affects “notes” usability)

| Item | Status |
| :--- | :--- |
| PDF render (`st.pdf`) | ✅ Fixed commit `8da434c` |
| Stats T1 Kaptein symlink on clone | ⚠️ Copy real file into `Books/` for Cloud |
| MFML in Streamlit course picker | ✅ Listed; hub thin without map slides |

---

## Recommended fix order (if continuing ingestion)

1. **MFML**: enrich `module_map.yaml` with slides + transcript paths from `CURRICULUM_MAPPING.md`.
2. **Stats**: add `**Syllabus module**` headers + optional M4 ANOVA numeric; transcribe CS8+.
3. **MFML**: refresh `AUDIT_INGESTION_STATUS.md`; optional transcript footers per module.
4. **Kaptein R1/R2 Stats books** on GitHub as real files.

---

## Artifact index

| Course | Audit file |
| :--- | :--- |
| ML | `Semester 1/Machine Learning/Notes/AI Notes/audit_ML_pass5.md` |
| DNN | `Semester 1/Deep Learning/Notes/AI Notes/audit_DNN_pass2.md` |
| Stats | `Semester 1/Statistics/Notes/AI Notes/audit_Stats_pass1.md` |
| MFML | `Semester 1/MFML/Notes/AI Notes/audit_MFML_pass1.md` |