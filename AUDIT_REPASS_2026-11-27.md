# Re-Audit — Newly Added / Updated Files (2026-11-27)

**Since last audit**: commits `fecb930` → `c079d1c` (MFML full tree, audit fixes, transcript enrichments, DNN M12 regularisation, Stats headers/ANOVA/Holt, Kaptein real PDF).

**Method**: `module_map.yaml` path validation (local), `git ls-files` (Streamlit Cloud), spot-check of pass-1 gaps.

---

## Executive verdict (revised)

| Course | Descriptive + QA in git | Local map links | Cloud hub (video/slides/transcripts) | Exam notes |
| :--- | :---: | :---: | :---: | :--- |
| **ML** | ✅ | **0 broken** | ⚠️ **28/28** hub refs not in git | Ready (pass 5) |
| **DNN** | ✅ | **0 broken** | ⚠️ **31/31** hub refs not in git; M12 **2/3** PDFs in git | Ready; M12 regularisation deck wired |
| **Stats** | ✅ | **0 broken** | ⚠️ **34/34** hub refs not in git | Ready; M04 ANOVA + M05 Holt added |
| **MFML** | ✅ (42 tracked under `Notes/`) | **0 broken** | ⚠️ **55/55** hub refs not in git | **Strong**; EC paper index in QA |

> **Cloud reality**: `Inbox/` and `Videos/` are **gitignored**. The module hub works **fully on your NVME workspace** after Drive sync; on **Streamlit Cloud** you get **markdown + committed PDFs only** (e.g. ML handwritten slides, DNN `Notes/Lecture Slides & Companions/*.pdf`, Stats/DNN books), not inbox videos or annotated decks.

---

## MFML (largest new drop) — `audit_MFML_pass2.md`

| Check | Pass 1 (Nov 23) | Pass 2 (Nov 27) |
| :--- | :--- | :--- |
| Modules M01–M14 | ✅ | ✅ (+20–50 lines §7 faculty per module) |
| `**Syllabus module**` headers | ❌ | ✅ all 14 |
| §7 Faculty Pedagogy block | ❌ | ✅ all 14 |
| `module_map` slides/lectures | ❌ | ✅ 29 slides, 18 lecture blocks |
| Transcript cite in body | ❌ | ✅ via §7 + headers |
| QA EC-2/EC-3 cross-index | — | ✅ §4 Master Cross-Validation Index |
| Portal `descriptive_note` | ⚠️ | ✅ Streamlit Tab 2 loads all modules |
| Books T1,R1,R2 | ✅ | ✅ |
| **Line growth** | ~177–272/mod | **~198–298/mod** (M14 largest) |

**Residual**: Tier 3 QA vs every sub-part of official PDF keys not machine-verified; manual spot-check still advised for M10 KKT branches.

---

## Statistics — `audit_Stats_pass2.md`

| Check | Pass 1 | Pass 2 |
| :--- | :--- | :--- |
| `**Syllabus module**` (all 6) | ❌ | ✅ |
| `**Canonical video**` in headers | ❌ | ✅ |
| M04 one-way ANOVA partition (§4.4) | ❌ | ✅ (~250 lines module) |
| M05 Holt / Holt–Winters (§5.4–5.5) | ⚠️ thin | ✅ |
| M02 Q8, M05 MA numeric | ✅ | ✅ |
| T1 Kaptein symlink | ❌ | ✅ **15.5 MB real file in git** |
| R1 Miller & Freund | ❌ | ❌ still missing |
| Krishna **transcripts** CS8+ in repo | ❌ | ❌ (CS8–15 **slides** in map only, local Inbox) |

---

## Deep Learning — delta since `audit_DNN_pass2.md`

- **M12** enriched with `DNN_M11_Regularisation` taxonomy (Xavier/He, DropBlock, BN+dropout) — `c079d1c`.
- **`module_map.yaml` M12**: `DNN_M11_Regularisation (1)_261010_111237.pdf` ✅ **in git**; annotated inbox copy ❌ not in git (optional on Cloud).

No regression in descriptive depth; M08–M11 prior pass still valid.

---

## Machine Learning

No module body changes since pass 5 (`4f35766` / `8da434c`). **Re-audit: pass 5 still authoritative.**

---

## Automated scores (2026-11-27)

```
Local resolve_course_path (all courses): 0 failures
MFML assets (../assets/*.png):           0 broken
Stats T1 on disk:                      regular file (not symlink)
MFML QA questions:                     50
Stats QA questions:                    50
MFML §7 Faculty blocks:                14/14 modules
```

---

## Honest correction vs `AUDIT_NEW_GENERATION_2026-11-23.md`

That doc (updated in `a919997`) claims Stats lecture ground “CS 1–15” and “0 broken” **portal-wide**. Pass 2 agrees on **local path integrity**, but:

1. **Transcript depth for Stats CS8–15** is still **deck/map only** unless you sync Inbox locally.
2. **“Comprehensive-ready” on Cloud** applies to **notes + QA + committed textbooks/slide PDFs**, not to **inline video/transcript** in the hub.

---

## Automation (2026-11-27 pass 3)

- `scripts/mirror_portal_assets.py` — copies Inbox **slides (PDF)** + **transcripts (MD)** → `Notes/AI Notes/portal_mirror/` and repoints `module_map.yaml`.
- `scripts/audit_verify.py` — writes `AUDIT_OPEN_POINTS.md`; **exit 0** when fixable checks pass.
- Re-run after Drive sync; **10m loop** documented in `AUDIT_AUTOMATION.md`.

## Remaining (non-automated)

1. **Stats R1** Miller & Freund — optional reference; add to `Books/` when available.
2. **Videos** — remain gitignored; use local `_materialized/` or Drive.

---

## Artifact index

| File | Purpose |
| :--- | :--- |
| `AUDIT_REPASS_2026-11-27.md` | This re-audit |
| `Semester 1/MFML/Notes/AI Notes/audit_MFML_pass2.md` | MFML detail |
| `Semester 1/Statistics/Notes/AI Notes/audit_Stats_pass2.md` | Stats detail |
| `AUDIT_NEW_GENERATION_2026-11-23.md` | Pass-1 → fix changelog (partially superseded) |