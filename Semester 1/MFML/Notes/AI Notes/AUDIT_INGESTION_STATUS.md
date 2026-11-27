# MFML (ZC416) — Ingestion & Audit Status

**Last updated**: 2026-11-27 (pass 2 re-audit — `audit_MFML_pass2.md`, `AUDIT_REPASS_2026-11-27.md`)

| Phase | Status | Artifact | Remarks |
| :--- | :---: | :--- | :--- |
| **1 Syllabus** | ✅ | `AIML_ZC416_COURSE_HANDOUT.md` & `Syllabus.md` | Handout, evaluation scheme (EC-1 15%, EC-2 35%, EC-3 50%), and full syllabus |
| **2 Books** | ✅ | `BOOKS_VERIFICATION.md` | All 3 prescribed open-access textbooks (T1, R1, R2) downloaded and verified in `Books/` |
| **3 Faculty audit** | ✅ | `FACULTY_AUDIT.md` | Primary lead: Prof. Mandar Vijay Datar; Companion: Prof. Saurabh; Exam reference: Prof. Michael Alphonse |
| **4 Curriculum map** | ✅ | `CURRICULUM_MAPPING.md` | Comprehensive session-by-session breakdown for all 16 contact sessions |
| **5 Keyframes & Assets**| ✅ | `assets/M01`–`M14` (19 PNGs) | High-res 200 DPI vector diagrams extracted from official 549-slide deck |
| **6 Dual notes** | ✅ | M01–M14 descriptive notes; 50-QA bank; master formula sheet | 14 textbook-grade descriptive modules with standardized `**Syllabus module**` headers |
| **7 Portal map** | ✅ | `module_map.yaml` | 100% clean (0 broken links): all 14 modules wired with `descriptive_note`, `slides`, and `lectures` |
| **8 Streamlit Hub** | 🟡 | `streamlit_app.py` & `portal_lib.py` | Notes/QA/books in git; **Inbox** video/slide hub refs **gitignored** on Cloud |

---

## Audit Verification & Ingestion Summary
- **Syllabus Module Convention**: Standardized across all 14 modules with `**Syllabus module**`, `**Canonical video**`, and `**Contact sessions**`.
- **Examination Cross-Validation**: Official BITS Pilani examinations (`EC2R`, `EC2M`, `EC3R`, `EC3M`) mapped directly to the 50-Question Bank and descriptive theory modules.
- **Portal Navigation**: **Local workspace**: 0 broken map paths. **GitHub/Streamlit Cloud**: mirror slide PDFs into `Notes/` or sync Drive — see `AUDIT_REPASS_2026-11-27.md`.
- **Pass 2 enrichment** (`45a79eb`): §7 Faculty Pedagogy on all 14 modules.
