# 🎓 BITS Pilani WILP — Academic Study Repository & Examination Portal

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Centralized academic repository and study portal for **BITS Pilani Work Integrated Learning Programmes (WILP)** — **M.Tech in Artificial Intelligence & Machine Learning (AI/ML)**.

---

## 🏛️ Curriculum Overview

### Semester 1
| Course Code | Course Title | Notes Status | Books & References | Question Bank |
| :--- | :--- | :---: | :---: | :---: |
| **AIML ZG565** | **Machine Learning** | ✅ Complete (11 Modules) | ✅ Verified (T1, R1, R2, R3) | ✅ 50Q Bank + Formula Sheet |
| **AIML ZG511** | **Deep Learning** | ✅ Complete (12 Modules) | ✅ Verified (T1, T3, Refs) | ✅ Comprehensive Bank |
| **AIML ZC416** | **Mathematical Foundations for ML** | ⏳ Ingestion Stage | ✅ Textbooks Preserved | ⏳ Syllabus Ready |
| **AIML ZC418** | **Introduction to Statistical Methods** | ⏳ Ingestion Stage | ✅ Textbooks Preserved | ⏳ Syllabus Ready |

---

## 📂 Repository Architecture

```text
BITS/Resources/
├── .gitignore                          # Excludes raw video streams and bulky Inbox dumps
├── README.md                           # Master repository documentation
├── requirements.txt                    # Python dependencies (Streamlit >= 1.30.0)
├── streamlit_app.py                    # Multi-course interactive web portal
├── app.py                              # Entrypoint symlink for Streamlit Community Cloud
└── Semester 1/
    ├── Deep Learning/                  # AIML ZG511
    │   ├── AIML ZG511 COURSE HANDOUT.md
    │   ├── Syllabus.md
    │   ├── Books/                      # Prescribed reference textbooks
    │   ├── Notes/
    │   │   ├── AI Notes/               # Descriptive notes & Q&A banks
    │   │   ├── Handwitten Notes/       # Tablet derivations
    │   │   ├── Lecture Slides & Companions/
    │   │   └── Question Papers & Solutions/
    │   └── Assignments/
    ├── Machine Learning/               # AIML ZG565
    │   ├── AIML_ZG565_COURSE_HANDOUT.md
    │   ├── Syllabus.md
    │   ├── Curriculum_Mapping_and_Faculty_Audit.md
    │   ├── Books/                      # Tom Mitchell, Bishop, Tan, Burges
    │   ├── Notes/
    │   │   ├── AI Notes/
    │   │   │   ├── 01_Descriptive_Notes/     # 11 Modular high-yield study guides
    │   │   │   ├── 02_Question_Answer_Notes/ # 50Q Exam Bank & Master Formula Sheet
    │   │   │   └── assets/                   # Lecture visual keyframes
    │   │   ├── Handwitten Notes/             # 8 annotated tablet derivation PDFs
    │   │   ├── Lecture Slides & Companions/  # Practice worksheets & companion code
    │   │   └── Question Papers & Solutions/  # 14 authentic practice question papers & solutions
    │   ├── Assignments/
    │   └── streamlit_app.py            # Standalone single-course runner
    ├── MFML/                           # Mathematical Foundations for ML
    │   ├── Books/
    │   ├── Notes/
    │   └── Assignments/
    └── Statistics/                     # Introduction to Statistical Methods
        ├── Books/
        ├── Notes/
        └── Assignments/
```

---

## 🚀 Interactive Streamlit Web Portal

This repository includes a mobile-friendly, multi-course web portal equipped with KaTeX math rendering, expandable exam question banks, and instant PDF downloads.

### Run Locally:
```bash
git clone https://github.com/aemmanuval/BITS-Resources.git
cd BITS-Resources
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy to Streamlit Community Cloud:
1. Navigate to [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub account (**aemmanuval**).
3. Select repository `aemmanuval/BITS-Resources`, branch `main`, and main file `streamlit_app.py`.
4. Click **Deploy**!

---

## 🔒 Media Policy & Git Hygiene
To maintain a high-speed, lightweight Git tree:
- **Strictly Excluded**: All raw recorded video containers (`*.mp4`, `*.mkv`, `*.avi`, `*.mov`) and raw meeting dump directories (`Inbox/`, `Videos/`) are excluded via `.gitignore`.
- **Preserved in Git**: All converted transcripts (`.md`), descriptive notes, textbook PDFs, handwritten tablet notes, assignment notebooks, and lecture keyframe assets.

