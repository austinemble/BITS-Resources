"""
BITS Pilani WILP — Universal Academic Resources & Learning Portal
Streamlit Web Application for Multi-Course Study Materials, Transcripts, Textbooks, and Exam Banks
"""

import os
import re
from pathlib import Path
import streamlit as st

from portal_lib import (
    load_course_modules,
    render_pdf_inline,
    render_pptx_outline,
    resolve_course_path,
    resolve_lecture_video,
)

# Configure page settings
st.set_page_config(
    page_title="BITS Pilani WILP Academic Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 0.375rem;
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 600;
    }
    .highlight-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin-bottom: 1.2rem;
    }
    .module-card {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


def discover_semesters_and_courses():
    """Dynamically scan for semester and course folders."""
    sem_dict = {}
    for sem_dir in sorted(ROOT_DIR.iterdir()):
        if sem_dir.is_dir() and sem_dir.name.startswith("Semester"):
            courses = []
            for course_dir in sorted(sem_dir.iterdir()):
                if course_dir.is_dir() and not course_dir.name.startswith("."):
                    courses.append(course_dir.name)
            if courses:
                sem_dict[sem_dir.name] = courses
    return sem_dict


def render_markdown_with_assets(md_text: str, base_course_dir: Path):
    """
    Renders markdown while replacing relative asset links with actual Streamlit images.
    """
    img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    pos = 0
    for match in img_pattern.finditer(md_text):
        st.markdown(md_text[pos:match.start()], unsafe_allow_html=True)
        alt_text = match.group(1)
        rel_path = match.group(2)
        
        # Resolve path
        clean_path = rel_path.lstrip("./").lstrip("/")
        candidate_paths = [
            base_course_dir / clean_path,
            base_course_dir / "Notes" / "AI Notes" / clean_path,
            base_course_dir / "Notes" / "AI Notes" / "assets" / Path(clean_path).name,
            base_course_dir / "assets" / Path(clean_path).name
        ]
        
        found = False
        for p in candidate_paths:
            if p.exists() and p.is_file():
                st.image(str(p), caption=alt_text, use_container_width=True)
                found = True
                break
        
        if not found:
            st.info(f"🖼️ *Visual Keyframe: {alt_text}*")
            
        pos = match.end()
    
    if pos < len(md_text):
        st.markdown(md_text[pos:], unsafe_allow_html=True)


# Discover available semesters and courses
SEMESTER_DATA = discover_semesters_and_courses()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=90)
    st.title("BITS Pilani WILP")
    st.caption("M.Tech AI & ML Study Materials & Exam Portal")
    st.markdown("---")
    
    if SEMESTER_DATA:
        selected_sem = st.selectbox("📅 Select Semester", list(SEMESTER_DATA.keys()))
        course_list = SEMESTER_DATA[selected_sem]
        
        # Course display labels
        course_display = {
            "Machine Learning": "🤖 AIML ZG565: Machine Learning",
            "Deep Learning": "🧠 AIML ZG511: Deep Learning",
            "MFML": "📐 Mathematical Foundations for ML",
            "Statistics": "📊 Introduction to Statistical Methods"
        }
        
        selected_course = st.selectbox(
            "📖 Select Course",
            course_list,
            format_func=lambda x: course_display.get(x, x)
        )
    else:
        st.error("No semester folders found.")
        st.stop()
        
    st.markdown("---")
    st.subheader("📚 Quick Links")
    st.markdown("""
    - [GitHub Academic Repository](https://github.com/aemmanuval/BITS-Resources)
    - [BITS WILP Student Portal](https://elearn.bits-pilani.ac.in)
    - [Streamlit Community Cloud](https://share.streamlit.io)
    """)
    st.info("💡 **Tip**: Math formulas are KaTeX rendered. You can read, search, or download any material directly.")


# Selected Course Paths
COURSE_DIR = ROOT_DIR / selected_sem / selected_course
NOTES_DIR = COURSE_DIR / "Notes" / "AI Notes"
DESCRIPTIVE_DIR = NOTES_DIR / "01_Descriptive_Notes"
QA_DIR = NOTES_DIR / "02_Question_Answer_Notes"
ASSETS_DIR = NOTES_DIR / "assets"
BOOKS_DIR = COURSE_DIR / "Books"
HANDWRITTEN_DIR = COURSE_DIR / "Notes" / "Handwitten Notes"
QP_DIR = COURSE_DIR / "Notes" / "Question Papers & Solutions"
COMPANIONS_DIR = COURSE_DIR / "Notes" / "Lecture Slides & Companions"
ASSIGNMENTS_DIR = COURSE_DIR / "Assignments"

# Main Page Header
disp_title = course_display.get(selected_course, selected_course)
st.markdown(f'<div class="main-title">{disp_title}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">BITS Pilani Work Integrated Learning Programmes — {selected_sem}</div>', unsafe_allow_html=True)

# Overview Badges
badges = []
if DESCRIPTIVE_DIR.exists():
    n_mods = len(list(DESCRIPTIVE_DIR.glob("*.md")))
    if n_mods > 0:
        badges.append(f"📖 {n_mods} AI Study Modules")
if QA_DIR.exists():
    n_qa = len(list(QA_DIR.glob("*.md")))
    if n_qa > 0:
        badges.append("🎯 Exam Q&A Banks")
if BOOKS_DIR.exists():
    n_bks = len(list(BOOKS_DIR.glob("*.pdf")))
    if n_bks > 0:
        badges.append(f"📚 {n_bks} Textbooks & Refs")
if HANDWRITTEN_DIR.exists():
    n_hw = len(list(HANDWRITTEN_DIR.glob("*.pdf")))
    if n_hw > 0:
        badges.append(f"✍️ {n_hw} Handwritten Notes")

if badges:
    badge_html = "".join([f'<span class="metric-badge">{b}</span>' for b in badges])
    st.markdown(f'<div style="margin-bottom: 1.5rem;">{badge_html}</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Curriculum & Syllabus",
    "📦 Syllabus Module Hub",
    "📖 Descriptive Notes",
    "🎯 Exam Q&A Bank",
    "📐 Formula Sheet & Study Guide",
    "📚 Textbooks & Handouts",
    "✍️ Slides, PDFs & Numericals",
])

# TAB 1: CURRICULUM & SYLLABUS
with tab1:
    st.header("Course Curriculum & Handout")
    
    # Check for syllabus files
    handout_files = list(COURSE_DIR.glob("*HANDOUT*.md")) + list(COURSE_DIR.glob("Syllabus.md")) + list(COURSE_DIR.glob("Curriculum*.md"))
    if handout_files:
        chosen_doc = st.selectbox(
            "Select Document to View",
            handout_files,
            format_func=lambda p: p.name
        )
        if chosen_doc:
            with open(chosen_doc, "r", encoding="utf-8", errors="replace") as f:
                st.markdown(f.read(), unsafe_allow_html=True)
    else:
        st.info("No markdown syllabus document found in this course folder yet.")


# TAB 2: SYLLABUS MODULE HUB
with tab2:
    st.header("Syllabus Module Hub")
    modules = load_course_modules(COURSE_DIR)
    if not modules:
        st.info("Add `Notes/AI Notes/module_map.yaml` for module-wise navigation.")
    else:
        labels = {m["id"]: f"{m['id']}: {m.get('title', '')}" for m in modules}
        pick_id = st.selectbox("Syllabus module", list(labels.keys()), format_func=lambda k: labels[k])
        mod = next(m for m in modules if m["id"] == pick_id)
        cs = mod.get("contact_sessions")
        st.caption(f"Contact sessions (reference): {cs if cs else '—'} | Webinar: {mod.get('webinar') or '—'}")
        note_rel = mod.get("descriptive_note", "")
        note_path = None
        if note_rel:
            candidate = NOTES_DIR / note_rel
            note_path = candidate if candidate.exists() else DESCRIPTIVE_DIR / Path(note_rel).name
        if note_path and note_path.exists():
            st.subheader("Descriptive notes")
            render_markdown_with_assets(note_path.read_text(encoding="utf-8", errors="replace"), COURSE_DIR)
        for lec in mod.get("lectures") or []:
            vid_rel = lec.get("video", "")
            st.subheader(lec.get("title") or vid_rel)
            vpath = resolve_lecture_video(COURSE_DIR, vid_rel)
            if vpath and vpath.is_file():
                st.video(str(vpath))
            else:
                st.caption("No local video — copy via `scripts/materialize_lecture_videos.py` into `Videos/_materialized/`.")
            tr = lec.get("transcript")
            if tr:
                tp = resolve_course_path(COURSE_DIR, tr)
                if tp:
                    with st.expander("Transcript"):
                        st.markdown(tp.read_text(encoding="utf-8", errors="replace")[:16000])
        for sl in mod.get("slides") or []:
            sp = resolve_course_path(COURSE_DIR, sl.get("path", ""))
            if not sp:
                continue
            st.subheader(sl.get("label") or sp.name)
            if sp.suffix.lower() == ".pdf":
                render_pdf_inline(sp)
            elif sp.suffix.lower() in {".pptx", ".ppt"}:
                render_pptx_outline(sp)


# TAB 3: DESCRIPTIVE NOTES
with tab3:
    st.header("All descriptive note files")
    if DESCRIPTIVE_DIR.exists() and any(DESCRIPTIVE_DIR.glob("Module_*.md")):
        notes_files = sorted(DESCRIPTIVE_DIR.glob("Module_*.md"))
        selected_note_file = st.selectbox(
            "Module file", notes_files, format_func=lambda p: p.stem.replace("_", " ")
        )
        render_markdown_with_assets(
            selected_note_file.read_text(encoding="utf-8", errors="replace"), COURSE_DIR
        )
    else:
        st.info(f"Descriptive notes are not available yet for {selected_course}.")


# TAB 4: EXAM Q&A BANK
with tab4:
    st.header("Examination Question-Answer Bank")
    if QA_DIR.exists() and any(QA_DIR.glob("*.md")):
        qa_files = sorted(list(QA_DIR.glob("*.md")))
        selected_qa_file = st.selectbox(
            "Select Q&A Bank Document",
            qa_files,
            format_func=lambda p: p.stem.replace("_", " ")
        )
        
        if selected_qa_file:
            with open(selected_qa_file, "r", encoding="utf-8", errors="replace") as f:
                qa_content = f.read()
                
            search_query = st.text_input("🔍 Search Questions or Keywords:", "")
            
            if re.search(r"^### Q\d+\.", qa_content, re.MULTILINE):
                questions = re.split(r"(?=^### Q\d+\.)", qa_content, flags=re.MULTILINE)
                header_part = questions[0]
                q_list = [q for q in questions[1:] if q.strip().startswith("### Q")]
                if search_query:
                    q_list = [q for q in q_list if search_query.lower() in q.lower()]
                    st.caption(f"Found {len(q_list)} matching questions.")
                st.markdown(header_part, unsafe_allow_html=True)
                for q_block in q_list:
                    first_line = q_block.strip().split("\n")[0].replace("###", "").strip()
                    body = "\n".join(q_block.strip().split("\n")[1:])
                    with st.expander(first_line, expanded=False):
                        st.markdown(body)
            elif "### Question" in qa_content:
                questions = re.split(r'(?=### Question \d+)', qa_content)
                header_part = questions[0]
                q_list = questions[1:]
                
                if search_query:
                    q_list = [q for q in q_list if search_query.lower() in q.lower()]
                    st.caption(f"Found {len(q_list)} matching questions.")
                    
                st.markdown(header_part, unsafe_allow_html=True)
                for q_block in q_list:
                    lines = q_block.strip().split("\n")
                    first_line = lines[0].replace("###", "").strip() if lines else "Question"
                    body = "\n".join(lines[1:])
                    with st.expander(first_line, expanded=False):
                        st.markdown(body, unsafe_allow_html=True)
            else:
                if search_query:
                    filtered_lines = [l for l in qa_content.split("\n") if search_query.lower() in l.lower()]
                    st.caption(f"Showing {len(filtered_lines)} matches.")
                    st.markdown("\n".join(filtered_lines))
                else:
                    st.markdown(qa_content, unsafe_allow_html=True)
    else:
        st.info(f"Exam Q&A Bank is currently in preparation for {selected_course}.")


# TAB 5: FORMULA SHEET & STUDY GUIDE
with tab5:
    st.header("Master Formula Sheet & Oral Viva Guide")
    formula_files = []
    if QA_DIR.exists():
        formula_files.extend(list(QA_DIR.glob("*Formula*.md")) + list(QA_DIR.glob("*Study_Guide*.md")))
    if NOTES_DIR.exists():
        formula_files.extend(list(NOTES_DIR.glob("*Formula*.md")))
        
    if formula_files:
        chosen_fs = st.selectbox(
            "Select Study Guide / Formula Sheet",
            formula_files,
            format_func=lambda p: p.stem.replace("_", " ")
        )
        if chosen_fs:
            with open(chosen_fs, "r", encoding="utf-8", errors="replace") as f:
                st.markdown(f.read(), unsafe_allow_html=True)
    else:
        st.info("Formula sheet and study guide will appear here once generated.")


# TAB 6: TEXTBOOKS & HANDOUTS (inline PDF)
with tab6:
    st.header("Prescribed Textbooks & Reference Books")
    if BOOKS_DIR.exists() and any(BOOKS_DIR.glob("*.pdf")):
        book_files = sorted(BOOKS_DIR.glob("*.pdf"))
        pick = st.selectbox("Textbook", book_files, format_func=lambda p: p.name)
        if pick:
            st.caption(f"{pick.stat().st_size / (1024**2):.1f} MB")
            render_pdf_inline(pick)
            with open(pick, "rb") as f_bytes:
                st.download_button("Download", f_bytes.read(), pick.name, mime="application/pdf", key=f"dl_{pick.name}")
    else:
        st.info(f"No PDFs found in Books directory for {selected_course}.")


# TAB 7: SLIDES, HANDWRITTEN, QP
with tab7:
    st.header("Slides, companions & practice PDFs")

    doc_buckets: list[tuple[str, list[Path]]] = []
    if HANDWRITTEN_DIR.exists():
        doc_buckets.append(("Handwritten derivations", sorted(HANDWRITTEN_DIR.glob("*.pdf"))))
    if COMPANIONS_DIR.exists():
        doc_buckets.append(
            ("Lecture slides & companions", sorted(COMPANIONS_DIR.rglob("*.pdf"))[:80])
        )
    if QP_DIR.exists():
        doc_buckets.append(("Question papers & solutions", sorted(QP_DIR.glob("*.pdf"))))

    flat: list[Path] = []
    flat_labels: list[str] = []
    for label, files in doc_buckets:
        for f in files:
            flat.append(f)
            flat_labels.append(f"{label}: {f.name}")

    if not flat:
        st.caption("No PDF companions in this course folder.")
    else:
        idx = st.selectbox("Document", range(len(flat)), format_func=lambda i: flat_labels[i])
        chosen = flat[idx]
        st.caption(str(chosen.relative_to(COURSE_DIR)))
        if chosen.suffix.lower() == ".pdf":
            render_pdf_inline(chosen)
        elif chosen.suffix.lower() in {".pptx", ".ppt"}:
            render_pptx_outline(chosen)
        with open(chosen, "rb") as fb:
            st.download_button("Download", fb.read(), chosen.name, key=f"dl7_{chosen.name}")

