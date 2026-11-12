"""
Streamlit Web Portal for AIML ZG565: Machine Learning
BITS Pilani WILP — Academic Study Repository & Examination Preparation Hub
"""

import os
import re
from pathlib import Path
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="AIML ZG565: Machine Learning Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
NOTES_DIR = BASE_DIR / "Notes" / "AI Notes"
DESCRIPTIVE_DIR = NOTES_DIR / "01_Descriptive_Notes"
QA_DIR = NOTES_DIR / "02_Question_Answer_Notes"
ASSETS_DIR = NOTES_DIR / "assets"
BOOKS_DIR = BASE_DIR / "Books"
HANDWRITTEN_DIR = BASE_DIR / "Notes" / "Handwitten Notes"
QP_DIR = BASE_DIR / "Notes" / "Question Papers & Solutions"
COMPANIONS_DIR = BASE_DIR / "Notes" / "Lecture Slides & Companions"
ASSIGNMENTS_DIR = BASE_DIR / "Assignments"

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
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
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

def render_markdown_with_assets(md_text: str, current_dir: Path):
    """Replaces relative image paths with absolute or base paths so Streamlit displays them."""
    # Pattern: ![alt](../assets/filename.jpg)
    pattern = re.compile(r'!\[([^\]]*)\]\((?:\.\./assets/|assets/)?([^)]+)\)')
    
    parts = []
    last_idx = 0
    for match in pattern.finditer(md_text):
        # markdown text before image
        parts.append(md_text[last_idx:match.start()])
        alt_text = match.group(1)
        img_name = Path(match.group(2)).name
        img_path = ASSETS_DIR / img_name
        
        # Output markdown before
        st.markdown(parts[-1], unsafe_allow_html=True)
        parts = []
        
        if img_path.exists():
            st.image(str(img_path), caption=alt_text or img_name, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_name}")
            
        last_idx = match.end()
        
    if last_idx < len(md_text):
        st.markdown(md_text[last_idx:], unsafe_allow_html=True)

def get_file_download_button(file_path: Path, label: str = None):
    """Helper to render a clean download button for any binary or text file."""
    if not file_path.exists():
        return
    size_mb = file_path.stat().st_size / (1024 * 1024)
    btn_label = label or f"📥 Download {file_path.name} ({size_mb:.1f} MB)"
    with open(file_path, "rb") as f:
        st.download_button(
            label=btn_label,
            data=f.read(),
            file_name=file_path.name,
            mime="application/octet-stream",
            key=f"dl_{file_path.name}_{file_path.stat().st_size}"
        )

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("AIML ZG565: ML")
st.sidebar.caption("BITS Pilani Work Integrated Learning Programmes")

nav_choice = st.sidebar.radio(
    "Navigation Portal",
    [
        "🏠 Course Overview & Syllabus",
        "🗺️ Curriculum Mapping & Faculty Audit",
        "📖 Topic-Wise Descriptive Notes",
        "🎯 50-Question Exam Q&A Bank",
        "📐 Master Formula Sheet & Viva Guide",
        "📚 Prescribed Textbooks & References",
        "✍️ Faculty Handwritten Derivations",
        "📋 Exam Practice Sets & Solutions",
        "💻 Assignments & Demo Notebooks"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Course Metadata**:
- **Code**: AIML ZG565 (4 Credits)
- **Primary Faculty**: Prof. Pritee Narendra Raole
- **Companions**: Prof. Nivethitha Somu, Prof. Devanshu Tayal
- **Textbooks**: Mitchell (T1), Bishop (R1), Tan (R2), Burges (R3)
""")

# --- MAIN CONTENT ROUTING ---

# 1. COURSE OVERVIEW & SYLLABUS
if nav_choice == "🏠 Course Overview & Syllabus":
    st.markdown('<div class="main-title">Machine Learning (AIML ZG565)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Birla Institute of Technology & Science, Pilani — WILP Division</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Credit Units", "4 (3-1-0)")
    col2.metric("Syllabus Modules", "11 Modules")
    col3.metric("Contact Sessions", "13 Lectures")
    col4.metric("Exam Questions", "50 Solved Q&As")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📋 Standard Syllabus (Syllabus.md)", "📑 Formal Course Handout"])
    
    with tab1:
        syllabus_file = BASE_DIR / "Syllabus.md"
        if syllabus_file.exists():
            st.markdown(syllabus_file.read_text(encoding="utf-8"))
        else:
            st.error("Syllabus.md not found.")
            
    with tab2:
        handout_file = BASE_DIR / "AIML_ZG565_COURSE_HANDOUT.md"
        if handout_file.exists():
            st.markdown(handout_file.read_text(encoding="utf-8"))
        else:
            st.error("Handout file not found.")

# 2. CURRICULUM MAPPING & FACULTY AUDIT
elif nav_choice == "🗺️ Curriculum Mapping & Faculty Audit":
    st.markdown('<div class="main-title">Curriculum Mapping & Faculty Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Faculty Audit, Teams Recording Anomaly Resolution & 1-to-1 Mapping</div>', unsafe_allow_html=True)
    
    audit_file = BASE_DIR / "Curriculum_Mapping_and_Faculty_Audit.md"
    if audit_file.exists():
        st.markdown(audit_file.read_text(encoding="utf-8"))
    else:
        st.error("Curriculum_Mapping_and_Faculty_Audit.md not found.")

# 3. MODULAR DESCRIPTIVE NOTES
elif nav_choice == "📖 Topic-Wise Descriptive Notes":
    st.markdown('<div class="main-title">Modular Descriptive Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Publication-Grade Study Guides with Formal Derivations, Python Implementations & Visual Keyframes</div>', unsafe_allow_html=True)
    
    modules = sorted(list(DESCRIPTIVE_DIR.glob("Module_*.md")))
    if not modules:
        st.warning("No modular descriptive notes found.")
    else:
        module_titles = {
            m.name: f"{m.stem.replace('_', ' ')}"
            for m in modules
        }
        
        selected_mod_name = st.selectbox(
            "Select Syllabus Module:",
            options=list(module_titles.keys()),
            format_func=lambda x: module_titles[x]
        )
        
        selected_file = DESCRIPTIVE_DIR / selected_mod_name
        st.markdown("---")
        
        # Download button for this note
        get_file_download_button(selected_file, f"📥 Download {selected_mod_name} (.md)")
        st.markdown("<br>", unsafe_allow_html=True)
        
        content = selected_file.read_text(encoding="utf-8")
        render_markdown_with_assets(content, DESCRIPTIVE_DIR)

# 4. 50-QUESTION EXAM Q&A BANK
elif nav_choice == "🎯 50-Question Exam Q&A Bank":
    st.markdown('<div class="main-title">Comprehensive Examination Q&A Bank</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">50 University Examination Questions (15 Easy, 25 Medium, 10 Tough) with 6-Marks Model Answers</div>', unsafe_allow_html=True)
    
    qa_file = QA_DIR / "50_Exam_Question_Answer_Bank.md"
    if not qa_file.exists():
        st.error("Exam Q&A Bank not found.")
    else:
        content = qa_file.read_text(encoding="utf-8")
        
        # Search & Difficulty Filter
        col1, col2 = st.columns([3, 1])
        search_kw = col1.text_input("🔍 Search Questions & Keywords (e.g., 'Normal Equation', 'Entropy', 'Slack', 'Laplace', 'McNemar'):", "")
        diff_filter = col2.selectbox("Filter Difficulty:", ["All", "Easy", "Medium", "Tough"])
        
        get_file_download_button(qa_file, "📥 Download Complete 50-Question Bank (.md)")
        st.markdown("---")
        
        # Parse questions out of markdown
        # Pattern: ### Q\d+\. \[(Easy|Medium|Tough)\] ...
        q_blocks = re.split(r'\n(?=### Q\d+\.)', content)
        
        displayed_count = 0
        for block in q_blocks:
            if not block.strip().startswith("### Q"):
                continue
            
            header_line = block.strip().split("\n")[0]
            m_diff = re.search(r'\[(Easy|Medium|Tough)\]', header_line)
            diff = m_diff.group(1) if m_diff else "Unknown"
            
            # Apply filters
            if diff_filter != "All" and diff.lower() != diff_filter.lower():
                continue
            if search_kw.strip() and search_kw.lower() not in block.lower():
                continue
                
            displayed_count += 1
            
            # Badge color
            badge_color = "#10B981" if diff == "Easy" else ("#F59E0B" if diff == "Medium" else "#EF4444")
            
            with st.expander(f"{header_line.replace('### ', '')}"):
                # Render the rest of the question content
                body = "\n".join(block.strip().split("\n")[1:])
                st.markdown(body)
                
        if displayed_count == 0:
            st.info(f"No questions match search keyword '{search_kw}' with difficulty '{diff_filter}'.")

# 5. MASTER FORMULA SHEET & VIVA GUIDE
elif nav_choice == "📐 Master Formula Sheet & Viva Guide":
    st.markdown('<div class="main-title">Master Formula Sheet & Study Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quick-Reference Formula Lookup Table, Tensor Dimensions, Bounds, and Oral Exam Checklist</div>', unsafe_allow_html=True)
    
    formula_file = QA_DIR / "Master_Formula_Sheet_and_Study_Guide.md"
    if formula_file.exists():
        get_file_download_button(formula_file, "📥 Download Formula Sheet (.md)")
        st.markdown("---")
        st.markdown(formula_file.read_text(encoding="utf-8"))
    else:
        st.error("Formula sheet not found.")

# 6. PRESCRIBED TEXTBOOKS & REFERENCES
elif nav_choice == "📚 Prescribed Textbooks & References":
    st.markdown('<div class="main-title">Prescribed Textbooks & Reference Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Official Prescribed Textbooks (T1) and Core References (R1, R2, R3)</div>', unsafe_allow_html=True)
    
    books = sorted(list(BOOKS_DIR.glob("*.pdf")))
    if not books:
        st.warning("No textbooks found in Books directory.")
    else:
        for b in books:
            size_mb = b.stat().st_size / (1024 * 1024)
            with st.container():
                st.subheader(f"📖 {b.name.replace('_', ' ').replace('.pdf', '')}")
                st.caption(f"File Size: {size_mb:.1f} MB | Location: Books/{b.name}")
                get_file_download_button(b)
                st.markdown("---")

# 7. FACULTY HANDWRITTEN DERIVATIONS
elif nav_choice == "✍️ Faculty Handwritten Derivations":
    st.markdown('<div class="main-title">Faculty Handwritten Derivations & Annotated PPTs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Prof. Pritee Narendra Raole (Parwekar) Digital Tablet Derivations & Solved Examples</div>', unsafe_allow_html=True)
    
    hw_files = sorted(list(HANDWRITTEN_DIR.glob("*.pdf")))
    if not hw_files:
        st.warning("No handwritten notes found.")
    else:
        for f in hw_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            with st.container():
                st.write(f"📝 **{f.name}** ({size_mb:.1f} MB)")
                get_file_download_button(f)
                st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# 8. EXAM PRACTICE SETS & SOLUTIONS
elif nav_choice == "📋 Exam Practice Sets & Solutions":
    st.markdown('<div class="main-title">Examination Practice Sets & Solutions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Authentic BITS Pilani Practice Numericals, Regularization Problems, CART Gini Sets & Step-by-Step Solutions</div>', unsafe_allow_html=True)
    
    qp_files = sorted(list(QP_DIR.glob("*.pdf")))
    if not qp_files:
        st.warning("No practice question papers found.")
    else:
        col_q, col_s = st.columns(2)
        with col_q:
            st.subheader("📑 Question Worksheets")
            for f in qp_files:
                if "solution" not in f.name.lower() and "step_by_step" not in f.name.lower():
                    st.write(f"• **{f.name}**")
                    get_file_download_button(f)
        with col_s:
            st.subheader("✅ Solved Solutions")
            for f in qp_files:
                if "solution" in f.name.lower() or "step_by_step" in f.name.lower():
                    st.write(f"• **{f.name}**")
                    get_file_download_button(f)

# 9. ASSIGNMENTS & DEMO NOTEBOOKS
elif nav_choice == "💻 Assignments & Demo Notebooks":
    st.markdown('<div class="main-title">Experiential Labs & Code Demonstrations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive Jupyter Notebooks, Assignment Solutions & Practical Python Implementations</div>', unsafe_allow_html=True)
    
    tab_asn, tab_demo = st.tabs(["💼 Official Assignment-2", "🐍 Companion Demo Notebooks"])
    
    with tab_asn:
        asn2_readme = ASSIGNMENTS_DIR / "Assignment-2" / "README.md"
        asn2_sub = ASSIGNMENTS_DIR / "Assignment-2" / "SUBMISSION_DOCUMENT.md"
        if asn2_readme.exists():
            st.markdown(asn2_readme.read_text(encoding="utf-8"))
        if asn2_sub.exists():
            st.markdown("---")
            st.markdown(asn2_sub.read_text(encoding="utf-8"))
            
    with tab_demo:
        demo_dir = COMPANIONS_DIR / "Demo codes"
        if demo_dir.exists():
            notebooks = sorted(list(demo_dir.glob("*.ipynb")))
            st.write(f"Found {len(notebooks)} hands-on demonstration notebooks from Prof. Nivethitha Somu:")
            for nb in notebooks:
                st.write(f"• 📓 **{nb.name}**")
                get_file_download_button(nb)
        else:
            st.info("Demo notebooks directory not found.")

