"""Shared helpers for BITS Resources Streamlit portal (module-wise navigation)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_course_modules(course_dir: Path) -> list[dict[str, Any]]:
    """Load syllabus modules from Notes/AI Notes/module_map.yaml."""
    path = course_dir / "Notes" / "AI Notes" / "module_map.yaml"
    if yaml is None or not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    modules = data.get("modules") or []
    return sorted(modules, key=lambda m: m.get("id", ""))


def resolve_course_path(course_dir: Path, rel: str) -> Path | None:
    if not rel:
        return None
    p = (course_dir / rel.strip()).resolve()
    if p.is_file():
        return p
    if p.is_symlink() or p.exists():
        try:
            r = p.resolve(strict=False)
            if r.is_file():
                return r
        except OSError:
            pass
    return None


def resolve_lecture_video(course_dir: Path, video_rel: str) -> Path | None:
    """Prefer materialized copy for players / Drive; fall back to symlink target."""
    if not video_rel:
        return None
    name = Path(video_rel).name
    mat = course_dir / "Videos" / "_materialized" / name
    if mat.is_file():
        return mat
    return resolve_course_path(course_dir, video_rel)


def render_pdf_inline(pdf_path: Path, *, height: int = 780, max_embed_mb: float = 14.0) -> None:
    """Show PDF in-app via native Streamlit viewer; legacy iframe/text as fallback."""
    del max_embed_mb  # kept for call-site compatibility; st.pdf handles large files
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    try:
        st.pdf(pdf_path, height=height)
        if size_mb >= 12:
            st.caption(f"{size_mb:.1f} MB — if the viewer is slow, use the download button below.")
        return
    except Exception as primary:
        st.warning(f"PDF viewer unavailable ({primary}). Trying fallback preview.")
    # Fallback: small PDFs only (base64 iframe); huge files → first-page text
    if size_mb <= 14.0:
        try:
            b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}" '
                f'width="100%" height="{height}px" style="border:1px solid #e2e8f0;"></iframe>',
                unsafe_allow_html=True,
            )
            return
        except Exception:
            pass
    st.info(f"PDF is {size_mb:.1f} MB — showing text outline only. Use download for full file.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        st.caption(f"{len(reader.pages)} pages — first page text:")
        if reader.pages:
            st.text(reader.pages[0].extract_text() or "(no extractable text)")
    except Exception as e:
        st.caption(str(e))


def render_pptx_outline(pptx_path: Path) -> None:
    try:
        from pptx import Presentation
        prs = Presentation(str(pptx_path))
        for i, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            title = texts[0][:120] if texts else f"Slide {i}"
            with st.expander(f"Slide {i}: {title}"):
                if texts:
                    st.markdown("\n\n".join(texts[:5]))
    except ImportError:
        st.warning("Install python-pptx for PPTX outline view.")
    except Exception as e:
        st.warning(f"Could not read PPTX: {e}")