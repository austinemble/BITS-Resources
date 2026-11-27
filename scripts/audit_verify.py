#!/usr/bin/env python3
"""Verify audit checklist; write AUDIT_OPEN_POINTS.md and exit 1 if blockers remain."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from portal_lib import resolve_course_path  # noqa: E402

COURSES = ["Machine Learning", "Deep Learning", "Statistics", "MFML"]
OUT = ROOT / "AUDIT_OPEN_POINTS.md"


def git_tracked(rel: str) -> bool:
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=ROOT,
        capture_output=True,
    )
    return r.returncode == 0


def check_paths() -> list[str]:
    fails = []
    cloud_slides_missing = []
    for course in COURSES:
        course_dir = ROOT / "Semester 1" / course
        ypath = course_dir / "Notes/AI Notes/module_map.yaml"
        if not ypath.is_file():
            continue
        data = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        for m in data.get("modules") or []:
            mid = m.get("id", "?")
            note = m.get("descriptive_note", "")
            if note:
                p = course_dir / "Notes/AI Notes" / note
                if not p.is_file():
                    fails.append(f"[{course}] {mid} missing note {note}")
            for sl in m.get("slides") or []:
                rel = sl.get("path", "")
                sp = resolve_course_path(course_dir, rel)
                if not sp:
                    fails.append(f"[{course}] {mid} missing slide {rel}")
                elif rel.endswith(".pdf") and not git_tracked(str(sp.relative_to(ROOT))):
                    cloud_slides_missing.append(f"[{course}] {mid} slide not in git: {rel}")
            for lec in m.get("lectures") or []:
                for key in ("video", "transcript"):
                    rel = lec.get(key, "")
                    if not rel:
                        continue
                    rp = resolve_course_path(course_dir, rel)
                    if not rp:
                        fails.append(f"[{course}] {mid} missing {key} {rel}")
    return fails, cloud_slides_missing


def check_content() -> list[str]:
    issues = []
    r1 = ROOT / "Semester 1/Statistics/Books/R1_Miller_Freund_Probability_Statistics_Engineers.pdf"
    if not r1.is_file():
        issues.append(
            "OPTIONAL_BOOK: Stats R1 Miller & Freund missing — notes use T1–T3; add PDF to Books/ when available"
        )
    mfml = ROOT / "Semester 1/MFML/Notes/AI Notes/01_Descriptive_Notes"
    if mfml.is_dir():
        mods = list(mfml.glob("Module_*.md"))
        for m in mods:
            t = m.read_text(encoding="utf-8", errors="replace")
            if "Faculty Pedagogy" not in t:
                issues.append(f"MFML {m.name} missing §7 Faculty Pedagogy")
        if len(mods) != 14:
            issues.append(f"MFML expected 14 modules, found {len(mods)}")
    stats = ROOT / "Semester 1/Statistics/Notes/AI Notes/01_Descriptive_Notes"
    for m in sorted(stats.glob("Module_*.md")):
        t = m.read_text(encoding="utf-8", errors="replace")
        if "**Syllabus module**" not in t:
            issues.append(f"Stats {m.name} missing Syllabus module header")
    return issues


def write_report(fails, cloud, content_issues):
    lines = [
        "# Open audit points (auto-generated)",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"- Local path failures: **{len(fails)}**",
        f"- Slide PDFs not in git: **{len(cloud)}**",
        f"- Content / book blockers: **{len(content_issues)}**",
        "",
    ]
    optional = [x for x in content_issues if x.startswith("OPTIONAL_BOOK:")]
    if not fails and not cloud and not [x for x in content_issues if not x.startswith("OPTIONAL_BOOK:")]:
        lines.append("**Status: ALL FIXABLE AUDIT CHECKS PASS** (videos local/Drive only by design).")
        if optional:
            lines.append("")
            lines.append("Optional (not counted as failure):")
            lines.extend(f"- {x}" for x in optional)
    else:
        for title, items in [
            ("Local path failures", fails),
            ("Slides not in git (Streamlit Cloud)", cloud),
            ("Content / books", content_issues),
        ]:
            if items:
                lines.append(f"## {title}")
                lines.extend(f"- {x}" for x in items)
                lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    fails, cloud = check_paths()
    content = check_content()
    write_report(fails, cloud, content)
    print(OUT.read_text(encoding="utf-8")[:800])
    fixable_content = [x for x in content if not x.startswith("OPTIONAL_BOOK:")]
    if fails:
        return 2
    if cloud:
        return 3
    if fixable_content:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())