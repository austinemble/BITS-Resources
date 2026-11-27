#!/usr/bin/env python3
"""Copy hub slides (PDF) and transcripts (MD) into git-tracked portal_mirror paths."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COURSES = [
    "Machine Learning",
    "Deep Learning",
    "Statistics",
    "MFML",
]
MIRROR_SUB = ("Notes", "AI Notes", "portal_mirror")


def safe_name(rel: str) -> str:
    h = hashlib.sha1(rel.encode()).hexdigest()[:10]
    base = Path(rel).name
    safe = re.sub(r"[^\w.\-]+", "_", base)
    return f"{h}_{safe}"


def mirror_file(course_dir: Path, rel: str, kind: str) -> str | None:
    if not rel:
        return None
    src = (course_dir / rel).resolve()
    if not src.is_file():
        return None
    ext = src.suffix.lower()
    if kind == "slide" and ext != ".pdf":
        return None
    if kind == "transcript" and ext not in {".md", ".txt"}:
        return None
    dest_dir = course_dir.joinpath(*MIRROR_SUB, kind + "s")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name(rel)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return str(dest.relative_to(course_dir))
    shutil.copy2(src, dest)
    return str(dest.relative_to(course_dir))


def process_course(course: str) -> dict:
    course_dir = ROOT / "Semester 1" / course
    ypath = course_dir / "Notes/AI Notes/module_map.yaml"
    if not ypath.is_file():
        return {"course": course, "skipped": "no map"}
    data = yaml.safe_load(ypath.read_text(encoding="utf-8"))
    modules = data.get("modules") or []
    mirrored = 0
    updated = False
    for m in modules:
        for sl in m.get("slides") or []:
            rel = sl.get("path", "")
            if not rel.startswith("Inbox/"):
                continue
            new_rel = mirror_file(course_dir, rel, "slide")
            if new_rel:
                sl["path_cloud"] = new_rel
                sl["path_inbox"] = rel
                sl["path"] = new_rel
                mirrored += 1
                updated = True
        for lec in m.get("lectures") or []:
            tr = lec.get("transcript", "")
            if tr.startswith("Inbox/"):
                new_tr = mirror_file(course_dir, tr, "transcript")
                if new_tr:
                    lec["transcript_inbox"] = tr
                    lec["transcript"] = new_tr
                    mirrored += 1
                    updated = True
    if updated:
        ypath.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"course": course, "mirrored": mirrored, "updated_map": updated}


def main() -> None:
  extras = []
  dnn_ann = ROOT / "Semester 1/Deep Learning/Inbox/Prof. Shakshi Sharma/Shakshi Sharma_annotated ppts/DNN_M11_Regularisation (1)_annotated.pdf"
  dnn_dest = ROOT / "Semester 1/Deep Learning/Notes/Lecture Slides & Companions/DNN_M11_Regularisation_annotated.pdf"
  if dnn_ann.is_file():
      dnn_dest.parent.mkdir(parents=True, exist_ok=True)
      if not dnn_dest.is_file() or dnn_dest.stat().st_size != dnn_ann.stat().st_size:
          shutil.copy2(dnn_ann, dnn_dest)
          extras.append(str(dnn_dest.relative_to(ROOT)))
  results = [process_course(c) for c in COURSES]
  for r in results:
      print(r)
  if extras:
      print("extras:", extras)


if __name__ == "__main__":
    main()