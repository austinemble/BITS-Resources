#!/usr/bin/env python3
"""Build Videos/module_map.json from Notes/AI Notes/module_map.yaml for materialize manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
COURSES = ("Machine Learning", "Deep Learning", "Statistics")


def build_video_index(course_dir: Path) -> dict[str, dict]:
    yaml_path = course_dir / "Notes" / "AI Notes" / "module_map.yaml"
    if not yaml_path.is_file():
        return {}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict] = {}
    for mod in data.get("modules") or []:
        mid = mod.get("id", "")
        title = mod.get("title", "")
        for lec in mod.get("lectures") or []:
            vid = lec.get("video") or ""
            if not vid.endswith(".mp4"):
                continue
            basename = Path(vid).name
            out[basename] = {
                "module_id": mid,
                "module_title": title,
                "lecture_title": lec.get("title", ""),
                "transcript": lec.get("transcript"),
            }
    return out


def main() -> int:
    for name in COURSES:
        course_dir = ROOT / "Semester 1" / name
        videos_dir = course_dir / "Videos"
        if not videos_dir.is_dir():
            continue
        index = build_video_index(course_dir)
        if not index:
            print(f"{name}: no module_map.yaml lectures — skipped")
            continue
        out_path = videos_dir / "module_map.json"
        out_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: wrote {len(index)} entries -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())