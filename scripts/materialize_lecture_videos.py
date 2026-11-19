#!/usr/bin/env python3
"""Copy lecture .mp4 files from Videos/ (symlinks or real files) into Videos/_materialized/."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

COURSE_NAMES = ("Machine Learning", "Deep Learning", "Statistics")
MATERIALIZED_DIR = "_materialized"
MODULE_MAP_FILENAME = "module_map.json"


@dataclass
class CourseStats:
    mp4_count: int = 0
    would_copy: int = 0
    copied: int = 0
    skipped_unchanged: int = 0
    broken_symlinks: list[str] = field(default_factory=list)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_module_map(videos_dir: Path) -> dict[str, object]:
    map_path = videos_dir / MODULE_MAP_FILENAME
    if not map_path.is_file():
        return {}
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read {map_path}: {exc}", file=sys.stderr)
        return {}
    if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], dict):
        return data["entries"]
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def resolve_source(video_entry: Path) -> tuple[Path | None, str | None]:
    """Return (resolved real file, error message)."""
    if not video_entry.exists() and not video_entry.is_symlink():
        return None, "missing (not a symlink and not a file)"

    if video_entry.is_symlink():
        target = video_entry.resolve(strict=False)
        if not target.is_file():
            link_text = os.readlink(video_entry)
            return None, f"broken symlink -> {link_text}"
        return target, None

    if video_entry.is_file():
        return video_entry.resolve(), None

    return None, "not a regular file"


def rel_for_manifest(path: Path, course_dir: Path, resources_root: Path) -> str:
    try:
        return str(path.relative_to(course_dir))
    except ValueError:
        try:
            return str(path.relative_to(resources_root))
        except ValueError:
            return str(path)


def dest_matches_source(dest: Path, source: Path) -> bool:
    if not dest.is_file():
        return False
    try:
        if dest.stat().st_size != source.stat().st_size:
            return False
    except OSError:
        return False
    return file_sha256(dest) == file_sha256(source)


def process_course(
    course_dir: Path,
    resources_root: Path,
    *,
    dry_run: bool,
) -> tuple[list[dict], CourseStats]:
    videos_dir = course_dir / "Videos"
    stats = CourseStats()
    manifest_entries: list[dict] = []

    if not videos_dir.is_dir():
        return manifest_entries, stats

    module_map = load_module_map(videos_dir)
    out_dir = videos_dir / MATERIALIZED_DIR

    mp4_paths = sorted(
        p
        for p in videos_dir.iterdir()
        if p.name.lower().endswith(".mp4") and (p.is_file() or p.is_symlink())
    )

    stats.mp4_count = len(mp4_paths)

    for entry in mp4_paths:
        basename = entry.name
        source, err = resolve_source(entry)
        if err:
            stats.broken_symlinks.append(f"{basename}: {err}")
            continue
        assert source is not None

        materialized_path = out_dir / basename
        rel_materialized = f"Videos/{MATERIALIZED_DIR}/{basename}"
        source_inbox = rel_for_manifest(source, course_dir, resources_root)

        row: dict = {
            "basename": basename,
            "source_inbox_path": source_inbox,
            "materialized_path": rel_materialized,
            "bytes": source.stat().st_size,
        }
        if basename in module_map:
            row["module_map"] = module_map[basename]

        if dest_matches_source(materialized_path, source):
            stats.skipped_unchanged += 1
            row["status"] = "skipped_unchanged"
            manifest_entries.append(row)
            continue

        if dry_run:
            stats.would_copy += 1
            row["status"] = "would_copy"
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, materialized_path)
            stats.copied += 1
            row["status"] = "copied"

        manifest_entries.append(row)

    if not dry_run and manifest_entries:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "course": course_dir.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entries": manifest_entries,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    return manifest_entries, stats


def print_course_report(course_name: str, stats: CourseStats, dry_run: bool) -> None:
    print(f"\n=== {course_name} ===")
    print(f"  mp4 in Videos/:     {stats.mp4_count}")
    if dry_run:
        print(f"  would copy:         {stats.would_copy}")
    else:
        print(f"  copied:             {stats.copied}")
    print(f"  skip (same hash):   {stats.skipped_unchanged}")
    print(f"  broken symlinks:    {len(stats.broken_symlinks)}")
    for msg in stats.broken_symlinks:
        print(f"    - {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize lecture videos into Videos/_materialized/ for cloud sync."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without copying files or writing manifest.json.",
    )
    parser.add_argument(
        "--course",
        action="append",
        dest="courses",
        metavar="NAME",
        help=f"Limit to course(s). Choices: {', '.join(COURSE_NAMES)}. Repeatable.",
    )
    args = parser.parse_args()

    resources_root = repo_root_from_script()
    semester = resources_root / "Semester 1"

    selected = list(COURSE_NAMES)
    if args.courses:
        unknown = [c for c in args.courses if c not in COURSE_NAMES]
        if unknown:
            print(f"error: unknown course(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"valid: {', '.join(COURSE_NAMES)}", file=sys.stderr)
            return 2
        selected = args.courses

    totals = CourseStats()
    exit_code = 0

    for name in selected:
        course_dir = semester / name
        if not (course_dir / "Videos").is_dir():
            print(f"\n=== {name} ===\n  (no Videos/ folder — skipped)")
            continue

        _, stats = process_course(course_dir, resources_root, dry_run=args.dry_run)
        print_course_report(name, stats, args.dry_run)

        totals.mp4_count += stats.mp4_count
        totals.would_copy += stats.would_copy
        totals.copied += stats.copied
        totals.skipped_unchanged += stats.skipped_unchanged
        totals.broken_symlinks.extend(f"{name}: {b}" for b in stats.broken_symlinks)

    print("\n=== totals ===")
    print(f"  mp4:              {totals.mp4_count}")
    if args.dry_run:
        print(f"  would copy:       {totals.would_copy}")
    else:
        print(f"  copied:           {totals.copied}")
    print(f"  skip (same hash): {totals.skipped_unchanged}")
    print(f"  broken symlinks:  {len(totals.broken_symlinks)}")

    if totals.broken_symlinks:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())