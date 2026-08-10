#!/usr/bin/env python3
"""Move curated lecture-figure source audits after exact asset renaming.

The mapping is produced during a PowerPoint-native migration. Every row must
identify the old and new asset paths and assert that the new asset is an exact
copy of the recorded PowerPoint source object. The script refuses missing
records, path collisions, duplicate mappings, and SHA drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSIGNMENTS = (
    ROOT / ".agents" / "lecture-adaptation" / "figure-source-assignments.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    args = parser.parse_args()

    rows = json.loads(args.mapping.read_text(encoding="utf-8"))
    assignments = json.loads(args.assignments.read_text(encoding="utf-8"))
    figures = assignments["figures"]

    old_paths = [row["old_asset_path"] for row in rows]
    new_paths = [row["new_asset_path"] for row in rows]
    if len(old_paths) != len(set(old_paths)):
        raise SystemExit("mapping repeats an old asset path")
    if len(new_paths) != len(set(new_paths)):
        raise SystemExit("mapping repeats a new asset path")

    for row in rows:
        old_path = row["old_asset_path"]
        new_path = row["new_asset_path"]
        if old_path not in figures:
            raise SystemExit(f"missing curated source record: {old_path}")
        if new_path != old_path and new_path in figures:
            raise SystemExit(f"curated source record already exists: {new_path}")
        if row.get("exact_source_sha_match") is not True:
            raise SystemExit(f"mapping is not an exact source match: {new_path}")
        asset_path = ROOT / new_path
        if not asset_path.is_file():
            raise SystemExit(f"new asset does not exist: {new_path}")
        actual_sha = sha256(asset_path)
        if actual_sha != row.get("new_asset_sha256"):
            raise SystemExit(f"new asset SHA drift: {new_path}")
        if actual_sha != row.get("source_sha256"):
            raise SystemExit(f"new asset differs from its PowerPoint source: {new_path}")

    for row in rows:
        old_path = row["old_asset_path"]
        new_path = row["new_asset_path"]
        figures[new_path] = figures.pop(old_path)

    assignments["figures"] = dict(sorted(figures.items()))
    args.assignments.write_text(
        json.dumps(assignments, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"remapped {len(rows)} curated lecture-figure source records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
