#!/usr/bin/env python3
"""Finalize a mixed PowerPoint-native lecture migration from curated records.

The selection JSON is a list of explicit ``pptx_figures`` records.  This is
used when readable asset names or PowerPoint shape-group exports prevent the
source from being inferred from the filename.  The command verifies every
published path, media relationship, and source shape before removing the old
PDF-region records.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from inventory_lecture_pptx_objects import inventory_slide


FIGURE_RE = re.compile(r'{%\s*include\s+figure\.liquid\b(?P<attrs>.*?)%}')
ATTR_RE = re.compile(r'(?P<key>[\w-]+)="(?P<value>[^"]*)"')
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
VALID_METHODS = {
    "pptx-media-copy",
    "pptx-picture-export",
    "pptx-shape-group-export",
}
VALID_ROLES = {
    "architecture",
    "benchmark-figure",
    "composite",
    "diagram",
    "illustration",
    "photograph",
    "plot",
    "scientific-image",
}


def figures_in_post(post_path: Path) -> set[str]:
    text = post_path.read_text(encoding="utf-8")
    figures = {
        attrs["path"]
        for match in FIGURE_RE.finditer(text)
        for attrs in [
            {
                item.group("key"): item.group("value")
                for item in ATTR_RE.finditer(match.group("attrs"))
            }
        ]
        if attrs.get("path")
    }
    return figures


def slide_media(archive: ZipFile, pptx_slide: int) -> set[str]:
    rel_path = f"ppt/slides/_rels/slide{pptx_slide}.xml.rels"
    if rel_path not in archive.namelist():
        return set()
    root = ET.fromstring(archive.read(rel_path))
    base = posixpath.dirname(f"ppt/slides/slide{pptx_slide}.xml")
    return {
        posixpath.normpath(posixpath.join(base, rel.attrib.get("Target", "")))
        for rel in root.findall(f"{{{REL_NS}}}Relationship")
        if rel.attrib.get("TargetMode") != "External"
        and posixpath.normpath(
            posixpath.join(base, rel.attrib.get("Target", ""))
        ).startswith("ppt/media/")
    }


def shape_names(pptx_path: Path, pptx_slide: int) -> set[str]:
    inventory = inventory_slide(pptx_path, pptx_slide)
    names: set[str] = set()

    def visit(objects: list[dict]) -> None:
        for item in objects:
            if item.get("shape_name"):
                names.add(item["shape_name"])
            visit(item.get("children", []))

    visit(inventory["objects"])
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--post", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = json.loads(args.selection.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("selection JSON must be a list")

    pptx_path = Path(manifest["source_pptx"])
    logical_to_pptx = {
        int(item["slide"]): int(item.get("pptx_slide") or item["slide"])
        for item in manifest["slides"]
    }
    expected_prefix = f"assets/img/blog/lectures/{manifest['deck_id']}/"
    selected_paths: set[str] = set()
    shape_cache: dict[int, set[str]] = {}

    with ZipFile(pptx_path) as archive:
        archive_names = set(archive.namelist())
        for record in records:
            slide = int(record["slide"])
            pptx_slide = logical_to_pptx.get(slide)
            if pptx_slide is None or int(record["pptx_slide"]) != pptx_slide:
                raise SystemExit(f"invalid PowerPoint slide mapping for slide {slide}")
            asset_path = record["asset_path"]
            if not asset_path.startswith(expected_prefix):
                raise SystemExit(f"{asset_path}: expected prefix {expected_prefix}")
            if asset_path in selected_paths:
                raise SystemExit(f"duplicate selected asset: {asset_path}")
            selected_paths.add(asset_path)
            if not Path(asset_path).exists():
                raise SystemExit(f"missing selected asset: {asset_path}")
            if record.get("extraction_method") not in VALID_METHODS:
                raise SystemExit(f"invalid extraction method for {asset_path}")
            if record.get("content_role") not in VALID_ROLES:
                raise SystemExit(f"invalid content role for {asset_path}")

            related_media = slide_media(archive, pptx_slide)
            for media_path in record.get("source_media_paths", []):
                if media_path not in archive_names:
                    raise SystemExit(f"missing PPTX media source: {media_path}")
                if media_path not in related_media:
                    raise SystemExit(
                        f"{media_path} is not related to PPTX slide {pptx_slide}"
                    )

            shape_name = record.get("source_shape_name")
            if shape_name:
                if pptx_slide not in shape_cache:
                    shape_cache[pptx_slide] = shape_names(pptx_path, pptx_slide)
                if shape_name not in shape_cache[pptx_slide]:
                    raise SystemExit(
                        f"shape {shape_name!r} is absent from PPTX slide {pptx_slide}"
                    )

    reused_media = {
        item["asset_path"]
        for item in manifest.get("media", [])
        if item.get("reuse_status") == "reused"
    }
    post_paths = figures_in_post(args.post)
    recorded_paths = reused_media | selected_paths
    if post_paths != recorded_paths:
        raise SystemExit(
            "manifest/post mismatch: "
            f"missing={sorted(recorded_paths - post_paths)}, "
            f"extra={sorted(post_paths - recorded_paths)}"
        )

    manifest["pptx_figures"] = records
    manifest.pop("pdf_regions", None)
    manifest["asset_migration_status"] = "pptx-native-complete"
    manifest["figure_coverage"] = {
        "unique_substantive_figures": len(recorded_paths),
        "reused_figures": len(recorded_paths),
        "redundant_figures": 0,
        "reuse_ratio": 1.0,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"recorded {len(records)} curated PPTX figures in {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
